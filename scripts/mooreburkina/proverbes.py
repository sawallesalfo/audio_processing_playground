import os
import re
import gc
from loguru import logger
import torch
import numpy as np
from pydub import AudioSegment, silence
from spleeter.separator import Separator
from pathlib import Path
import shutil

from datasets import load_from_disk, Dataset, Audio, concatenate_datasets
from langdetect import detect
from utils import build_dataset, crawl_and_collect

MIN_SILENCE_LEN = 1000  # ms
SILENCE_THRESH   = -40  # dBFS
KEEP_SILENCE     = 200  # ms

BASE_TMP_DIR = os.getenv("TMPDIR", "/tmp") + "/clean_audio_pool"
os.makedirs(BASE_TMP_DIR, exist_ok=True)

import torchaudio

import soundfile as sf
import librosa
import noisereduce as nr

# S3 storage options
storage_options = {
    "key": os.getenv("AWS_ACCESS_KEY_ID"),
    "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
    "client_kwargs": {"endpoint_url": os.getenv("AWS_ENDPOINT_URL_S3")}
}

def force_cleanup():
    """Aggressive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def cleanup_temp_directory():
    """Clean up temporary directory to free disk space"""
    try:
        if os.path.exists(BASE_TMP_DIR):
            shutil.rmtree(BASE_TMP_DIR)
            os.makedirs(BASE_TMP_DIR, exist_ok=True)
            logger.info(f"Cleaned up temporary directory: {BASE_TMP_DIR}")
    except Exception as e:
        logger.warning(f"Could not clean temp directory: {e}")

# Init Spleeter once but recreate if needed
separator = None

def get_separator():
    """Get separator instance, creating if needed"""
    global separator
    if separator is None:
        separator = Separator('spleeter:2stems')
    return separator

def clean_audio(example, idx):
    audio_np = example["audio"]["array"]
    sr = example["audio"]["sampling_rate"]
    basename = f"audio_{idx:06d}"
    wav_path = os.path.join(BASE_TMP_DIR, f"{basename}.wav")
    out_dir = os.path.join(BASE_TMP_DIR, basename)

    try:
        sf.write(wav_path, audio_np, sr)
        
        # Get separator instance
        sep = get_separator()
        sep.separate_to_file(wav_path, BASE_TMP_DIR)

        vocals_path = os.path.join(out_dir, "vocals.wav")
        if not os.path.isfile(vocals_path):
            raise FileNotFoundError(f"{vocals_path} missing")

        y, sr_loaded = librosa.load(vocals_path, sr=None)
        if y.ndim == 2:
            y = y.mean(axis=1)

        y_denoised = nr.reduce_noise(y=y, sr=sr_loaded)
        
        # Clear intermediate variables
        del y
        
    finally:
        # Clean up all temporary files
        try:
            if os.path.exists(wav_path):
                os.remove(wav_path)
            if os.path.exists(out_dir):
                shutil.rmtree(out_dir)
        except OSError:
            pass

    pcm16 = np.clip(y_denoised * 32767, -32768, 32767).astype(np.int16)
    
    # Clear intermediate variable
    del y_denoised
    
    seg = AudioSegment(
        pcm16.tobytes(),
        frame_rate=sr_loaded,
        sample_width=2,
        channels=1
    )
    
    # Clear intermediate variable
    del pcm16

    chunks = silence.split_on_silence(
        seg,
        min_silence_len=MIN_SILENCE_LEN,
        silence_thresh=SILENCE_THRESH,
        keep_silence=KEEP_SILENCE
    )
    seg_clean = sum(chunks) if chunks else seg
    
    # Clear intermediate variables
    del seg, chunks

    arr = np.array(seg_clean.get_array_of_samples(), dtype=np.float32) / 32767.0
    del seg_clean
    
    # Force garbage collection after each audio processing
    force_cleanup()
    
    return {"clean": {"array": arr, "sampling_rate": sr_loaded}}


def is_french(text: str) -> bool:
    try:
        return detect(text) == "fr" or text.startswith("(") or text.endswith(")")
    except:
        return text.startswith("(") or text.endswith(")")


def extraire_id(texte):
    m = re.search(r"(\d+)[A-Za-z]$", texte)
    return m.group(1) if m else None


def calculate_duration(audio_array, sr):
    return round(len(audio_array) / sr, 2)


def find_language_and_group_segments(ds: Dataset) -> Dataset:
    """Regroupe les segments consécutifs de même langue & même groupe."""
    change_idxs = [0]
    curr_lang  = ds[0]["french_map"]
    curr_grp   = ds[0]["group"]

    for i in range(1, len(ds)):
        if ds[i]["french_map"] != curr_lang or ds[i]["group"] != curr_grp:
            change_idxs.append(i)
            curr_lang = ds[i]["french_map"]
            curr_grp  = ds[i]["group"]
    change_idxs.append(len(ds))

    segments = []
    for start, end in zip(change_idxs, change_idxs[1:]):
        text_concat = " ".join(ds[j]["text"] for j in range(start, end))
        audio_arrays = [ds[j]["audio"]["array"] for j in range(start, end)]
        combined = np.concatenate(audio_arrays).astype(np.float32)
        sr = ds[start]["audio"]["sampling_rate"]

        segments.append({
            "group": ds[start]["group"],
            "is_french": ds[start]["french_map"],
            "text": text_concat,
            "audio": {"array": combined, "sampling_rate": sr},
            "duration": calculate_duration(combined, sr),
            "Genre": ds[start]["Genre"],
            "Auteurs": ds[start]["Auteurs"]
        })
        
        # Clear intermediate variables
        del audio_arrays, combined

    new_ds = Dataset.from_dict({
        "group":      [s["group"] for s in segments],
        "french_map": [s["is_french"] for s in segments],
        "text":       [s["text"] for s in segments],
        "audio":      [s["audio"] for s in segments],
        "duration":   [s["duration"] for s in segments],
        "Genre":      [s["Genre"] for s in segments],
        "Auteurs":    [s["Auteurs"] for s in segments],
    })
    
    # Clear segments list
    del segments
    force_cleanup()
    
    return new_ds.cast_column("audio", Audio(sampling_rate=ds[0]["audio"]["sampling_rate"]))


def add_duration_to_dataset(example):
    arr = example["audio"]["array"]
    sr  = example["audio"]["sampling_rate"]
    return {"duration": len(arr) / sr}


def crawl_and_save_thimote():
    """Crawl Thimote data and save immediately"""
    logger.info("=== PHASE 1: Crawling THIMOTE data ===")
    BASE_URLS_THIMOTE = [f"https://media.ipsapps.org/mos/ora/p{i}/01-001-001.html" for i in range(1, 12)]
    datasets = []
    
    for i, url in enumerate(BASE_URLS_THIMOTE):
        logger.info(f"Scraping {url} ({i+1}/{len(BASE_URLS_THIMOTE)})")
        recs = crawl_and_collect(url)
        if recs:
            ds = build_dataset(recs)
            if ds: 
                datasets.append(ds)
                logger.info(f"Added dataset with {len(ds)} samples")
            # Clear memory after each URL
            del recs, ds
            force_cleanup()
        
        # Periodic cleanup during crawling
        if (i + 1) % 3 == 0:
            logger.info(f"Periodic cleanup after {i+1} URLs")
            force_cleanup()
    
    if datasets:
        logger.info("Combining Thimote datasets")
        ds_full_thimote = concatenate_datasets(datasets)
        logger.info(f"Thimote dataset: {len(ds_full_thimote)} samples")
        
        # Clear datasets list before saving
        del datasets
        force_cleanup()
        
        thimote_raw_path = "s3://burkimbia/audios/cooked/mooreburkina/thimote_raw"
        ds_full_thimote.save_to_disk(thimote_raw_path, storage_options=storage_options)
        logger.info(f"Saved Thimote raw dataset to {thimote_raw_path}")
        
        del ds_full_thimote
        force_cleanup()
        return True
    else:
        logger.warning("No Thimote datasets were created")
        return False


def remove_duplicates(dataset):
    """Supprime les doublons basés sur le texte en gardant la première occurrence."""
    seen_texts = set()
    indices_to_keep = []
    
    for i, text in enumerate(dataset['text']):
        # Normaliser le texte pour la comparaison
        normalized_text = ' '.join(text.lower().split())
        if normalized_text not in seen_texts:
            seen_texts.add(normalized_text)
            indices_to_keep.append(i)
    
    logger.info(f"Samples before deduplication: {len(dataset)}")
    logger.info(f"Samples after deduplication: {len(indices_to_keep)}")
    
    # Clear seen_texts set
    del seen_texts
    force_cleanup()
    
    return dataset.select(indices_to_keep)


def crawl_and_save_rachida():
    """Crawl Rachida data and save immediately"""
    logger.info("=== PHASE 2: Crawling RACHIDA data ===")
    
    versions = [
        {"version": "v09", "range": (1, 15)},
        {"version": "v10", "range": (1, 22)},
        {"version": "v11", "range": (1, 22)}
    ]
    
    datasets = []
    
    for version_info in versions:
        version = version_info["version"]
        start, end = version_info["range"]
        BASE_URL_RACHIDA = f"https://media.ipsapps.org/mos/ora/prv-{version}/"
        
        logger.info(f"=== Crawling Rachida {version} ===")
        for i in range(start, end):
            url = f"{BASE_URL_RACHIDA}{i:02d}-B{i:03d}-001.html"
            logger.info(f"Scraping Rachida {url}")
            recs = crawl_and_collect(url)
            if recs:
                ds = build_dataset(recs)
                if ds:
                    # Remove duplicates in each subset
                    ds = remove_duplicates(ds)
                    # Add version information
                    ds = ds.add_column("version", [version] * len(ds))
                    datasets.append(ds)
                    logger.info(f"Added dataset with {len(ds)} samples")
                # Clear memory after each URL
                del recs, ds
                force_cleanup()
            
            # Periodic cleanup
            if i % 5 == 0:
                logger.info(f"Periodic cleanup at URL {i}")
                force_cleanup()
    
    if datasets:
        logger.info("Combining Rachida datasets")
        ds_full_rachida = concatenate_datasets(datasets)
        
        # Clear datasets list before deduplication
        del datasets
        force_cleanup()
        
        # Remove duplicates in the complete dataset
        ds_full_rachida = remove_duplicates(ds_full_rachida)
        logger.info(f"Rachida dataset final: {len(ds_full_rachida)} samples")
        
        rachida_raw_path = "s3://burkimbia/audios/cooked/mooreburkina/rachida_raw"
        ds_full_rachida.save_to_disk(rachida_raw_path, storage_options=storage_options)
        logger.info(f"Saved Rachida raw dataset to {rachida_raw_path}")
        
        del ds_full_rachida
        force_cleanup()
        return True
    else:
        logger.warning("No Rachida datasets were created")
        return False


def process_saved_datasets():
    """Load saved datasets and process them"""
    logger.info("=== PHASE 3: Processing saved datasets ===")
    datasets_to_combine = []

    # Load Thimote if exists
    try:
        thimote_raw_path = "s3://burkimbia/audios/cooked/mooreburkina/thimote_raw"
        ds_thimote = load_from_disk(thimote_raw_path, storage_options=storage_options)
        logger.info(f"Loaded Thimote dataset: {len(ds_thimote)} samples")
        
        # Process Thimote
        ds_thimote = ds_thimote.map(lambda x: {"group": extraire_id(x["id"])})
        force_cleanup()
        
        ds_thimote = ds_thimote.map(lambda x: {"french_map": is_french(x["text"])})
        force_cleanup()
        
        ds_thimote = ds_thimote.map(add_duration_to_dataset)
        force_cleanup()
        
        ds_thimote = ds_thimote.add_column("Genre", ["male"]*len(ds_thimote))
        ds_thimote = ds_thimote.add_column("Auteurs", ["Thimote"]*len(ds_thimote))
        
        logger.info("Grouping language segments")
        ds_thimote = find_language_and_group_segments(ds_thimote)
        force_cleanup()
        
        logger.info("Filtering Moore samples")
        logger.info(f"Dataset length before: {len(ds_thimote)}")
        ds_thimote = ds_thimote.filter(lambda x: is_french(x["text"])==False)
        logger.info(f"Dataset length after: {len(ds_thimote)}")
        force_cleanup()

        logger.info("Processed Thimote dataset")
        datasets_to_combine.append(ds_thimote)
        
    except Exception as e:
        logger.warning(f"Could not load Thimote dataset: {e}")
        ds_thimote = None
    
    # Load and process Rachida if exists
    try:
        rachida_raw_path = "s3://burkimbia/audios/cooked/mooreburkina/rachida_raw"
        ds_rachida = load_from_disk(rachida_raw_path, storage_options=storage_options)
        logger.info(f"Loaded Rachida dataset: {len(ds_rachida)} samples")
        
        # Process Rachida
        ds_rachida = ds_rachida.add_column("Genre", ["female"]*len(ds_rachida))
        ds_rachida = ds_rachida.add_column("Auteurs", ["Rachida"]*len(ds_rachida))
        ds_rachida = ds_rachida.map(lambda x: {"group": extraire_id(x["id"])})
        force_cleanup()
        
        ds_rachida = ds_rachida.map(lambda x: {"french_map": is_french(x["text"])})
        force_cleanup()

        # Process duration in smaller batches with aggressive cleanup
        ds_rachida_tmps = []
        batch_size = 50  # Reduced batch size
        
        for i in range(0, len(ds_rachida), batch_size):
            start = i
            end = min(i + batch_size, len(ds_rachida))
            logger.info(f"Processing Rachida duration batch {start} to {end}")
            
            ds_rachida_tmp = ds_rachida.select(range(start, end)).map(
                add_duration_to_dataset, 
                num_proc=2  # Reduced from 4 to save memory
            )
            ds_rachida_tmps.append(ds_rachida_tmp)
            del ds_rachida_tmp
            force_cleanup()
        
        # Combine duration-processed batches
        ds_rachida = concatenate_datasets(ds_rachida_tmps)
        del ds_rachida_tmps
        force_cleanup()
        
        # Process language grouping in smaller batches
        ds_rachida_tmps = []
        batch_size = 200  # Reduced batch size
        
        for i in range(0, len(ds_rachida), batch_size):
            start = i
            end = min(i + batch_size, len(ds_rachida))
            logger.info(f"Grouping language segments {start} to {end}")
            
            ds_rachida_tmp = find_language_and_group_segments(
                ds_rachida.select(range(start, end))
            )
            ds_rachida_tmp = ds_rachida_tmp.filter(lambda x: is_french(x["text"])==False)
            ds_rachida_tmps.append(ds_rachida_tmp)
            del ds_rachida_tmp
            force_cleanup()
        
        # Combine language-grouped batches
        ds_rachida = concatenate_datasets(ds_rachida_tmps)
        del ds_rachida_tmps
        force_cleanup()
        
        logger.info("Processed Rachida dataset")
        datasets_to_combine.append(ds_rachida)
        
    except Exception as e:
        logger.warning(f"Could not load Rachida dataset: {e}")
        ds_rachida = None

    if not datasets_to_combine:
        logger.error("No datasets available to combine")
        return False
    
    logger.info(f"Combining {len(datasets_to_combine)} datasets")
    ds_combined = concatenate_datasets(datasets_to_combine)
    
    # Clear datasets_to_combine
    del datasets_to_combine
    if 'ds_thimote' in locals():
        del ds_thimote
    if 'ds_rachida' in locals():
        del ds_rachida
    force_cleanup()
    
    # Save combined raw dataset
    combined_raw_path = "s3://burkimbia/audios/cooked/mooreburkina/proverbes_raw"
    ds_combined.save_to_disk(combined_raw_path, storage_options=storage_options)
    logger.info(f"Saved combined raw dataset: {len(ds_combined)} samples")
    
    # Clean up temporary files before heavy processing
    cleanup_temp_directory()
    force_cleanup()
    
    # Split dataset into smaller parts for memory management
    total_samples = len(ds_combined)
    part_size = total_samples // 4  # Split into 4 parts instead of 2
    
    logger.info(f"Splitting dataset into 4 parts of ~{part_size} samples each")
    
    for part_num in range(4):
        start_idx = part_num * part_size
        end_idx = min((part_num + 1) * part_size, total_samples)
        
        if part_num == 3:  # Last part gets remaining samples
            end_idx = total_samples
            
        logger.info(f"=== Processing Part {part_num + 1} ({start_idx}-{end_idx}) ===")
        
        if start_idx >= end_idx:
            continue
            
        ds_part = ds_combined.select(range(start_idx, end_idx))
        logger.info(f"Part {part_num + 1} size: {len(ds_part)} samples")
        
        # Clean audio for this part
        logger.info(f"Starting audio cleaning for Part {part_num + 1}")
        ds_part = ds_part.cast_column("audio", Audio(sampling_rate=44000))
        
        # Process in even smaller batches for cleaning
        ds_part = ds_part.map(
            clean_audio, 
            batch_size=2,  # Very small batch size for memory
            with_indices=True,
            num_proc=1  # Single process to avoid memory issues
        )
        ds_part = ds_part.cast_column("clean", Audio(sampling_rate=44000))
        
        logger.info(f"Part {part_num + 1} cleaned duration: {sum(ds_part['duration']):.2f}s")
        
        # Save this part
        part_path = f"s3://burkimbia/audios/cooked/mooreburkina/proverbes_part_{part_num + 1}"
        ds_part.save_to_disk(part_path, storage_options=storage_options)
        logger.info(f"Saved Part {part_num + 1} to {part_path}")
        
        # Aggressive cleanup after each part
        del ds_part
        cleanup_temp_directory()
        force_cleanup()
    
    # Final cleanup
    del ds_combined
    force_cleanup()
    
    logger.info("Successfully processed all parts")
    return True


def combine_parts_if_needed():
    """Combine the parts back together if memory allows"""
    logger.info("=== COMBINING PARTS (Optional) ===")
    
    try:
        # Load all parts
        parts = []
        total_duration = 0
        
        for part_num in range(1, 5):
            part_path = f"s3://burkimbia/audios/cooked/mooreburkina/proverbes_part_{part_num}"
            try:
                ds_part = load_from_disk(part_path, storage_options=storage_options)
                parts.append(ds_part)
                part_duration = sum(ds_part['duration'])
                total_duration += part_duration
                logger.info(f"Loaded Part {part_num}: {len(ds_part)} samples, {part_duration:.2f}s")
                del ds_part
                force_cleanup()
            except Exception as e:
                logger.warning(f"Could not load Part {part_num}: {e}")
        
        if not parts:
            logger.error("No parts found to combine")
            return False
        
        # Combine parts
        logger.info(f"Combining {len(parts)} parts")
        ds_combined = concatenate_datasets(parts)
        
        # Clear parts list
        del parts
        force_cleanup()
        
        logger.info(f"Combined dataset: {len(ds_combined)} samples, {total_duration:.2f}s total")
        
        # Save combined dataset
        final_path = "s3://burkimbia/audios/cooked/mooreburkina/proverbes"
        ds_combined.save_to_disk(final_path, storage_options=storage_options)
        logger.info(f"Saved final combined dataset to {final_path}")
        
        del ds_combined
        force_cleanup()
        
        return True
        
    except Exception as e:
        logger.error(f"Could not combine parts: {e}")
        logger.info("Parts remain separate - you can use them individually")
        return False


if __name__ == "__main__":
    try:
        # Force initial cleanup
        force_cleanup()
        
        # thimote_success = crawl_and_save_thimote()
        # rachida_success = crawl_and_save_rachida()
        rachida_success, thimote_success = True, True
        
        if not thimote_success and not rachida_success:
            logger.error("All crawling phases failed")
            exit(1)
        
        process_success = process_saved_datasets()
        if process_success:
            logger.info("Split processing completed successfully!")
            
            combine_success = combine_parts_if_needed()
            if combine_success:
                logger.info("Pipeline completed successfully with combined dataset!")
            else:
                logger.info("Pipeline completed successfully with split datasets!")
        else:
            logger.error("Processing phase failed")
            exit(1)
            
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        exit(1)
    finally:
        # Final cleanup
        cleanup_temp_directory()
        force_cleanup()
        logger.info("Final cleanup completed")
