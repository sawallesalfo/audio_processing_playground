import os
import re
import gc
from loguru import logger

import torch
import numpy as np
from pydub import AudioSegment, silence

from datasets import load_from_disk, Dataset, Audio, concatenate_datasets
from resemble_enhance.enhancer.inference import denoise
from langdetect import detect

from utils import build_dataset, crawl_and_collect

MIN_SILENCE_LEN = 1000  # ms
SILENCE_THRESH   = -40  # dBFS
KEEP_SILENCE     = 200  # ms
import torchaudio

import tempfile
import soundfile as sf

# S3 storage options
storage_options = {
    "key": os.getenv("AWS_ACCESS_KEY_ID"),
    "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
    "client_kwargs": {"endpoint_url": os.getenv("AWS_ENDPOINT_URL_S3")}
}

def clean_audio(example):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # — Extraire les données audio
    audio_np = example["audio"]["array"]
    sr       = example["audio"]["sampling_rate"]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        sf.write(tmpfile.name, audio_np, sr)
        tmp_wav_path = tmpfile.name

    wav, sr = torchaudio.load(tmp_wav_path)  # wav : (1, n) ou (2, n)
    wav = wav.mean(dim=0)  # mono

    denoised, sr = denoise(wav.to(device), sr, device=device)
    denoised_np = denoised.cpu().numpy()

    denoised_int16 = (denoised_np * 32767).astype(np.int16)
    seg = AudioSegment(
        denoised_int16.tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=1
    )

    # 5️⃣ Split on silence
    chunks = silence.split_on_silence(
        seg,
        min_silence_len=MIN_SILENCE_LEN,
        silence_thresh=SILENCE_THRESH,
        keep_silence=KEEP_SILENCE
    )
    seg_clean = sum(chunks) if chunks else seg

    # 6️⃣ Retour en float32 normalisé
    arr = np.array(seg_clean.get_array_of_samples()).astype(np.float32) / 32767.0

    # Clean up temporary file
    os.unlink(tmp_wav_path)

    return {
        "clean": {
            "array": arr,
            "sampling_rate": sr
        }
    }


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
            "duration": calculate_duration(combined, sr)
        })

    new_ds = Dataset.from_dict({
        "group":      [s["group"] for s in segments],
        "french_map": [s["is_french"] for s in segments],
        "text":       [s["text"] for s in segments],
        "audio":      [s["audio"] for s in segments],
        "duration":   [s["duration"] for s in segments],
    })
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
    
    for url in BASE_URLS_THIMOTE:
        logger.info(f"Scraping {url}")
        recs = crawl_and_collect(url)
        if recs:
            ds = build_dataset(recs)
            if ds: 
                datasets.append(ds)
                # Clear memory after each URL
                del recs, ds
                gc.collect()
    
    if datasets:
        logger.info("Combining Thimote datasets")
        ds_full_thimote = concatenate_datasets(datasets)
        logger.info(f"Thimote dataset: {len(ds_full_thimote)} samples")
        
        thimote_raw_path = "s3://burkimbia/audios/cooked/mooreburkina/thimote_raw"
        ds_full_thimote.save_to_disk(thimote_raw_path, storage_options=storage_options)
        logger.info(f"Saved Thimote raw dataset to {thimote_raw_path}")
        
        del datasets, ds_full_thimote
        gc.collect()
        return True
    else:
        logger.warning("No Thimote datasets were created")
        return False


def crawl_and_save_rachida():
    """Crawl Rachida data and save immediately"""
    logger.info("=== PHASE 2: Crawling RACHIDA data ===")
    BASE_URL_RACHIDA = "https://media.ipsapps.org/mos/ora/prv-v10/"
    datasets = []
    
    for i in range(1, 22):  # 01 to 21
        url = f"{BASE_URL_RACHIDA}{i:02d}-B{i:03d}-001.html"
        logger.info(f"Scraping Rachida {url}")
        recs = crawl_and_collect(url)
        if recs:
            ds = build_dataset(recs)
            if ds: 
                datasets.append(ds)
                # Clear memory after each URL
                del recs, ds
                gc.collect()
    
    if datasets:
        logger.info("Combining Rachida datasets")
        ds_full_rachida = concatenate_datasets(datasets)
        logger.info(f"Rachida dataset: {len(ds_full_rachida)} samples")
        
        rachida_raw_path = "s3://burkimbia/audios/cooked/mooreburkina/rachida_raw"
        ds_full_rachida.save_to_disk(rachida_raw_path, storage_options=storage_options)
        logger.info(f"Saved Rachida raw dataset to {rachida_raw_path}")
        
        del datasets, ds_full_rachida
        gc.collect()
        return True
    else:
        logger.warning("No Rachida datasets were created")
        return False


def process_saved_datasets():
    """Load saved datasets and process them"""
    logger.info("=== PHASE 3: Processing saved datasets ===")
    
    # Load datasets from disk
    datasets_to_combine = []
    
    # Load Thimote if exists
    try:
        thimote_raw_path = "s3://burkimbia/audios/cooked/mooreburkina/thimote_raw"
        ds_thimote = load_from_disk(thimote_raw_path, storage_options=storage_options)
        logger.info(f"Loaded Thimote dataset: {len(ds_thimote)} samples")
        
        # Process Thimote
        ds_thimote = ds_thimote.map(lambda x: {"group": extraire_id(x["id"])})
        ds_thimote = ds_thimote.map(lambda x: {"french_map": is_french(x["text"])})
        ds_thimote = ds_thimote.map(add_duration_to_dataset)
        ds_thimote = ds_thimote.add_column("Genre", ["Homme"]*len(ds_thimote))
        ds_thimote = ds_thimote.add_column("Auteurs", ["Thimote"]*len(ds_thimote))
        logger.info("Grouping language segments")
        ds_thimote = find_language_and_group_segments(ds_thimote)
        logger.info("Processed Thimote dataset")
        
    except Exception as e:
        logger.warning(f"Could not load Thimote dataset: {e}")
    
    # Load Rachida if exists
    try:
        ds_rachida_tmps = []
        rachida_raw_path = "s3://burkimbia/audios/cooked/mooreburkina/rachida_raw"
        ds_rachida = load_from_disk(rachida_raw_path, storage_options=storage_options)
        logger.info(f"Loaded Rachida dataset: {len(ds_rachida)} samples")
        
        # Process Rachida
        ds_rachida = ds_rachida.add_column("Genre", ["Femme"]*len(ds_rachida))
        ds_rachida = ds_rachida.add_column("Auteurs", ["Rachida"]*len(ds_rachida))
        ds_rachida = ds_rachida.map(lambda x: {"group": extraire_id(x["id"])})
        ds_rachida = ds_rachida.map(lambda x: {"french_map": is_french(x["text"])})
        
        # Let's do loop to avaoid error 137

        for i in range(0, len(ds_rachida), 100):
            start = i
            end = min(i + 100, len(ds_rachida))  # Avoid going out of bounds
            logger.info(f"Processing Rachida segment {start} to {end}")
            ds_rachida_tmp = ds_rachida.select(range(start, end)).map(add_duration_to_dataset, num_proc=4)
            ds_rachida_tmps.append(ds_rachida_tmp)
            del ds_rachida_tmp
            gc.collect()
        ds_rachida = concatenate_datasets(ds_rachida_tmps)
        ds_rachida_tmps = []
        for i in range(0, len(ds_rachida), 500):
            logger.info(f"Grouping language segments {start} to {end}")
            start = i
            end = min(i + 500, len(ds_rachida))
            ds_rachida_tmps.append(find_language_and_group_segments(ds_rachida.select(range(start, end))))
        ds_rachida = concatenate_datasets(ds_rachida_tmps)
        logger.info("Processed Rachida dataset")
        
    except Exception as e:
        logger.warning(f"Could not load Rachida dataset: {e}")
    
    if not datasets_to_combine:
        logger.error("No datasets could be loaded for processing")
        return False
    
    # Combine datasets
    logger.info("Combining processed datasets")
    ds_combined = concatenate_datasets([ds_thimote, ds_rachida])
    
    # Save combined raw dataset
    combined_raw_path = "s3://burkimbia/audios/cooked/mooreburkina/proverbes_raw"
    ds_combined.save_to_disk(combined_raw_path, storage_options=storage_options)
    logger.info(f"Saved combined raw dataset: {len(ds_combined)} samples")
    
    # Clear memory before segmentation
    gc.collect()
    # Audio cleaning
    logger.info("Starting audio cleaning process")
    ds_combined = ds_combined.cast_column("audio", Audio(sampling_rate=16000))
    ds_combined = ds_combined.map(clean_audio, batch_size=1)  # Process one at a time to save memory    
    ds_combined = ds_combined.cast_column("clean", Audio(sampling_rate=16000))
    
    logger.info(f"Total cleaned duration: {sum(ds_combined['duration']):.2f}s")
    
    # Save final dataset
    final_path = "s3://burkimbia/audios/cooked/mooreburkina/proverbes"
    ds_combined.save_to_disk(final_path, storage_options=storage_options)
    logger.info(f"Saved final cleaned dataset to {final_path}")
    
    return True


if __name__ == "__main__":
    try:
        #thimote_success = crawl_and_save_thimote()
        #rachida_success = crawl_and_save_rachida()    
        # if not thimote_success and not rachida_success:
        #     logger.error("Both crawling phases failed")
        #     exit(1)
        
        process_success = process_saved_datasets()
        if process_success:
            logger.info("Pipeline completed successfully!")
        else:
            logger.error("Processing phase failed")
            exit(1)
            
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        exit(1)
