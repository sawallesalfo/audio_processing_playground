import os
import re
import gc
from loguru import logger
import torch
import numpy as np
from pydub import AudioSegment, silence

from datasets import load_from_disk, Dataset, Audio, concatenate_datasets
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
    sr = example["audio"]["sampling_rate"]

    # 1. Créer un fichier temporaire pour l'audio d'entrée
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        sf.write(tmpfile.name, audio_np, sr)
        input_wav_path = tmpfile.name

    # 2. Utiliser Spleeter pour séparer la voix de la musique
    from spleeter.separator import Separator
    separator = Separator('spleeter:2stems')
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        separator.separate_to_file(input_wav_path, tmpdirname)
        # Récupérer le fichier vocals.wav généré par Spleeter
        vocals_path = os.path.join(tmpdirname, os.path.splitext(os.path.basename(input_wav_path))[0], "vocals.wav")
        # Charger le fichier vocal
        vocals_wav, sr = librosa.load(vocals_path, sr=None)

        # 3. Appliquer noise reduce sur la voix isolée
        import noisereduce as nr
        vocals_denoised = nr.reduce_noise(y=vocals_wav, sr=sr)
        
        # Sauvegarder temporairement pour resemble-enhance
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            sf.write(tmpfile.name, vocals_denoised, sr)
            denoised_path = tmpfile.name

    # Supprimer le fichier d'entrée original
    os.unlink(input_wav_path)

    # Convertir le signal débruité en format pour pydub
    denoised_int16 = (vocals_denoised * 32767).astype(np.int16)
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
            "duration": calculate_duration(combined, sr),
            "Genre": ds[start]["Genre"],      # Add this line
            "Auteurs": ds[start]["Auteurs"]   # Add this line
        })

    new_ds = Dataset.from_dict({
        "group":      [s["group"] for s in segments],
        "french_map": [s["is_french"] for s in segments],
        "text":       [s["text"] for s in segments],
        "audio":      [s["audio"] for s in segments],
        "duration":   [s["duration"] for s in segments],
        "Genre":      [s["Genre"] for s in segments],     # Add this line
        "Auteurs":    [s["Auteurs"] for s in segments],   # Add this line
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


def remove_duplicates(dataset):
    """Supprime les doublons basés sur le texte en gardant la première occurrence."""
    seen_texts = {}
    indices_to_keep = []
    
    for i, text in enumerate(dataset['text']):
        # Normaliser le texte pour la comparaison (enlever espaces supplémentaires, mettre en minuscules)
        normalized_text = ' '.join(text.lower().split())
        if normalized_text not in seen_texts:
            seen_texts[normalized_text] = i
            indices_to_keep.append(i)
    
    logger.info(f"Nombre d'échantillons avant déduplication: {len(dataset)}")
    logger.info(f"Nombre d'échantillons après déduplication: {len(indices_to_keep)}")
    
    return dataset.select(indices_to_keep)

def crawl_and_save_devinettes():
    """Crawl Devinettes data and save immediately"""
    logger.info("=== PHASE: Crawling DEVINETTES data ===")
    BASE_URL_DEVINETTES = "https://media.ipsapps.org/mos/ora/devin/"
    datasets = []
    
    for i in range(1, 20):  # 01 to 19
        url = f"{BASE_URL_DEVINETTES}{i:02d}-B{i:03d}-001.html"
        logger.info(f"Scraping Devinettes {url}")
        recs = crawl_and_collect(url)
        if recs:
            ds = build_dataset(recs)
            if ds:
                # Supprimer les doublons dans chaque sous-ensemble
                ds = remove_duplicates(ds)
                # Ajouter l'information du type
                ds = ds.add_column("type", ["devinette"] * len(ds))
                datasets.append(ds)
                # Clear memory after each URL
                del recs, ds
                gc.collect()
    
    if datasets:
        logger.info("Combining Devinettes datasets")
        ds_full_devinettes = concatenate_datasets(datasets)
        # Supprimer les doublons dans le dataset complet
        ds_full_devinettes = remove_duplicates(ds_full_devinettes)
        logger.info(f"Devinettes dataset final: {len(ds_full_devinettes)} samples")
        
        devinettes_raw_path = "s3://burkimbia/audios/cooked/mooreburkina/devinettes_raw"
        ds_full_devinettes.save_to_disk(devinettes_raw_path, storage_options=storage_options)
        logger.info(f"Saved Devinettes raw dataset to {devinettes_raw_path}")
        
        del datasets, ds_full_devinettes
        gc.collect()
        return True
    else:
        logger.warning("No Devinettes datasets were created")
        return False

def crawl_and_save_rachida():
    """Crawl Rachida data and save immediately"""
    logger.info("=== PHASE 2: Crawling RACHIDA data ===")
    
    # Définir toutes les versions et leurs plages
    versions = [
        {"version": "v09", "range": (1, 15)},  # v09 a 14 fichiers
        {"version": "v10", "range": (1, 22)},  # v10 a 21 fichiers
        {"version": "v11", "range": (1, 22)}   # v11 a 21 fichiers
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
                    # Supprimer les doublons dans chaque sous-ensemble
                    ds = remove_duplicates(ds)
                    # Ajouter l'information de la version
                    ds = ds.add_column("version", [version] * len(ds))
                    datasets.append(ds)
                    # Clear memory after each URL
                    del recs, ds
                    gc.collect()
    
    if datasets:
        logger.info("Combining Rachida datasets")
        ds_full_rachida = concatenate_datasets(datasets)
        # Supprimer les doublons dans le dataset complet
        ds_full_rachida = remove_duplicates(ds_full_rachida)
        logger.info(f"Rachida dataset final: {len(ds_full_rachida)} samples")
        
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
    
    # Load Devinettes if exists
    try:
        devinettes_raw_path = "s3://burkimbia/audios/cooked/mooreburkina/devinettes_raw"
        ds_devinettes = load_from_disk(devinettes_raw_path, storage_options=storage_options)
        logger.info(f"Loaded Devinettes dataset: {len(ds_devinettes)} samples")
        
        # Process Devinettes
        ds_devinettes = ds_devinettes.map(lambda x: {"group": extraire_id(x["id"])})
        ds_devinettes = ds_devinettes.map(lambda x: {"french_map": is_french(x["text"])})
        ds_devinettes = ds_devinettes.map(add_duration_to_dataset)
        ds_devinettes = find_language_and_group_segments(ds_devinettes)
        ds_devinettes = ds_devinettes.filter(lambda x: is_french(x["text"])==False)
        logger.info(f"Processed Devinettes dataset: {len(ds_devinettes)} samples")
        
    except Exception as e:
        logger.warning(f"Could not load Devinettes dataset: {e}")
        ds_devinettes = None

    # Load Thimote if exists
    try:
        thimote_raw_path = "s3://burkimbia/audios/cooked/mooreburkina/thimote_raw"
        ds_thimote = load_from_disk(thimote_raw_path, storage_options=storage_options)
        logger.info(f"Loaded Thimote dataset: {len(ds_thimote)} samples")
        
        # Process Thimote
        ds_thimote = ds_thimote.map(lambda x: {"group": extraire_id(x["id"])})
        ds_thimote = ds_thimote.map(lambda x: {"french_map": is_french(x["text"])})
        ds_thimote = ds_thimote.map(add_duration_to_dataset)
        ds_thimote = ds_thimote.add_column("Genre", ["male"]*len(ds_thimote))
        ds_thimote = ds_thimote.add_column("Auteurs", ["Thimote"]*len(ds_thimote))
        logger.info("Grouping language segments")
        ds_thimote = find_language_and_group_segments(ds_thimote)
        logger.info("filtering moore sample")
        logger.info(f"dataset length before: {len(ds_thimote)}")
        ds_thimote = ds_thimote.filter(lambda x: is_french(x["text"])==False)
        logger.info(f"dataset length after: {len(ds_thimote)}")

        logger.info("Processed Thimote dataset")
        
    except Exception as e:
        logger.warning(f"Could not load Thimote dataset: {e}")
        ds_thimote = None
    
    # Load Rachida if exists
    try:
        ds_rachida_tmps = []
        rachida_raw_path = "s3://burkimbia/audios/cooked/mooreburkina/rachida_raw"
        ds_rachida = load_from_disk(rachida_raw_path, storage_options=storage_options)
        logger.info(f"Loaded Rachida dataset: {len(ds_rachida)} samples")
        
        # Process Rachida
        ds_rachida = ds_rachida.add_column("Genre", ["female"]*len(ds_rachida))
        ds_rachida = ds_rachida.add_column("Auteurs", ["Rachida"]*len(ds_rachida))
        ds_rachida = ds_rachida.map(lambda x: {"group": extraire_id(x["id"])})
        ds_rachida = ds_rachida.map(lambda x: {"french_map": is_french(x["text"])})

        # Let's do loop to avoid error 137
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
        for i in range(0, len(ds_rachida), 400):
            start = i
            end = min(i + 400, len(ds_rachida))
            logger.info(f"Grouping language segments {start} to {end}")
            ds_rachida_tmp = find_language_and_group_segments(ds_rachida.select(range(start, end)))
            ds_rachida_tmp = ds_rachida_tmp.filter(lambda x: is_french(x["text"])==False)
            ds_rachida_tmps.append(ds_rachida_tmp)
            del ds_rachida_tmp
            gc.collect()
        ds_rachida = concatenate_datasets(ds_rachida_tmps)
        logger.info("Processed Rachida dataset")
        
    except Exception as e:
        logger.warning(f"Could not load Rachida dataset: {e}")
        ds_rachida = None
    
    # Combine all available datasets
    datasets_to_combine = []
    if ds_thimote is not None:
        datasets_to_combine.append(ds_thimote)
    if ds_rachida is not None:
        datasets_to_combine.append(ds_rachida)
    if ds_devinettes is not None:
        datasets_to_combine.append(ds_devinettes)

    if datasets_to_combine:
        logger.info(f"Combining {len(datasets_to_combine)} datasets")
        ds_combined = concatenate_datasets(datasets_to_combine)
    else:
        logger.error("No datasets available to combine")
        return False
    
    # Save combined raw dataset
    combined_raw_path = "s3://burkimbia/audios/cooked/mooreburkina/proverbes_raw"
    ds_combined.save_to_disk(combined_raw_path, storage_options=storage_options)
    logger.info(f"Saved combined raw dataset: {len(ds_combined)} samples")
    
    # Clear memory before segmentation
    gc.collect()
    
    # Split dataset into two parts to handle memory issues
    total_samples = len(ds_combined)
    mid_point = total_samples // 2
    
    logger.info(f"Splitting dataset into two parts: Part 1 (0-{mid_point}), Part 2 ({mid_point}-{total_samples})")
    
    # Process Part 1
    logger.info("=== Processing Part 1 ===")
    ds_part1 = ds_combined.select(range(0, mid_point))
    logger.info(f"Part 1 size: {len(ds_part1)} samples")
    
    # Clean audio for Part 1
    logger.info("Starting audio cleaning process for Part 1")
    ds_part1 = ds_part1.cast_column("audio", Audio(sampling_rate=16000))
    ds_part1 = ds_part1.map(clean_audio, batch_size=4)  # Reduced batch size for memory
    ds_part1 = ds_part1.cast_column("clean", Audio(sampling_rate=16000))
    
    logger.info(f"Part 1 cleaned duration: {sum(ds_part1['duration']):.2f}s")
    
    # Save Part 1
    part1_path = "s3://burkimbia/audios/cooked/mooreburkina/proverbes_part_1"
    ds_part1.save_to_disk(part1_path, storage_options=storage_options)
    logger.info(f"Saved Part 1 cleaned dataset to {part1_path}")
    
    # Store part1 duration before deleting
    part1_duration = sum(ds_part1['duration'])
    
    # Clear memory after Part 1
    del ds_part1
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Process Part 2
    logger.info("=== Processing Part 2 ===")
    ds_part2 = ds_combined.select(range(mid_point, total_samples))
    logger.info(f"Part 2 size: {len(ds_part2)} samples")
    
    # Clear memory
    del ds_combined
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Clean audio for Part 2
    logger.info("Starting audio cleaning process for Part 2")
    ds_part2 = ds_part2.cast_column("audio", Audio(sampling_rate=16000))
    ds_part2 = ds_part2.map(clean_audio, batch_size=4)  # Reduced batch size for memory
    ds_part2 = ds_part2.cast_column("clean", Audio(sampling_rate=16000))
    
    logger.info(f"Part 2 cleaned duration: {sum(ds_part2['duration']):.2f}s")
    
    # Save Part 2
    part2_path = "s3://burkimbia/audios/cooked/mooreburkina/proverbes_part_2"
    ds_part2.save_to_disk(part2_path, storage_options=storage_options)
    logger.info(f"Saved Part 2 cleaned dataset to {part2_path}")
    
    # Calculate total duration
    total_duration = sum(ds_part2['duration']) + part1_duration
    logger.info(f"Total cleaned duration across both parts: {total_duration:.2f}s")
    
    # Clear memory
    del ds_part2
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    logger.info("Successfully split and processed dataset into two parts")
    return True


def combine_parts_if_needed():
    """Optional function to combine the parts back together if memory allows later"""
    logger.info("=== COMBINING PARTS (Optional) ===")
    
    try:
        # Load both parts
        part1_path = "s3://burkimbia/audios/cooked/mooreburkina/proverbes_part_1"
        part2_path = "s3://burkimbia/audios/cooked/mooreburkina/proverbes_part_2"
        
        ds_part1 = load_from_disk(part1_path, storage_options=storage_options)
        ds_part2 = load_from_disk(part2_path, storage_options=storage_options)
        
        logger.info(f"Loaded Part 1: {len(ds_part1)} samples")
        logger.info(f"Loaded Part 2: {len(ds_part2)} samples")
        
        # Combine
        ds_combined = concatenate_datasets([ds_part1, ds_part2])
        logger.info(f"Combined dataset: {len(ds_combined)} samples")
        
        # Save combined
        final_path = "s3://burkimbia/audios/cooked/mooreburkina/proverbes"
        ds_combined.save_to_disk(final_path, storage_options=storage_options)
        logger.info(f"Saved final combined dataset to {final_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Could not combine parts: {e}")
        logger.info("Parts remain separate - you can use them individually")
        return False


if __name__ == "__main__":
    try:
        thimote_success = crawl_and_save_thimote()
        rachida_success = crawl_and_save_rachida()
        devinettes_success = crawl_and_save_devinettes()
        
        if not thimote_success and not rachida_success and not devinettes_success:
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
