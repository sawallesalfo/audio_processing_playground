import os
import re
from loguru import logger

import torch
import numpy as np
from pydub import AudioSegment, silence

from datasets import load_dataset, Dataset, Audio, concatenate_datasets
from resemble_enhance.enhancer.inference import denoise
from langdetect import detect

from utils import build_dataset, crawl_and_collect

MIN_SILENCE_LEN = 1000  # ms
SILENCE_THRESH   = -40  # dBFS
KEEP_SILENCE     = 200  # ms
import torchaudio

import tempfile
import soundfile as sf

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


if __name__ == "__main__":
    # THIMOTE
    BASE_URLS_THIMOTE = [f"https://media.ipsapps.org/mos/ora/p{i}/01-001-001.html" for i in range(1, 12)]
    datasets = []
    for url in BASE_URLS_THIMOTE:
        logger.info(f"=== Scraping {url} ===")
        recs = crawl_and_collect(url)
        if recs:
            ds = build_dataset(recs)
            if ds: datasets.append(ds)
    
    if datasets:
        ds_full_thimote = concatenate_datasets(datasets)
        # Add processing for Thimote dataset
        ds_full_thimote = ds_full_thimote.map(lambda x: {"group": extraire_id(x["id"])})
        ds_full_thimote = ds_full_thimote.map(lambda x: {"french_map": is_french(x["text"])})
        ds_full_thimote = ds_full_thimote.map(add_duration_to_dataset)
        ds_full_thimote = ds_full_thimote.add_column("Genre", ["Homme"]*len(ds_full_thimote))
        ds_full_thimote = ds_full_thimote.add_column("Auteurs", ["Thimote"]*len(ds_full_thimote))
    else:
        ds_full_thimote = None
    
    # RACHIDA
    BASE_URL_RACHIDA = "https://media.ipsapps.org/mos/ora/prv-v10/"
    datasets = []
    for i in range(1, 22):  # 01 to 21
        url = f"{BASE_URL_RACHIDA}{i:02d}-B{i:03d}-001.html"
        logger.info(f"=== Scraping Rachida {url} ===")
        recs = crawl_and_collect(url)
        if recs:
            ds = build_dataset(recs)
            if ds: datasets.append(ds)
    
    if datasets:
        ds_full_rachida = concatenate_datasets(datasets)
        ds_full_rachida = ds_full_rachida.add_column("Genre", ["Femme"]*len(ds_full_rachida))
        ds_full_rachida = ds_full_rachida.add_column("Auteurs", ["Rachida"]*len(ds_full_rachida))
        ds_full_rachida = ds_full_rachida.map(lambda x: {"group": extraire_id(x["id"])})
        ds_full_rachida = ds_full_rachida.map(lambda x: {"french_map": is_french(x["text"])})
        ds_full_rachida = ds_full_rachida.map(add_duration_to_dataset)
    else:
        ds_full_rachida = None

    # Combine datasets
    logger.info("Combining datasets")
    if ds_full_thimote is not None and ds_full_rachida is not None:
        ds_combined = concatenate_datasets([ds_full_rachida, ds_full_thimote])
    elif ds_full_rachida is not None:
        ds_combined = ds_full_rachida
    elif ds_full_thimote is not None:
        ds_combined = ds_full_thimote
    else:
        logger.error("No datasets were successfully created")
        exit(1)
    
    logger.info("saving datasets")
    ds_combined.save_to_disk("s3://burkimbia/audios/cooked/mooreburkina/proverbes_raw", storage_options=storage_options)
    logger.info(f"Combined dataset: {len(ds_combined)} samples")
    logger.info(f"Rachidat dataset: {len(ds_full_rachida)} samples")
    logger.info(f"Thimote dataset: {len(ds_full_thimote)} samples")

    ds_segments = find_language_and_group_segments(ds_combined)

    ds_cleaned = ds_segments.cast_column("audio", Audio(sampling_rate=16000)) \
                             .map(clean_audio)

    ds_cleaned = ds_cleaned.cast_column("clean", Audio(sampling_rate=16000))

    logger.info(f"Durée totale nettoyée : {sum(ds_cleaned['duration']):.2f}s")
    storage_options = {
        "key": os.getenv("AWS_ACCESS_KEY_ID"),
        "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "client_kwargs": {"endpoint_url": os.getenv("AWS_ENDPOINT_URL_S3")}
    }
    OUTPUT_DATASET_PATH = "s3://burkimbia/audios/cooked/mooreburkina/proverbes"
    ds_cleaned.save_to_disk(OUTPUT_DATASET_PATH, storage_options=storage_options)
