import os
import re
import gc
import torch
import numpy as np
from pathlib import Path
from loguru import logger

from datasets import load_from_disk, Dataset, Audio, concatenate_datasets
from pydub import AudioSegment, silence
from spleeter.separator import Separator

import soundfile as sf
import librosa
import noisereduce as nr

# Silences params (ms / dB)
MIN_SILENCE_LEN = 1000
SILENCE_THRESH  = -40
KEEP_SILENCE    = 200

# Tmp dir
BASE_TMP_DIR = os.getenv("TMPDIR", "/tmp") + "/clean_audio_pool"
os.makedirs(BASE_TMP_DIR, exist_ok=True)

# AWS credentials for HuggingFace dataset I/O
storage_options = {
    "key": os.getenv("AWS_ACCESS_KEY_ID"),
    "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
    "client_kwargs": {"endpoint_url": os.getenv("AWS_ENDPOINT_URL_S3")}
}

# Init Spleeter once
separator = Separator('spleeter:2stems')


def clean_audio(example, idx):
    audio_np = example["audio"]["array"]
    sr = example["audio"]["sampling_rate"]
    basename = f"audio_{idx:06d}"
    wav_path = os.path.join(BASE_TMP_DIR, f"{basename}.wav")
    out_dir = os.path.join(BASE_TMP_DIR, basename)

    try:
        sf.write(wav_path, audio_np, sr)
        separator.separate_to_file(wav_path, BASE_TMP_DIR)

        vocals_path = os.path.join(out_dir, "vocals.wav")
        if not os.path.isfile(vocals_path):
            raise FileNotFoundError(f"{vocals_path} missing")

        y, sr_loaded = librosa.load(vocals_path, sr=None)
        if y.ndim == 2:
            y = y.mean(axis=1)

        y_denoised = nr.reduce_noise(y=y, sr=sr_loaded)

    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass

    pcm16 = np.clip(y_denoised * 32767, -32768, 32767).astype(np.int16)
    seg = AudioSegment(
        pcm16.tobytes(),
        frame_rate=sr_loaded,
        sample_width=2,
        channels=1
    )

    chunks = silence.split_on_silence(
        seg,
        min_silence_len=MIN_SILENCE_LEN,
        silence_thresh=SILENCE_THRESH,
        keep_silence=KEEP_SILENCE
    )
    seg_clean = sum(chunks) if chunks else seg

    arr = np.array(seg_clean.get_array_of_samples(), dtype=np.float32) / 32767.0
    return {"clean": {"array": arr, "sampling_rate": sr_loaded}}


def extraire_id(texte):
    m = re.search(r"(\d+)[A-Za-z]$", texte)
    return m.group(1) if m else None


def calculate_duration(audio_array, sr):
    return round(len(audio_array) / sr, 2)


def add_duration(example):
    example["duration"] = calculate_duration(example["audio"]["array"], example["audio"]["sampling_rate"])
    return example


def remove_duplicates(dataset):
    seen_texts = {}
    indices_to_keep = []

    for i, text in enumerate(dataset['text']):
        normalized_text = ' '.join(text.lower().split())
        if normalized_text not in seen_texts:
            seen_texts[normalized_text] = i
            indices_to_keep.append(i)

    logger.info(f"Before dedup: {len(dataset)} | After: {len(indices_to_keep)}")
    return dataset.select(indices_to_keep)


def process_saved_devinettes():
    logger.info("=== Processing Devinettes dataset ===")
    try:
        raw_path = "s3://burkimbia/audios/cooked/mooreburkina/devinettes_raw"
        ds = load_from_disk(raw_path, storage_options=storage_options).select(range(20))

        ds = ds.map(lambda x: {"group": extraire_id(x["id"])})
        ds = ds.map(add_duration)

        logger.info("Audio cleaning...")
        ds = ds.cast_column("audio", Audio(sampling_rate=44000))
        ds = ds.map(clean_audio, with_indices=True)
        ds = ds.cast_column("clean", Audio(sampling_rate=44000))

        processed_path = "s3://burkimbia/audios/cooked/mooreburkina/devinettes_processed"
        ds.save_to_disk(processed_path, storage_options=storage_options)

        logger.info(f"Saved to {processed_path}")
        return True

    except Exception as e:
        logger.error(f"Error during processing: {e}")
        return False


if __name__ == "__main__":
    try:
        # crawl_success = crawl_and_save_devinettes()
        # if not crawl_success:
        #     logger.error("Crawling phase failed")
        #     exit(1)
            
        process_success = process_saved_devinettes()
        if process_success:
            logger.info("All done.")
        else:
            logger.error("Process failed.")
            exit(1)
    except Exception:
        logger.error("Fatal error")
        exit(1)
