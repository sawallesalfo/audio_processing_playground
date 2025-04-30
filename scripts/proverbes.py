import os
import re
from loguru import logger

import torch
import numpy as np
from pydub import AudioSegment, silence

from datasets import load_dataset, Dataset, Audio, concatenate_datasets
from resemble_enhance.enhancer.inference import denoise
from langdetect import detect

from mooreburkina.utils import build_dataset, crawl_and_collect


MIN_SILENCE_LEN = 1000  # ms
SILENCE_THRESH   = -40  # dBFS
KEEP_SILENCE     = 200  # ms

def clean_audio(example):
    # 1️⃣ Denoising
    # ———— Convert device to a torch.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ———— Turn the NumPy array into a float32 Torch tensor on the right device
    audio_np = example["audio"]["array"]
    sr       = example["audio"]["sampling_rate"]
    audio_t  = torch.from_numpy(audio_np).float().to(device)

    # ———— Call denoise with a proper torch.Tensor
    denoised_t? sr  = denoise(audio_t, sr, device=device)

    # ———— Bring it back to CPU NumPy for Pydub
    denoised_np = denoised_t.detach().cpu().numpy()

    # 2️⃣ To AudioSegment
    denoised_int16 = (denoised_np * 32767).astype(np.int16)
    seg = AudioSegment(
        denoised_int16.tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=1
    )

    # 3️⃣ Split on long silences
    chunks = silence.split_on_silence(
        seg,
        min_silence_len=MIN_SILENCE_LEN,
        silence_thresh=SILENCE_THRESH,
        keep_silence=KEEP_SILENCE
    )
    seg_clean = sum(chunks) if chunks else seg

    # 4️⃣ Back to normalized float32 NumPy
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
    BASE_URLS = [f"https://media.ipsapps.org/mos/ora/p{i}/01-001-001.html" for i in range(1, 12)]
    datasets = []
    for url in BASE_URLS:
        logger.info(f"=== Scraping {url} ===")
        recs = crawl_and_collect(url)
        if recs:
            ds = build_dataset(recs)
            if ds: datasets.append(ds)
        break
    ds_full = concatenate_datasets(datasets)

    ds_full = ds_full.map(lambda x: {"group": extraire_id(x["id"])})
    ds_full = ds_full.map(lambda x: {"french_map": is_french(x["text"])})
    ds_full = ds_full.map(add_duration_to_dataset)

    ds_segments = find_language_and_group_segments(ds_full)

    ds_cleaned = ds_segments.cast_column("audio", Audio(sampling_rate=16000)) \
                             .map(clean_audio)

    ds_cleaned = ds_cleaned.cast_column("clean", Audio(sampling_rate=16000))

    # 6. Stats & push
    logger.info(f"Durée totale nettoyée : {sum(ds_cleaned['duration']):.2f}s")
    ds_cleaned.push_to_hub(
        "sawadogosalif/proverbes_clean",
        private=True, token=os.environ["HF_TOKEN"]
    )
    ds_full.push_to_hub(
        "sawadogosalif/proverbes",
        private=True, token=os.environ["HF_TOKEN"]
    )
