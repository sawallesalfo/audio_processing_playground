import os
import re
import gc
from loguru import logger
import torch
import numpy as np
from pydub import AudioSegment, silence

from datasets import load_from_disk, Dataset, Audio, concatenate_datasets

from utils import build_dataset, crawl_and_collect

MIN_SILENCE_LEN = 1000  # ms
SILENCE_THRESH   = -40  # dBFS
KEEP_SILENCE     = 200  # ms
import torchaudio
import tempfile
import soundfile as sf
import librosa
import noisereduce as nr
storage_options = {
    "key": os.getenv("AWS_ACCESS_KEY_ID"),
    "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
    "client_kwargs": {"endpoint_url": os.getenv("AWS_ENDPOINT_URL_S3")}
}

def clean_audio(example):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    audio_np = example["audio"]["array"]
    sr = example["audio"]["sampling_rate"]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        sf.write(tmpfile.name, audio_np, sr)
        input_wav_path = tmpfile.name

    # Utiliser Spleeter pour séparer la voix de la musique
    from spleeter.separator import Separator
    separator = Separator('spleeter:2stems')
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        separator.separate_to_file(input_wav_path, tmpdirname)
        # Récupérer le fichier vocals.wav généré par Spleeter
        vocals_path = os.path.join(tmpdirname, os.path.splitext(os.path.basename(input_wav_path))[0], "vocals.wav")
        # Charger le fichier vocal
        vocals_wav, sr = librosa.load(vocals_path, sr=None)

        # 3. Appliquer noise reduce sur la voix isolée
        vocals_denoised = nr.reduce_noise(y=vocals_wav, sr=sr)
        
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

    # Split on silence
    chunks = silence.split_on_silence(
        seg,
        min_silence_len=MIN_SILENCE_LEN,
        silence_thresh=SILENCE_THRESH,
        keep_silence=KEEP_SILENCE
    )
    seg_clean = sum(chunks) if chunks else seg

    # Retour en float32 normalisé
    arr = np.array(seg_clean.get_array_of_samples()).astype(np.float32) / 32767.0

    return {
        "clean": {
            "array": arr,
            "sampling_rate": sr
        }
    }

def extraire_id(texte):
    m = re.search(r"(\d+)[A-Za-z]$", texte)
    return m.group(1) if m else None

def calculate_duration(audio_array, sr):
    return round(len(audio_array) / sr, 2)

def remove_duplicates(dataset):
    """Supprime les doublons basés sur le texte en gardant la première occurrence."""
    seen_texts = {}
    indices_to_keep = []
    
    for i, text in enumerate(dataset['text']):
        # Normaliser le texte pour la comparaison
        normalized_text = ' '.join(text.lower().split())
        if normalized_text not in seen_texts:
            seen_texts[normalized_text] = i
            indices_to_keep.append(i)
    
    logger.info(f"Nombre d'échantillons avant déduplication: {len(dataset)}")
    logger.info(f"Nombre d'échantillons après déduplication: {len(indices_to_keep)}")
    
    return dataset.select(indices_to_keep)

def crawl_and_save_devinettes():
    """Crawl les devinettes et sauvegarde immédiatement"""
    logger.info("=== Crawling DEVINETTES data ===")
    
    # Générer les URLs pour les devinettes (de 1 à 19)
    BASE_URLS = []
    for i in range(1, 20):  # de 1 à 19
        volume_num = str(i).zfill(2)  # Pad avec des zéros: 1 -> 01, 2 -> 02, etc.
        volume_name = f"devin_{volume_num}"
        url = f"https://media.ipsapps.org/mos/ora/devin/{volume_num}-B{volume_num}-001.html"
        BASE_URLS.append((volume_name, url))
    
    logger.info(f"Nombre total d'URLs à traiter: {len(BASE_URLS)}")
    datasets = []
    
    
    for volume, url in BASE_URLS:
        logger.info(f"Scraping Devinettes {volume} - {url}")
        recs = crawl_and_collect(url)
        if recs:
            ds = build_dataset(recs)
            if ds:
                ds = remove_duplicates(ds)
                ds = ds.add_column("genre", ["female"] * len(ds))
                ds = ds.add_column("auteur", ["Rachida"] * len(ds))
                datasets.append(ds)
                del recs, ds
                gc.collect()
    
    if datasets:
        logger.info("Combining Devinettes datasets")
        ds_full = concatenate_datasets(datasets)
        ds_full = remove_duplicates(ds_full)
        logger.info(f"Devinettes dataset final: {len(ds_full)} samples")
        
        raw_path = "s3://burkimbia/audios/cooked/mooreburkina/devinettes_raw"
        ds_full.save_to_disk(raw_path, storage_options=storage_options)
        logger.info(f"Saved Devinettes raw dataset to {raw_path}")
        
        del datasets, ds_full
        gc.collect()
        return True
    else:
        logger.warning("No Devinettes datasets were created")
        return False

def process_saved_devinettes():
    """Process saved devinettes dataset"""
    logger.info("=== Processing Devinettes dataset ===")
    
    try:
        raw_path = "s3://burkimbia/audios/cooked/mooreburkina/devinettes_raw"
        ds = load_from_disk(raw_path, storage_options=storage_options)
        logger.info(f"Loaded dataset: {len(ds)} samples")
        
        # Process dataset
        ds = ds.map(lambda x: {"group": extraire_id(x["id"])})
        ds = ds.map(calculate_duration)
        
        # Clean audio
        logger.info("Starting audio cleaning process")
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        ds = ds.map(clean_audio, batch_size=4)
        ds = ds.cast_column("clean", Audio(sampling_rate=16000))
        
        # Save processed dataset
        processed_path = "s3://burkimbia/audios/cooked/mooreburkina/devinettes_processed"
        ds.save_to_disk(processed_path, storage_options=storage_options)
        logger.info(f"Saved processed dataset to {processed_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Processing failed with error: {e}")
        return False

if __name__ == "__main__":
    try:
        crawl_success = crawl_and_save_devinettes()
        if not crawl_success:
            logger.error("Crawling phase failed")
            exit(1)
            
        process_success = process_saved_devinettes()
        if process_success:
            logger.info("Processing completed successfully!")
        else:
            logger.error("Processing phase failed")
            exit(1)
            
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        exit(1)
