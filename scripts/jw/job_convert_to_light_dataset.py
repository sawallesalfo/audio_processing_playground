"""
Module de conversion et de préparation d'un dataset audio pour du deep learning.

WAV (PCM 16-bit ou 32-bit float) ➝ Meilleur choix pour l'entraînement deep learning
MP3 (Compression avec perte) ➝ À éviter sauf si nécessaire pour économiser de l'espace

Ce module réalise les opérations suivantes :
1. Convertit les fichiers audio présents dans le dataset Hugging Face en fichiers WAVE avec une fréquence d'échantillonnage de 48000 Hz.
2. Enregistre ces fichiers WAV dans le dossier "audios".
3. Récupère les chemins des fichiers convertis et calcule la durée de chaque audio.
4. Construit un nouveau dataset Hugging Face en associant à chaque audio son transcript, la page associée, la séquence audio et la durée.
5. Affiche la durée totale de l'ensemble des fichiers audio et le dataset final.

Assurez-vous d'avoir installé les dépendances via pip (par exemple, `pip install datasets soundfile pydub tqdm ffmpeg`) et que ffmpeg est installé sur votre système.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import logging
from datasets import Dataset, Features, Value, Audio, load_from_disk
import soundfile as sf
from pydub import AudioSegment
from tqdm import tqdm
from loguru import logger
from shelpers.collectors import get_audio_paths

def convert_audio_to_wav(dataset_path, output_dir="audios", storage_options=None):
    """
    Convertit les fichiers audio d'un dataset Hugging Face en fichiers WAV.
    
    Args:
        dataset_path: Chemin vers le dataset (local ou S3)
        output_dir: Dossier de sortie pour les fichiers WAV
        s3_credentials: Credentials pour accéder à S3 (si nécessaire)
    
    Returns:
        Le dataset Hugging Face avec les chemins audio et durées
    """
    os.makedirs(output_dir, exist_ok=True)            
    dataset = load_from_disk(dataset_path, storage_options=storage_options)
    logger.info(f"Dataset chargé: {len(dataset)} exemples, {dataset.data.nbytes/1e6:.2f} MB")
    
    # Conversion et sauvegarde des fichiers audio au format WAV
    for i, audio in enumerate(tqdm(dataset["audio"], desc="Conversion audio"), start=1):
        array = audio["array"]
        filename = f"{output_dir}/segment_{i}.wav"
        try:
            sf.write(filename, array, samplerate=48000)
        except Exception as e:
            logger.error(f"Erreur lors de la conversion du fichier {filename}: {e}")
    
    logger.info("Tous les fichiers audio ont été enregistrés.")
  
    # Récupération des données du dataset initial
    paths = get_audio_paths(output_dir, "wav")
    logger.info(f"Nombre de fichiers audio détectés : {len(paths)}")

    transcripts = dataset["transcript"]
    pages = dataset["page"]
    audio_sequences = dataset["audio_sequence"]
    
    features = Features({
        "audio": Audio(sampling_rate=48000),
        "transcript": Value("string"),
        "page": Value("string"),
        "audio_sequence": Value("string"),
        "duration": Value("float")
    })
    
    # Construction du nouveau dataset avec calcul des durées
    data_dict = []
    durations = []
    
    for transcript, page, audio_seq, audio_path in tqdm(zip(transcripts, pages, audio_sequences, paths), 
                                                       desc="Calcul des durées", 
                                                       total=len(paths)):
        audio = AudioSegment.from_file(audio_path)
        duration = len(audio) / 1000.0  # Conversion millisecondes -> secondes
        durations.append(duration)
        data_dict.append({
            "audio": audio_path,
            "transcript": transcript,
            "page": page,
            "audio_sequence": audio_seq,
            "duration": duration
        })
    logger.info(f"Nombre d'entrées dans data_dict: {len(data_dict)}")
    logger.info(f"Durée totale des audios : {sum(durations):.2f} secondes")
    
    new_dataset = Dataset.from_list(data_dict).cast(features)
    logger.info(f"Dataset final: {len(new_dataset)} exemples, {new_dataset.data.nbytes/1e6:.2f} MB")
    
    return new_dataset

if __name__ == "__main__":

    
    BUCKET_NAME = "moore-collection"
    DATASET_PATH = f"s3://{BUCKET_NAME}/hf_datasets/verbatim_Ɛsdras"
    OUTPUT_PATH = f"s3://{BUCKET_NAME}/hf_datasets/verbatim_Ɛsdras"
    storage_options = {
        "key": os.getenv("AWS_ACCESS_KEY_ID"),
        "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "client_kwargs": {"endpoint_url": os.getenv("AWS_ENDPOINT_URL_S3")
    }
    }

    dataset = convert_audio_to_wav(DATASET_PATH, "audios", storage_options)
    dataset.save_to_disk(OUTPUT_PATH, storage_options=storage_options)
