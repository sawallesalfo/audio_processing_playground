import os
from datasets import concatenate_datasets
from loguru import logger
import numpy as np
from utils import build_dataset, crawl_and_collect

def compute_duration(example):
    duration = round(len(example["audio"]["array"]) / example["audio"]["sampling_rate"],2)
    example["duration"] = duration
    return example


if __name__ == "__main__":
    datasets = []
    logger.info("=== Début du scraping des contes en mooré ===")
    BASE_URLS = [
        "https://media.ipsapps.org/mos/ora/co1/01-B001-001.html",
        "https://media.ipsapps.org/mos/ora/co2/01-B001-001.html",
        'https://media.ipsapps.org/mos/ora/vol3/01-B001-001.html',
        "https://media.ipsapps.org/mos/ora/vol4//01-B001-001.html",
        "https://media.ipsapps.org/mos/ora/vol5//01-B021-001.html"
    ]
    READERS_NAMES = ["Patrick OUEDRAOGO", "Patrick OUEDRAOGO", "Ruth Ouedraogo", "Ruth Ouedraogo", "Ruth Ouedraogo"]
    GENRES = ["masculin", "masculin", "féminin", "féminin", "féminin"]  

    for BASE_URL, READER_NAME, GENRE in zip(BASE_URLS, READERS_NAMES, GENRES):
        logger.info(f"=== Début du scraping pour {BASE_URL} ===")
        all_recs = crawl_and_collect(BASE_URL)
        logger.info(f"Total d'enregistrements collectés: {len(all_recs)}")

        if all_recs:
            dataset = build_dataset(all_recs)
            if dataset:
                dataset = dataset.add_column("auteur", [READER_NAME] * len(dataset))
                dataset = dataset.add_column("genre", [GENRE] * len(dataset))
                datasets.append(dataset)
    logger.info("Scraping terminé")
    datasets = concatenate_datasets(datasets)
    datasets = datasets.map(compute_duration)

    # Afficher quelques statistiques sur les durées
    durations = datasets['duration']
    logger.info(f"Durée totale des audios: {sum(durations):.2f} secondes")
    logger.info(f"Durée moyenne: {np.mean(durations):.2f} secondes")
    logger.info(f"Durée minimale: {min(durations):.2f} secondes")
    logger.info(f"Durée maximale: {max(durations):.2f} secondes")
    
    storage_options = {
        "key": os.getenv("AWS_ACCESS_KEY_ID"),
        "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "client_kwargs": {"endpoint_url": os.getenv("AWS_ENDPOINT_URL_S3")}
    }
    OUTPUT_DATASET_PATH = "s3://burkimbia/audios/cooked/mooreburkina/contes"
    datasets.save_to_disk(OUTPUT_DATASET_PATH, storage_options=storage_options)