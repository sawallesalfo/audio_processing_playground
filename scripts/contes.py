import os
from datasets import concatenate_datasets
from loguru import logger
from mooreburkina.utils import build_dataset, crawl_and_collect

def compute_duration(example):
    duration = round(len(example["audio"]["array"]) / example["audio"]["sampling_rate"],2)
    example["duration"] = duration
    return example

if __name__ == "__main__":
    datasets = []
    BASE_URLS = [
        "https://media.ipsapps.org/mos/ora/co2/01-B001-001.html",
        'https://media.ipsapps.org/mos/ora/vol3/01-B001-001.html',
        "https://media.ipsapps.org/mos/ora/vol4//01-B001-001.html",
        "https://media.ipsapps.org/mos/ora/vol5//01-B021-001.html"
    ]

    for BASE_URL in BASE_URLS:
        logger.info(f"=== Début du scraping pour {BASE_URL} ===")
        all_recs = crawl_and_collect(BASE_URL)
        logger.info(f"Total d'enregistrements collectés: {len(all_recs)}")

        if all_recs:
            dataset = build_dataset(all_recs)
            if dataset:
                datasets.append(dataset)
    logger.info("Scraping terminé")
    datasets = concatenate_datasets(datasets)
    datasets = datasets.map(compute_duration)
    datasets.push_to_hub("sawadogosalif/contes", private=True, token=os.environ["HF_TOKEN"] 
)
