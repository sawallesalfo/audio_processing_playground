import os
import re
from loguru import logger
from datasets import Dataset, Audio
from itertools import groupby
import numpy as np
from datasets import concatenate_datasets
from mooreburkina.utils import build_dataset, crawl_and_collect
from langdetect import detect
from datasets import load_dataset, Dataset


def is_french(text: str) -> bool:
    text = text.strip()
    try:
        return (
            detect(text) == 'fr' or
            text.startswith('(') or
            text.endswith(')')
        )
    except:
        return text.startswith('(') or text.endswith(')')
      
def extraire_id(texte):
    pattern = r'(\d+)[a-zA-Z]$'
    match = re.search(pattern, texte)
    if match:
        return match.group(1)
    else:
        return None


def find_language_and_group_segments(dataset):

    change_indices = [0]
    current_lang = dataset[0]['french_map']
    current_group = dataset[0]['group']
    
    for i in range(1, len(dataset)):
        # Vérifier si la langue OU le groupe change
        if dataset[i]['french_map'] != current_lang or dataset[i]['group'] != current_group:
            change_indices.append(i)
            current_lang = dataset[i]['french_map']
            current_group = dataset[i]['group']
    
    change_indices.append(len(dataset))
    
    segments = []
    for i in range(len(change_indices) - 1):
        start = change_indices[i]
        end = change_indices[i+1]
        
        segment_text = " ".join([dataset[j]['text'] for j in range(start, end)])
        
        audio_arrays = [dataset[j]['audio']['array'] for j in range(start, end)]
        combined_audio = np.concatenate(audio_arrays)
        
        if np.issubdtype(combined_audio.dtype, np.floating):
            combined_audio = combined_audio.astype(np.float32)
        else:
            combined_audio = combined_audio.astype(np.int16)
        
        segments.append({
            'group': dataset[start]['group'],  
            'is_french': dataset[start]['french_map'],
            'text': segment_text,
            'audio': {
                'array': combined_audio,
                'sampling_rate': dataset[start]['audio']['sampling_rate']
            }
        })
    
    new_dataset = Dataset.from_dict({
        'group': [s['group'] for s in segments],  
        'is_french': [s['is_french'] for s in segments],
        'text': [s['text'] for s in segments],
        'audio': [s['audio'] for s in segments]
    })
    
    sampling_rate = dataset[0]['audio']['sampling_rate']
    new_dataset = new_dataset.cast_column('audio', Audio(sampling_rate=sampling_rate))
    
    return new_dataset


if __name__ == "__main__":
    datasets = []
    BASE_URLS = [
        "https://media.ipsapps.org/mos/ora/p{i}/01-001-001.html" for i in range(1, 12)
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
    datasets = datasets.map(lambda x: {"group": extraire_id(x["id"])})
    datasets = datasets.map(lambda x: {"french_map": is_french(x["text"])})
    clean = find_language_and_group_segments(datasets)
    clean.push_to_hub("sawadogosalif/proverbes_clean", private=True, token=os.environ["HF_TOKEN"])
    datasets.push_to_hub("sawadogosalif/proverbes", private=True, token=os.environ["HF_TOKEN"])
