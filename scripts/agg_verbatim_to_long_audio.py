"""
Audio Dataset Aggregation Module


As you know Whisper fine tuning prefer audi with leng of 30 s. 
Papers suggested to use audio with lenght of 15 to 30 s for betting finetuning
This module processes JSON files containing audio segments and their transcripts,
concatenates audio segments with silence between them, and uploads the aggregated
dataset to the Hugging Face Hub. It handles batches of 13 audio segments at a time
and joins their transcripts with commas. 
"""

import numpy as np
from datasets import Features, Value, Audio, Dataset, concatenate_datasets
from pathlib import Path
from tqdm import tqdm
import json

# Missing function that needs to be added
def create_dataset_from_json(json_file):
    """
    Create a dataset from a JSON file containing audio data and transcripts.
    
    Args:
        json_file (str): Path to the JSON file
        
    Returns:
        Dataset: A Hugging Face dataset object
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Assuming the JSON structure has keys for audio, transcript, and page
    return Dataset.from_dict({
        "audio": data["audio"],
        "transcript": data["transcript"],
        "page": data["page"]
    })

def mapper_function2(batch):
    silence_duration = 0.3  
    sampling_rate = 48000 
    silence_samples = int(silence_duration * sampling_rate)
    silence_array = np.zeros(silence_samples, dtype=np.float32)
    
    concatenated_audio = []
    for i, audio_segment in enumerate(batch["audio"]):
        concatenated_audio.extend(audio_segment["array"].tolist())
        if i < len(batch["audio"]) - 1:  # Add silence between segments
            concatenated_audio.extend(silence_array.tolist())
    
    concatenated_audio = np.array(concatenated_audio, dtype=np.float32)
    return {
        "audio": [{"array": concatenated_audio, "sampling_rate": sampling_rate}],
        "transcript": [", ".join(batch["transcript"])],
        "page": [batch["page"][0]],
    }

features = Features({
    "audio": Audio(sampling_rate=48000),
    "transcript": Value("string"),
    "page": Value("string"),
    "audio_sequence": Value("string")
})

datasets = []
files = Path("output").glob("*.json")
for file in tqdm(files):
    try:
        file = file.as_posix()
        dataset = create_dataset_from_json(file)
        agg_dataset = dataset.map(mapper_function2, batched=True, batch_size=13, remove_columns=list(dataset.features))
        agg_dataset = agg_dataset.add_column("audio_sequence", list(range(1, len(agg_dataset) + 1)))
        datasets.append(agg_dataset)
    
    except Exception as e:
        print(f"Error processing {file}: {e}")

datasets = concatenate_datasets(datasets)
datasets = datasets.cast(features)

datasets.push_to_hub("faso-ai/audio-dataset-aggregated", commit_message="🚀 batch of 13 - all page eXCEPT 41 and 42")
