"""
Audio Dataset Aggregation Module

As you know Whisper fine tuning prefer audi with leng of 30 s. 
Papers suggested to use audio with lenght of 15 to 30 s for betting finetuning
This module processes JSON files containing audio segments and their transcripts,
concatenates audio segments with silence between them, and uploads the aggregated
dataset to the Hugging Face Hub. It handles batches of 13 audio segments at a time
and joins their transcripts with commas. 
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import numpy as np
from datasets import Features, Value, Audio, Dataset, concatenate_datasets
from pathlib import Path
from tqdm import tqdm
import json
import boto3
import s3fs
from loguru import logger

from shelpers.s3 import list_s3_files, download_folder_from_s3
from shelpers.hugginface import get_audio_lengths, create_dataset_from_json

def mapper_function2(batch):
    silence_duration = SILENCE_DURATION
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

def process_s3_audio_data(bucket_name: str, folder_to_process: str, output_path: str, BATCH_SIZE):
    """
    Processes audio data from an S3 bucket, creates a Hugging Face Dataset, and saves it to S3.

    Args:
        bucket_name (str): The name of the S3 bucket.
        folder_to_process (str): The folder within the bucket containing JSON files.
        output_path (str): The S3 path to save the resulting Hugging Face Dataset.
    """

    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    endpoint_url = os.getenv("AWS_ENDPOINT_URL_S3")

    if not all([access_key, secret_key, endpoint_url]):
        raise ValueError("AWS credentials or endpoint URL not set as environment variables")

    s3_client = boto3.client(
        "s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        endpoint_url=endpoint_url,
    )
    
    fs = s3fs.S3FileSystem(key=access_key, secret=secret_key, client_kwargs={"endpoint_url":endpoint_url})
    
    features = Features({
        "audio": Audio(sampling_rate=48000),
        "transcript": Value("string"),
        "page": Value("string"),
        "audio_sequence": Value("string")
    })

    combined_dataset = []
    files = list_s3_files(s3_client, bucket_name, folder_to_process)[1:]  # Skip the folder itself
    logger.info(f"Number of page to process : {len(files)}")

    for file in tqdm(files):
        segmented_audio_folder = f"fasoai-segmented_audios/{CHAPTER}/{file.split('/')[-1].replace('.json','')}/"
        file_path = f"{bucket_name}/{file}"
        dataset = create_dataset_from_json(file_path, fs)
        download_folder_from_s3(s3_client, bucket_name, segmented_audio_folder)
        agg_dataset = dataset.map(mapper_function2, batched=True, batch_size=BATCH_SIZE, remove_columns=list(dataset.features))
        agg_dataset = agg_dataset.add_column("audio_sequence", list(range(1, len(agg_dataset) + 1)))
        combined_dataset.append(agg_dataset)

    if combined_dataset:
        final_dataset = concatenate_datasets(combined_dataset)
        final_dataset = final_dataset.cast(features)

        final_dataset.save_to_disk(
            output_path,
            storage_options={"key": access_key, "secret": secret_key, "client_kwargs":{"endpoint_url":endpoint_url}},
        )
        print(f"Dataset saved to {output_path}")
    else:
        print("No datasets were successfully created")

if __name__ == "__main__":
    
    BUCKET_NAME = "moore-collection"
    ##############################CHANGE ME ######################################
    CHAPTER = "ɛsdras"
    FOLDER_TO_PROCESS = "output_jsons_yikri"
    OUTPUT_PATH = f"s3://{BUCKET_NAME}/hf_datasets/audio-dataset-aggregated_yikri"
    BATCH_SIZE= 7
    SILENCE_DURATION = 0.6
    #######################################################################################
    process_s3_audio_data(BUCKET_NAME, FOLDER_TO_PROCESS, OUTPUT_PATH, BATCH_SIZE)
