import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import boto3
import s3fs
from datasets import concatenate_datasets, Audio, Features, Value
from loguru import logger
from shelpers.s3 import list_s3_files, download_folder_from_s3
from shelpers.hugginface import get_audio_lengths, create_dataset_from_json

def process_s3_audio_data(bucket_name: str, folder_to_process: str, output_path: str):
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
        "audio_sequence": Value("string"),
        "page":  Value("string"),
    })

    files = list_s3_files(s3_client, bucket_name, folder_to_process)[1:]  # Skip the folder itself
    logger.info(f"Number of page to process : {len(files)}")
    datasets = []
    for file in files:
        segmented_audio_folder = f"fasoai-segmented_audios/{CHAPTER}/{file.split('/')[-1].replace('.json','')}/"
        print(segmented_audio_folder)
        download_folder_from_s3(s3_client, bucket_name, segmented_audio_folder)
        file_path = f"{bucket_name}/{file}"
        dataset = create_dataset_from_json(file_path, fs)
        datasets.append(dataset)

    if datasets:
        combined_dataset = concatenate_datasets(datasets)
        combined_dataset = combined_dataset.cast(features)
        combined_dataset.save_to_disk(
            output_path,
            storage_options={"key": access_key, "secret": secret_key, "client_kwargs":{"endpoint_url":endpoint_url}},
        )
        print(f"Dataset saved to {output_path}")
    else:
        print("No datasets were successfully created.")


if __name__ == "__main__":
    
    BUCKET_NAME = "moore-collection"
    ######################### CHANGE ME #################################
    FOLDER_TO_PROCESS = "output_jsons_Ebre"
    OUTPUT_PATH = f"s3://{BUCKET_NAME}/hf_datasets/verbatim_Ebre"
    CHAPTER = "Ebre-rãmbã"
    ########################################################################
    process_s3_audio_data(BUCKET_NAME, FOLDER_TO_PROCESS, OUTPUT_PATH)
