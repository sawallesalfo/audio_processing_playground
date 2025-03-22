import os
from loguru import logger
import sf3s
from datasets import Dataset


def download_file_from_s3(s3_client, bucket_name, s3_key, local_path):
    """Download a single file from S3."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3_client.download_file(bucket_name, s3_key, local_path)
    logger.info(f"Downloaded {s3_key} to {local_path}")

def download_folder_from_s3(s3_client, bucket_name, s3_key):
    """Download a single file from S3."""
    # os.makedirs(os.path.dirname(local_folder), exist_ok=True)
    os.makedirs(os.path.dirname(s3_key), exist_ok=True)

    segments = list_s3_files(s3_client, bucket_name, s3_key)
    for segment in segments:
        download_file_from_s3(s3_client, bucket_name, segment,segment)
        logger.info(f"Downloaded {segment} to {s3_key}")
    logger.info(f"End")
    

def list_s3_files(s3_client, bucket_name, prefix):
    """List all files in an S3 bucket under a given prefix."""
    paginator = s3_client.get_paginator("list_objects_v2")
    files = []
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get("Contents", []):
            files.append(obj["Key"])
    return files



def create_dataset_from_json(json_file_path, fs=sf3s.S3FileSystem):
    """
    Create a Hugging Face dataset from a JSON file containing audio paths and transcripts.
    
    Args:
        json_file_path: Path to the JSON file with format:
                        {"audio": [path1, path2, ...], "transcript": [text1, text2, ...]}
    
    Returns:
        A Hugging Face dataset with audio and transcript features
    """
    # Load the JSON data
    if fs:
        with fs.open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else: 
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    
    if 'audio' not in data or 'transcript' not in data:
        raise ValueError("JSON file must contain 'audio' and 'transcript' keys")
    
    audio_paths = data['audio']
    transcripts = data['transcript']
    
    if len(audio_paths) != len(transcripts):
        raise ValueError("The number of audio paths and transcripts must match")
    
    
    dataset = Dataset.from_dict({
        "audio": audio_paths,
        "transcript": transcripts,
        "audio_sequence": list(range(1, len(audio_paths) + 1))

    })
    
    # Cast the audio column to Audio type
    page = (json_file_path.split("/")[-1]).replace("page_","").replace(".json","")
    dataset = dataset.cast_column("audio", Audio()).add_column("page", [page]*len(dataset))
    
    return dataset