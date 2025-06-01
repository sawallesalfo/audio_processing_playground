#!/usr/bin/env python3
import os
import subprocess
import shutil
import tempfile

# Set up cache directories in a writable location
cache_dir = "/tmp/cache"
config_dir = "/tmp/config"

# Create cache directories
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(config_dir, exist_ok=True)
os.makedirs(f"{cache_dir}/torch/hub/torchaudio/models", exist_ok=True)
os.makedirs(f"{config_dir}/matplotlib", exist_ok=True)

# Set environment variables for cache directories
os.environ["TORCH_HOME"] = f"{cache_dir}/torch"
os.environ["HF_HOME"] = f"{cache_dir}/huggingface"
os.environ["TRANSFORMERS_CACHE"] = f"{cache_dir}/transformers"
os.environ["MPLCONFIGDIR"] = f"{config_dir}/matplotlib"
os.environ["FONTCONFIG_PATH"] = f"{config_dir}/fontconfig"
os.environ["XDG_CACHE_HOME"] = cache_dir

print(f"Cache directories set up in: {cache_dir}")
print(f"Config directories set up in: {config_dir}")

path_1 = "s3://burkimbia/audios/cooked/mooreburkina/contes"
enriched_path = f"{path_1}_enriched"

# Change to dataspeech directory
os.chdir("dataspeech")

# STEP 1: Run DataSpeech to enrich the dataset with audio metadata
print("="*60)
print("STEP 1: Running DataSpeech to enrich dataset with audio metadata")
print("="*60)

command_step1 = [
    "python", "main.py", path_1,
    "--from_disk",
    "--aws_access_key_id", os.environ["AWS_ACCESS_KEY_ID"],
    "--aws_secret_access_key", os.environ["AWS_SECRET_ACCESS_KEY"],
    "--aws_endpoint_url", os.environ["AWS_ENDPOINT_URL_S3"],
    "--text_column_name", "text",
    "--audio_column_name", "audio",
    "--cpu_num_workers", "2",
    "--rename_column",
    "--output_dir", enriched_path,
    "--apply_squim_quality_estimation"
]

print(f"Running dataspeech with dataset: {path_1}")
try:
    result = subprocess.run(command_step1, check=True, env=os.environ.copy())
    print(f"Step 1 Success! Enriched data saved to: {enriched_path}")
except subprocess.CalledProcessError as e:
    print(f"Error in Step 1 - running dataspeech: {e}")
    exit(1)
except KeyError as e:
    print(f"Missing environment variable: {e}")
    exit(1)

# STEP 2: Convert metadata to text tags
print("\n" + "="*60)
print("STEP 2: Converting metadata to text tags")
print("="*60)

# Output path for the final tagged dataset
tagged_path = f"{path_1}_tagged"

command_step2 = [
    "python", "./scripts/metadata_to_text.py",
    enriched_path,
    "--from_disk",
    "--aws_access_key_id", os.environ["AWS_ACCESS_KEY_ID"],
    "--aws_secret_access_key", os.environ["AWS_SECRET_ACCESS_KEY"],
    "--aws_endpoint_url", os.environ["AWS_ENDPOINT_URL_S3"],
    "--output_dir", tagged_path,
    "--cpu_num_workers", "2",
    "--apply_squim_quality_estimation",
    # Since your dataset has multiple speakers and genders, we include pitch computation
    "--speaker_id_column_name", "auteur",
    "--gender_column_name", "genre",
    "--pitch_std_tolerance", "2.0",
    "--speaking_rate_std_tolerance", "4.0",
    "--snr_std_tolerance", "3.5",
    "--reverberation_std_tolerance", "4.0",
    "--speech_monotony_std_tolerance", "4.0"
]

print(f"Converting metadata to text tags for dataset: {enriched_path}")
try:
    result = subprocess.run(command_step2, check=True, env=os.environ.copy())
    print(f"Step 2 Success! Tagged data saved to: {tagged_path}")
except subprocess.CalledProcessError as e:
    print(f"Error in Step 2 - converting metadata to text: {e}")
    exit(1)
except KeyError as e:
    print(f"Missing environment variable: {e}")
    exit(1)

print("\n" + "="*60)
print("PIPELINE COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"Original dataset: {path_1}")
print(f"Enriched dataset: {enriched_path}")
print(f"Final tagged dataset: {tagged_path}")
print("="*60)
