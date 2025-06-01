#!/usr/bin/env python3
import os
import subprocess
import shutil
import tempfile
import json
from datasets import load_from_disk

# Set up cache directories in a writable location
cache_dir = "/tmp/cache"
config_dir = "/tmp/config"

#  cache directories
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(config_dir, exist_ok=True)
os.makedirs(f"{cache_dir}/torch/hub/torchaudio/models", exist_ok=True)
os.makedirs(f"{config_dir}/matplotlib", exist_ok=True)

os.environ["TORCH_HOME"] = f"{cache_dir}/torch"
os.environ["HF_HOME"] = f"{cache_dir}/huggingface"
os.environ["TRANSFORMERS_CACHE"] = f"{cache_dir}/transformers"
os.environ["MPLCONFIGDIR"] = f"{config_dir}/matplotlib"
os.environ["FONTCONFIG_PATH"] = f"{config_dir}/fontconfig"
os.environ["XDG_CACHE_HOME"] = cache_dir
os.environ["SAFETENSORS_FAST_GPU"] = "1"
print(f"Cache directories set up in: {cache_dir}")
print(f"Config directories set up in: {config_dir}")

# Dataset paths
path_1 = "s3://burkimbia/audios/cooked/mooreburkina/contes"
enriched_path = f"{path_1}_enriched"
tagged_path = f"{path_1}_tagged"
final_path = f"{path_1}_final"

os.chdir("dataspeech")

# Prep speaker names dict
speaker_id_colum = "auteur"
speaker_ids_to_name_json= "./speakers.json"
storage_options = {
      "key": os.environ["AWS_ACCESS_KEY_ID"],
      "secret": os.environ["AWS_SECRET_ACCESS_KEY"],
      "client_kwargs": {"endpoint_url": os.environ["AWS_ENDPOINT_URL_S3"]}
  }
dataset = load_from_disk(path_1, storage_options=storage_options).select_columns([speaker_id_colum])
speaker_dict = {sid: sid for sid in dataset.unique(speaker_id_colum)}
with open("speakers.json", "w") as f:
   json.dump(speaker_dict, f)

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
    # Since  dataset has multiple speakers and genders, we include pitch computation
    "--speaker_id_column_name", speaker_id_colum,
    "--gender_column_name", "genre",
    "--pitch_std_tolerance", "2.0",
    "--speaking_rate_std_tolerance", "4.0",
    "--snr_std_tolerance", "3.5",
    "--max_new_tokens", "1",
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

# STEP 3: Create natural language descriptions from text bins
print("\n" + "="*60)
print("STEP 3: Creating natural language descriptions from text bins")
print("="*60)

command_step3 = [
    "python", "./scripts/run_prompt_creation.py", 
    "--speaker_id_column", speaker_id_colum,
    "--speaker_ids_to_name_json", speaker_ids_to_name_json,
    "--dataset_name", tagged_path,
    "--output_dir", final_path,
    "--from_disk",
    "--model_name_or_path", "google/gemma-2-2b-it",
   "--per_device_eval_batch_size", "5", 
   "--attn_implementation","sdpa", 
    "--dataloader_num_workers", "2", 
    "--preprocessing_num_workers", "2",
    "--aws_access_key_id", os.environ["AWS_ACCESS_KEY_ID"],
    "--aws_secret_access_key", os.environ["AWS_SECRET_ACCESS_KEY"],
    "--aws_endpoint_url", os.environ["AWS_ENDPOINT_URL_S3"]
]

print(f"Creating natural language descriptions for dataset: {tagged_path}")
try:
    result = subprocess.run(command_step3, check=True, env=os.environ.copy())
    print(f"Step 3 Success! Final dataset with descriptions saved to: {final_path}")
except subprocess.CalledProcessError as e:
    print(f"Error in Step 3 - creating natural language descriptions: {e}")
    exit(1)
except KeyError as e:
    print(f"Missing environment variable: {e}")
    exit(1)

# Final summary
print("\n" + "="*60)
print("COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
print("="*60)
print(f"Original dataset: {path_1}")
print(f"Step 1 - Enriched dataset: {enriched_path}")
print(f"Step 2 - Tagged dataset: {tagged_path}")
print(f"Step 3 - Final dataset with descriptions: {final_path}")
print("="*60)
print("\nPipeline Summary:")
print("1. ✓ Annotated  dataset with continuous speech characteristics")
print("2. ✓ Mapped annotations to text bins characterizing speech")
print("3. ✓ Created natural language descriptions from text bins")
print("="*60)
