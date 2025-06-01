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

# Change to dataspeech directory and run the command
os.chdir("dataspeech")

# Build command as a list
command = [
    "python", "main.py", path_1,
    "--from_disk",
    "--aws_access_key_id", os.environ["AWS_ACCESS_KEY_ID"],
    "--aws_secret_access_key", os.environ["AWS_SECRET_ACCESS_KEY"],
    "--aws_endpoint_url", os.environ["AWS_ENDPOINT_URL_S3"],
    "--text_column_name", "text",
    "--audio_column_name", "audio",
    "--cpu_num_workers", "2",
    "--rename_column",
    "--output_dir", f"{path_1}_enriched",
    "--apply_squim_quality_estimation"
]

# Run the dataspeech command
print(f"Running dataspeech with dataset: {path_1}")
try:
    result = subprocess.run(command, check=True, env=os.environ.copy())
    print(f"Success! Output saved to: {path_1}_enriched")
except subprocess.CalledProcessError as e:
    print(f"Error running dataspeech: {e}")
except KeyError as e:
    print(f"Missing environment variable: {e}")
