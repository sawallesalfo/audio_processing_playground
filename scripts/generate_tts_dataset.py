#!/usr/bin/env python3
import os
import subprocess
import shutil

# Clone dataspeech repository if it doesn't exist
if not os.path.exists("dataspeech"):
    print("Cloning dataspeech repository...")
    try:
        subprocess.run(["git", "clone", "https://github.com/sawallesalfo/dataspeech.git"], check=True)
        print("Repository cloned successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error cloning repository: {e}")
        exit(1)
else:
    print("Dataspeech repository already exists, skipping clone...")


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
    result = subprocess.run(command, check=True)
    print(f"Success! Output saved to: {path_1}_enriched")
except subprocess.CalledProcessError as e:
    print(f"Error running dataspeech: {e}")
except KeyError as e:
    print(f"Missing environment variable: {e}")
