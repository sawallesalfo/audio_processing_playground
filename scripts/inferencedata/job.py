import os
from datasets import load_dataset
import soundfile as sf
from gradio_client import Client, handle_file
import tempfile

client = Client(
    "sawadogosalif/Sachi-ASR-demo",
    hf_token=os.environ["HF_TOKEN"]
)

ds = load_dataset("sawadogosalif/contes", split="train")

def transcribe(example):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        sf.write(tmp_path, example["audio"]["array"], example["audio"]["sampling_rate"])
        
    try:
        res = client.predict(
            handle_file(tmp_path),
            apply_enhance=False,
            api_name="/transcribe_and_update"
        )
        example["text_transcript"] = res
    except Exception as e:
        print(f"Erreur sur exemple {example['id'] if 'id' in example else 'unknown'}: {e}")
        example["text_transcript"] = ""
    finally:
        # Ensure the temporary file is removed
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    
    return example

dataset = ds.map(transcribe)
dataset.push_to_hub("sawadogosalif/contes", token=os.environ["HF_TOKEN"])
