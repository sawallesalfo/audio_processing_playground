
import s3fs
from datasets import Dataset

def create_dataset_from_json(json_file_path, fs=s3fs.S3FileSystem):
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

def get_audio_lengths(dataset):
  """
  Calculates the length of each audio sample in a Hugging Face dataset.

  Args:
    dataset: A Hugging Face Dataset object containing audio data.

  Returns:
    A list of audio lengths in seconds. Returns None if the audio column is not found
  """
  if "audio" not in dataset.features:
    print("Error: Dataset does not contain an 'audio' column.")
    return None

  audio_lengths = []
  for audio_data in dataset["audio"]:
    if audio_data is None:
      audio_lengths.append(0.0) #Handle empty audio samples
      continue
    file_path = audio_data["path"]
    try:
      audio, samplerate = sf.read(file_path)
      length_seconds = len(audio) / samplerate
      audio_lengths.append(length_seconds)
    except Exception as e:
      print(f"Error processing {file_path}: {e}")
      audio_lengths.append(None)  # Indicate error with None.

  return audio_lengths