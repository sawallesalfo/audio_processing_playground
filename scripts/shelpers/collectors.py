import re
from pathlib import Path

def get_audio_paths(folder: str, format="mp3") -> list[str]:
    def extract_number(file_path: str) -> int:
        match = re.search(r"segment_(\d+)", file_path)
        return int(match.group(1)) if match else float("inf")
    audio_paths = list(Path(folder).glob(f"*.{format}"))
    audio_paths = [audio_path.as_posix() for audio_path in audio_paths]
    audio_paths = sorted(audio_paths, key=extract_number)
    return audio_paths
