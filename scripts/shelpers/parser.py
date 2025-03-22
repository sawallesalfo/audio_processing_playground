import re
import urllib


def extract_audio_identifier(url):
    parts = url.strip("/").split("/")
    return urllib.parse.unquote(parts[-2]), int(parts[-1].replace("page_",""))

def remove_digits_and_numbers(text):
  """
  Removes digits and numbers from the beginning of a string and keeps the remaining text.
  """
  match = re.match(r'^[\d\s]+', text)
  if match:
    return text[match.end():].strip()
  else:
    return text
  

def time_to_milliseconds(time_str):
    """Converts time string (MM:SS or HH:MM:SS) to milliseconds."""
    try:
        parts = time_str.split(":")
        if len(parts) == 2:  # MM:SS format
            minutes, seconds = map(int, parts)
            return (minutes * 60 + seconds) * 1000
        elif len(parts) == 3:  # HH:MM:SS format
            minutes, seconds, milli_second = map(int, parts)
            return (minutes * 60 + seconds  + milli_second/10000) * 1000
        else:
            print(f"Error: Invalid time format '{time_str}'")
            return 0  # Return 0 milliseconds if invalid format
    except ValueError as e:
        print(f"Error parsing time string '{time_str}': {e}")
        return 0  
    


def get_verse_id(verse_number, base_id="v1041"):
    return f"{base_id}{verse_number:03d}"
