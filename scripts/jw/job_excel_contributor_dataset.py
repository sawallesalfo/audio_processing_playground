import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import boto3
from loguru import logger
import pandas as pd
from pydub import AudioSegment
import openpyxl
from tqdm import tqdm
from datasets import concatenate_datasets, Audio, Features, Value, load_dataset, DownloadConfig, Dataset


from shelpers.collectors import get_audio_paths
from shelpers.parser import time_to_milliseconds, extract_audio_identifier, remove_digits_and_numbers
from shelpers.matcher import get_matches
from shelpers.s3 import download_file_from_s3

def segment_audio_from_excel(df_sheet, audio, sheet_name, output_folder="segments"):
    """ Segmente un fichier audio en fonction des timestamps d'une feuille Excel. """
    sheet_output_folder = os.path.join(output_folder, sheet_name)
    os.makedirs(sheet_output_folder, exist_ok=True)

    for index, row in df_sheet.iterrows():
        start_time = row["debut partie"]
        end_time = row["fin partie"]

        start_ms = time_to_milliseconds(str(start_time))
        end_ms = time_to_milliseconds(str(end_time))

        if start_ms < end_ms:  # Vérification pour éviter les erreurs
            segment = audio[start_ms:end_ms]
            filename = f"{sheet_output_folder}/segment_{index + 1}.wav"
            segment.export(filename, format="wav")
            print(f"✅ Segment sauvegardé: {filename}")
        else:
            print(f"⚠️ Erreur: start_time ({start_time}) >= end_time ({end_time})")

def infer_matching(dataset, chapter, excel_file, audio_files, output_folder="segments"):
    """
    Traite chaque feuille d'un fichier Excel en utilisant son fichier audio correspondant.
    :param excel_file: Fichier Excel contenant les timestamps (une feuille = un fichier audio).
    :param audio_files: Dictionnaire associant chaque feuille à son fichier audio.
    :param output_folder: Dossier où sauvegarder les segments audios.
    """
    wb = openpyxl.load_workbook(excel_file)
    sheet_names = wb.sheetnames
    
    results = []
    for sheet_name in tqdm(sheet_names[PAGE_START-1:], desc="📄 Traitement des feuilles"):
        logger.info(f"Sheet :{sheet_name}")

        if sheet_name in audio_files:
            audio_file = audio_files[sheet_name]
            if os.path.exists(audio_file):
                print(f"🔹 Traitement de la feuille '{sheet_name}' avec l'audio '{audio_file}'")
                df_sheet = pd.read_excel(excel_file, sheet_name=sheet_name)
                audio = AudioSegment.from_file(audio_file)
                segment_audio_from_excel(df_sheet, audio, sheet_name, output_folder)
            else:
                print(f"❌ Fichier audio '{audio_file}' introuvable pour la feuille '{sheet_name}'")
        else:
            print(f"⚠️ Aucun fichier audio spécifié pour la feuille '{sheet_name}'")
        
        # Extraction de l'ID de la page
        logger.info(f"end of sheets processing")
        try:
            page_id = int(sheet_name.replace("page_", ""))
        except ValueError as e:
            logger.error(f"⚠️ Impossible d'extraire un ID de page depuis '{sheet_name}'")
            raise e

        segments = get_audio_paths(f"{output_folder}/{sheet_name}", "wav")
        logger.info(f"count of audio files:  {len(segments)}")

        # Filtrage des transcriptions correspondant à cette page et ce chapitre
        sub_transcription_df = dataset[(dataset["page"] == page_id) & (dataset["chapter"] == chapter)]
        transcriptions = get_matches(df_sheet, sub_transcription_df)
        logger.info(f"count of transcription:  {len(transcriptions)}")
        audio_sequence = list(range(1, len(transcriptions) + 1))

        # Sauvegarde des résultats
        results.append({
            "audio": segments,
            "transcript": transcriptions,
            "page": [page_id] * len(segments),
            "audio_sequence": audio_sequence
        })

    return results

def generate_audio_dict(page_start, page_end, base_path, subfolder, file_pattern):
    audio_files = {
        f"page_{page}": os.path.join(base_path, subfolder) + "\\" + file_pattern.format(page=page)
        for page in range(page_start, page_end + 1)
    }
    return audio_files

if __name__ == "__main__":
    
    BUCKET_NAME = "moore-collection"
    DATA_FILE = "sawadogosalif/MooreFRCollections_BibleOnlyText"
    
    ################################### CHANGE ME ########################
    CHAPTER= "abdiyaas"
    EXCEL_FILE= "contributor_files/abdiyaas.xlsx"
    PAGE_START = 1
    PAGE_END = 1
    ####################
    #######################################################################
    file_pattern = "page_{page}.mp3"
    audio_files  = generate_audio_dict(PAGE_START, PAGE_END, "raw_data", CHAPTER, file_pattern)

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
    
    logger.info("Reading dataset")
    dataset = load_dataset(DATA_FILE, split="train", download_config=DownloadConfig(token=os.environ["HF_TOKEN"])).to_pandas()
    dataset[["chapter", "page"]] = dataset["moore_source_url"].apply(
        lambda x: pd.Series(extract_audio_identifier(x))
    )
    dataset["moore_verse_text"] = dataset["moore_verse_text"].apply(remove_digits_and_numbers)

    logger.info("Download files")
    download_file_from_s3(s3_client, BUCKET_NAME, EXCEL_FILE, EXCEL_FILE)
    for audio_file in audio_files.values():
        print(audio_file)
        download_file_from_s3(s3_client, BUCKET_NAME, audio_file, audio_file)


    results  = infer_matching(dataset, CHAPTER, EXCEL_FILE, audio_files, "output1")

    logger.info("Creating hugginface")
    dataset_list = []
    for result in tqdm(results):
        logger.info(f"creating hf dataset for page {result['page']}")
        dataset = Dataset.from_dict(result)
        features = Features({
                "audio": Audio(sampling_rate=48000),
                "transcript": Value("string"),
                "page": Value("string"),
                "audio_sequence": Value("string")
            })
        dataset = dataset.cast(features)

        durations  = []
        for example in dataset:
            audio_path = example["audio"]['path']
            audio = AudioSegment.from_file(audio_path)
            duration = len(audio) / 1000.0  # Convertir
            durations.append(duration)
        dataset = dataset.add_column("duration", durations)
        dataset_list.append(dataset)
        

    logger.info("SAVING hugginface")
    datasets = concatenate_datasets(dataset_list)
    datasets.save_to_disk(f"s3://{BUCKET_NAME}/hf_datasets/contribution_dataset_{CHAPTER}",
    storage_options={
        "key": access_key,
        "secret": secret_key,
        "client_kwargs": {"endpoint_url": endpoint_url}
    }
)
