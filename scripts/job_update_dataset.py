import os
from datasets import load_dataset, load_from_disk, concatenate_datasets, Features, Value, Audio, Dataset, DownloadConfig
from loguru import logger

def process_dataset(current_dataset_path, incoming_dataset_path, output_dataset_path, storage_options, hf_token, commit_message):
    """
    Charge, vérifie et fusionne deux datasets audio, puis pousse le résultat sur le Hub Hugging Face

    Args:
        current_dataset_path (str): Chemin du dataset actuel sur Hugging Face Hub.
        incoming_dataset_path (str): Chemin du nouveau dataset à intégrer.
        output_dataset_path (str): Chemin du dataset final (Hub).
        storage_options (dict): Options pour accéder au stockage distant
        hf_token (str): Jeton d'authentification Hugging Face.
        commit_message (str): Message de commit pour le push sur le Hub.

    Returns:
        None
    """
    
    # Chargement du dataset actuel depuis le Hub
    logger.info("Chargement du dataset actuel depuis le Hub...")
    current_dataset = load_dataset(
        current_dataset_path, 
        split="train", 
        download_config=DownloadConfig(token=hf_token)
    )
    # Chargement du dataset entrant depuis un stockage
    logger.info("Chargement du dataset entrant depuis le stockage...")
    incoming_dataset = load_from_disk(incoming_dataset_path, storage_options=storage_options)

    # Vérification des colonnes
    if not current_dataset.features.keys()==incoming_dataset.features.keys():
        raise ValueError("Les colonnes du dataset entrant ne correspondent pas aux colonnes attendues.")
    # logger.info("Vérification des colonnes réussie ✅")


    # Vérification du nombre de lignes
    logger.info(f"Nombre de lignes - Dataset actuel: {len(current_dataset)}, Dataset entrant: {len(incoming_dataset)}")

    if VERBATIM: 
        expected_features = Features({
            "audio": Audio(sampling_rate=48000),
            "transcript": Value("string"),
            "audio_sequence": Value("string"),
            "page": Value("string"),
        })
    else:
        expected_features = Features({
        "audio": Audio(sampling_rate=48000),
        "transcript": Value("string"),
        "page": Value("string"),
        "audio_sequence": Value("string"),
        "duration": Value("float")
    })
    final_dataset = concatenate_datasets([current_dataset, incoming_dataset.cast(expected_features)])
    logger.info(f"Dataset final après fusion: {len(final_dataset)} lignes")

    if not VERBATIM:
        total_duration = sum(final_dataset["duration"])
        logger.info(f"Durée totale des fichiers audio : {total_duration:.2f} secondes")

    # Push sur le Hub
    logger.info(f"Push du dataset final sur {output_dataset_path}...")
    final_dataset.save_to_disk(output_dataset_path)
    logger.info("Push terminé avec succès ✅")

if __name__ == "__main__":

    BUCKET_NAME = "moore-collection"

    ########################## Change me ######################################
    VERBATIM = False 
    CURRENT_DATASET_PATH = "sawadogosalif/MooreFRCollectionsAudios"
    COMMIT_MESSAGE = "add ebre ramba 👨‍🍳"
    INCOMING_DATASET_PATH = f"s3://{BUCKET_NAME}/hf_datasets/light-verbatim_zakari"
    OUTPUT_DATASET_PATH = "s3://burkimbia/audios/cooked/bible-jw"

    ############################################################################

    storage_options = {
        "key": os.getenv("AWS_ACCESS_KEY_ID"),
        "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "client_kwargs": {"endpoint_url": os.getenv("AWS_ENDPOINT_URL_S3")}
    }

    HF_TOKEN = os.getenv("HF_TOKEN")
    process_dataset(
        CURRENT_DATASET_PATH, 
        INCOMING_DATASET_PATH, 
        OUTPUT_DATASET_PATH, 
        storage_options, 
        HF_TOKEN, 
        COMMIT_MESSAGE
    )
