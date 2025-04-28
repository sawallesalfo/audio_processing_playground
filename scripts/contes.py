# -*- coding: utf-8 -*-


import os
import re
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from pydub import AudioSegment
from datasets import Dataset, Features, Value, Audio, concatenate_datasets
from loguru import logger

# -------------------------------------------------------
# Configuration principale
# -------------------------------------------------------
SAMPLING_RATE = 48000
MAX_RETRIES = 3
RETRY_DELAY = 2  # secondes entre chaque nouvelle tentative

def fetch_html(url: str) -> str:
    """Télécharge et retourne le HTML d'une URL."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()  # Raise an exception for bad status codes
        resp.encoding = 'utf-8'  # Explicitly set encoding to UTF-8
        return resp.text
    except requests.exceptions.RequestException as e:
        logger.error(f"Échec de récupération pour {url}: {e}")
        return None

def parse_timings(html: str) -> list:
    """Extrait les minutages audio (label, start, end) depuis le HTML."""
    if not html:
        return []

    # Méthode 1: Utiliser un regex pour extraire les timings
    pattern = r"\{\s*label:\s*\"?([^\"]+)\"?,\s*start:\s*([\d\.]+),\s*end:\s*([\d\.]+)\s*\}"
    timings_regex = [{"label": m[0], "start": float(m[1]), "end": float(m[2])}
            for m in re.findall(pattern, html)]

    # Si la première méthode ne fonctionne pas, essayer une approche alternative
    if not timings_regex:
        pattern = r'var timings = \[(.*?)\];'
        match = re.search(pattern, html, re.DOTALL)
        if match:
            content = match.group(1)
            timings_regex = [{"label": m[0], "start": float(m[1]), "end": float(m[2])}
                    for m in re.findall(pattern, content)]

    return timings_regex

def parse_audio_url(html: str, base_url: str) -> str:
    """Récupère l'URL de l'audio à partir de la page HTML."""
    if not html:
        return None

    soup = BeautifulSoup(html, "html.parser", from_encoding='utf-8')
    source = soup.find("source", id="audio1")

    if source and source.get("src"):
        return urljoin(base_url, source["src"])

    # Recherche alternative si l'ID n'est pas trouvé
    audio_tag = soup.find("audio")
    if audio_tag:
        source = audio_tag.find("source")
        if source and source.get("src"):
            return urljoin(base_url, source["src"])

    return None

def parse_texts(html: str, timings: list) -> dict:
    """Associe chaque label temporel à son texte correspondant avec préfixe T."""
    if not html or not timings:
        return {}

    soup = BeautifulSoup(html, "html.parser", from_encoding='utf-8')
    texts = {}

    for t in timings:
        # Ajouter le préfixe 'T' pour correspondre aux IDs dans le HTML
        element_id = f"T{t['label']}"
        el = soup.find(id=element_id)

        if el:
            texts[t["label"]] = el.get_text(strip=True)
        else:
            logger.warning(f"Élément avec ID {element_id} non trouvé dans le HTML")
            texts[t["label"]] = ""

    return texts

def download_audio(audio_url: str, folder: str = "audio") -> str:
    """Télécharge un fichier audio depuis une URL, avec cache local."""
    if not audio_url:
        return None

    os.makedirs(folder, exist_ok=True)
    filename = os.path.basename(audio_url)
    dest = os.path.join(folder, filename)

    if os.path.exists(dest):
        logger.info(f"Fichier audio déjà téléchargé: {dest}")
        return dest

    logger.info(f"Téléchargement de {audio_url}")
    try:
        resp = requests.get(audio_url, stream=True, timeout=30)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info(f"Téléchargement réussi: {dest}")
        return dest
    except requests.exceptions.RequestException as e:
        logger.error(f"Échec de téléchargement pour {audio_url}: {e}")
        return None

def segment_audio(audio_path: str, timings: list, texts: dict, out_dir: str) -> list:
    """
    Découpe un fichier audio complet en clips selon les minutages.
    """
    if not audio_path:
        return []

    # Charger l'audio
    try:
        audio = AudioSegment.from_file(audio_path)
    except Exception as e:
        logger.error(f"Erreur lors du chargement de l'audio {audio_path}: {e}")
        return []

    os.makedirs(out_dir, exist_ok=True)
    records = []
    base_name = os.path.splitext(os.path.basename(audio_path))[0]

    for t in timings:
        start_ms = int(t["start"] * 1000)
        end_ms = int(t["end"] * 1000)

        if start_ms >= end_ms:
            logger.warning(f"Timing invalide pour {t['label']}: start={start_ms}ms, end={end_ms}ms")
            continue

        clip = audio[start_ms:end_ms]

        seg_name = f"{base_name}_{t['label']}"
        out_path = os.path.join(out_dir, f"{seg_name}.mp3")

        # Export avec compression mp3
        try:
            clip.export(out_path, format="mp3")
            # Récupérer le texte associé
            text = texts.get(t["label"], "")

            records.append({
                "id": seg_name,
                "text": text,
                "audio": out_path
            })

            logger.info(f"Segment créé: {seg_name}")
        except Exception as e:
            logger.error(f"Erreur lors de l'export du segment {seg_name}: {e}")

    return records

def crawl_and_collect(base_url: str) -> list:
    """
    Crawler récursivement les chapitres/livres à partir d'une URL de base,
    collecter tous les segments audio/textes.
    """
    visited = set()
    to_visit = [base_url]
    all_records = []

    while to_visit:
        url = to_visit.pop(0)
        if url in visited:
            continue

        logger.info(f"Traitement de: {url}")
        visited.add(url)
        html = fetch_html(url)

        if not html:
            continue

        # Extraire les infos nécessaires
        timings = parse_timings(html)

        if not timings:
            logger.warning(f"Aucun timing trouvé dans {url}")
            continue

        logger.info(f"Trouvé {len(timings)} segments de timing")

        audio_url = parse_audio_url(html, url)

        if audio_url:
            logger.info(f"URL audio trouvée: {audio_url}")
            texts = parse_texts(html, timings)
            audio_path = download_audio(audio_url)

            if not audio_path:
                continue

            page_id = os.path.splitext(os.path.basename(url.rstrip('/')))[0]

            recs = segment_audio(
                audio_path, timings, texts,
                out_dir=os.path.join("segments", page_id)
            )

            all_records.extend(recs)
            logger.info(f"Collecté {len(recs)} segments pour {url}")
        else:
            logger.warning(f"Aucun audio trouvé pour {url}")

        soup = BeautifulSoup(html, "html.parser")
        next_ch = soup.find("a", title="Next Chapter")
        next_bk = soup.find("a", title="Next Book")

        for link in (next_ch, next_bk):
            if link and link.get("href"):
                nxt = urljoin(url, link["href"])
                if nxt not in visited:
                    to_visit.append(nxt)
                    logger.info(f"Page ajoutée à la file: {nxt}")
                break  # on ne suit qu'un seul lien à la fois

    return all_records

def build_dataset(records: list):
    """
    Construit un Hugging Face Dataset à partir des segments collectés.
    """
    if not records:
        return None

    logger.info(f"Construction du dataset avec {len(records)} enregistrements")

    features = Features({
        "id": Value("string"),
        "text": Value("string"),
        "audio": Audio(sampling_rate=SAMPLING_RATE)
    })

    ds = Dataset.from_dict({
        "id": [r["id"] for r in records],
        "text": [r["text"] for r in records],
        "audio": [r["audio"] for r in records]
    }, features=features)

    logger.info("Dataset construit avec succès")
    return ds

if __name__ == "__main__":
    datasets = []
    BASE_URLS = [
        "https://media.ipsapps.org/mos/ora/co2/01-B001-001.html",
        'https://media.ipsapps.org/mos/ora/vol3/01-B001-001.html',
        "https://media.ipsapps.org/mos/ora/vol4//01-B001-001.html",
        "https://media.ipsapps.org/mos/ora/vol5//01-B021-001.html"
    ]

    for BASE_URL in BASE_URLS:
        logger.info(f"=== Début du scraping pour {BASE_URL} ===")
        all_recs = crawl_and_collect(BASE_URL)
        logger.info(f"Total d'enregistrements collectés: {len(all_recs)}")

        if all_recs:
            dataset = build_dataset(all_recs)
            if dataset:
                datasets.append(dataset)
                # Sauvegarde locale du dataset
                output_dir = f"dataset_{os.path.basename(BASE_URL).replace('.html', '')}"
                dataset.save_to_disk(output_dir)
                logger.info(f"Dataset sauvegardé dans {output_dir}")
        break
    logger.info("Scraping terminé")
    concatenate_datasets(datasets).push_to_hub("sawadogosalif/contes", private=True,)
