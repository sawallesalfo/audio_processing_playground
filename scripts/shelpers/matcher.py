def get_verse_id(verse_number, base_id):
    """
    Creates a verse ID by maintaining the base prefix and chapter number,
    then appending the formatted verse number.
    
    Parameters:
    - verse_number (int): The verse number to use
    - base_id (str): A reference base ID (e.g., "v54001001")
    
    Returns:
    - str: A complete verse ID with the correct chapter and verse format
    """
    # Extract the prefix (everything except the last 4 digits)
    prefix = base_id[:-4]
    
    # Extract the chapter number (the first digit of the last 4 digits)
    chapter = base_id[-4:-3]
    
    # Format the verse with leading zeros (3 digits)
    return f"{prefix}{chapter}{verse_number:03d}"



def get_matches(segment_data, transcription_df):
    """
    Associe chaque segment audio à son texte de transcription en fonction des IDs de versets.

    Cette fonction prend un DataFrame contenant les segments audio avec des plages de versets 
    ("debut verset", "fin verset") et un DataFrame contenant les transcriptions associées 
    (avec une colonne "verse_id"). Elle extrait les versets correspondant aux plages définies 
    et les fusionne pour chaque segment audio.

    ### Paramètres :
    - `segment_data` (pd.DataFrame) : DataFrame contenant les segments avec colonnes :
        - "debut verset" : ID du premier verset du segment.
        - "fin verset" : ID du dernier verset du segment.
    - `transcription_df` (pd.DataFrame) : DataFrame contenant les transcriptions avec colonnes :
        - "verse_id" : ID du verset.
        - "moore_verse_text" : Texte en langue mooré correspondant au verset.
    """

    # Extraction du préfixe de l'ID de verset pour normaliser les identifiants
    sample_id = transcription_df.verse_id.iloc[0]

    try:
        # Normalisation des IDs des versets de début et de fin
        segment_data = segment_data[["debut verset", "fin verset"]].copy()
        segment_data.loc[:, "fin verset"] = segment_data["fin verset"].apply(lambda x: get_verse_id(x, sample_id))
        segment_data.loc[:, "debut verset"] = segment_data["debut verset"].apply(lambda x: get_verse_id(x, sample_id))
    except Exception as e:
        raise ValueError(f"Erreur lors de la normalisation des versets : {e}")

    # Extraction des numéros de versets sous forme d'entiers
    segment_data.loc[:, "debut_num"] = segment_data["debut verset"].str[1:].astype(int)
    segment_data.loc[:, "fin_num"] = segment_data["fin verset"].str[1:].astype(int)
    transcription_df.loc[:, "verse_num"] = transcription_df["verse_id"].str[1:].astype(int)

    # Création d'une jointure croisée pour associer chaque segment à tous les versets
    cross_df = segment_data.assign(key=1).merge(transcription_df.assign(key=1), on="key").drop("key", axis=1)

    # Filtrage pour ne garder que les versets inclus dans la plage [début verset, fin verset]
    filtered_df = cross_df[
        (cross_df["verse_num"] >= cross_df["debut_num"]) &
        (cross_df["verse_num"] <= cross_df["fin_num"])
    ]

    # Agrégation des transcriptions associées à chaque segment
    result_df = filtered_df.groupby(
        ["debut verset", "fin verset"]
    ).agg({
        "moore_verse_text": lambda x: " ".join(x)
    }).reset_index()

    return result_df["moore_verse_text"].to_list()
