import os
import pandas as pd

v_map = {
    "v02": 0,
    "v04": 3,
    "v05": 6,
    "v06": 6,
    "v11": 12*1,
    "v21": 12*2,
    "v31": 12*3,
    "v41": 12*4,
    "v51": 12*5,
}

def build_mri_npy_file_path(row, preprocessed_root, image_type="mri"):
    """
    Bygger sökväg till MRI .npy-filen: <root>/<subject>/mri/<exam_id>.npy
    """
    subject = row["individual_id"]
    exam_id = row["exam_id"]

    npy_path = os.path.join(
        preprocessed_root,
        subject,
        image_type,
        f"{exam_id}.npy"
    )
    return npy_path

def build_pet_npy_file_path(row, preprocessed_root, pet_type="amy"):
    """
    Bygger sökväg till PET .npy-filen: <root>/<subject>/amy/<exam_id>.npy
    """
    subject = row["individual_id"]
    exam_id = row["exam_id"]

    npy_path = os.path.join(
        preprocessed_root,
        subject,
        pet_type,
        f"{exam_id}.npy"
    )
    return npy_path

def _get_first_entry(subject_data, viscode: str, column: str):
    entries = subject_data[subject_data["VISCODE2"] == viscode]
    if entries.empty:
        return pd.NA
    # if len(entries) > 1:
        # logger.debug("Multiple %s entries for subject %s", viscode, subject_data["individual_id"].iloc[0])
    return entries.iloc[0][column]

def get_baseline_date_for_subject(subject_id: str, dxsum: pd.DataFrame) -> pd.Timestamp:
    """Hämtar baseline-datum för en patient."""
    subject_data = dxsum[dxsum["individual_id"] == subject_id]
    if subject_data.empty:
        return pd.NaT
    
    date = _get_first_entry(subject_data, "bl", "EXAMDATE")
    if pd.isna(date):
        date = _get_first_entry(subject_data, "sc", "EXAMDATE")

    if pd.isna(date):
        logger.warning(f"No valid baseline/screening date found for subject {subject_id}.")
        return pd.NaT

    return date

def get_months_from_baseline(timepoint: str) -> float:
    """Konverterar timepoint-sträng (t.ex. 'm24') till antal månader (float)."""
    if pd.isna(timepoint):
        return pd.NA
    
    tp = str(timepoint).lower()
    
    if tp in ["sc", "init", "bl"]:
        return 0
    if tp.startswith("m"):
        try:
            return int(tp[1:])
        except ValueError:
            return pd.NA
    if tp.startswith("y"):
        try:
            years = int(tp[1:])
            return years * 12
        except ValueError:
            return pd.NA
    if tp.startswith("v"):
        if tp in v_map:
            return v_map[tp]
    if timepoint in ["tau"]:
        return pd.NA
    return pd.NA

def get_session_date(row: pd.Series) -> pd.Timestamp:
    """Räknar ut besöksdatum baserat på baseline + månader om exam_date saknas."""
    if pd.notna(row.get("exam_date")):
        return row["exam_date"]

    baseline_date = row.get("baseline_date")
    months_from_bl = row.get("months_from_baseline")
    
    if pd.isna(baseline_date) or pd.isna(months_from_bl):
        return pd.NA
        
    try:
        session_date = baseline_date + pd.DateOffset(months=months_from_bl)
        return session_date
    except Exception:
        return pd.NA

# -----------------------------------------------------------------------------
# LOAD MRI
# -----------------------------------------------------------------------------
def load_mri_csv(adni_path="/home/maia-user/ADNI_npy"):
    """
    Laddar MRI-data baserat på OutputCollection.csv.
    """
    expected_cols = ["exam_id", "individual_id", "time_point", "file_path", "baseline_date", "months_from_baseline", "exam_date"]
    
    COLLECTION_PATH = os.path.join(adni_path, "csv", "OutputCollection.csv")
    DXSUM_PATH = os.path.join(adni_path, "csv", "DXSUM_10Feb2026.csv")
    
    if not os.path.exists(COLLECTION_PATH):
        print(f"[WARN] MRI CSV not found at {COLLECTION_PATH}. Returning empty DataFrame.")
        return pd.DataFrame(columns=expected_cols)
    
    # Load csv
    csv_output_collection = pd.read_csv(COLLECTION_PATH)
    csv_dxsum = pd.read_csv(DXSUM_PATH)
    valid_ptids = set(csv_dxsum["PTID"])
    # Load dxsum for validation (optional but good practice)
    # if os.path.exists(DXSUM_PATH):
    #     csv_dxsum = pd.read_csv(DXSUM_PATH)
    #     valid_ptids = set(csv_dxsum["PTID"])
    # else:
    #     valid_ptids = set(csv_output_collection["Individual's ID"]) # Fallback

    # Standardize columns
    mri = csv_output_collection.rename(columns={
        "Output collection GUID": "exam_id",
        "Individual's ID": "individual_id",
        "TimePoint": "time_point"
    }).copy()

    # Build .npy file path
    mri["file_path"] = mri.apply(build_mri_npy_file_path, args=(adni_path, "mri"), axis=1)
    
    # Filtering
    mri["included"] = (
        (mri["Job status"] == "completed") &
        mri["file_path"].apply(os.path.exists) &
        (mri["time_point"] != "tau") &
        mri["individual_id"].isin(valid_ptids)
    )

    mri = mri[mri["included"]].copy()
    
    # Prepare Diagnosis Summary for date calculations
    if os.path.exists(DXSUM_PATH):
        dxsum = csv_dxsum.rename(columns={"PTID": "individual_id"})
        dxsum["EXAMDATE"] = pd.to_datetime(dxsum["EXAMDATE"].fillna(dxsum["USERDATE"]), errors='coerce')
        
        # Calculate dates
        mri["baseline_date"] = mri["individual_id"].apply(get_baseline_date_for_subject, args=(dxsum,))
        mri["months_from_baseline"] = mri["time_point"].apply(get_months_from_baseline)
        mri["exam_date"] = mri.apply(get_session_date, axis=1)
    else:
        mri["exam_date"] = pd.NaT

    return mri[expected_cols]

# -----------------------------------------------------------------------------
# LOAD PET
# -----------------------------------------------------------------------------
def load_pet_csv(adni_path="/home/maia-user/ADNI_npy", pet_type="amy"):
    """
    Laddar PET-data. Prioriterar 'generated_pet_registry.csv' om den finns.
    """
    # 1. Försök ladda det nya registret vi skapade (Bäst)
    generated_registry_path = os.path.join(adni_path, "csv", "generated_pet_registry.csv")
    
    # # 2. Fallback: Gamla Berkeley-filen (Sämre, ID kan diffa)
    # berkeley_csv_name = "UCBERKELEY_AMY_6MM_02Apr2025.csv"
    # berkeley_path = os.path.join(adni_path, "csv", berkeley_csv_name)
    
    if os.path.exists(generated_registry_path):
        print(f"[INFO] Loading PET data from generated registry: {generated_registry_path}")
        pet = pd.read_csv(generated_registry_path)
        # generated_pet_registry har kolumnerna: individual_id, exam_id, exam_date, tracer, file_path
        
    # elif os.path.exists(berkeley_path):
    #     print(f"[WARN] Generated registry not found. Falling back to Berkeley CSV: {berkeley_path}")
    #     print("[WARN] NOTE: This might fail if .npy files were named using the new ID format.")
        
    #     pet_csv = pd.read_csv(berkeley_path, dtype={"VISCODE": str, "LONIUID": str, "PTID": str})
    #     pet = pet_csv.rename(columns={
    #         "LONIUID": "exam_id",
    #         "VISCODE": "viscode",
    #         "PTID": "individual_id",
    #         "SCANDATE": "exam_date",
    #     })
    else:
        print(f"[ERROR] No PET CSV found in {os.path.join(adni_path, 'csv')}")
        return pd.DataFrame(columns=["individual_id", "exam_id", "exam_date", "file_path"])

    # Gemensam processering
    pet["exam_date"] = pd.to_datetime(pet["exam_date"], errors='coerce')
    
    # BYGG .NPY SÖKVÄG
    # Här räknas file_path ut och pekar på den processade .npy-filen
    pet["file_path"] = pet.apply(build_pet_npy_file_path, args=(adni_path, pet_type), axis=1)
    
    # Filtrera: Bara filer som faktiskt finns på disken (.npy)
    pet["included"] = pet["file_path"].apply(os.path.exists)
    
    pet_df = pet[pet["included"]].copy()
    
    return pet_df[["individual_id", "exam_id", "exam_date", "file_path"]]

# -----------------------------------------------------------------------------
# CLINICAL STAGE LOGIC
# -----------------------------------------------------------------------------
def determine_clinical_stage(row: pd.Series) -> str:
    """Bestämmer kliniskt stadie (CN, MCI, AD) baserat på diagnos-koder."""
    dxcur = row.get("DXCURREN", pd.NA)
    dxchange = row.get("DXCHANGE", pd.NA)
    diagnosis = row.get("DIAGNOSIS", pd.NA)
    
    # ADNI3 (DIAGNOSIS)
    if pd.notna(diagnosis):
        if diagnosis == 1: return "CN"
        elif diagnosis == 2: return "MCI"
        elif diagnosis == 3: return "AD"
        
    # # ADNIGO/2 (DXCHANGE)
    # if pd.notna(dxchange):
    #     if dxchange in [1, 7, 9]: return "CN"
    #     elif dxchange in [2, 4, 8]: return "MCI"
    #     elif dxchange in [3, 5, 6]: return "AD"
        
    # # ADNI1 (DXCURREN)
    # if pd.notna(dxcur):
    #     if dxcur == 1: return "CN"
    #     elif dxcur == 2: return "MCI"
    #     elif dxcur == 3: return "AD"

    return pd.NA

def get_baseline_class_for_subject(subject_id: str, dxsum: pd.DataFrame) -> str:
    """Hämtar baslinje-diagnos för en patient."""
    subject_data = dxsum[dxsum["individual_id"] == subject_id]
    if subject_data.empty:
        return "Unknown"
    
    stage = _get_first_entry(subject_data, "bl", "clinical_stage")
    if pd.isna(stage):
        stage = _get_first_entry(subject_data, "sc", "clinical_stage")

    return stage if pd.notna(stage) else "Unknown"


# -----------------------------------------------------------------------------
# MAIN LOAD FUNCTION
# -----------------------------------------------------------------------------
def load_dataset(
    classes=["CN", "MCI", "AD"], 
    adni_path_mri="/home/maia-user/ADNI_npy", 
    adni_path_pet="/home/maia-user/ADNI_npy", 
    modalities=["mri", "amy"],
    seed=42,
    balanced=False 
):
    # 1. Ladda Modaliteter
    mri = load_mri_csv(adni_path_mri)
    pet = load_pet_csv(adni_path_pet, pet_type="amy")

    # 2. Ladda Diagnos-data
    DXSUM_PATH = os.path.join(adni_path_mri, "csv", "DXSUM_10Feb2026.csv")
    
    if not os.path.exists(DXSUM_PATH):
        print(f"[WARN] DXSUM not found. Creating dummy.")
        dxsum = pd.DataFrame(columns=["individual_id", "EXAMDATE", "clinical_stage"])
    else:
        csv_dxsum = pd.read_csv(DXSUM_PATH)
        dxsum = csv_dxsum.rename(columns={"PTID": "individual_id"})
        dxsum["EXAMDATE"] = pd.to_datetime(dxsum["EXAMDATE"].fillna(dxsum["USERDATE"]), errors='coerce')
        dxsum["clinical_stage"] = dxsum.apply(determine_clinical_stage, axis=1)
        dxsum = dxsum.dropna(subset=["clinical_stage"])

    # 3. Typkonvertering
    for df in [mri, pet]:
        if not df.empty:
            df["exam_date"] = pd.to_datetime(df["exam_date"])
            df["individual_id"] = df["individual_id"].astype(str)
        else:
            df["exam_date"] = pd.to_datetime([])
            df["individual_id"] = pd.Series([], dtype=str)

    # 4. MATCHNING (Strategi: Alla MRI får en PET om det finns)
    
    # A. Huvud-matchning: Varje MRI hämtar närmaste PET
    # Eftersom vi tillåter dubletter på PET-sidan (flera MRI mot samma PET), 
    # är detta vår huvudsakliga datakälla.
    mri_to_pet = pd.merge_asof(
        mri.sort_values("exam_date"), 
        pet.sort_values("exam_date"),
        by="individual_id", 
        left_on="exam_date", 
        right_on="exam_date",
        direction="nearest", 
        suffixes=("_mri", "_amy")
    )
    
    # B. Hitta "Föräldralösa" PET-bilder
    # Vi kollar åt andra hållet BARA för att hitta PET-bilder där patienten 
    # inte har någon MRI alls (eftersom mri_to_pet missar dessa).
    pet_to_mri = pd.merge_asof(
        pet.sort_values("exam_date"), 
        mri.sort_values("exam_date"),
        by="individual_id", 
        left_on="exam_date", 
        right_on="exam_date",
        direction="nearest", 
        suffixes=("_amy", "_mri")
    )
    
    # Säkra kolumner inför ihopslagning
    req_cols = ["individual_id", "exam_id_amy", "exam_id_mri", "file_path_amy", "file_path_mri"]
    for df in [mri_to_pet, pet_to_mri]:
        for col in req_cols:
            if col not in df.columns: df[col] = pd.NA

    # C. Sätt ihop allt
    # 1. Ta ALLA rader från mri_to_pet (Detta inkluderar 'Paired' och 'MRI-only')
    #    Här kan samma PET-bild dyka upp på flera rader, vilket var det du ville.
    main_dataset = mri_to_pet[req_cols + ["exam_date"]]
    
    # 2. Hitta PET-bilder som inte matchade någon MRI alls (unmatched PETs)
    unmatched_pet = pet_to_mri[pet_to_mri["file_path_mri"].isna()]
    
    # 3. Lägg ihop
    combined = pd.concat([main_dataset, unmatched_pet[req_cols + ["exam_date"]]], ignore_index=True)

    if combined.empty:
        print("[WARN] Dataset is empty after merge.")
        return pd.DataFrame()

    # 5. Matcha med Diagnos
    combined["anchor_date"] = combined["exam_date"]
    matched = pd.merge_asof(
        combined.sort_values("anchor_date"),
        dxsum[["individual_id", "EXAMDATE", "clinical_stage"]].sort_values("EXAMDATE"),
        by="individual_id", 
        left_on="anchor_date", 
        right_on="EXAMDATE",
        direction="nearest", 
        # tolerance=pd.Timedelta("365D")
    )

    final_df = matched[matched["clinical_stage"].isin(classes)].reset_index(drop=True)

    # -------------------------------------------------------------------------
    # BALANSERING (Uppdaterad för din nya logik)
    # -------------------------------------------------------------------------
    if balanced and len(modalities) > 1:
        print("--- [BALANCING ACTIVATED] ---")
        
        # Paired = Har både MRI och PET (Här ingår nu de MRI som delar på samma PET)
        paired = final_df[final_df["file_path_mri"].notna() & final_df["file_path_amy"].notna()]
        
        # MRI Only = Har MRI men hittade ingen PET för patienten
        mri_only = final_df[final_df["file_path_mri"].notna() & final_df["file_path_amy"].isna()]
        
        # PET Only = Har PET men ingen MRI (Extremt ovanligt med din volym, men möjligt)
        pet_only = final_df[final_df["file_path_amy"].notna() & final_df["file_path_mri"].isna()]

        n_paired = len(paired)
        n_mri_only = len(mri_only)
        n_pet_only = len(pet_only)

        print(f"Status: Paired (MRI+PET): {n_paired}, MRI-only: {n_mri_only}, PET-only: {n_pet_only}")

        # LOGIK: Vi vill använda ALLA parade exempel eftersom de är guld värda.
        # Vi fyller sedan på med "rena" MRI-bilder (mri_only) så att vi inte dränker PET-datan.
        
        # Exempel: Om du har 2000 parade (där 900 unika PET återanvänds) och 18000 MRI-only.
        # Då kanske vi vill ha max lika många MRI-only som vi har Paired.
        
        limit = n_paired + n_pet_only
        
        if n_mri_only > limit:
            mri_only_sampled = mri_only.sample(n=limit, random_state=seed)
            final_df = pd.concat([paired, pet_only, mri_only_sampled], ignore_index=True)
            print(f"Balansering: Begränsade MRI-only till {limit} st för att matcha mängden PET-data.")
        else:
            print("Balansering: Ingen downsampling behövdes (MRI-only < Paired).")

        print(f"Slutligt antal rader: {len(final_df)}")

    return final_df

def load_npy_dataset(classes=["CN", "MCI", "AD"], adni_path="/home/maia-user/ADNI_npy", modalities=["mri", "amy"], seed=42, balanced=False):
    """Wrapper med det nya argumentet."""
    return load_dataset(classes=classes, adni_path_mri=adni_path, adni_path_pet=adni_path, modalities=modalities, seed=seed, balanced=balanced)
