import os
import pandas as pd
import numpy as np 

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

# -----------------------------------------------------------------------------
# HELPERS (Paths & Dates)
# -----------------------------------------------------------------------------
def build_mri_npy_file_path(row, preprocessed_root, image_type="mri"):
    return os.path.join(preprocessed_root, row["individual_id"], image_type, f"{row['exam_id']}.npy")

def build_pet_npy_file_path(row, preprocessed_root, pet_type="amy"):
    return os.path.join(preprocessed_root, row["individual_id"], pet_type, f"{row['exam_id']}.npy")

def _get_first_entry(subject_data, viscode: str, column: str):
    entries = subject_data[subject_data["VISCODE2"] == viscode]
    return entries.iloc[0][column] if not entries.empty else pd.NA

def get_baseline_date_for_subject(subject_id: str, dxsum: pd.DataFrame) -> pd.Timestamp:
    subject_data = dxsum[dxsum["individual_id"] == subject_id]
    if subject_data.empty: return pd.NaT
    date = _get_first_entry(subject_data, "bl", "EXAMDATE")
    if pd.isna(date): date = _get_first_entry(subject_data, "sc", "EXAMDATE")
    return date if pd.notna(date) else pd.NaT

def get_months_from_baseline(timepoint: str) -> float:
    if pd.isna(timepoint): return pd.NA
    tp = str(timepoint).lower()
    if tp in ["sc", "init", "bl"]: return 0
    if tp.startswith("m"):
        try: return int(tp[1:])
        except: return pd.NA
    if tp.startswith("y"):
        try: return int(tp[1:]) * 12
        except: return pd.NA
    if tp.startswith("v") and tp in v_map: return v_map[tp]
    return pd.NA

def get_session_date(row: pd.Series) -> pd.Timestamp:
    if pd.notna(row.get("exam_date")): return row["exam_date"]
    try:
        return row["baseline_date"] + pd.DateOffset(months=row["months_from_baseline"])
    except: return pd.NA

def determine_clinical_stage(row: pd.Series) -> str:
    diagnosis = row.get("DIAGNOSIS", pd.NA)
    if pd.notna(diagnosis):
        if diagnosis == 1: return "CN"
        elif diagnosis == 2: return "MCI"
        elif diagnosis == 3: return "AD"
    return pd.NA

# -----------------------------------------------------------------------------
# LOADERS (CSV)
# -----------------------------------------------------------------------------
def load_mri_csv(adni_path):
    expected_cols = ["exam_id", "individual_id", "time_point", "file_path", "baseline_date", "months_from_baseline", "exam_date"]
    col_path = os.path.join(adni_path, "csv", "OutputCollection.csv")
    dx_path = os.path.join(adni_path, "csv", "DXSUM_10Feb2026.csv")
    
    if not os.path.exists(col_path): return pd.DataFrame(columns=expected_cols)
    
    mri = pd.read_csv(col_path).rename(columns={
        "Output collection GUID": "exam_id", "Individual's ID": "individual_id", "TimePoint": "time_point"
    })
    
    # Validation against DXSUM
    valid_ids = set(mri["individual_id"])
    if os.path.exists(dx_path):
        dx = pd.read_csv(dx_path)
        if "PTID" in dx.columns: valid_ids = set(dx["PTID"])

    mri["file_path"] = mri.apply(build_mri_npy_file_path, args=(adni_path, "mri"), axis=1)
    mri = mri[(mri["Job status"] == "completed") & mri["file_path"].apply(os.path.exists) & (mri["time_point"] != "tau") & mri["individual_id"].isin(valid_ids)].copy()

    # Date logic
    if os.path.exists(dx_path):
        dx = pd.read_csv(dx_path).rename(columns={"PTID": "individual_id"})
        dx["EXAMDATE"] = pd.to_datetime(dx["EXAMDATE"].fillna(dx["USERDATE"]), errors='coerce')
        mri["baseline_date"] = mri["individual_id"].apply(get_baseline_date_for_subject, args=(dx,))
        mri["months_from_baseline"] = mri["time_point"].apply(get_months_from_baseline)
        mri["exam_date"] = mri.apply(get_session_date, axis=1)
    else:
        mri["exam_date"] = pd.NaT

    return mri[expected_cols]

def load_pet_csv(adni_path, pet_type="amy"):
    reg_path = os.path.join(adni_path, "csv", "generated_pet_registry.csv")
    if not os.path.exists(reg_path): return pd.DataFrame(columns=["individual_id", "exam_id", "exam_date", "file_path"])
    
    pet = pd.read_csv(reg_path)
    pet["exam_date"] = pd.to_datetime(pet["exam_date"], errors='coerce')
    pet["file_path"] = pet.apply(build_pet_npy_file_path, args=(adni_path, pet_type), axis=1)
    return pet[pet["file_path"].apply(os.path.exists)].copy()[["individual_id", "exam_id", "exam_date", "file_path"]]


def load_single_modality_dataset(classes, adni_path_mri, adni_path_pet, modality="mri", seed=42):
    print(f"--- Loading SINGLE MODALITY dataset: {modality} ---")
    
    if modality == "mri": df = load_mri_csv(adni_path_mri)
    elif modality == "amy": df = load_pet_csv(adni_path_pet, pet_type="amy")
    else: return pd.DataFrame()

    if df.empty: return pd.DataFrame()

    # Diagnosis
    dx_path = os.path.join(adni_path_mri, "csv", "DXSUM_10Feb2026.csv")
    if not os.path.exists(dx_path): return pd.DataFrame()
    
    dxsum = pd.read_csv(dx_path)
    if "PTID" in dxsum.columns: dxsum = dxsum.rename(columns={"PTID": "individual_id"})
    dxsum["EXAMDATE"] = pd.to_datetime(dxsum["EXAMDATE"].fillna(dxsum["USERDATE"]), errors='coerce')
    dxsum["clinical_stage"] = dxsum.apply(determine_clinical_stage, axis=1)
    dxsum = dxsum.dropna(subset=["clinical_stage"])

    df["exam_date"] = pd.to_datetime(df["exam_date"])
    df["individual_id"] = df["individual_id"].astype(str)
    
    # Match diagnosis
    df["anchor_date"] = df["exam_date"]
    matched = pd.merge_asof(
        df.sort_values("anchor_date"),
        dxsum[["individual_id", "EXAMDATE", "clinical_stage"]].sort_values("EXAMDATE"),
        by="individual_id", left_on="anchor_date", right_on="EXAMDATE",
        direction="nearest", tolerance=pd.Timedelta("365D")
    )

    final_df = matched[matched["clinical_stage"].isin(classes)].reset_index(drop=True)
    
    # Deduplication for Single Modality
    col_name = "file_path" if "file_path" in final_df.columns else f"file_path_{modality}"
    if col_name in final_df.columns:
        final_df["time_diff"] = (final_df["exam_date"] - final_df["anchor_date"]).abs()
        final_df = final_df.sort_values(by=["individual_id", "time_diff"])
        final_df = final_df.drop_duplicates(subset=[col_name], keep='first').drop(columns=["time_diff"])

    # Rename to standard format
    if "file_path" in final_df.columns:
        final_df = final_df.rename(columns={"file_path": f"file_path_{modality}", "exam_id": f"exam_id_{modality}"})

    print(f"Loaded {len(final_df)} images for {modality}.")
    return final_df


def load_multimodal_dataset(classes, adni_path_mri, adni_path_pet, modalities, seed=42, balanced=False):
    print(f"--- Loading MULTIMODAL dataset: {modalities} ---")

    mri = load_mri_csv(adni_path_mri)
    pet = load_pet_csv(adni_path_pet, pet_type="amy")

    dx_path = os.path.join(adni_path_mri, "csv", "DXSUM_10Feb2026.csv")
    if not os.path.exists(dx_path): return pd.DataFrame()
    
    dxsum = pd.read_csv(dx_path)
    if "PTID" in dxsum.columns: dxsum = dxsum.rename(columns={"PTID": "individual_id"})
    dxsum["EXAMDATE"] = pd.to_datetime(dxsum["EXAMDATE"].fillna(dxsum["USERDATE"]), errors='coerce')
    dxsum["clinical_stage"] = dxsum.apply(determine_clinical_stage, axis=1)
    dxsum = dxsum.dropna(subset=["clinical_stage"])

    for df in [mri, pet]:
        df["exam_date"] = pd.to_datetime(df["exam_date"])
        df["individual_id"] = df["individual_id"].astype(str)

    # Matching
    mri_to_pet = pd.merge_asof(
        mri.sort_values("exam_date"), pet.sort_values("exam_date"),
        by="individual_id", left_on="exam_date", right_on="exam_date",
        direction="nearest", suffixes=("_mri", "_amy")
    )
    
    pet_to_mri = pd.merge_asof(
        pet.sort_values("exam_date"), mri.sort_values("exam_date"),
        by="individual_id", left_on="exam_date", right_on="exam_date",
        direction="nearest", suffixes=("_amy", "_mri")
    )
    
    req_cols = ["individual_id", "exam_id_amy", "exam_id_mri", "file_path_amy", "file_path_mri"]
    for df in [mri_to_pet, pet_to_mri]:
        for col in req_cols:
            if col not in df.columns: df[col] = pd.NA

    combined = pd.concat([
        mri_to_pet[req_cols + ["exam_date"]], 
        pet_to_mri[pet_to_mri["file_path_mri"].isna()][req_cols + ["exam_date"]]
    ], ignore_index=True)

    if combined.empty: return pd.DataFrame()

    # Diagnosis
    combined["anchor_date"] = combined["exam_date"]
    matched = pd.merge_asof(
        combined.sort_values("anchor_date"),
        dxsum[["individual_id", "EXAMDATE", "clinical_stage"]].sort_values("EXAMDATE"),
        by="individual_id", left_on="anchor_date", right_on="EXAMDATE",
        direction="nearest", tolerance=pd.Timedelta("365D")
    )

    final_df = matched[matched["clinical_stage"].isin(classes)].reset_index(drop=True)

    # -------------------------------------------------------------------------
    # BALANCING (Only PAIRED DATA if balanced=True)
    # -------------------------------------------------------------------------
    if balanced:
        print("--- [BALANCING ACTIVATED: STRICT PAIRED ONLY] ---")
        n_before = len(final_df)
        
        required_cols = [f"file_path_{mod}" for mod in modalities]
        
        final_df = final_df.dropna(subset=required_cols, how='any').reset_index(drop=True)
        
        n_after = len(final_df)
        print(f"Balancing: Kept only paired data. Rows: {n_before} -> {n_after}")

    # Filter out completely empty rows
    cols_to_check = [f"file_path_{mod}" for mod in modalities]
    valid_cols = [c for c in cols_to_check if c in final_df.columns]
    if valid_cols:
        final_df = final_df.dropna(subset=valid_cols, how='all').reset_index(drop=True)

    return final_df


def load_npy_dataset(
    classes=["CN", "MCI", "AD"], 
    adni_path="/home/maia-user/ADNI_npy",
    modalities=["mri", "amy"],
    seed=42,
    balanced=False 
):
    if len(modalities) == 1:
        return load_single_modality_dataset(classes, adni_path, adni_path, modalities[0], seed)
    else:
        return load_multimodal_dataset(classes, adni_path, adni_path, modalities, seed, balanced)