
import os
from sklearn.model_selection import StratifiedKFold

import logging
import os
import pandas as pd

logger = logging.getLogger(__name__)

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


def build_file_path(row: pd.Series, data_path, mri_filename) -> str:
    """Build the file path for a given row in the dataframe."""
    guid = row["Output collection GUID"]
    file_path = os.path.join(data_path, guid, mri_filename)
    return file_path


def determine_clinical_stage(row: pd.Series) -> str:
    """Determine the clinical stage based on diagnosis variables."""
    # Extract diagnosis variables
    dxcur = row.get("DXCURREN", pd.NA)
    dxchange = row.get("DXCHANGE", pd.NA)
    diagnosis = row.get("DIAGNOSIS", pd.NA)
    
    # Determine clinical stage based on ADNI3 (DIAGNOSIS)
    if pd.notna(diagnosis):
        if diagnosis == 1:
            return "CN"  # Cognitively Normal
        elif diagnosis == 2:
            return "MCI"  # Mild Cognitive Impairment
        elif diagnosis == 3:
            return "AD"  # Alzheimer's Disease
        
    # Determine clinical stage based on ADNIGO/2 (DXCHANGE)
    if pd.notna(dxchange):
        if dxchange in [1, 7, 9]:
            return "CN"  # Stable/Returned to Normal
        elif dxchange in [2, 8]:
            return "MCI"  # Stable/Returned to MCI
        elif dxchange in [3, 5, 6]:
            return "AD"  # Stable/Converted to Dementia
        elif dxchange == 4:
            return "MCI"  # Conversion to MCI
        
    # Determine clinical stage based on ADNI1 (DXCURREN)
    if pd.notna(dxcur):
        if dxcur == 1:
            return "CN"  # Cognitively Normal
        elif dxcur == 2:
            return "MCI"  # Mild Cognitive Impairment
        elif dxcur == 3:
            return "AD"  # Alzheimer's Disease

    return pd.NA  # If no valid data


def _get_first_entry(subject_data, viscode: str, column: str):
    entries = subject_data[subject_data["VISCODE2"] == viscode]
    if entries.empty:
        return pd.NA
    if len(entries) > 1:
        logger.warning("Multiple %s entries for subject %s", viscode, subject_data["individual_id"].iloc[0])
    return entries.iloc[0][column]


def get_baseline_date_for_subject(subject_id: str, dxsum: pd.DataFrame) -> pd.Timestamp:
    """Get the baseline date for a given subject ID."""
    subject_data = dxsum[dxsum["individual_id"] == subject_id]
    if subject_data.empty:
        logger.warning(f"No data found for subject {subject_id}.")
        return pd.NaT
    
    date = _get_first_entry(subject_data, "bl", "EXAMDATE")
    if pd.isna(date):
        date = _get_first_entry(subject_data, "sc", "EXAMDATE")

    if pd.isna(date):
        logger.warning(f"No valid baseline/screening date found for subject {subject_id}.")
        return pd.NaT

    return date

def get_baseline_class_for_subject(subject_id: str, dxsum: pd.DataFrame) -> str:
    """Get the baseline clinical stage for a given subject ID."""
    subject_data = dxsum[dxsum["individual_id"] == subject_id]
    if subject_data.empty:
        logger.warning(f"No data found for subject {subject_id}.")
        return "Unknown"
    
    stage = _get_first_entry(subject_data, "bl", "clinical_stage")
    if pd.isna(stage):
        stage = _get_first_entry(subject_data, "sc", "clinical_stage")

    if pd.isna(stage):
        logger.warning(f"No baseline/screening clinical stage for subject {subject_id}.")
        return "Unknown"
    
    return stage


def get_months_from_baseline(timepoint: str) -> float:
    """Convert timepoint string to months from baseline."""

    if pd.isna(timepoint):
        return pd.NA
    if timepoint in ["sc", "init"]:
        return 0
    if timepoint == "bl":
        return 0
    if timepoint.startswith("m"):
        try:
            return int(timepoint[1:])
        except ValueError:
            return pd.NA
    if timepoint.startswith("y"):
        try:
            years = int(timepoint[1:])
            return years * 12
        except ValueError:
            return pd.NA
    if timepoint.startswith("v"):
        if timepoint in v_map:
            return v_map[timepoint]
    if timepoint in ["tau"]:
        return pd.NA
    return pd.NA


def get_session_date(row: pd.Series) -> pd.Timestamp:
    """Calculate the session date based on baseline date and months from baseline."""
    baseline_date = row["baseline_date"]
    months_from_bl = row["months_from_baseline"]
    if pd.isna(baseline_date) or pd.isna(months_from_bl):
        logger.warning(f"Missing data for row {row.name} for subject {row['individual_id']}: baseline_date={baseline_date}, months_from_baseline={months_from_bl}")
        # detta är de som är tau istället för tidsperiod
        return pd.NA
    try:
        session_date = baseline_date + pd.DateOffset(months=months_from_bl)
        return session_date
    except Exception as e:
        logger.warning(f"Error calculating session date for row {row.name}: {e}")
        return pd.NA


def setup_mri_dataframe(classes=["CN", "MCI", "AD"], adni_path="/home/maia-user/ADNI_complete"):

    # ADNI_PATH = "/home/maia-user/ADNI_complete"
    #ADNI_PATH = "/proj/berzbiomedicalimagingkth/users/x_julwe/ADNI/ADNI_complete"
    DATA_PATH = os.path.join(adni_path, "adni")
    COLLECTION_PATH = os.path.join(adni_path, "OutputCollection.csv")
    # DEMOGRAPHICS_PATH = os.path.join(ADNI_PATH, "participant_demographics.csv")
    DXSUM_PATH = os.path.join(adni_path, "DXSUM_PDXCONV_ADNIALL.csv")
    MRI_FILENAME = "mri/rescaled_align_norm.nii.gz"

    # -- Load csv --
    csv_output_collection = pd.read_csv(COLLECTION_PATH)
    csv_dxsum = pd.read_csv(DXSUM_PATH)

    # -- Build filepath and inclusion flags --
    csv_output_collection["file_path"] = csv_output_collection.apply(build_file_path, args=(DATA_PATH, MRI_FILENAME), axis=1)
    csv_output_collection["included"] = (
        (csv_output_collection["Job status"] == "completed")
        & csv_output_collection["file_path"].apply(os.path.exists)
        & (csv_output_collection["TimePoint"] != "tau")
        & csv_output_collection["Individual's ID"].isin(csv_dxsum["PTID"])
    )

    # -- Prepare mri dataset --
    mri = csv_output_collection[csv_output_collection["included"]].copy()
    mri = mri[[
        "Output collection GUID", 
        "Individual's ID", 
        "TimePoint", 
        "file_path"
    ]].rename(columns={
        "Output collection GUID": "exam_id",
        "Individual's ID": "individual_id",
        "TimePoint": "time_point"
    })

    # -- Prepare diagnosis summary --
    dxsum = csv_dxsum[[
        "PTID",
        "EXAMDATE",
        "USERDATE",
        "VISCODE", 
        "VISCODE2",
        "DXCURREN",
        "DXCHANGE",
        "DIAGNOSIS"
    ]].rename(columns={
        "PTID": "individual_id",
    })
    dxsum["EXAMDATE"] = dxsum["EXAMDATE"].fillna(dxsum["USERDATE"])
    dxsum["EXAMDATE"] = pd.to_datetime(dxsum["EXAMDATE"], errors='coerce')
    dxsum["clinical_stage"] = dxsum.apply(determine_clinical_stage, axis=1)
    dxsum = dxsum.dropna(subset=["clinical_stage"])

    # -- Add baseline info --
    mri["baseline_date"] = mri["individual_id"].apply(get_baseline_date_for_subject, args=(dxsum,))
    mri["baseline_class"] = mri["individual_id"].apply(get_baseline_class_for_subject, args=(dxsum,))
    mri["months_from_baseline"] = mri["time_point"].apply(get_months_from_baseline)
    mri["est_exam_date"] = mri.apply(get_session_date, axis=1)

    # -- Merge mri with diagnosis info --
    matched = pd.merge_asof(
        mri.dropna(subset=["est_exam_date"]).sort_values("est_exam_date"),
        dxsum[["individual_id", "EXAMDATE", "VISCODE2", "clinical_stage"]]
            .sort_values("EXAMDATE"),
        by="individual_id",
        left_on="est_exam_date",
        right_on="EXAMDATE",
        direction="nearest"   
    ).dropna(subset=["clinical_stage"])

    mri_merged = matched[[
        "exam_id", 
        "individual_id", 
        "EXAMDATE",
        "baseline_class",
        "baseline_date",
        "clinical_stage",
        "file_path",
    ]].rename(columns={
        "EXAMDATE": "exam_date"
    })
    mri_merged = mri_merged[mri_merged["clinical_stage"].isin(classes)].reset_index(drop=True)

    return mri_merged #[:100]  # TODO: remove. for testing purposes, use only first 100 entries

def build_npy_file_path(row, preprocessed_root):
    """
    Bygger paths till .npy och omvandlar clinical_stage → label.
    Returnerar:
        X_paths : np.array med paths
        y       : np.array med int labels
    """

    subject = row["Individual's ID"]
    exam_id = row["Output collection GUID"]

    npy_path = os.path.join(
        preprocessed_root,
        subject,
        "mri",
        f"{exam_id}.npy"
    )

    return npy_path

def setup_npy_mri_dataframe(classes=["CN", "MCI", "AD"], adni_path="/home/maia-user/ADNI_npy"):

    # ADNI_PATH = "/home/maia-user/ADNI_complete"
    #ADNI_PATH = "/proj/berzbiomedicalimagingkth/users/x_julwe/ADNI/ADNI_complete"
    # DATA_PATH = os.path.join(ADNI_PATH, "adni")
    COLLECTION_PATH = os.path.join(adni_path, "csv", "OutputCollection.csv")
    # DEMOGRAPHICS_PATH = os.path.join(ADNI_PATH, "participant_demographics.csv")
    DXSUM_PATH = os.path.join(adni_path, "csv", "DXSUM_PDXCONV_ADNIALL.csv")
    # MRI_FILENAME = "mri/rescaled_align_norm.nii.gz"

    # -- Load csv --
    csv_output_collection = pd.read_csv(COLLECTION_PATH)
    csv_dxsum = pd.read_csv(DXSUM_PATH)

    # -- Build filepath and inclusion flags --
    csv_output_collection["file_path"] = csv_output_collection.apply(build_npy_file_path, args=(adni_path,), axis=1)
    csv_output_collection["included"] = (
        (csv_output_collection["Job status"] == "completed")
        & csv_output_collection["file_path"].apply(os.path.exists)
        & (csv_output_collection["TimePoint"] != "tau")
        & csv_output_collection["Individual's ID"].isin(csv_dxsum["PTID"])
    )

    # -- Prepare mri dataset --
    mri = csv_output_collection[csv_output_collection["included"]].copy()
    mri = mri[[
        "Output collection GUID", 
        "Individual's ID", 
        "TimePoint", 
        "file_path"
    ]].rename(columns={
        "Output collection GUID": "exam_id",
        "Individual's ID": "individual_id",
        "TimePoint": "time_point"
    })

    # -- Prepare diagnosis summary --
    dxsum = csv_dxsum[[
        "PTID",
        "EXAMDATE",
        "USERDATE",
        "VISCODE", 
        "VISCODE2",
        "DXCURREN",
        "DXCHANGE",
        "DIAGNOSIS"
    ]].rename(columns={
        "PTID": "individual_id",
    })
    dxsum["EXAMDATE"] = dxsum["EXAMDATE"].fillna(dxsum["USERDATE"])
    dxsum["EXAMDATE"] = pd.to_datetime(dxsum["EXAMDATE"], errors='coerce')
    dxsum["clinical_stage"] = dxsum.apply(determine_clinical_stage, axis=1)
    dxsum = dxsum.dropna(subset=["clinical_stage"])

    # -- Add baseline info --
    mri["baseline_date"] = mri["individual_id"].apply(get_baseline_date_for_subject, args=(dxsum,))
    mri["baseline_class"] = mri["individual_id"].apply(get_baseline_class_for_subject, args=(dxsum,))
    mri["months_from_baseline"] = mri["time_point"].apply(get_months_from_baseline)
    mri["est_exam_date"] = mri.apply(get_session_date, axis=1)

    # -- Merge mri with diagnosis info --
    matched = pd.merge_asof(
        mri.dropna(subset=["est_exam_date"]).sort_values("est_exam_date"),
        dxsum[["individual_id", "EXAMDATE", "VISCODE2", "clinical_stage"]]
            .sort_values("EXAMDATE"),
        by="individual_id",
        left_on="est_exam_date",
        right_on="EXAMDATE",
        direction="nearest"   
    ).dropna(subset=["clinical_stage"])

    mri_merged = matched[[
        "exam_id", 
        "individual_id", 
        "EXAMDATE",
        "baseline_class",
        "baseline_date",
        "clinical_stage",
        "file_path",
    ]].rename(columns={
        "EXAMDATE": "exam_date"
    })
    mri_merged = mri_merged[mri_merged["clinical_stage"].isin(classes)].reset_index(drop=True)

    return mri_merged #[:110]  # TODO: remove. for testing purposes, use only first 100 entries