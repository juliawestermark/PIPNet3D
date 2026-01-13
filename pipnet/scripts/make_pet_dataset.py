import pandas as pd
import os
from pathlib import Path
import re
from datetime import datetime


def find_pet_file(individual_id, exam_id, exam_date, identifier, adni_path):
    pet_path = Path(adni_path)
    string_date = exam_date.strftime("%Y%m%d")
    search_path = pet_path / individual_id
    pattern = f"*{identifier}*{string_date}*.nii.gz"
    # pattern = rf"*{exam_id}*.nii.gz"
    # pattern = rf"{exam_id}.*{amy_identifier}.*\.nii\.gz"
    # print(f"Söker i: {search_path}, mönster: {pattern}")
    matches = list(search_path.glob(pattern))
    # if len(matches) > 1:
    #     print(f"Varning: Flera träffar för {individual_id}, {exam_id}: {matches}")
    # matches = [f for f in search_path.rglob("*.nii.gz") if re.search(pattern, f.name)]
    if matches:
        return str(matches[0])
    return None


def setup_pet_dataframe(identifier="AV45", adni_path="/home/maia-user/ADNI_PET/ADNI", adni_pet_file_name="UCBERKELEY_AMY_6MM_02Apr2025.csv"):
    # amy_csv_file = "/home/maia-user/ADNI_PET/ADNI/UCBERKELEY_AMY_6MM_02Apr2025.csv"
    # tau_csv_file = "/home/maia-user/ADNI_PET/ADNI/UCBERKELEY_TAU_6MM_02Apr2025.csv"
    adni_file_path = os.path.join(adni_path, adni_pet_file_name)
    pet_csv = pd.read_csv(adni_file_path, dtype={"VISCODE": str, "LONIUID": str, "PTID": str})
    # amy_csv.head()
    pet = pet_csv[[
        "LONIUID", 
        "VISCODE", 
        "PTID", 
        "SCANDATE",
        # "PROCESSDATE"
    ]].rename(columns={
        "LONIUID": "exam_id",
        "VISCODE": "viscode",
        "PTID": "individual_id",
        "SCANDATE": "exam_date",
        # "PROCESSDATE": "process_date"
    }).copy()
    pet["exam_date"] = pd.to_datetime(pet["exam_date"], errors='coerce')

    pet["file_path"] = pet.apply(
        lambda row: find_pet_file(row["individual_id"], row["exam_id"], row["exam_date"], identifier, adni_path=adni_path), axis=1
    )

    pet_df = pet[pet["file_path"].notna()].copy()

    return pet_df

def build_npy_file_path(row, preprocessed_root, pet_type):
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
        pet_type,
        f"{exam_id}.npy"
    )

    return npy_path

def setup_npy_pet_dataframe(adni_path="/home/maia-user/ADNI_npy", adni_pet_file_name="UCBERKELEY_AMY_6MM_02Apr2025.csv", pet_type="amy"):
    # amy_csv_file = "/home/maia-user/ADNI_PET/ADNI/UCBERKELEY_AMY_6MM_02Apr2025.csv"
    # tau_csv_file = "/home/maia-user/ADNI_PET/ADNI/UCBERKELEY_TAU_6MM_02Apr2025.csv"
    adni_file_path = os.path.join(adni_path, adni_pet_file_name)
    pet_csv = pd.read_csv(adni_file_path, dtype={"VISCODE": str, "LONIUID": str, "PTID": str})
    # amy_csv.head()
    pet = pet_csv[[
        "LONIUID", 
        "VISCODE", 
        "PTID", 
        "SCANDATE",
        # "PROCESSDATE"
    ]].rename(columns={
        "LONIUID": "exam_id",
        "VISCODE": "viscode",
        "PTID": "individual_id",
        "SCANDATE": "exam_date",
        # "PROCESSDATE": "process_date"
    }).copy()
    pet["exam_date"] = pd.to_datetime(pet["exam_date"], errors='coerce')

    # pet["file_path"] = pet.apply(
    #     lambda row: find_pet_file(row["individual_id"], row["exam_id"], row["exam_date"], identifier, adni_path=adni_path), axis=1
    # )
    pet["file_path"] = pet.apply(build_npy_file_path, args=(adni_path, pet_type), axis=1)

    pet_df = pet[pet["file_path"].notna()].copy()

    return pet_df
