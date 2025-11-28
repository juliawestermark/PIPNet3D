import os
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
import shutil
from make_mri_dataset import setup_mri_dataframe
from datetime import datetime

def convert_single_file(nii_path, save_path):
    """Load NIfTI → normalize → save as .npy."""
    img = nib.load(nii_path)
    arr = img.get_fdata().astype(np.float32)

    # normalize safely
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, arr)


def convert_all_mri(df, output_root):
    """
    df = mri_merged returned by setup_mri_dataframe()
    Converts all NIfTI MRI files to .npy in structure:
    
        <output_root>/<subject_id>/mri/<exam_id>.npy
    """

    converted = 0
    skipped = 0
    failed = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        nii_path = row["file_path"]
        subject = row["individual_id"]
        exam_id = row["exam_id"]

        # output path
        save_path = os.path.join(output_root, subject, "mri", f"{exam_id}.npy")

        # SKIP if file already exists
        if os.path.exists(save_path):
            # If you want to verify file integrity, you can load it here.
            # Try loading and skip only if load succeeds:
            try:
                np.load(save_path)
                skipped += 1
                continue  # file exists and is valid → skip
            except Exception:
                print(f"Corrupted file detected, rewriting: {save_path}")

        try:
            convert_single_file(nii_path, save_path)
            converted += 1
        except Exception as e:
            failed += 1
            print(f"ERROR processing {nii_path}: {e}")
    return converted, skipped, failed


def copy_csv_files(output_path, collection_path, demographics_path, dxsum_path):
    """Copy relevant ADNI CSVs into OUTPUT_ROOT/csv/."""
    os.makedirs(output_path, exist_ok=True)

    files = {
        "OutputCollection.csv": collection_path,
        "participant_demographics.csv": demographics_path,
        "DXSUM_PDXCONV_ADNIALL.csv": dxsum_path,
    }

    for filename, src in files.items():
        dst = os.path.join(CSV_OUTPUT_DIR, filename)
        if os.path.exists(dst):
            print(f"File already exists, skipping: {dst}")
            continue
        print(f"Copying {src} → {dst}")
        shutil.copy2(src, dst)


if __name__ == "__main__":

    print("Start time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # Paths
    ADNI_PATH = "/home/maia-user/ADNI_complete"
    COLLECTION_PATH = os.path.join(ADNI_PATH, "OutputCollection.csv")
    DEMOGRAPHICS_PATH = os.path.join(ADNI_PATH, "participant_demographics.csv")
    DXSUM_PATH = os.path.join(ADNI_PATH, "DXSUM_PDXCONV_ADNIALL.csv")
    OUTPUT_ROOT = "/home/maia-user/ADNI_npy"
    CSV_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "csv")

    # Prepare dataframe
    df = setup_mri_dataframe(ADNI_PATH="/home/maia-user/ADNI_complete")

    # Convert all MRI files
    converted, skipped, failed = convert_all_mri(df, OUTPUT_ROOT)
    # converted, skipped, failed = convert_all_mri(df.iloc[0:200], OUTPUT_ROOT)

    # Copy CSV files
    copy_csv_files(CSV_OUTPUT_DIR, COLLECTION_PATH, DEMOGRAPHICS_PATH, DXSUM_PATH)

    # Done
    print("All done!")
    print(f"Preprocessed data saved to {OUTPUT_ROOT}")
    print(f"Converted: {converted}, Skipped: {skipped}, Failed: {failed}")
    print("End time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


