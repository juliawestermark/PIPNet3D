import os
import errno
import sys
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm.auto import tqdm
import shutil
from datetime import datetime

from make_pet_dataset import create_pet_dataframe_from_filesystem

def convert_single_file(nii_path, save_path):
    """Load NIfTI → Normalize → Save as .npy."""
    img = nib.load(nii_path)
    arr = img.get_fdata().astype(np.float32)
    
    # Min-Max normalization (0 to 1)
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)
    
    # Cast to float16 to save disk space
    arr = arr.astype(np.float16)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, arr)


def convert_all_pet(df, output_root, pet_type):
    """
    Converts NIfTI PET files to .npy using the structure:
    <output_root>/<subject_id>/<pet_type>/<exam_id>.npy
    """
    processed = 0
    converted = 0
    skipped = 0
    failed = 0

    print(f"[INFO] Starting conversion of {len(df)} files...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        processed += 1
        nii_path = row["file_path"]
        subject = row["individual_id"]
        exam_id = row["exam_id"]

        save_path = os.path.join(output_root, subject, pet_type, f"{exam_id}.npy")

        if os.path.exists(save_path):
            try:
                # Verify file integrity without loading fully into RAM
                np.load(save_path, mmap_mode='r')
                skipped += 1
                continue 
            except Exception:
                print(f"[WARN] Corrupted file found, rewriting: {save_path}")

        try:
            convert_single_file(nii_path, save_path)
            converted += 1
        except OSError as e:
            if e.errno in (errno.ENOSPC, errno.EDQUOT, errno.EIO, errno.EROFS):
                print(f"[CRITICAL] Stopping — filesystem error: {e}")
                sys.exit(1)

            print(f"[ERROR] OS error on {nii_path}: {e}")
            failed += 1
        except Exception as e:
            failed += 1
            print(f"[ERROR] Unexpected error on {nii_path}: {e}")

    return processed, converted, skipped, failed


def manage_csv_files(output_csv_dir, pet_df, other_csv_paths=[]):
    """Saves the PET registry and copies relevant metadata CSVs."""
    os.makedirs(output_csv_dir, exist_ok=True)

    pet_reg_path = os.path.join(output_csv_dir, "generated_pet_registry.csv")
    pet_df.to_csv(pet_reg_path, index=False)
    print(f"[INFO] Saved PET registry to: {pet_reg_path}")

    for src in other_csv_paths:
        if src and os.path.exists(src):
            filename = os.path.basename(src)
            dst = os.path.join(output_csv_dir, filename)
            
            if not os.path.exists(dst):
                print(f"[INFO] Copying {filename} -> {output_csv_dir}")
                shutil.copy2(src, dst)
        elif src:
            print(f"[WARN] Source CSV not found: {src}")


if __name__ == "__main__":
    print("Start time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # --- CONFIGURATION ---
    ADNI_PET_PATH = "/proj/berzbiomedicalimagingkth/ADNI_NIFTI_AMYLOID_CoregAvg_StdImgVox"
    #ADNI_PET_PATH = "/home/maia-user/ADNI_PET/ADNI_NEW"
    # OUTPUT_ROOT = "/home/maia-user/ADNI_npy"
    OUTPUT_ROOT = "/proj/berzbiomedicalimagingkth/users/x_julwe/ADNI_npy"
    
    # CSV_FOLDER_PATH = "/home/maia-user/ADNI_PET/csv"
    CSV_FOLDER_PATH = "/proj/berzbiomedicalimagingkth/users/x_julwe/ADNI/csv"
    DXSUM_CSV_SOURCE = os.path.join(CSV_FOLDER_PATH, "DXSUM_10Feb2026.csv")
    PTDEM_CSV_SOURCE = os.path.join(CSV_FOLDER_PATH, "PTDEMOG_10Feb2026.csv")

    CSV_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "csv")
    PET_TYPE = "amy"

    # --- EXECUTION ---
    print("--- Step 1: Scanning Filesystem ---")
    df = create_pet_dataframe_from_filesystem(ADNI_PET_PATH)
    
    if len(df) == 0:
        print("[ERROR] No PET files found! Check path.")
        sys.exit(1)

    print("--- Step 2: Managing CSV files ---")
    manage_csv_files(
        CSV_OUTPUT_DIR, 
        df, 
        other_csv_paths=[DXSUM_CSV_SOURCE, PTDEM_CSV_SOURCE]
    )
    
    print(f"--- Step 3: Converting {len(df)} PET files to .npy ---")
    processed, converted, skipped, failed = convert_all_pet(df, OUTPUT_ROOT, PET_TYPE)

    print("All done!")
    print(f"Preprocessed data saved to {OUTPUT_ROOT}")
    print(f"Summary: Processed: {processed}, Converted: {converted}, Skipped: {skipped}, Failed: {failed}")
    print("End time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))