import os
import errno
import sys
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm.auto import tqdm
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
    arr = arr.astype(np.float16)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, arr)


def convert_all_mri(df, output_root):
    """
    df = mri_merged returned by setup_mri_dataframe()
    Converts all NIfTI MRI files to .npy in structure:
    
        <output_root>/<subject_id>/mri/<exam_id>.npy
    """

    processed = 0
    converted = 0
    skipped = 0
    failed = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        processed += 1
        nii_path = row["file_path"]
        subject = row["individual_id"]
        exam_id = row["exam_id"]

        # output path
        save_path = os.path.join(output_root, subject, "mri", f"{exam_id}.npy")

        # SKIP if file already exists
        if os.path.exists(save_path):
            try:
                np.load(save_path)
                skipped += 1
                continue  # file exists and is valid → skip
            except Exception:
                print(f"[WARN] Corrupted file found, rewriting: {save_path}")

        try:
            convert_single_file(nii_path, save_path)
            converted += 1
        except OSError as e:
            # === Critical filesystem errors: stop entire run ===
            if e.errno in (errno.ENOSPC, errno.EDQUOT, errno.EIO, errno.EROFS):
                not_processed = len(df) - processed
                print(f"[CRITICAL] Stopping — disk or filesystem error: {e}")
                print(f"Converted: {converted}, Skipped: {skipped}, Failed: {failed}, Not processed: {not_processed}")
                sys.exit(1)

            # === Non-critical filesystem errors — skip file ===
            if e.errno in (errno.ENOENT, errno.EACCES, errno.ENOTDIR,
                           errno.EISDIR, errno.EINVAL, errno.EFBIG):
                print(f"[ERROR] OS error on {nii_path}: {e}")
                failed += 1
                continue

            # Unknown os error → rethrow
            raise
        except Exception as e:
            failed += 1
            print(f"[ERROR] Unexpected error on {nii_path}: {e}")
    return processed, converted, skipped, failed


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
    # ADNI_PATH = "/proj/berzbiomedicalimagingkth/users/x_julwe/ADNI/ADNI_complete"
    OUTPUT_ROOT = "/home/maia-user/ADNI_npy"
    # OUTPUT_ROOT = "/proj/berzbiomedicalimagingkth/users/x_julwe/ADNI_npy"

    COLLECTION_PATH = os.path.join(ADNI_PATH, "OutputCollection.csv")
    DEMOGRAPHICS_PATH = os.path.join(ADNI_PATH, "participant_demographics.csv")
    DXSUM_PATH = os.path.join(ADNI_PATH, "DXSUM_PDXCONV_ADNIALL.csv")
    CSV_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "csv")

    # Copy CSV files
    copy_csv_files(CSV_OUTPUT_DIR, COLLECTION_PATH, DEMOGRAPHICS_PATH, DXSUM_PATH)
    
    # Prepare dataframe
    df = setup_mri_dataframe(adni_path=ADNI_PATH)
    print(f"Total MRI files to process: {len(df)}")
    # Convert all MRI files
    # processed, converted, skipped, failed = convert_all_mri(df, OUTPUT_ROOT)
    processed, converted, skipped, failed = convert_all_mri(df.iloc[0:10], OUTPUT_ROOT)

    # Done
    print("All done!")
    print(f"Preprocessed data saved to {OUTPUT_ROOT}")
    print(f"Processed: {processed}, Converted: {converted}, Skipped: {skipped}, Failed: {failed}")
    print("End time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


