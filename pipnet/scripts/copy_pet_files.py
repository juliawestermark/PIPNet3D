import os
import errno
import sys
import numpy as np
import pandas as pd
import shutil
from tqdm.auto import tqdm
from datetime import datetime
from make_pet_dataset import setup_pet_dataframe

def copy_files_keeping_structure(df, source_root, output_root):
    """
    Copies files from source_root to output_root while maintaining the internal directory structure.
    """
    processed = 0
    copied = 0
    skipped = 0
    failed = 0

    print(f"Copying from: {source_root}")
    print(f"To: {output_root}")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        processed += 1
        nii_path = row["file_path"]

        # 1. Calculate the relative path (remove the source_root prefix)
        # Result example: "002_S_0295/I239487_ADNI_Brain_PET_Raw_FDG_....nii.gz"
        try:
            rel_path = os.path.relpath(nii_path, source_root)
        except ValueError:
            print(f"[ERROR] Could not calculate relative path for: {nii_path}")
            failed += 1
            continue

        # 2. Create the new full destination path
        dest_path = os.path.join(output_root, rel_path)

        # 3. If file already exists, skip (modify logic here if you want to overwrite)
        if os.path.exists(dest_path):
            skipped += 1
            continue

        # 4. Create the destination directory if it doesn't exist
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        try:
            # shutil.copy2 preserves metadata (timestamps, etc.)
            shutil.copy2(nii_path, dest_path)
            copied += 1
        except OSError as e:
            # === Critical errors (Disk full, etc.) ===
            if e.errno in (errno.ENOSPC, errno.EDQUOT, errno.EIO, errno.EROFS):
                print(f"[CRITICAL] Stopping execution – disk/system error: {e}")
                sys.exit(1)

            # === Non-critical errors ===
            print(f"[ERROR] OS error when copying {nii_path}: {e}")
            failed += 1
        except Exception as e:
            failed += 1
            print(f"[ERROR] Unexpected error at {nii_path}: {e}")

    return processed, copied, skipped, failed


def copy_csv_files(output_path, amy_csv, tau_csv):
    """Copy relevant ADNI CSVs into OUTPUT_ROOT/."""
    os.makedirs(output_path, exist_ok=True)
    
    # Mapping filenames to their source paths
    files = {
        os.path.basename(amy_csv): amy_csv,
        os.path.basename(tau_csv): tau_csv,
    }

    for filename, src in files.items():
        dst = os.path.join(output_path, filename) 
        if os.path.exists(dst):
            print(f"File already exists, skipping: {dst}")
            continue
        print(f"Copying {src} → {dst}")
        shutil.copy2(src, dst)


if __name__ == "__main__":
    print("Start time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    ADNI_PATH_SOURCE = "/home/maia-user/ADNI_PET/ADNI"

    OUTPUT_ROOT = "/home/maia-user/ADNI_PET/AMY" 

    amy_file_name = "UCBERKELEY_AMY_6MM_02Apr2025.csv"
    amy_csv_file = os.path.join(ADNI_PATH_SOURCE, amy_file_name)
    tau_csv_file = os.path.join(ADNI_PATH_SOURCE, "UCBERKELEY_TAU_6MM_02Apr2025.csv")

    # 1. Copy CSV files
    copy_csv_files(OUTPUT_ROOT, amy_csv_file, tau_csv_file)
    
    # 2. Prepare dataframe (to get the list of files to move)
    df = setup_pet_dataframe(identifier="AV45", adni_path=ADNI_PATH_SOURCE, adni_pet_file_name=amy_file_name)
    print(f"Total files to copy: {len(df)}")
    
    # 3. Copy files maintaining structure
    processed, copied, skipped, failed = copy_files_keeping_structure(df, ADNI_PATH_SOURCE, OUTPUT_ROOT)

    # Done
    print("All done!")
    print(f"Data copied to {OUTPUT_ROOT}")
    print(f"Summary - Processed: {processed}, Copied: {copied}, Skipped: {skipped}, Failed: {failed}")
    print("End time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))