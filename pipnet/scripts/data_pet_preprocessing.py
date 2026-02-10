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
    # Ladda data som float32
    arr = img.get_fdata().astype(np.float32)
    
    # Normalize safely (Min-Max scaling 0 to 1)
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min)
    else:
        # Om bilden är helt platt (bara nollor), behåll nollor
        arr = np.zeros_like(arr, dtype=np.float32)
    
    # Spara plats genom att konvertera till float16
    arr = arr.astype(np.float16)

    # Skapa mappen och spara
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, arr)


def convert_all_pet(df, output_root, pet_type):
    """
    df = DataFrame från create_pet_dataframe_from_filesystem
    Converts all NIfTI PET files to .npy in structure:
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
        exam_id = row["exam_id"] # Detta är nu vårt unika ID (Subj_Tracer_Date)

        # Output path: .../Subject/amy/Subject_Tracer_Date.npy
        save_path = os.path.join(output_root, subject, pet_type, f"{exam_id}.npy")

        # SKIP if file already exists and looks valid
        if os.path.exists(save_path):
            try:
                # Testa att ladda den snabbt för att se att den inte är korrupt
                # mmap_mode='r' gör att vi inte läser in hela i minnet
                np.load(save_path, mmap_mode='r')
                skipped += 1
                continue 
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
                print(f"Converted: {converted}, Skipped: {skipped}, Failed: {failed}")
                sys.exit(1)

            # === Non-critical errors ===
            print(f"[ERROR] OS error on {nii_path}: {e}")
            failed += 1
        except Exception as e:
            failed += 1
            print(f"[ERROR] Unexpected error on {nii_path}: {e}")

    return processed, converted, skipped, failed


def manage_csv_files(output_csv_dir, pet_df, other_csv_paths=[]):
    """
    1. Sparar den genererade PET-registret.
    2. Kopierar andra viktiga CSV-filer (t.ex. DXSUM, OutputCollection) om de finns.
    """
    os.makedirs(output_csv_dir, exist_ok=True)

    # 1. Spara PET-registret vi nyss skapade
    pet_reg_path = os.path.join(output_csv_dir, "generated_pet_registry.csv")
    pet_df.to_csv(pet_reg_path, index=False)
    print(f"[INFO] Saved generated PET registry to: {pet_reg_path}")

    # 2. Kopiera andra filer (t.ex. MRI info eller Diagnoser)
    for src in other_csv_paths:
        if src and os.path.exists(src):
            filename = os.path.basename(src)
            dst = os.path.join(output_csv_dir, filename)
            
            # Kopiera bara om den inte redan finns (eller om du vill skriva över, ta bort if-satsen)
            if not os.path.exists(dst):
                print(f"[INFO] Copying {filename} -> {output_csv_dir}")
                shutil.copy2(src, dst)
            else:
                print(f"[INFO] {filename} already exists in output CSV dir.")
        else:
            if src: print(f"[WARN] Could not find source CSV to copy: {src}")


if __name__ == "__main__":
    print("Start time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # --- KONFIGURATION ---
    # Sökväg till mappstrukturen med PET NIfTI-filer
    ADNI_PET_PATH = "/proj/berzbiomedicalimagingkth/ADNI_NIFTI_AMYLOID_CoregAvg_StdImgVox"
    #ADNI_PET_PATH = "/home/maia-user/ADNI_PET/ADNI_NEW"
    
    # Vart ska .npy filerna och CSV-filerna sparas?
    # OUTPUT_ROOT = "/home/maia-user/ADNI_npy"
    OUTPUT_ROOT = "/proj/berzbiomedicalimagingkth/users/x_julwe/ADNI_npy"
    
    # Här anger du sökvägar till eventuella andra CSV-filer du vill kopiera med (från MRI-steget)
    # Har du dem inte kvar, lämna listan tom.
    # CSV_FOLDER_PATH = "/home/maia-user/ADNI_PET/csv"
    CSV_FOLDER_PATH = "/proj/berzbiomedicalimagingkth/users/x_julwe/ADNI/csv"
    DXSUM_CSV_SOURCE = os.path.join(CSV_FOLDER_PATH, "DXSUM_10Feb2026.csv")
    PTDEM_CSV_SOURCE = os.path.join(CSV_FOLDER_PATH, "PTDEMOG_10Feb2026.csv")

    CSV_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "csv")
    PET_TYPE = "amy" # Mappen det sparas i (t.ex. .../subject/amy/...)

    # ---------------------

    # 1. Skapa Dataframe genom att skanna filsystemet
    print("--- Step 1: Scanning Filesystem ---")
    df = create_pet_dataframe_from_filesystem(ADNI_PET_PATH)
    
    if len(df) == 0:
        print("[ERROR] No PET files found! Check path.")
        sys.exit(1)

    # 2. Spara CSV och kopiera andra filer
    print("--- Step 2: Managing CSV files ---")
    manage_csv_files(
        CSV_OUTPUT_DIR, 
        df, 
        other_csv_paths=[DXSUM_CSV_SOURCE, PTDEM_CSV_SOURCE]
    )
    
    # 3. Konvertera NIfTI till .npy
    print(f"--- Step 3: converting {len(df)} PET files to .npy ---")
    
    # OBS: Kör convertern. 
    # Ta bort .iloc[0:20] när du vill köra hela datasetet!
    processed, converted, skipped, failed = convert_all_pet(df, OUTPUT_ROOT, PET_TYPE)
    # processed, converted, skipped, failed = convert_all_pet(df.iloc[0:20], OUTPUT_ROOT, PET_TYPE)

    # Done
    print("All done!")
    print(f"Preprocessed data saved to {OUTPUT_ROOT}")
    print(f"Processed: {processed}, Converted: {converted}, Skipped: {skipped}, Failed: {failed}")
    print("End time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
