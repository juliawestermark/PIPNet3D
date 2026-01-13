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
    Kopierar filer från source_root till output_root men behåller den interna strukturen.
    
    Exempel:
    Källa: /home/user/ADNI/001/bild.nii
    Source_root: /home/user/ADNI
    Output_root: /home/user/NY_MAPP
    
    Resultat: /home/user/NY_MAPP/001/bild.nii
    """

    processed = 0
    copied = 0
    skipped = 0
    failed = 0

    print(f"Kopierar från: {source_root}")
    print(f"Till: {output_root}")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        processed += 1
        nii_path = row["file_path"]

        # 1. Räkna ut den relativa sökvägen (ta bort /home/maia-user/ADNI_PET/ADNI från början)
        # Detta ger t.ex: "002_S_0295/I239487_ADNI_Brain_PET_Raw_FDG_....nii.gz"
        try:
            rel_path = os.path.relpath(nii_path, source_root)
        except ValueError:
            # Om filen ligger på en helt annan disk/mount kan relpath ibland misslyckas i Windows,
            # men i Linux brukar det gå bra så länge de delar root.
            print(f"[ERROR] Kan inte beräkna relativ sökväg för: {nii_path}")
            failed += 1
            continue

        # 2. Skapa den nya fullständiga sökvägen
        dest_path = os.path.join(output_root, rel_path)

        # 3. Om filen redan finns, hoppa över (eller skriv över om du vill ändra logiken)
        if os.path.exists(dest_path):
            skipped += 1
            continue

        # 4. Skapa mappen om den inte finns
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        try:
            # shutil.copy2 behåller metadata (tidsstämplar etc.)
            shutil.copy2(nii_path, dest_path)
            copied += 1
        except OSError as e:
            # === Kritiska fel (Disk full etc) ===
            if e.errno in (errno.ENOSPC, errno.EDQUOT, errno.EIO, errno.EROFS):
                print(f"[CRITICAL] Stoppar körning – disk/systemfel: {e}")
                sys.exit(1)

            # === Icke-kritiska fel ===
            print(f"[ERROR] OS error vid kopiering av {nii_path}: {e}")
            failed += 1
        except Exception as e:
            failed += 1
            print(f"[ERROR] Oväntat fel vid {nii_path}: {e}")

    return processed, copied, skipped, failed


def copy_csv_files(output_path, amy_csv, tau_csv):
    """Copy relevant ADNI CSVs into OUTPUT_ROOT/."""
    os.makedirs(output_path, exist_ok=True)
    
    # Här vill du kanske lägga dem direkt i roten eller i en undermapp, ändra vid behov.
    files = {
        os.path.basename(amy_csv): amy_csv,
        os.path.basename(tau_csv): tau_csv,
    }

    for filename, src in files.items():
        dst = os.path.join(output_path, filename) # Lade dem direkt i output_path här
        if os.path.exists(dst):
            print(f"File already exists, skipping: {dst}")
            continue
        print(f"Copying {src} → {dst}")
        shutil.copy2(src, dst)


if __name__ == "__main__":
    print("Start time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # --- KONFIGURATION ---
    # Detta är roten för källan. Allt EFTER denna del i sökvägen kommer behållas.
    # Om filen är: /home/maia-user/ADNI_PET/ADNI/002_S...
    # Så blir relativa delen: 002_S...
    ADNI_PATH_SOURCE = "/home/maia-user/ADNI_PET/ADNI"
    
    # Hit ska filerna flyttas
    OUTPUT_ROOT = "/home/maia-user/ADNI_PET/AMY" 

    amy_file_name = "UCBERKELEY_AMY_6MM_02Apr2025.csv"
    amy_csv_file = os.path.join(ADNI_PATH_SOURCE, amy_file_name)
    tau_csv_file = os.path.join(ADNI_PATH_SOURCE, "UCBERKELEY_TAU_6MM_02Apr2025.csv")

    # 1. Kopiera CSV-filer
    copy_csv_files(OUTPUT_ROOT, amy_csv_file, tau_csv_file)
    
    # 2. Förbered dataframe (för att få listan på filer vi ska flytta)
    df = setup_pet_dataframe(identifier="AV45", adni_path=ADNI_PATH_SOURCE, adni_pet_file_name=amy_file_name)
    print(f"Totalt antal filer att kopiera: {len(df)}")
    
    # 3. Kopiera filerna med struktur
    # Notera: Jag skickar in source_root här så funktionen vet vad den ska "klippa bort"
    processed, copied, skipped, failed = copy_files_keeping_structure(df, ADNI_PATH_SOURCE, OUTPUT_ROOT)

    # Done
    print("All done!")
    print(f"Data kopierad till {OUTPUT_ROOT}")
    print(f"Hanterade: {processed}, Kopierade: {copied}, Hoppade över: {skipped}, Misslyckades: {failed}")
    print("End time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))