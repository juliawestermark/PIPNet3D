import os
import shutil
from tqdm import tqdm

def clean_and_prune(output_root, folder_to_remove):

    if not os.path.exists(output_root):
        print(f"Could not find root: {output_root}")
        return

    subjects = [d for d in os.listdir(output_root) if os.path.isdir(os.path.join(output_root, d))]
    
    print(f"Scan {len(subjects)} subjects in {output_root}...")

    removed_amy_count = 0
    removed_subject_count = 0

    for subject in tqdm(subjects, desc="Cleaning"):
        subject_path = os.path.join(output_root, subject)
        amy_path = os.path.join(subject_path, folder_to_remove)

        # 1. Ta bort amy-mappen om den finns
        if os.path.exists(amy_path):
            try:
                shutil.rmtree(amy_path)
                removed_amy_count += 1
            except Exception as e:
                print(f"Error deleting {amy_path}: {e}")

        # 2. Kolla om subject-mappen nu är tom
        # Vi använder os.listdir för att se om det finns något kvar (t.ex. mri-mapp)
        try:
            if os.path.exists(subject_path) and not os.listdir(subject_path):
                os.rmdir(subject_path) # os.rmdir fungerar bara på tomma mappar (säkert)
                removed_subject_count += 1
        except Exception as e:
            print(f"Could not remove empty folder {subject_path}: {e}")

    print("\n--- Results ---")
    print(f"'{folder_to_remove}' folders deleted: {removed_amy_count}")
    print(f"Empty folders deleted: {removed_subject_count}")
    print("Done!")

if __name__ == "__main__":
    #output_root = "/home/maia-user/ADNI_npy"
    output_root = "/proj/berzbiomedicalimagingkth/users/x_julwe/ADNI_npy"
    folder_to_remove = "amy"

    clean_and_prune(output_root, folder_to_remove)