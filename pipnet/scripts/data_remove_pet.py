import os
import shutil
from tqdm import tqdm

def clean_and_prune(output_root, folder_to_remove):

    if not os.path.exists(output_root):
        print(f"Hittar inte roten: {output_root}")
        return

    subjects = [d for d in os.listdir(output_root) if os.path.isdir(os.path.join(output_root, d))]
    
    print(f"Scannar {len(subjects)} subjects i {output_root}...")

    # Bekräfta innan start
    # confirm = input(f"Detta kommer radera alla '{folder_to_remove}'-mappar och ta bort subject-mappen om den blir tom. Fortsätt? (ja/nej): ")
    # if confirm.lower() != "ja":
    #     print("Avbryter.")
    #     return

    removed_amy_count = 0
    removed_subject_count = 0

    for subject in tqdm(subjects, desc="Rensar"):
        subject_path = os.path.join(output_root, subject)
        amy_path = os.path.join(subject_path, folder_to_remove)

        # 1. Ta bort amy-mappen om den finns
        if os.path.exists(amy_path):
            try:
                shutil.rmtree(amy_path)
                removed_amy_count += 1
            except Exception as e:
                print(f"Fel vid radering av {amy_path}: {e}")

        # 2. Kolla om subject-mappen nu är tom
        # Vi använder os.listdir för att se om det finns något kvar (t.ex. mri-mapp)
        try:
            if os.path.exists(subject_path) and not os.listdir(subject_path):
                os.rmdir(subject_path) # os.rmdir fungerar bara på tomma mappar (säkert)
                removed_subject_count += 1
        except Exception as e:
            print(f"Kunde inte ta bort tom mapp {subject_path}: {e}")

    print("\n--- Resultat ---")
    print(f"Antal '{folder_to_remove}'-mappar raderade: {removed_amy_count}")
    print(f"Antal tomma subject-mappar raderade: {removed_subject_count}")
    print("Klart!")

if __name__ == "__main__":
    output_root = "/home/maia-user/ADNI_npy"
    #output_root = "/proj/berzbiomedicalimagingkth/users/x_julwe/ADNI_npy"
    folder_to_remove = "amy"

    clean_and_prune(output_root, folder_to_remove)