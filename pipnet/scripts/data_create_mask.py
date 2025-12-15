import os
import numpy as np
import nibabel as nib
from nilearn.image import mean_img, threshold_img
from make_mri_dataset import setup_mri_dataframe # Din befintliga import

ADNI_PATH = "/proj/berzbiomedicalimagingkth/users/x_julwe/ADNI/ADNI_complete"
OUTPUT_ROOT = "/proj/berzbiomedicalimagingkth/users/x_julwe/ADNI_npy"
#ADNI_PATH = "/home/maia-user/ADNI_complete"
#OUTPUT_ROOT = "/home/maia-user/ADNI_npy"
GLOBAL_MASK_PATH = os.path.join(OUTPUT_ROOT, "global_brain_mask_nilearn.npy")

MASK_FILENAME_NII = "global_mask.nii.gz"
MASK_FILENAME_NPY = "global_mask.npy"

def create_global_mask():
    print("1. Laddar lista över filer...")
    # Hämta filvägar med din befintliga funktion
    df = setup_mri_dataframe(adni_path=ADNI_PATH)
    
    # OBS: Om datasetet är enormt, ta ett robust stickprov (t.ex. 200 slumpmässiga bilder) 
    # för att spara RAM. Det brukar räcka för en global mask.
    nifti_files = df["file_path"].sample(n=min(200, len(df)), random_state=42).tolist()
    print(f"Använder {len(nifti_files)} bilder för att bygga genomsnittet.")

    # --- Steg A: Skapa mask med Nilearn (NIfTI) ---
    print("2. Beräknar genomsnittlig hjärna (Nilearn)...")
    mean_brain = mean_img(nifti_files)
    
    # Spara medelbilden för att kunna dubbelkolla visuellt senare
    mean_brain.to_filename(os.path.join(OUTPUT_ROOT, "mean_brain_reference.nii.gz"))

    print("3. Trösklar och skapar binär mask...")
    # threshold=0.1 är ofta bra för normaliserad data, justera vid behov
    mask_nii = threshold_img(mean_brain, threshold=0.1, copy=True)
    
    # Spara NIfTI-versionen (bra för visualisering i program som ITK-SNAP/FSL)
    nii_save_path = os.path.join(OUTPUT_ROOT, MASK_FILENAME_NII)
    mask_nii.to_filename(nii_save_path)
    print(f"NIfTI-mask sparad till: {nii_save_path}")

    # --- Steg B: Konvertera till .npy (Matcha din pipeline) ---
    print("4. Konverterar till .npy (Nibabel)...")
    
    # VIKTIGT: Vi laddar om den nyss sparade masken med nibabel
    # Detta simulerar exakt vad din 'convert_single_file' gör.
    img = nib.load(nii_save_path)
    
    # Hämta data (precis som i ditt preprocessing-skript)
    # Vi castar till uint8 eftersom en mask bara är 0 och 1 (sparar minne)
    mask_arr = img.get_fdata().astype(np.float32) # Läs in som float först för säkerhets skull
    mask_arr = (mask_arr > 0.001).astype(np.uint8) # Gör den strikt binär (0 eller 1)

    # Spara som .npy
    npy_save_path = os.path.join(OUTPUT_ROOT, MASK_FILENAME_NPY)
    np.save(npy_save_path, mask_arr)
    
    print(f"✅ Klar! Global mask sparad som .npy: {npy_save_path}")
    print(f"Maskens dimensioner: {mask_arr.shape}")

if __name__ == "__main__":
    create_global_mask()