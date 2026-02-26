import os
import numpy as np
import nibabel as nib
from nilearn.image import mean_img, threshold_img
from make_mm_dataset import load_dataset

def create_global_mask(modality, ADNI_PATH_MRI, ADNI_PATH_PET, OUTPUT_ROOT):
    MASK_FILENAME_NII = "global_mask.nii.gz"
    MASK_FILENAME_NPY = "global_mask.npy"
    
    # Setup directory structure
    mask_dir = os.path.join(OUTPUT_ROOT, "masks")
    os.makedirs(mask_dir, exist_ok=True)
    
    modality_dir = os.path.join(mask_dir, modality)
    os.makedirs(modality_dir, exist_ok=True)

    print("Loading file list...")
    file_path_col = f"file_path_{modality}"
    dataset = load_dataset(mode="nii", classes=["CN", "MCI", "AD"], adni_path_mri=ADNI_PATH_MRI, adni_path_pet=ADNI_PATH_PET)
    df = dataset[dataset[file_path_col].notna()]
    
    # Use a robust sample to save RAM while maintaining mask quality
    nifti_files = df[file_path_col].sample(n=min(10, len(df)), random_state=42).tolist()
    print(f"Building average brain from {len(nifti_files)} images.")

    # --- Step A: Create Mask with Nilearn ---
    mean_brain = mean_img(nifti_files)
    mean_brain.to_filename(os.path.join(modality_dir, "mean_brain_reference.nii.gz"))

    # Thresholding to create a binary mask (adjust 0.1 if needed for normalized data)
    mask_nii = threshold_img(mean_brain, threshold=0.1, copy=True)
    
    nii_save_path = os.path.join(modality_dir, MASK_FILENAME_NII)
    mask_nii.to_filename(nii_save_path)

    # --- Step B: Convert to .npy for Pipeline Compatibility ---
    img = nib.load(nii_save_path)
    
    # Load data and ensure strictly binary uint8 format (0 or 1)
    mask_arr = img.get_fdata().astype(np.float32)
    mask_arr = (mask_arr > 0.001).astype(np.uint8)

    npy_save_path = os.path.join(modality_dir, MASK_FILENAME_NPY)
    np.save(npy_save_path, mask_arr)
    
    print(f"Global mask saved: {npy_save_path}")
    print(f"Mask dimensions: {mask_arr.shape}")

if __name__ == "__main__":
    BASE_PATH = "/proj/berzbiomedicalimagingkth/users/x_julwe"
    MRI_ADNI_PATH = os.path.join(BASE_PATH, "ADNI", "ADNI_complete")
    PET_ADNI_PATH = os.path.join(BASE_PATH, "ADNI", "AMY")

    #BASE_PATH = "/home/maia-user"
    #MRI_ADNI_PATH = os.path.join(BASE_PATH, "ADNI_complete")
    #PET_ADNI_PATH = os.path.join(BASE_PATH, "ADNI_PET", "ADNI")
    
    MODALITY = "amy"
    OUTPUT_ROOT = os.path.join(BASE_PATH, "ADNI_npy")

    create_global_mask(MODALITY, MRI_ADNI_PATH, PET_ADNI_PATH, OUTPUT_ROOT)