import os
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
import re
from nilearn.image import mean_img, threshold_img

def create_pet_dataframe_from_filesystem(adni_path):
    """Scans the directory structure and returns a DataFrame with PET file information."""
    root = Path(adni_path)
    data_records = []
    
    subjects = sorted([p for p in root.iterdir() if p.is_dir()])
    print(f"Scanning {len(subjects)} subject directories in {adni_path}...")

    for subj_dir in subjects:
        subject_id = subj_dir.name
        if not re.match(r'\d{3}_S_\d+', subject_id):
            continue

        for desc_dir in subj_dir.iterdir():
            if not desc_dir.is_dir(): continue
            dir_name = desc_dir.name
            
            tracer = None
            if "AV45" in dir_name: tracer = "AV45"
            elif "FBB" in dir_name: tracer = "FBB"
            
            if tracer is None or "Coreg" not in dir_name:
                continue

            for date_dir in desc_dir.iterdir():
                if not date_dir.is_dir(): continue
                
                date_str_raw = date_dir.name.split('_')[0]
                try: 
                    pd.to_datetime(date_str_raw)
                except: 
                    continue

                nifti_files = list(date_dir.rglob("*.nii.gz"))
                if not nifti_files: continue
                
                file_path = str(nifti_files[0])
                synthetic_exam_id = f"{subject_id}_{tracer}_{date_str_raw}"

                data_records.append({
                    "individual_id": subject_id,
                    "exam_id": synthetic_exam_id,
                    "exam_date": pd.to_datetime(date_str_raw),
                    "tracer": tracer,
                    "file_path": file_path,
                    "included": True
                })

    df = pd.DataFrame(data_records)
    print(f"Found {len(df)} PET images.")
    return df

def create_global_pet_mask(ADNI_PATH_PET, OUTPUT_ROOT, num_samples=50):
    """Generates a global binary mask based on a sample of PET images."""
    modality = "amy"
    MASK_FILENAME_NII = "global_mask.nii.gz"
    MASK_FILENAME_NPY = "global_mask.npy"
    
    modality_dir = os.path.join(OUTPUT_ROOT, "masks", modality)
    os.makedirs(modality_dir, exist_ok=True)

    df = create_pet_dataframe_from_filesystem(ADNI_PATH_PET)
    
    # Sample files to manage RAM usage
    nifti_files = df['file_path'].sample(n=min(num_samples, len(df)), random_state=42).tolist()
    print(f"Generating mean brain from {len(nifti_files)} images...")

    mean_brain = mean_img(nifti_files)
    mean_brain.to_filename(os.path.join(modality_dir, "mean_brain_reference.nii.gz"))

    # Create binary mask using thresholding
    mask_nii = threshold_img(mean_brain, threshold=0.1, copy=True)
    nii_save_path = os.path.join(modality_dir, MASK_FILENAME_NII)
    mask_nii.to_filename(nii_save_path)

    # Convert to .npy format for pipeline compatibility
    img = nib.load(nii_save_path)
    mask_arr = img.get_fdata().astype(np.float32)
    mask_arr = (mask_arr > 0.001).astype(np.uint8) 

    npy_save_path = os.path.join(modality_dir, MASK_FILENAME_NPY)
    np.save(npy_save_path, mask_arr)
    
    print(f"Success! Global mask saved to: {npy_save_path}")
    print(f"Mask shape: {mask_arr.shape}")

if __name__ == "__main__":
    BASE_PATH = "/home/maia-user"
    PET_ADNI_PATH = os.path.join(BASE_PATH, "ADNI_PET", "ADNI_NEW")
    OUTPUT_ROOT = os.path.join(BASE_PATH, "ADNI_npy")

    create_global_pet_mask(PET_ADNI_PATH, OUTPUT_ROOT, num_samples=200)