import pandas as pd
import os
from pathlib import Path
import re

def create_pet_dataframe_from_filesystem(adni_path):
    """
    Scans the directory structure:
    ROOT / SubjectID / Tracer_Coreg... / YYYY-MM-DD_... / file.nii.gz
    
    Returns a DataFrame with columns:
    [individual_id, exam_id, exam_date, tracer, file_path, included]
    """
    root = Path(adni_path)
    data_records = []
    
    subjects = sorted([p for p in root.iterdir() if p.is_dir()])
    
    print(f"[INFO] Scanning {len(subjects)} patient directories in {adni_path}...")

    for subj_dir in subjects:
        subject_id = subj_dir.name
        
        if not re.match(r'\d{3}_S_\d+', subject_id):
            continue

        for desc_dir in subj_dir.iterdir():
            if not desc_dir.is_dir(): continue
            
            dir_name = desc_dir.name
            
            tracer = None
            if "AV45" in dir_name:
                tracer = "AV45"
            elif "FBB" in dir_name:
                tracer = "FBB"
            
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
                
                if not nifti_files:
                    continue
                
                file_path = str(nifti_files[0])
                
                # Create a unique ID: Subject_Tracer_Date
                synthetic_exam_id = f"{subject_id}_{tracer}_{date_str_raw}"

                data_records.append({
                    "individual_id": subject_id,
                    "exam_id": synthetic_exam_id,
                    "exam_date": pd.to_datetime(date_str_raw),
                    "tracer": tracer,
                    "viscode": None, 
                    "file_path": file_path,
                    "included": True
                })

    df = pd.DataFrame(data_records)
    print("\n--- Results ---")
    print(f"[INFO] Found a total of {len(df)} PET images.")
    print(f"Tracer distribution:\n{df['tracer'].value_counts()}")
    return df