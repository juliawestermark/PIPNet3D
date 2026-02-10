import pandas as pd
import os
from pathlib import Path
import re

def create_pet_dataframe_from_filesystem(adni_path):
    """
    Skannar mappstrukturen:
    ROOT / SubjectID / Tracer_Coreg... / YYYY-MM-DD_... / fil.nii.gz
    
    Returnerar en DataFrame med kolumnerna:
    [individual_id, exam_id, exam_date, tracer, file_path, included]
    """
    root = Path(adni_path)
    data_records = []
    
    # Hämta alla subject-mappar
    # Vi kollar bara mappar som faktiskt ser ut som Subject ID (XXX_S_XXXX)
    subjects = sorted([p for p in root.iterdir() if p.is_dir()])
    
    print(f"[INFO] Skannar {len(subjects)} patient-mappar i {adni_path}...")

    for subj_dir in subjects:
        subject_id = subj_dir.name
        
        if not re.match(r'\d{3}_S_\d+', subject_id):
            continue

        # Loopa igenom undermappar (Beskrivning/Tracer)
        for desc_dir in subj_dir.iterdir():
            if not desc_dir.is_dir(): continue
            
            dir_name = desc_dir.name
            
            # Identifiera Tracer (AV45 eller FBB)
            tracer = None
            if "AV45" in dir_name:
                tracer = "AV45"
            elif "FBB" in dir_name:
                tracer = "FBB"
            
            # Om det inte är en mapp vi bryr oss om (måste vara Coreg), hoppa över
            if tracer is None or "Coreg" not in dir_name:
                continue

            # Loopa igenom Datum-mappar
            for date_dir in desc_dir.iterdir():
                if not date_dir.is_dir(): continue
                
                # Mappnamn är typ: "2025-10-30_15_40_09.0" -> "2025-10-30"
                date_str_raw = date_dir.name.split('_')[0]
                
                try:
                    # Validera att det är ett datum
                    pd.to_datetime(date_str_raw)
                except:
                    continue

                # Hitta själva .nii.gz filen (rekursivt i datum-mappen)
                nifti_files = list(date_dir.rglob("*.nii.gz"))
                
                if not nifti_files:
                    continue
                
                # Ta den första filen
                file_path = str(nifti_files[0])
                
                # SKAPA ETT UNIKT ID (Ersätter det gamla LONIUID)
                # Format: Subject_Tracer_Datum (t.ex. 381_S_10563_FBB_2025-10-30)
                synthetic_exam_id = f"{subject_id}_{tracer}_{date_str_raw}"

                data_records.append({
                    "individual_id": subject_id,
                    "exam_id": synthetic_exam_id,
                    "exam_date": pd.to_datetime(date_str_raw),
                    "tracer": tracer,
                    "viscode": None, # Vi vet inte viscode utan DXSUM-matchning än
                    "file_path": None,
                    "included": True
                })

    df = pd.DataFrame(data_records)
    print("\n--- Resultat ---")
    print(f"[INFO] Hittade totalt {len(df)} PET-bilder.")
    print(f"Fördelning tracers:\n{df['tracer'].value_counts()}")
    return df