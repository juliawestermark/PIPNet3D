import pandas as pd
import numpy as np
import os

from make_mm_dataset import load_npy_dataset

def get_demographics_summary(
    main_df, 
    demo_csv_path="/home/maia-user/ADNI_npy/csv/PTDEMOG_10Feb2026.csv",
    target_stages=None 
):
    print(f"--- Bearbetar Demografi ---")
    
    # 1. Läs in
    if not os.path.exists(demo_csv_path):
        raise FileNotFoundError(f"Kunde inte hitta {demo_csv_path}")
    
    # Header=1 för att hoppa över 'PTDEMOG_10Feb...' raden
    demo_raw = pd.read_csv(demo_csv_path)
    demo_raw.columns = [col.strip() for col in demo_raw.columns]

    # Mappa namn
    if "PTID" in demo_raw.columns:
        demo_raw = demo_raw.rename(columns={"PTID": "individual_id"})

    # Extrahera månad och år
    if "PTDOB" in demo_raw.columns:
        demo_raw["PTDOBMM"] = demo_raw["PTDOB"].astype(str).apply(
            lambda x: x.split('/')[0] if '/' in x else np.nan
        )

    if "PTDOBYY" in demo_raw.columns:
        temp_dates = pd.to_datetime(demo_raw["PTDOBYY"], errors='coerce')
        demo_raw["PTDOBYY"] = temp_dates.dt.year

    # 2. Välj och städa kolumner
    subset_cols = ["individual_id", "PTGENDER", "PTDOBYY", "PTDOBMM", "PTEDUCAT"]
    # Plocka bara de som finns
    valid_cols = [c for c in subset_cols if c in demo_raw.columns]
    demo_clean = demo_raw[valid_cols].copy()
    
    # Konvertera till numeric
    for col in ["PTGENDER", "PTEDUCAT", "PTDOBYY", "PTDOBMM"]:
        if col in demo_clean.columns:
            demo_clean[col] = pd.to_numeric(demo_clean[col], errors='coerce')
            # Sätt ogiltiga värden (<0) till NaN
            demo_clean.loc[demo_clean[col] < 0, col] = np.nan
    
    demo_clean["individual_id"] = demo_clean["individual_id"].astype(str).str.strip()
    main_df["individual_id"] = main_df["individual_id"].astype(str).str.strip()

    # Aggregrea per patient (Ta första giltiga värdet)
    # Om en patient har flera rader i demografifilen tar vi den första.
    demo_aggregated = demo_clean.groupby("individual_id").first().reset_index()
    
    # 3. Slå ihop (Left join behåller ALLA rader från ditt dataset)
    df_merged = pd.merge(main_df, demo_aggregated, on="individual_id", how="left")

    # Filtrera på stadier om önskat
    if target_stages is not None:
        df_merged = df_merged[df_merged["clinical_stage"].isin(target_stages)].reset_index(drop=True)

    # 4. Beräkna Ålder (Här är ändringen: Vi tar INTE bort rader)
    
    # Vi försöker skapa datum. Om år eller månad saknas (NaN), blir resultatet NaT.
    # Vi sätter dag=15 som standard.
    df_merged["birth_date"] = pd.to_datetime(
        dict(year=df_merged.PTDOBYY, month=df_merged.PTDOBMM, day=15), 
        errors='coerce'
    )
    
    # Räkna ut ålder. 
    # Om birth_date är NaT eller exam_date är NaT -> age_at_scan blir NaN.
    # Men raden finns kvar!
    if "exam_date" in df_merged.columns:
        df_merged["age_at_scan"] = (df_merged["exam_date"] - df_merged["birth_date"]).dt.days / 365.25
    else:
        df_merged["age_at_scan"] = np.nan

    # 5. Mappa Kön
    gender_map = {1.0: "Male", 2.0: "Female", 1: "Male", 2: "Female"}
    df_merged["gender_str"] = df_merged["PTGENDER"].map(gender_map)
    
    # --- SKAPA TABELL ---
    stats_list = []
    
    if "clinical_stage" in df_merged.columns:
        available_groups = sorted([x for x in df_merged["clinical_stage"].unique() if pd.notna(x)])
    else:
        available_groups = []
        
    groups = ["Total"] + available_groups
    
    for group in groups:
        if group == "Total":
            sub_df = df_merged
        else:
            sub_df = df_merged[df_merged["clinical_stage"] == group]
            
        n_scans = len(sub_df)
        n_subjects = sub_df["individual_id"].nunique()
        
        # Unika patienter för demografisk statistik
        subj_df = sub_df.drop_duplicates(subset=["individual_id"])
        
        # Kön
        n_male = len(subj_df[subj_df["gender_str"] == "Male"])
        n_female = len(subj_df[subj_df["gender_str"] == "Female"])
        n_unknown = len(subj_df) - (n_male + n_female) # Räknar med de som saknar kön
        perc_male = (n_male / n_subjects * 100) if n_subjects > 0 else 0
        
        # Ålder (Pandas ignorerar automatiskt NaN när den räknar mean/std)
        # Vi lägger till info om hur många som saknas, för tydlighetens skull
        n_age_missing = sub_df["age_at_scan"].isna().sum()
        age_mean = sub_df["age_at_scan"].mean()
        age_std = sub_df["age_at_scan"].std()
        
        age_str = f"{age_mean:.1f} ± {age_std:.1f}"
        if n_age_missing > 0:
            # T.ex: "75.2 ± 5.1 (2 missing)"
            age_str += f" ({n_age_missing} miss)"
        
        # Utbildning
        n_edu_missing = subj_df["PTEDUCAT"].isna().sum()
        edu_mean = subj_df["PTEDUCAT"].mean()
        edu_std = subj_df["PTEDUCAT"].std()
        
        edu_str = f"{edu_mean:.1f} ± {edu_std:.1f}"
        if n_edu_missing > 0:
            edu_str += f" ({n_edu_missing} miss)"

        # Scans
        n_mri = sub_df["file_path_mri"].notna().sum() if "file_path_mri" in sub_df.columns else 0
        n_pet = sub_df["file_path_amy"].notna().sum() if "file_path_amy" in sub_df.columns else 0
        
        n_paired = 0
        if "file_path_mri" in sub_df.columns and "file_path_amy" in sub_df.columns:
            n_paired = sub_df[["file_path_mri", "file_path_amy"]].notna().all(axis=1).sum()
        
        stats_list.append({
            "Group": group,
            "Subjects (N)": n_subjects,
            "Scans (Total)": n_scans,
            "Age (Mean ± SD)": age_str,
            "Gender (M/F/U)": f"{n_male}/{n_female}/{n_unknown} ({perc_male:.1f}% M)",
            "Education (Years)": edu_str,
            "MRI Scans": n_mri,
            "PET Scans": n_pet,
            "Paired (MRI+PET)": n_paired
        })
        
    stats_df = pd.DataFrame(stats_list)
    stats_df = stats_df.set_index("Group")
    
    return stats_df, df_merged

# --- (Samma main block som förut) ---


def run_demographics(adni_path, pipnet_path, classes, modalities):
    path_to_demo = os.path.join(adni_path, "csv", "PTDEMOG_10Feb2026.csv")
    experiment_folder = os.path.join(pipnet_path, "results")
    
    # Ladda datasetet (se till att load_npy_dataset är korrekt importerad/definierad)
    df = load_npy_dataset(classes=classes, adni_path=adni_path, modalities=modalities)
    
    if df.empty:
        print("Varning: Datasetet är tomt. Kontrollera sökvägar.")
        return

    # Kör funktionen
    summary_table, df_with_demographics = get_demographics_summary(df, path_to_demo, target_stages=classes)

    # Skriv ut tabellen snyggt
    print("\n=== DATASET DEMOGRAPHICS ===")
    print(summary_table.T) 

    # Spara till CSV
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)
        
    class_names = "_".join(cl for cl in classes)
    output_name = f"NEW_dataset_demographics_table_{class_names}.csv"
    output_file = os.path.join(experiment_folder, output_name)
    summary_table.to_csv(output_file)
    print(f"\nSparade demografisk tabell till: {output_file}")

if __name__ == "__main__":

    #pipnet_path = "/home/maia-user/PIPNet3D"
    #adni_path = "/home/maia-user/ADNI_npy"

    pipnet_path = "/proj/berzbiomedicalimagingkth/users/x_julwe/PIPNet3D/"
    adni_path = "/proj/berzbiomedicalimagingkth/users/x_julwe/ADNI_npy"

    modalities = ["mri", "amy"]
    classes = ["CN", "MCI", "AD"]
    
    run_demographics(adni_path, pipnet_path, classes, modalities)