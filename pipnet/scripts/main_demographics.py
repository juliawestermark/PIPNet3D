import pandas as pd
import numpy as np
import os

from make_mm_dataset import load_npy_dataset

def get_demographics_summary(
    main_df, 
    demo_csv_path="/home/maia-user/ADNI_npy/csv/participant_demographics.csv",
    target_stages=None  # <--- NYTT ARGUMENT (lista med strängar, t.ex. ["CN", "AD"])
):
    print(f"--- Bearbetar Demografi ---")
    
    # 1. Läs in och städa
    if not os.path.exists(demo_csv_path):
        alt_path = "/home/maia-user/ADNI_npy/csv/PTDEMOG.csv"
        if os.path.exists(alt_path):
            demo_csv_path = alt_path
        else:
            raise FileNotFoundError(f"Kunde inte hitta {demo_csv_path}")
    
    demo_raw = pd.read_csv(demo_csv_path)
    demo_raw.columns = [col.split('(')[0].strip() for col in demo_raw.columns]
    
    subset_cols = ["Individual", "PTGENDER", "PTDOBYY", "PTDOBMM", "PTEDUCAT"]
    demo_clean = demo_raw[subset_cols].rename(columns={"Individual": "individual_id"})
    
    # 2. Konvertera och hantera felkoder (-4)
    cols_to_numeric = ["PTGENDER", "PTEDUCAT", "PTDOBYY", "PTDOBMM"]
    for col in cols_to_numeric:
        demo_clean[col] = pd.to_numeric(demo_clean[col], errors='coerce')

    demo_clean.loc[demo_clean["PTEDUCAT"] < 0, "PTEDUCAT"] = np.nan
    demo_clean.loc[demo_clean["PTGENDER"] < 0, "PTGENDER"] = np.nan
    demo_clean.loc[demo_clean["PTDOBYY"] < 0, "PTDOBYY"] = np.nan
    
    # Städa ID
    demo_clean["individual_id"] = demo_clean["individual_id"].astype(str).str.strip()
    main_df["individual_id"] = main_df["individual_id"].astype(str).str.strip()

    # 3. Aggregrea per patient (Robust metod: fyller i hål)
    demo_aggregated = demo_clean.groupby("individual_id")[["PTGENDER", "PTDOBYY", "PTDOBMM", "PTEDUCAT"]].first().reset_index()
    
    # 4. Slå ihop
    df_merged = pd.merge(main_df, demo_aggregated, on="individual_id", how="left")

    # --- NYTT: FILTRERA PÅ VALDA KLINISKA STADIER ---
    if target_stages is not None:
        print(f"Filtrerar på stadier: {target_stages}")
        df_merged = df_merged[df_merged["clinical_stage"].isin(target_stages)].reset_index(drop=True)
    # -----------------------------------------------

    # 5. Räkna Ålder
    df_merged["PTDOBMM"] = df_merged["PTDOBMM"].fillna(7)
    fallback_year = df_merged["exam_date"].dt.year - 75
    df_merged["PTDOBYY"] = df_merged["PTDOBYY"].fillna(fallback_year)
    
    df_merged["birth_date"] = pd.to_datetime(dict(year=df_merged.PTDOBYY, month=df_merged.PTDOBMM, day=15), errors='coerce')
    df_merged["age_at_scan"] = (df_merged["exam_date"] - df_merged["birth_date"]).dt.days / 365.25
    
    # 6. Mappa Kön
    gender_map = {1.0: "Male", 2.0: "Female", 1: "Male", 2: "Female"}
    df_merged["gender_str"] = df_merged["PTGENDER"].map(gender_map)
    
    # --- SKAPA TABELL ---
    stats_list = []
    # Hämta de grupper som faktiskt finns kvar i datan
    available_groups = sorted([x for x in df_merged["clinical_stage"].unique() if pd.notna(x)])
    groups = ["Total"] + available_groups
    
    for group in groups:
        if group == "Total":
            sub_df = df_merged
        else:
            sub_df = df_merged[df_merged["clinical_stage"] == group]
            
        n_scans = len(sub_df)
        n_subjects = sub_df["individual_id"].nunique()
        
        subj_df = sub_df.drop_duplicates(subset=["individual_id"])
        
        n_male = len(subj_df[subj_df["gender_str"] == "Male"])
        n_female = len(subj_df[subj_df["gender_str"] == "Female"])
        n_unknown = len(subj_df[~subj_df["gender_str"].isin(["Male", "Female"])])
        perc_male = (n_male / n_subjects * 100) if n_subjects > 0 else 0
        
        age_mean = sub_df["age_at_scan"].mean()
        age_std = sub_df["age_at_scan"].std()
        
        edu_mean = subj_df["PTEDUCAT"].mean()
        edu_std = subj_df["PTEDUCAT"].std()
        
        n_missing_edu = subj_df["PTEDUCAT"].isna().sum()
        edu_str = f"{edu_mean:.1f} ± {edu_std:.1f}"
        if n_missing_edu > 0:
            edu_str += f" ({n_missing_edu} missing)"

        n_mri = sub_df["file_path_mri"].notna().sum() if "file_path_mri" in sub_df.columns else 0
        n_pet = sub_df["file_path_amy"].notna().sum() if "file_path_amy" in sub_df.columns else 0
        
        n_paired = 0
        if "file_path_mri" in sub_df.columns and "file_path_amy" in sub_df.columns:
            n_paired = sub_df[["file_path_mri", "file_path_amy"]].notna().all(axis=1).sum()
        
        stats_list.append({
            "Group": group,
            "Subjects (N)": n_subjects,
            "Scans (Total)": n_scans,
            "Age (Mean ± SD)": f"{age_mean:.1f} ± {age_std:.1f}",
            "Gender (M/F/U)": f"{n_male}/{n_female}/{n_unknown} ({perc_male:.1f}% M)",
            "Education (Years)": edu_str,
            "MRI Scans": n_mri,
            "PET Scans": n_pet,
            "Paired (MRI+PET)": n_paired
        })
        
    stats_df = pd.DataFrame(stats_list)
    stats_df = stats_df.set_index("Group")
    
    return stats_df, df_merged


def run_demographics(adni_path, pipnet_path, classes, modalitites):
    path_to_demo = os.path.join(adni_path, "csv", "participant_demographics.csv")
    experiment_folder = os.path.join(pipnet_path, "results")
    
    df = load_npy_dataset(classes=classes, adni_path=adni_path, modalities=modalities)
    # Kör funktionen
    # summary_table, df_with_demographics = get_demographics_summary(df, path_to_demo)
    summary_table, df_with_demographics = get_demographics_summary(df, path_to_demo, target_stages=classes)

    # Skriv ut tabellen snyggt
    print("\n=== DATASET DEMOGRAPHICS ===")
    print(summary_table.T) # Transponera för att få grupper som kolumner (snyggare)

    # Spara till CSV om du vill ha med i rapporten
    class_names = "_".join(cl for cl in classes)
    output_name = f"dataset_demographics_table_{class_names}.csv"
    output_file = os.path.join(experiment_folder, output_name)
    summary_table.to_csv(output_file)

if __name__ == "__main__":

    #pipnet_path = "/home/maia-user/PIPNet3D"
    #adni_path = "/home/maia-user/ADNI_npy"

    pipnet_path = "/proj/berzbiomedicalimagingkth/users/x_julwe/PIPNet3D/"
    adni_path = "/proj/berzbiomedicalimagingkth/users/x_julwe/ADNI_npy"

    modalities = ["mri", "amy"]
    classes = ["CN", "AD"]
    run_demographics(adni_path, pipnet_path, classes, modalities)
