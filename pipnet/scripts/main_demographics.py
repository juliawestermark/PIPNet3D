import pandas as pd
import numpy as np
import os
from make_mm_dataset import load_npy_dataset

def get_demographics_summary(
    main_df, 
    demo_csv_path="/home/maia-user/ADNI_npy/csv/PTDEMOG_10Feb2026.csv",
    target_stages=None 
):
    if not os.path.exists(demo_csv_path):
        raise FileNotFoundError(f"Could not find {demo_csv_path}")
    
    # Load demographics and clean column names
    demo_raw = pd.read_csv(demo_csv_path)
    demo_raw.columns = [col.strip() for col in demo_raw.columns]

    if "PTID" in demo_raw.columns:
        demo_raw = demo_raw.rename(columns={"PTID": "individual_id"})

    # Extract birth month and year
    if "PTDOB" in demo_raw.columns:
        demo_raw["PTDOBMM"] = demo_raw["PTDOB"].astype(str).apply(
            lambda x: x.split('/')[0] if '/' in x else np.nan
        )

    if "PTDOBYY" in demo_raw.columns:
        temp_dates = pd.to_datetime(demo_raw["PTDOBYY"], errors='coerce')
        demo_raw["PTDOBYY"] = temp_dates.dt.year

    # Select and clean numeric columns
    subset_cols = ["individual_id", "PTGENDER", "PTDOBYY", "PTDOBMM", "PTEDUCAT"]
    valid_cols = [c for c in subset_cols if c in demo_raw.columns]
    demo_clean = demo_raw[valid_cols].copy()
    
    for col in ["PTGENDER", "PTEDUCAT", "PTDOBYY", "PTDOBMM"]:
        if col in demo_clean.columns:
            demo_clean[col] = pd.to_numeric(demo_clean[col], errors='coerce')
            demo_clean.loc[demo_clean[col] < 0, col] = np.nan
    
    demo_clean["individual_id"] = demo_clean["individual_id"].astype(str).str.strip()
    main_df["individual_id"] = main_df["individual_id"].astype(str).str.strip()

    # Aggregate per patient (take first valid entry)
    demo_aggregated = demo_clean.groupby("individual_id").first().reset_index()
    
    # Merge with main dataset
    df_merged = pd.merge(main_df, demo_aggregated, on="individual_id", how="left")

    if target_stages is not None:
        df_merged = df_merged[df_merged["clinical_stage"].isin(target_stages)].reset_index(drop=True)

    # Calculate Age (Handling missing values without dropping rows)
    df_merged["birth_date"] = pd.to_datetime(
        dict(year=df_merged.PTDOBYY, month=df_merged.PTDOBMM, day=15), 
        errors='coerce'
    )
    
    if "exam_date" in df_merged.columns:
        df_merged["age_at_scan"] = (df_merged["exam_date"] - df_merged["birth_date"]).dt.days / 365.25
    else:
        df_merged["age_at_scan"] = np.nan

    # Map Gender
    gender_map = {1.0: "Male", 2.0: "Female", 1: "Male", 2: "Female"}
    df_merged["gender_str"] = df_merged["PTGENDER"].map(gender_map)
    
    # Generate Stats Table
    stats_list = []
    available_groups = sorted([x for x in df_merged["clinical_stage"].unique() if pd.notna(x)]) if "clinical_stage" in df_merged.columns else []
    groups = ["Total"] + available_groups
    
    for group in groups:
        sub_df = df_merged if group == "Total" else df_merged[df_merged["clinical_stage"] == group]
            
        n_scans = len(sub_df)
        n_subjects = sub_df["individual_id"].nunique()
        subj_df = sub_df.drop_duplicates(subset=["individual_id"])
        
        # Gender stats
        n_male = len(subj_df[subj_df["gender_str"] == "Male"])
        n_female = len(subj_df[subj_df["gender_str"] == "Female"])
        n_unknown = len(subj_df) - (n_male + n_female)
        perc_male = (n_male / n_subjects * 100) if n_subjects > 0 else 0
        
        # Age and Education stats
        def format_stat_str(df_col, sub_df_col):
            n_missing = sub_df_col.isna().sum()
            mean, std = sub_df_col.mean(), sub_df_col.std()
            s = f"{mean:.1f} ± {std:.1f}"
            return s + f" ({n_missing} missing)" if n_missing > 0 else s

        age_str = format_stat_str(None, sub_df["age_at_scan"])
        edu_str = format_stat_str(None, subj_df["PTEDUCAT"])

        # Scan counts
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
        
    stats_df = pd.DataFrame(stats_list).set_index("Group")
    return stats_df, df_merged

def run_demographics(adni_path, pipnet_path, classes, modalities):
    path_to_demo = os.path.join(adni_path, "csv", "PTDEMOG_10Feb2026.csv")
    experiment_folder = os.path.join(pipnet_path, "results")
    
    df = load_npy_dataset(classes=classes, adni_path=adni_path, modalities=modalities)
    
    if df.empty:
        print("Warning: Dataset is empty. Check paths.")
        return

    summary_table, _ = get_demographics_summary(df, path_to_demo, target_stages=classes)

    print("\n=== DATASET DEMOGRAPHICS ===")
    print(summary_table.T) 

    os.makedirs(experiment_folder, exist_ok=True)
    class_names = "_".join(cl for cl in classes)
    output_file = os.path.join(experiment_folder, f"dataset_demographics_{class_names}.csv")
    summary_table.to_csv(output_file)
    print(f"\nSaved demographic table to: {output_file}")

if __name__ == "__main__":
    pipnet_path = "/proj/berzbiomedicalimagingkth/users/x_julwe/PIPNet3D/"
    adni_path = "/proj/berzbiomedicalimagingkth/users/x_julwe/ADNI_npy"
    modalities = ["mri", "amy"]
    classes = ["CN", "MCI", "AD"]
    
    run_demographics(adni_path, pipnet_path, classes, modalities)