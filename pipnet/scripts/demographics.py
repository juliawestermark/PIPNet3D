import os
import pandas as pd
# import logging

from make_dataset import setup_mri_dataframe

# logger = logging.getLogger(__name__)

ADNI_PATH = "/home/maia-user/ADNI_complete"
DEMOGRAPHICS_PATH = os.path.join(ADNI_PATH, "participant_demographics.csv")

def calculate_age(row: pd.Series) -> int:
    """Calculate age at baseline date."""
    birth_year = row["PTDOBYY"]
    birth_month = row["PTDOBMM"]
    baseline_date = row["baseline_date"]
    if pd.isna(birth_year) or pd.isna(birth_month) or pd.isna(baseline_date):
        return pd.NA
    try:
        birth_date = pd.Timestamp(year=int(birth_year), month=int(birth_month), day=15)  # Use 15:e as an approximation
        age = (baseline_date - birth_date).days / 365  # In years
        return int(age)
    except Exception as e:
        print(f"Error calculating age for birth_year={birth_year}, birth_month={birth_month}, baseline_date={baseline_date}: {e}")
        return pd.NA


def get_nbr_of_exams(subject_id: str, dataframe: pd.DataFrame) -> int:
    """Returns the number of MRI exams for a given subject ID."""
    exams = dataframe[dataframe["individual_id"] == subject_id]
    return len(exams)
    

def get_demographics(dataframe: pd.DataFrame, csv_demographics: pd.DataFrame) -> pd.DataFrame:
    """Extract and merge demographic data for subjects in the given dataframe."""

    df_subjects = dataframe.groupby("individual_id").first().reset_index().copy()
    df_subjects = df_subjects[["individual_id", "baseline_date", "baseline_class"]]


    # Rename and select relevant columns from demographics CSV
    cols = {
        "Individual": "individual_id",
        "PTGENDER(1. Participant Gender T: 1=Male; 2=Female)": "PTGENDER",
        "PTDOBMM(Participant Month of Birth)": "PTDOBMM",
        "PTDOBYY(Participant Year of Birth)": "PTDOBYY",
    }
    df_demographics = csv_demographics[list(cols.keys())].rename(columns=cols)

    df_demographics = df_demographics.replace(-4, pd.NA).replace("-4", pd.NA)

    df_demographics = df_demographics.dropna(subset=["PTGENDER", "PTDOBMM", "PTDOBYY"], how="all")

    df_demographics = df_demographics.groupby("individual_id").first().reset_index()

    df_subjects = df_subjects.merge(df_demographics, on="individual_id", how="left")

    missing = df_subjects[df_subjects[["PTGENDER", "PTDOBMM", "PTDOBYY"]].isna().any(axis=1)]

    print(f"Subjects in df {len(df_subjects)}, missing in df {len(missing)}")

    df_subjects["AGE"] = df_subjects.apply(calculate_age, axis=1)
    
    return df_subjects


def print_demographics_info(df_subjects: pd.DataFrame, exams_dataframe: pd.DataFrame, info_class: str = "ALL"):
    """Print demographics info for a given class (ALL, CN, MCI, AD)."""
    subset = df_subjects
    if info_class in ["CN", "MCI", "AD"]:
        subset = df_subjects[df_subjects["baseline_class"] == info_class].copy()

    min_age = subset["AGE"].min()
    max_age = subset["AGE"].max()
    mean_age = subset["AGE"].mean()
    unknown_age = subset["AGE"].isna().sum()
    gender_counts = subset["PTGENDER"].value_counts() # Male 1, Female 2
    unknown_gender = subset["PTGENDER"].isna().sum()

    nbr_of_exams = subset["individual_id"].apply(get_nbr_of_exams, args=(exams_dataframe,))

    print(f"\n--- {info_class} ---")
    print(f"Number of subjects: {len(subset)}")
    print(f"Number of exams: {nbr_of_exams.sum()}")
    print(f"Min age: {min_age:.1f}")
    print(f"Max age: {max_age:.1f}")
    print(f"Mean age: {mean_age:.1f}")
    print(f"Gender (M/F): {gender_counts["1"]}/{gender_counts["2"]}")
    print(f"Number with unknown age: {unknown_age} ({unknown_age/len(subset):.2%})")
    print(f"Number with unknown gender: {unknown_gender}")


def print_all_info():
    csv_demographics = pd.read_csv(DEMOGRAPHICS_PATH)
    info_classes = ["CN", "MCI", "AD"]
    mri = setup_mri_dataframe(classes=info_classes)
    df_subjects = get_demographics(mri, csv_demographics)
    for info_class in ["ALL"] + info_classes:
        print_demographics_info(df_subjects, mri, info_class)

if __name__ == "__main__":
    print_all_info()
