# Handles everything before the classifiers touch the data.
# Zeros in medical columns like Glucose or BMI aren't real zeros — they're missing values.
# We replace them with class-specific medians so diabetic and non-diabetic groups
# get their own reference, which is more accurate than a single global median.
# Also adds two binary features from EDA that turned out to improve accuracy.

import pandas as pd
import numpy as np

# Columns where a zero value is biologically impossible
ZERO_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

FEATURE_COLS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
    "High_BP_Glucose", "High_BP",
]


def load_data(path: str = "diabetes.csv") -> pd.DataFrame:
    """Load the raw CSV and return a DataFrame."""
    return pd.read_csv(path)


def explore(df: pd.DataFrame) -> None:
    """Print dataset overview, class distribution, per-feature stats, and correlations."""
    W = 65
    features = [c for c in df.columns if c != "Outcome"]

    # Class distribution
    diabetic   = int((df["Outcome"] == 1).sum())
    non_diab   = int((df["Outcome"] == 0).sum())
    print("\n" + "=" * W)
    print("CLASS DISTRIBUTION")
    print("=" * W)
    print(f"  Tested Positive (Diabetic): {diabetic}")
    print(f"  Tested Negative (Healthy) : {non_diab}")
    print(f"  Total records             : {len(df)}")

    # Dataset info
    print("\n" + "=" * W)
    print("DATASET INFO")
    print("=" * W)
    df.info()

    # Per-feature stats by class
    diab    = df[df["Outcome"] == 1]
    nondiab = df[df["Outcome"] == 0]
    print("\n" + "=" * W)
    print("TABLE 2 — Per-Feature Statistics by Class")
    print("=" * W)
    hdr = (f"{'Feature':<28} {'Mean(D)':>10} {'Mean(ND)':>10}"
           f" {'Std(D)':>10} {'Std(ND)':>10} {'Min':>8} {'Max':>8}")
    print(hdr)
    print("-" * len(hdr))
    for feat in features:
        print(
            f"{feat:<28}"
            f" {diab[feat].mean():>10.2f}"
            f" {nondiab[feat].mean():>10.2f}"
            f" {diab[feat].std():>10.2f}"
            f" {nondiab[feat].std():>10.2f}"
            f" {df[feat].min():>8.2f}"
            f" {df[feat].max():>8.2f}"
        )

    # Zero-value analysis
    print("\n" + "=" * W)
    print("Missing Value Analysis (Biologically Invalid Zeros)")
    print("=" * W)
    for col in ZERO_COLS:
        cnt = int((df[col] == 0).sum())
        pct = cnt / len(df) * 100
        print(f"  {col:<25} zeros={cnt:>3}  ({pct:5.1f}%)")

    # Correlation with Outcome
    print("\n" + "=" * W)
    print("Feature Correlations with Outcome (absolute)")
    print("=" * W)
    corr = df.corr(numeric_only=True)["Outcome"].drop("Outcome").abs().sort_values(ascending=False)
    for feat, val in corr.items():
        print(f"  {feat:<28} {val:.6f}")


def impute_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace zeros in ZERO_COLS with the median computed separately
    for diabetic (Outcome=1) and non-diabetic (Outcome=0) patients —
    the exact strategy described in the paper.
    """
    df = df.copy()
    for col in ZERO_COLS:
        for outcome_val in [0, 1]:
            mask_zero    = (df[col] == 0) & (df["Outcome"] == outcome_val)
            mask_nonzero = (df[col] != 0) & (df["Outcome"] == outcome_val)
            median_val   = df.loc[mask_nonzero, col].median()
            df.loc[mask_zero, col] = df[col].dtype.type(median_val)
    return df

# Lightweight feature engineering based on EDA insights
def add_eda_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 2 binary features extracted via EDA (as described in the paper):
      High_BP_Glucose : BP > 80  AND  Glucose > 105
      High_BP         : BP > 80
    """
    df = df.copy()
    df["High_BP_Glucose"] = (
        (df["BloodPressure"] > 80) & (df["Glucose"] > 105)
    ).astype(int)
    df["High_BP"] = (df["BloodPressure"] > 80).astype(int)
    return df


