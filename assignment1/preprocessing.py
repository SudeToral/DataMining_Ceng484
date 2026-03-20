import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Columns where a zero value is biologically impossible
ZERO_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

FEATURE_COLS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
    "High_BP_Glucose", "High_BP",
]


def load_data(path: str = "diabetes.csv") -> pd.DataFrame:
    """Load the raw CSV and return a DataFrame."""
    df = pd.read_csv(path)
    print(f"[load]  shape={df.shape}")
    return df


def explore(df: pd.DataFrame) -> None:
    """Print a quick overview of the dataset."""
    print(df.dtypes)
    print(df.head())
    print(df.describe().round(2))
    print("\nZero counts in medically invalid columns:")
    for col in ZERO_COLS:
        print(f"  {col:25s}: {(df[col] == 0).sum()} zeros")


def impute_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace zeros in ZERO_COLS with the median computed separately
    for diabetic (Outcome=1) and non-diabetic (Outcome=0) patients —
    the exact strategy described in the paper.
    """
    df = df.copy()
    for col in ZERO_COLS:
        for outcome_val, label in [(0, "non-diabetic"), (1, "diabetic")]:
            mask_zero    = (df[col] == 0) & (df["Outcome"] == outcome_val)
            mask_nonzero = (df[col] != 0) & (df["Outcome"] == outcome_val)
            median_val   = df.loc[mask_nonzero, col].median()
            df.loc[mask_zero, col] = df[col].dtype.type(median_val)
            print(f"  [impute] {col} | {label:12s} → {mask_zero.sum():>2d} zeros → median {median_val:.2f}")
    return df


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


def split(df: pd.DataFrame, test_size: float = 0.30, random_state: int = 42):
    """
    Return X_train, X_test, y_train, y_test, X_train_sc, X_test_sc.
    Scaled versions are needed by KNN, SVM, and ANN.
    """
    X = df[FEATURE_COLS]
    y = df["Outcome"]

    # stratify=y preserves the class ratio in both splits (paper's test set:
    # 150 non-diabetic, 81 diabetic — exactly what stratified 30 % gives)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print(f"[split] train={X_train.shape}  test={X_test.shape}")
    return X_train, X_test, y_train, y_test, X_train_sc, X_test_sc


def prepare(path: str = "diabetes.csv", verbose: bool = False):
    """Full pipeline: load → explore → impute → feature-engineer → split."""
    df = load_data(path)
    if verbose:
        explore(df)
    df = impute_zeros(df)
    df = add_eda_features(df)
    return split(df)
