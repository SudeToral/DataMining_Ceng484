# Main entry point. Loads data, runs preprocessing, then trains all five
# classifiers across three feature scenarios (Tables 3, 4, 5 from the paper).
# Results are printed side by side with the paper's reported values.
# Confusion matrices are only saved for Table 3 to keep output manageable.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import preprocessing as pp
import utils
import models


PAPER = {
    "table3": {
        "KNN": {"Accuracy": 0.81,   "Precision": 0.80, "Recall": 0.77, "F1-Score": 0.78},
        "RF" : {"Accuracy": 0.8831, "Precision": 0.88, "Recall": 0.86, "F1-Score": 0.87},
        "SVM": {"Accuracy": 0.77,   "Precision": 0.74, "Recall": 0.74, "F1-Score": 0.74},
        "ANN": {"Accuracy": 0.86,   "Precision": 0.85, "Recall": 0.85, "F1-Score": 0.85},
        "DT" : {"Accuracy": 0.84,   "Precision": 0.82, "Recall": 0.81, "F1-Score": 0.82},
    },
    "table4": {
        "KNN": {"Accuracy": 0.82, "Precision": 0.80, "Recall": 0.79, "F1-Score": 0.80},
        "RF" : {"Accuracy": 0.87, "Precision": 0.86, "Recall": 0.85, "F1-Score": 0.85},
        "SVM": {"Accuracy": 0.77, "Precision": 0.75, "Recall": 0.75, "F1-Score": 0.75},
        "ANN": {"Accuracy": 0.82, "Precision": 0.81, "Recall": 0.80, "F1-Score": 0.80},
        "DT" : {"Accuracy": 0.83, "Precision": 0.82, "Recall": 0.81, "F1-Score": 0.81},
    },
    "table5": {
        "KNN": {"Accuracy": 0.83, "Precision": 0.81, "Recall": 0.80, "F1-Score": 0.81},
        "RF" : {"Accuracy": 0.88, "Precision": 0.88, "Recall": 0.86, "F1-Score": 0.87},
        "SVM": {"Accuracy": 0.78, "Precision": 0.76, "Recall": 0.76, "F1-Score": 0.76},
        "ANN": {"Accuracy": 0.83, "Precision": 0.81, "Recall": 0.80, "F1-Score": 0.81},
        "DT" : {"Accuracy": 0.84, "Precision": 0.82, "Recall": 0.81, "F1-Score": 0.81},
    },
}

CLASSIFIER_CONFIGS = [
    (models.knn, "K-Nearest Neighbors (KNN)", "knn"),
    (models.rf,  "Random Forest (RF)",         "rf"),
    (models.svm, "Support Vector Machine (SVM)", "svm"),
    (models.ann, "Artificial Neural Network (ANN)", "ann"),
    (models.dt,  "Decision Tree (DT)",         "dt"),
]


def run_scenario(X_train, X_test, y_train, y_test,
                 X_train_sc, X_test_sc,
                 feature_cols, table_prefix, save_cm=True):
    results        = {}
    trained_models = {}

    for mod, display_name, prefix in CLASSIFIER_CONFIGS:
        print(f"    Training {display_name}...", end=" ", flush=True)
        clf = mod.build()
        if mod.USES_SCALED:
            clf.fit(X_train_sc, y_train)
            y_pred = clf.predict(X_test_sc)
        else:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

        if hasattr(clf, "best_params_"):
            print(f"(best k={clf.best_params_['n_neighbors']})", end=" ")

        results[mod.NAME]        = utils.evaluate(y_test, y_pred)
        print(f"-> Accuracy: {results[mod.NAME]['Accuracy']*100:.2f}%")
        trained_models[mod.NAME] = (clf, y_pred)

        if save_cm:
            utils.plot_confusion_matrix(
                y_test, y_pred,
                title    = f"{display_name} — Confusion Matrix ({table_prefix})",
                filename = f"{table_prefix}_{prefix}_confusion_matrix.png",
            )

    return results, trained_models


print("=" * 75)
print("DIABETES PREDICTION — REPRODUCING PAPER RESULTS")
print("=" * 75)

print("\n[1/4] Loading and exploring dataset...")
df_raw = pp.load_data("diabetes.csv")
pp.explore(df_raw)

print("\n[2/4] Preprocessing...")
print("      Replacing biologically impossible zero values with class-conditional medians...")
df_clean = pp.impute_zeros(df_raw)
print("\n" + "=" * 65)
print("PREPROCESSING — Zero Imputation Complete")
print("=" * 65)
print("      Remaining NaN values after imputation:")
print(df_clean.isna().sum().to_string())
print("      Engineering 2 additional features (High_BP_Glucose, High_BP)...")
df_eda   = pp.add_eda_features(df_clean)
print("      Feature engineering complete.")

y = df_eda["Outcome"]

SCENARIOS = {
    "table3": {
        "label"   : "Table 3 — 8 original + 2 EDA features",
        "features": pp.FEATURE_COLS,
    },
    "table4": {
        "label"   : "Table 4 — 8 original features only",
        "features": [c for c in pp.FEATURE_COLS
                     if c not in ("High_BP_Glucose", "High_BP")],
    },
    "table5": {
        "label"   : "Table 5 — 6 features (no SkinThickness, PedigreeFunc, EDA)",
        "features": [c for c in pp.FEATURE_COLS
                     if c not in ("SkinThickness", "DiabetesPedigreeFunction",
                                  "High_BP_Glucose", "High_BP")],
    },
}

all_results = {}

print("\n[3/4] Training and evaluating classifiers across 3 feature scenarios...")

for table_prefix, cfg in SCENARIOS.items():
    print(f"\n  Scenario: {cfg['label']}")
    print(f"  Features ({len(cfg['features'])}): {', '.join(cfg['features'])}")

    X = df_eda[cfg["features"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    print(f"  Train/test split (70/30, stratified): {X_train.shape[0]} train / {X_test.shape[0]} test samples")

    print("  Applying StandardScaler to distance/gradient-based models (KNN, SVM, ANN)...")
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print("  Running classifiers (KNN uses 5-fold GridSearch for k):")
    results, trained_models = run_scenario(
        X_train, X_test, y_train, y_test,
        X_train_sc, X_test_sc,
        cfg["features"], table_prefix,
        save_cm=(table_prefix == "table3"),
    )
    all_results[table_prefix] = results

    utils.print_benchmark(results, PAPER[table_prefix], title=cfg["label"])

print("\n[4/4] Generating accuracy comparison chart (Config A vs B vs C)...")
utils.plot_accuracy_comparison(
    all_results["table3"],
    all_results["table4"],
    all_results["table5"],
    filename="figure8_accuracy_comparison.png",
)
print("      Saved → figure8_accuracy_comparison.png")
print("\nAll scenarios complete.")
print("=" * 75)

