import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import preprocessing as pp
import utils
import models
from models import ann_keras


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
        print(f"\n{display_name}")

        clf = mod.build()
        if mod.USES_SCALED:
            clf.fit(X_train_sc, y_train)
            y_pred = clf.predict(X_test_sc)
        else:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

        if hasattr(clf, "best_params_"):
            print(f"  best params: {clf.best_params_}")

        results[mod.NAME]        = utils.evaluate(display_name, y_test, y_pred)
        trained_models[mod.NAME] = (clf, y_pred)

        if save_cm:
            utils.plot_confusion_matrix(
                y_test, y_pred,
                title    = f"{display_name} — Confusion Matrix ({table_prefix})",
                filename = f"{table_prefix}_{prefix}_confusion_matrix.png",
            )

    return results, trained_models


print("=" * 60)
print("DATA PREPARATION")
print("=" * 60)

df_raw   = pp.load_data("diabetes.csv")
pp.explore(df_raw)
df_clean = pp.impute_zeros(df_raw)
df_eda   = pp.add_eda_features(df_clean)

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

for table_prefix, cfg in SCENARIOS.items():
    print("\n" + "=" * 60)
    print(cfg["label"].upper())
    print(f"Features: {cfg['features']}")
    print("=" * 60)

    X = df_eda[cfg["features"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print(f"[split] train={X_train.shape}  test={X_test.shape}")

    results, trained_models = run_scenario(
        X_train, X_test, y_train, y_test,
        X_train_sc, X_test_sc,
        cfg["features"], table_prefix,
    )
    all_results[table_prefix] = results

    utils.plot_comparison(
        results,
        filename=f"{table_prefix}_comparison_chart.png",
        title=f"Classifier Comparison — {cfg['label']}",
    )

    utils.print_benchmark(results, PAPER[table_prefix], title=cfg["label"])

    if table_prefix == "table3":
        rf_clf, _ = trained_models["RF"]
        if rf_clf is not None:
            utils.plot_feature_importance(
                rf_clf, cfg["features"], filename="feature_importance.png"
            )

print("\n" + "=" * 60)
print("ALL SCENARIOS COMPLETE")
print("=" * 60)

print("\n" + "=" * 60)
print("ANN COMPARISON: sklearn (paper) vs Keras (enhancement)")
print("Feature set: Table 3  (8 original + 2 EDA features)")
print("=" * 60)

X_t3 = df_eda[SCENARIOS["table3"]["features"]]
X_tr, X_te, y_tr, y_te = train_test_split(
    X_t3, y, test_size=0.30, random_state=42, stratify=y
)
sc       = StandardScaler()
X_tr_sc  = sc.fit_transform(X_tr)
X_te_sc  = sc.transform(X_te)

print("\n--- sklearn MLPClassifier (paper reproduction) ---")
sk_ann = models.ann.build()
sk_ann.fit(X_tr_sc, y_tr)
y_sk   = sk_ann.predict(X_te_sc)
res_sk = utils.evaluate("ANN (sklearn)", y_te, y_sk)
utils.plot_confusion_matrix(
    y_te, y_sk,
    title    = "ANN sklearn — Confusion Matrix",
    filename = "ann_sklearn_confusion_matrix.png",
)

print("\n--- Keras ANN (Dropout + class_weight enhancement) ---")
y_ke   = ann_keras.fit_predict(X_tr_sc, y_tr.values, X_te_sc)
res_ke = utils.evaluate("ANN (Keras)", y_te, y_ke)
utils.plot_confusion_matrix(
    y_te, y_ke,
    title    = "ANN Keras — Confusion Matrix",
    filename = "ann_keras_confusion_matrix.png",
)

utils.plot_comparison(
    {"sklearn": res_sk, "Keras": res_ke},
    filename = "ann_sklearn_vs_keras.png",
    title    = "ANN: sklearn (paper) vs Keras (enhancement)",
)

print("\nANN Comparison Summary:")
header = f"{'Version':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}"
print(header)
for label, res in [("sklearn (paper)", res_sk), ("Keras (enhanced)", res_ke)]:
    print(f"{label:<20} {res['Accuracy']*100:>9.2f}%"
          f" {res['Precision']*100:>9.2f}%"
          f" {res['Recall']*100:>9.2f}%"
          f" {res['F1-Score']*100:>9.2f}%")
