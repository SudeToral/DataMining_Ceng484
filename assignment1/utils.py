# Shared evaluation and plotting helpers used by train.py.
# evaluate() returns the four metrics we care about.
# print_benchmark() shows our results and the paper's results side by side
# so it's easy to spot where we match and where we diverge.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)


def evaluate(y_true, y_pred) -> dict:
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1}


def plot_confusion_matrix(y_true, y_pred, title: str, filename: str) -> None:
    """Save a heatmap of the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Diabetes (0)", "Diabetes (1)"],
        yticklabels=["No Diabetes (0)", "Diabetes (1)"],
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_accuracy_comparison(results_with: dict, results_without: dict,
                             filename: str = "figure8_accuracy_comparison.png") -> None:
    """
    Grouped bar chart: accuracy WITH vs WITHOUT the 2 EDA features,
    matching Figure 8 style from the paper.
    """
    clf_keys    = ["KNN", "RF",  "SVM", "ANN", "DT"]
    x_labels    = ["KNN", "RF",  "SVM", "ANN", "DS"]

    acc_with    = [results_with[k]["Accuracy"]    for k in clf_keys]
    acc_without = [results_without[k]["Accuracy"] for k in clf_keys]

    x     = np.arange(len(x_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, acc_with,    width, color="#4472C4",
                   label="Accuracy with using two new extracted feature")
    bars2 = ax.bar(x + width / 2, acc_without, width, color="#9E3B2E",
                   label="Accuracy without using two new extracted feature")

    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0", "0,2", "0,4", "0,6", "0,8", "1"])
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=12)
    ax.set_xlabel("Classification Techniques", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def print_benchmark(results: dict, paper_results: dict,
                    title: str = "BENCHMARK SUMMARY") -> None:
    """Print two side-by-side tables: our results vs. paper results."""
    W = 75
    print("\n" + "=" * W)
    print(title.upper())
    print("=" * W)

    # Our results
    print(f"\n  {'[ OUR RESULTS ]'}")
    h1 = (f"  {'Classifier':<16} {'Accuracy':>10} {'Precision':>10}"
          f" {'Recall':>8} {'F1-Score':>10}")
    print(h1)
    print("  " + "-" * (len(h1) - 2))
    for clf, r in results.items():
        print(
            f"  {clf:<16} {r['Accuracy']*100:>9.2f}%"
            f" {r['Precision']*100:>9.2f}%"
            f" {r['Recall']*100:>7.2f}%"
            f" {r['F1-Score']*100:>9.2f}%"
        )

    # Paper results
    print(f"\n  {'[ PAPER RESULTS ]'}")
    print(h1)
    print("  " + "-" * (len(h1) - 2))
    for clf in results:
        p = paper_results.get(clf, {})
        print(
            f"  {clf:<16} {p.get('Accuracy', float('nan'))*100:>9.2f}%"
            f" {p.get('Precision', float('nan'))*100:>9.2f}%"
            f" {p.get('Recall', float('nan'))*100:>7.2f}%"
            f" {p.get('F1-Score', float('nan'))*100:>9.2f}%"
        )

    print("\n" + "=" * W)
