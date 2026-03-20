import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
)


def evaluate(name: str, y_true, y_pred) -> dict:
    """
    Print classification report + confusion matrix.
    Return a dict with Accuracy, Precision, Recall, F1-Score.
    """
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n{'─'*50}")
    print(f"  {name}  —  Accuracy: {acc*100:.2f}%")
    print(f"{'─'*50}")
    print(classification_report(y_true, y_pred,
                                target_names=["No Diabetes", "Diabetes"]))
    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion Matrix:\n{cm}\n")

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
    print(f"  [saved] {filename}")


def plot_comparison(results: dict, filename: str = "comparison_chart.png",
                    title: str = "Classifier Performance Comparison") -> None:
    """
    Grouped bar chart comparing Accuracy, Precision, Recall, F1-Score
    for all classifiers side by side.
    """
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
    classifiers  = list(results.keys())
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

    x     = np.arange(len(metric_names))
    width = 0.15

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (clf, color) in enumerate(zip(classifiers, colors)):
        scores = [results[clf][m] for m in metric_names]
        offset = (i - len(classifiers) / 2 + 0.5) * width
        bars   = ax.bar(x + offset, scores, width, label=clf, color=color)
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.1%}",
                ha="center", va="bottom", fontsize=6.5, rotation=45,
            )

    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  [saved] {filename}")


def plot_feature_importance(model, feature_cols: list,
                            filename: str = "feature_importance.png") -> None:
    """Horizontal bar chart of Random Forest feature importances."""
    importances = model.feature_importances_
    indices     = np.argsort(importances)
    labels      = [feature_cols[i] for i in indices]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(indices)), importances[indices], color="#4C72B0")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Feature Importance (Gini)")
    ax.set_title("Random Forest — Feature Importances")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  [saved] {filename}")


def print_benchmark(results: dict, paper_results: dict,
                    title: str = "BENCHMARK SUMMARY") -> None:
    """Print a summary table comparing our results vs. the paper."""
    header = (f"{'Classifier':<10} {'Acc (ours)':>12} {'Acc (paper)':>12}"
              f" {'Prec':>8} {'Recall':>8} {'F1':>8}")
    print("\n" + "=" * 60)
    print(title.upper())
    print("=" * 60)
    print(header)
    for clf, r in results.items():
        p = paper_results.get(clf, {})
        paper_acc = p.get("Accuracy", float("nan"))
        print(
            f"{clf:<10} {r['Accuracy']*100:>11.2f}%"
            f" {paper_acc*100:>11.2f}%"
            f" {r['Precision']*100:>7.2f}%"
            f" {r['Recall']*100:>7.2f}%"
            f" {r['F1-Score']*100:>7.2f}%"
        )
    print("=" * 60)
