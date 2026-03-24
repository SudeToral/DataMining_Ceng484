#generates a single PNG with distribution plots and statistics for each feature.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

TEAL  = "#1ABC9C"
RED   = "#C0392B"
GOLD  = "#C9A84C"
LIGHT = "#B0BEC5"
WHITE = "#FFFFFF"
BG    = "#0F1B2D"
CARD  = "#1A2940"

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
ZERO_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def plot_feature(ax_hist, ax_stats, col, series_all, series_neg, series_pos):
    # Histogram + KDE
    ax_hist.set_facecolor(CARD)
    bins = min(30, int(np.sqrt(len(series_all))))
    ax_hist.hist(series_neg, bins=bins, alpha=0.55, color=TEAL,
                 label="No Diabetes", density=True, edgecolor="none")
    ax_hist.hist(series_pos, bins=bins, alpha=0.55, color=RED,
                 label="Diabetes", density=True, edgecolor="none")
    for s, c in [(series_neg, TEAL), (series_pos, RED)]:
        if s.std() > 0:
            kde_x = np.linspace(s.min(), s.max(), 200)
            ax_hist.plot(kde_x, stats.gaussian_kde(s)(kde_x), color=c, lw=1.5)
    ax_hist.axvline(series_neg.mean(), color=TEAL, linestyle="--", lw=1, alpha=0.8)
    ax_hist.axvline(series_pos.mean(), color=RED,  linestyle="--", lw=1, alpha=0.8)
    ax_hist.set_title(col, color=WHITE, fontsize=10, fontweight="bold", pad=4)
    ax_hist.tick_params(colors=LIGHT, labelsize=7)
    for spine in ax_hist.spines.values():
        spine.set_edgecolor("#2C3E50")
    ax_hist.yaxis.set_visible(False)
    ax_hist.legend(fontsize=6, framealpha=0, labelcolor=LIGHT)

    # Stats text panel
    ax_stats.set_facecolor(CARD)
    ax_stats.axis("off")
    skew = series_all.skew()
    skew_label = ("right-skewed" if skew > 0.5
                  else "left-skewed" if skew < -0.5 else "normal")
    t, p = stats.ttest_ind(series_neg.dropna(), series_pos.dropna())
    sig  = "p < 0.05 ✓" if p < 0.05 else "p ≥ 0.05"

    lines = [
        ("Statistics", GOLD, 8, "bold"),
        (f"Mean:  {series_all.mean():.2f}   Std: {series_all.std():.2f}", LIGHT, 7, "normal"),
        (f"Min:   {series_all.min():.1f}     Max: {series_all.max():.1f}", LIGHT, 7, "normal"),
        (f"Skew:  {skew:.2f}  ({skew_label})", LIGHT, 7, "normal"),
    ]
    if col in ZERO_COLS:
        zero_pct = (series_all == 0).mean() * 100
        lines.append((f"Zeros: {zero_pct:.1f}%  (imputed)", LIGHT, 7, "normal"))
    lines += [
        ("", WHITE, 4, "normal"),
        ("By Class", TEAL, 8, "bold"),
        (f"No Diabetes  mean={series_neg.mean():.2f}  std={series_neg.std():.2f}", LIGHT, 7, "normal"),
        (f"Diabetes     mean={series_pos.mean():.2f}  std={series_pos.std():.2f}", LIGHT, 7, "normal"),
        (f"Difference   {abs(series_pos.mean()-series_neg.mean()):.2f}", LIGHT, 7, "normal"),
        ("", WHITE, 4, "normal"),
        ("t-test", GOLD, 8, "bold"),
        (f"t={t:.2f}  p={p:.4f}  {sig}", LIGHT, 7, "normal"),
    ]
    y = 0.97
    for text, color, size, weight in lines:
        ax_stats.text(0.05, y, text, transform=ax_stats.transAxes,
                      color=color, fontsize=size, fontweight=weight,
                      va="top", ha="left")
        y -= 0.09 if size >= 8 else 0.08


def build_png(df: pd.DataFrame):
    n_cols = 4
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 8))
    fig.patch.set_facecolor(BG)
    axes = axes.flatten()

    for i, col in enumerate(FEATURES):
        series_all = df[col]
        series_neg = df.loc[df["Outcome"] == 0, col]
        series_pos = df.loc[df["Outcome"] == 1, col]

        ax = axes[i]
        ax.set_facecolor(CARD)
        bins = min(30, int(np.sqrt(len(series_all))))
        ax.hist(series_neg, bins=bins, alpha=0.55, color=TEAL,
                label="No Diabetes", density=True, edgecolor="none")
        ax.hist(series_pos, bins=bins, alpha=0.55, color=RED,
                label="Diabetes", density=True, edgecolor="none")
        for s, c in [(series_neg, TEAL), (series_pos, RED)]:
            if s.std() > 0:
                kde_x = np.linspace(s.min(), s.max(), 200)
                ax.plot(kde_x, stats.gaussian_kde(s)(kde_x), color=c, lw=1.5)
        ax.axvline(series_neg.mean(), color=TEAL, linestyle="--", lw=1, alpha=0.8)
        ax.axvline(series_pos.mean(), color=RED,  linestyle="--", lw=1, alpha=0.8)
        ax.set_title(col, color=WHITE, fontsize=10, fontweight="bold", pad=4)
        ax.tick_params(colors=LIGHT, labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2C3E50")
        ax.yaxis.set_visible(False)
        if i == 0:
            ax.legend(fontsize=7, framealpha=0, labelcolor=LIGHT)

    fig.suptitle("Feature Distributions by Class", color=WHITE,
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout(h_pad=1.5, w_pad=1.0)
    out = "feature_distributions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv("diabetes.csv")
    build_png(df)
