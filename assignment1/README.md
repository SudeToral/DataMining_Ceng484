# Diabetes Prediction — CENG484 Assignment 1

Reproduction of the ML classification experiments from:

> Nahzat, S. & Yağanoğlu, M. (2021). *Diabetes Prediction Using Machine Learning Classification Algorithms*. European Journal of Science and Technology, (24), 53-59. https://doi.org/10.31590/ejosat.899716

Dataset: [UCI Pima Indians Diabetes Database](https://archive.ics.uci.edu/ml/datasets/pima+indians+diabetes)

---

## What this does

Trains five classifiers (KNN, Random Forest, SVM, ANN, Decision Tree) on the Pima Indians Diabetes dataset and reproduces the three benchmark tables from the paper:

- **Table 3** — all 8 original features + 2 EDA-derived features
- **Table 4** — 8 original features only
- **Table 5** — 6 features (SkinThickness and DiabetesPedigreeFunction removed)

As an extra, the ANN is also run with a Keras implementation (Dropout + class_weight) to address the class imbalance the paper does not discuss (500 healthy vs 268 diabetic).


Outputs: confusion matrices, comparison charts, and feature importance plots for all three scenarios.

---

## Results vs. paper

| | Table 3 | | Table 4 | | Table 5 | |
|---|---|---|---|---|---|---|
| | Ours | Paper | Ours | Paper | Ours | Paper |
| KNN | 79.65% | 81% | 80.09% | 82% | 81.82% | 83% |
| RF | 87.88% | 88.31% | 87.45% | 87% | 88.31% | 88% |
| SVM | 80.09% | 77% | 81.39% | 77% | 83.98% | 78% |
| ANN | 82.25% | 86% | 83.98% | 82% | 86.15% | 83% |
| DT | 84.42% | 84% | 83.55% | 83% | 83.98% | 84% |

RF and DT match the paper closely across all three scenarios. The ANN gap exists because the paper does not specify the architecture, library, or any hyperparameters.

---

## Findings

**RF is consistently the best classifier** across all three feature sets, which confirms the paper's main claim. DT is a close second in most scenarios.

**EDA features don't uniformly help.** The paper presents them as an improvement, but KNN actually scores slightly better without them (Table 4 vs Table 3). The benefit is model-dependent, not universal.

**Class imbalance is a real issue the paper ignores.** The dataset has 500 healthy vs 268 diabetic samples. With the standard sklearn ANN, diabetic recall sits at 70% — meaning 3 in 10 diabetic patients are missed. The Keras model with class_weight correction brings this up to 77%. For a medical application this matters more than overall accuracy.

**SVM results are hard to reproduce.** The paper reports 77-78% for SVM but our implementation scores higher (80-84%) across all scenarios. Without knowing the kernel, C, or gamma used, it's impossible to match their exact numbers.

**Removing SkinThickness and DiabetesPedigreeFunction doesn't hurt.** RF hits its best score (88.31%) with only 6 features in Table 5, suggesting those two columns add noise rather than signal for tree-based models.
