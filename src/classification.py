"""
Classification (assignment section 3).

Trains one interpretable classifier (logistic regression — chosen because
its coefficients are directly explainable to a non-technical cooperative
operations team, matching the "interpretable classifier" requirement) to
predict dispatch_attention. Uses a stratified split, reports confusion
matrix and standard metrics, and explains which error type is costlier.
"""

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils import RANDOM_SEED, ensure_dir, save_json

# In this scenario a False Negative (predicting "no dispatch attention
# needed" when the consignment actually needed it) risks spoiled potatoes
# and a failed collection — materially worse than a False Positive, which
# only costs the operations team a few minutes checking a consignment that
# turns out to be fine. Recall on the positive class therefore matters at
# least as much as raw accuracy.
COST_EXPLANATION = (
    "A false negative (missing a consignment that actually needs dispatch "
    "attention) is costlier than a false positive here: a missed flag can "
    "mean spoiled potatoes and a failed collection run, while a false "
    "positive only costs staff a few extra minutes double-checking a "
    "consignment that turns out to be fine. Recall on the positive class "
    "is therefore prioritized alongside accuracy when judging this model."
)


def run_classification(X: np.ndarray, y: np.ndarray, output_dir: str, models_dir: str,
                        seed: int = RANDOM_SEED) -> dict:
    start = time.time()

    # Stratified split where possible — falls back gracefully if a class
    # has too few members to stratify (e.g. a tiny or unbalanced hidden set).
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        stratified = True
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )
        stratified = False

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(random_state=seed, max_iter=1000)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    metrics = {
        "random_seed": seed,
        "model_type": "LogisticRegression",
        "stratified_split": stratified,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": cm,
        "confusion_matrix_labels": ["no_attention_needed(0)", "attention_needed(1)"],
        "cost_of_errors_explanation": COST_EXPLANATION,
        "runtime_seconds": round(time.time() - start, 4),
    }

    ensure_dir(output_dir)
    save_json(metrics, f"{output_dir}/classification_metrics.json")

    # Confusion matrix plot
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Oranges",
        xticklabels=["Pred: No Attention", "Pred: Attention"],
        yticklabels=["True: No Attention", "True: Attention"],
    )
    plt.title("Confusion Matrix — Dispatch Attention")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=150)
    plt.close()

    ensure_dir(models_dir)
    import joblib
    joblib.dump(
        {"model": model, "scaler": scaler},
        f"{models_dir}/classification_model.joblib",
    )

    return metrics
