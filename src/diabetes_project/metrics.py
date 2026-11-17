"""Shared binary-classification metrics and threshold policies."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, auc, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score,
)


def pr_auc_scores(y_true, probability, sample_weight=None) -> dict:
    """Return both PR-AUC conventions; average precision is primary."""
    precision, recall, _ = precision_recall_curve(
        y_true, probability, sample_weight=sample_weight,
    )
    return {
        "PR_AUC_AP": float(average_precision_score(
            y_true, probability, sample_weight=sample_weight,
        )),
        "PR_AUC_trapezoid": float(auc(recall, precision)),
    }


def best_f1_threshold(y_true, probability) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    if thresholds.size == 0:
        return 0.5
    scores = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-12
    )
    return float(thresholds[int(np.nanargmax(scores))])


def binary_metrics(y_true, probability, threshold: float) -> dict:
    prediction = (np.asarray(probability) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "AUROC": float(roc_auc_score(y_true, probability)),
        **pr_auc_scores(y_true, probability),
        "Accuracy": float(accuracy_score(y_true, prediction)),
        "Precision": float(precision_score(y_true, prediction, zero_division=0)),
        "Recall": float(recall_score(y_true, prediction, zero_division=0)),
        "F1": float(f1_score(y_true, prediction, zero_division=0)),
        "Brier": float(brier_score_loss(y_true, probability)),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
    }
