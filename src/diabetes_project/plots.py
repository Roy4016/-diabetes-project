"""Common diagnostic plots for binary classifiers."""

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "diabetes-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


def save_diagnostic_curves(y_true, probability, output_dir: str | Path, prefix: str) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, probability)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, label=f"AUROC = {roc_auc_score(y_true, probability):.3f}")
    ax.plot([0, 1], [0, 1], "--", linewidth=1)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve")
    ax.legend(); fig.tight_layout(); fig.savefig(output / f"{prefix}_roc.png", dpi=170); plt.close(fig)

    precision, recall, _ = precision_recall_curve(y_true, probability)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall, precision, label=f"AP = {average_precision_score(y_true, probability):.3f}")
    ax.axhline(sum(y_true) / len(y_true), linestyle="--", linewidth=1)
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision–Recall Curve")
    ax.legend(); fig.tight_layout(); fig.savefig(output / f"{prefix}_pr.png", dpi=170); plt.close(fig)

    observed, predicted = calibration_curve(y_true, probability, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([0, 1], [0, 1], "--", label="Perfect calibration")
    ax.plot(predicted, observed, "o-", label=prefix)
    ax.set(xlabel="Mean predicted probability", ylabel="Observed positive fraction",
           title="Calibration Curve", xlim=(0, 1), ylim=(0, 1))
    ax.legend(); fig.tight_layout(); fig.savefig(output / f"{prefix}_calibration.png", dpi=170); plt.close(fig)

