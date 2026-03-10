"""Visualization utilities for evaluation reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # non-interactive backend for server environments
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve

from urgency.utils.logging import get_logger

logger = get_logger(__name__)

FIGURE_DPI = 120


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    """Save a normalized confusion matrix as PNG.

    Args:
        y_true: True integer class labels.
        y_pred: Predicted integer class labels.
        class_names: Display names for each class index.
        output_path: Path to save the PNG file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=class_names,
        normalize="true",
        ax=ax,
        colorbar=False,
    )
    ax.set_title("Confusion Matrix (row-normalized)")
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved confusion matrix to %s", output_path)


def save_calibration_plot(
    probs: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    n_bins: int = 15,
) -> None:
    """Save a reliability (calibration) diagram as PNG.

    Args:
        probs: Predicted probabilities for the positive class [N].
        labels: Binary ground-truth labels [N].
        output_path: Path to save the PNG file.
        n_bins: Number of bins for the reliability diagram.
    """
    from sklearn.calibration import calibration_curve

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fraction_of_positives, mean_predicted_value = calibration_curve(
        labels, probs, n_bins=n_bins, strategy="uniform"
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
    ax.set_xlabel("Mean predicted probability (urgent)")
    ax.set_ylabel("Fraction of urgent cases")
    ax.set_title("Reliability Diagram (Urgent vs Rest)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved calibration plot to %s", output_path)


def save_roc_curve(
    p_urgent: np.ndarray,
    y_true_binary: np.ndarray,
    output_path: Path,
) -> None:
    """Save ROC curve for urgent vs rest as PNG.

    Args:
        p_urgent: Predicted probability of urgent class [N].
        y_true_binary: Binary labels (1=urgent, 0=rest) [N].
        output_path: Path to save the PNG file.
    """
    from sklearn.metrics import auc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true_binary, p_urgent)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (Urgent vs Rest)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Saved ROC curve to %s", output_path)
