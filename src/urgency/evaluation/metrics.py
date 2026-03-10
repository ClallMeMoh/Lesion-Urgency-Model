"""Evaluation metrics for urgency classification."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)

from urgency.data.splits import URGENCY_CLASSES

# Integer class index for "urgent"
URGENT_CLASS = 0


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    p_urgent: np.ndarray,
    triage_decisions: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute full evaluation metric suite.

    Args:
        y_true: True integer class labels [N].
        y_pred: Predicted integer class labels (argmax) [N].
        p_urgent: Probability of the urgent class [N].
        triage_decisions: Triage decision labels after threshold application [N].
        n_bootstrap: Number of bootstrap iterations for CIs (0 to skip).
        seed: Random seed for bootstrap sampling.

    Returns:
        Flat dictionary with metric names and values.
    """
    n = len(y_true)
    rng = np.random.default_rng(seed)

    # Core metrics using argmax predictions
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # Urgent vs rest (binary)
    y_binary = (y_true == URGENT_CLASS).astype(int)
    pred_binary = (triage_decisions == URGENT_CLASS).astype(int)

    urgent_sensitivity = _sensitivity(y_binary, pred_binary)
    urgent_specificity = _specificity(y_binary, pred_binary)

    try:
        roc_auc = float(roc_auc_score(y_binary, p_urgent))
    except ValueError:
        roc_auc = float("nan")

    try:
        pr_auc = float(average_precision_score(y_binary, p_urgent))
    except ValueError:
        pr_auc = float("nan")

    ece = compute_ece(p_urgent, y_binary)

    # Confusion matrix as list-of-lists for JSON serialization
    n_classes = len(URGENCY_CLASSES)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, triage_decisions):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1

    # Abstain rate: fraction predicted as uncertain by threshold logic
    abstain_rate = float((triage_decisions == 2).mean())

    result: dict[str, Any] = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "urgent_sensitivity": urgent_sensitivity,
        "urgent_specificity": urgent_specificity,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "ece": ece,
        "abstain_rate": abstain_rate,
        "confusion_matrix": cm.tolist(),
        "n_samples": n,
    }

    # Bootstrap confidence intervals
    if n_bootstrap > 0:
        ci = _bootstrap_ci(
            y_true=y_true,
            y_pred=y_pred,
            p_urgent=p_urgent,
            triage_decisions=triage_decisions,
            n_bootstrap=n_bootstrap,
            rng=rng,
        )
        result.update(ci)

    return result


def apply_triage(
    p_urgent: np.ndarray,
    t_high: float,
    t_low: float,
) -> np.ndarray:
    """Apply triage thresholds to produce a decision for each sample.

    Rules:
        p_urgent >= t_high -> 0 (urgent)
        p_urgent <= t_low  -> 1 (monitor)
        otherwise          -> 2 (uncertain)

    Args:
        p_urgent: Probability of the urgent class [N].
        t_high: Upper threshold for urgent decision.
        t_low: Lower threshold for monitor decision.

    Returns:
        Integer array of triage decisions [N].
    """
    decisions = np.full(len(p_urgent), 2, dtype=int)  # default: uncertain
    decisions[p_urgent >= t_high] = 0  # urgent
    decisions[p_urgent <= t_low] = 1   # monitor
    return decisions


def tune_thresholds(
    p_urgent: np.ndarray,
    y_true: np.ndarray,
    target_sensitivity: float = 0.90,
    grid_steps: int = 50,
) -> tuple[float, float]:
    """Grid-search triage thresholds on validation data.

    Objective (in priority order):
        1. Maximize urgent sensitivity to >= target_sensitivity
        2. Minimize abstain rate
        3. Maximize urgent specificity

    Args:
        p_urgent: Calibrated urgent probability [N].
        y_true: True integer class labels [N].
        target_sensitivity: Minimum required urgent sensitivity.
        grid_steps: Number of grid points per threshold dimension.

    Returns:
        Tuple of (t_high, t_low).
    """
    y_binary = (y_true == URGENT_CLASS).astype(int)
    grid = np.linspace(0.01, 0.99, grid_steps)

    best = {"t_high": 0.6, "t_low": 0.3, "sensitivity": 0.0, "abstain": 1.0, "specificity": 0.0}

    for t_high in grid:
        for t_low in grid:
            if t_low >= t_high:
                continue
            decisions = apply_triage(p_urgent, t_high, t_low)
            pred_binary = (decisions == URGENT_CLASS).astype(int)
            sens = _sensitivity(y_binary, pred_binary)
            spec = _specificity(y_binary, pred_binary)
            abstain = float((decisions == 2).mean())

            if sens < target_sensitivity:
                continue

            is_better = (
                abstain < best["abstain"]
                or (abstain == best["abstain"] and spec > best["specificity"])
            )
            if is_better:
                best = {"t_high": t_high, "t_low": t_low, "sensitivity": sens, "abstain": abstain, "specificity": spec}

    return float(best["t_high"]), float(best["t_low"])


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error.

    Args:
        probs: Predicted probabilities for the positive class [N].
        labels: Binary ground-truth labels [N].
        n_bins: Number of equal-width probability bins.

    Returns:
        ECE scalar.
    """
    if len(probs) == 0:
        return float("nan")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        bin_conf = probs[mask].mean()
        bin_acc = labels[mask].mean()
        ece += mask.sum() / n * abs(bin_conf - bin_acc)

    return float(ece)


def _sensitivity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """True positive rate (recall) for the positive class."""
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0


def _specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """True negative rate for the positive class."""
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0


def _bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    p_urgent: np.ndarray,
    triage_decisions: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Compute bootstrap confidence intervals for key metrics.

    Returns a flat dict with {metric}_ci_lower and {metric}_ci_upper.
    """
    n = len(y_true)
    f1_scores, sens_scores, auc_scores = [], [], []
    y_binary = (y_true == URGENT_CLASS).astype(int)

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        bt = y_true[idx]
        bp = y_pred[idx]
        bpu = p_urgent[idx]
        btd = triage_decisions[idx]
        bbin = y_binary[idx]
        bpred = (btd == URGENT_CLASS).astype(int)

        f1_scores.append(float(f1_score(bt, bp, average="macro", zero_division=0)))
        sens_scores.append(_sensitivity(bbin, bpred))
        try:
            auc_scores.append(float(roc_auc_score(bbin, bpu)))
        except ValueError:
            auc_scores.append(float("nan"))

    def ci(arr: list[float]) -> tuple[float, float]:
        a = np.array([x for x in arr if not np.isnan(x)])
        if len(a) == 0:
            return float("nan"), float("nan")
        lo = float(np.percentile(a, 100 * alpha / 2))
        hi = float(np.percentile(a, 100 * (1 - alpha / 2)))
        return lo, hi

    f1_lo, f1_hi = ci(f1_scores)
    sens_lo, sens_hi = ci(sens_scores)
    auc_lo, auc_hi = ci(auc_scores)

    return {
        "macro_f1_ci_lower": f1_lo,
        "macro_f1_ci_upper": f1_hi,
        "urgent_sensitivity_ci_lower": sens_lo,
        "urgent_sensitivity_ci_upper": sens_hi,
        "roc_auc_ci_lower": auc_lo,
        "roc_auc_ci_upper": auc_hi,
    }
