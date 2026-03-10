"""Error analysis: identify and report top false positives and false negatives."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from urgency.config import AppConfig
from urgency.data.splits import URGENCY_CLASSES
from urgency.utils.io import save_json
from urgency.utils.logging import get_logger

logger = get_logger(__name__)

URGENT_CLASS = 0


def write_error_analysis(
    df: pd.DataFrame,
    output_path: Path,
    n_top: int = 20,
) -> None:
    """Write a markdown error analysis report.

    Identifies and reports:
        - Top false negatives: urgent cases predicted as monitor or uncertain.
          Sorted by p_urgent ascending (most missed urgent cases first).
        - Top false positives: non-urgent predicted as urgent.
          Sorted by p_urgent descending (most confident wrong urgent predictions).

    Args:
        df: DataFrame with columns: image_path, y_true, triage_decision, p_urgent.
        output_path: Path to write the markdown report.
        n_top: Maximum number of cases to include per error category.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # False negatives: true urgent, predicted non-urgent
    fn_mask = (df["y_true"] == URGENT_CLASS) & (df["triage_decision"] != URGENT_CLASS)
    fn_df = (
        df[fn_mask]
        .sort_values("p_urgent", ascending=True)
        .head(n_top)
        .reset_index(drop=True)
    )

    # False positives: true non-urgent, predicted urgent
    fp_mask = (df["y_true"] != URGENT_CLASS) & (df["triage_decision"] == URGENT_CLASS)
    fp_df = (
        df[fp_mask]
        .sort_values("p_urgent", ascending=False)
        .head(n_top)
        .reset_index(drop=True)
    )

    lines = [
        "# Error Analysis",
        "",
        f"Generated: {datetime.now(tz=timezone.utc).isoformat()}",
        "",
        f"Total false negatives (urgent missed): {fn_mask.sum()}",
        f"Total false positives (non-urgent predicted urgent): {fp_mask.sum()}",
        "",
    ]

    lines += _render_table(
        fn_df,
        title=f"Top-{n_top} False Negatives (Urgent Missed)",
        note="Sorted by ascending p_urgent — most missed urgent cases at the top.",
    )
    lines += [""]
    lines += _render_table(
        fp_df,
        title=f"Top-{n_top} False Positives (Non-Urgent Predicted Urgent)",
        note="Sorted by descending p_urgent — most confidently wrong predictions at the top.",
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved error analysis to %s", output_path)


def _render_table(
    df: pd.DataFrame,
    title: str,
    note: str,
) -> list[str]:
    """Render a DataFrame as a markdown table section."""
    lines = [f"## {title}", "", f"_{note}_", ""]
    if df.empty:
        lines.append("No cases in this category.")
        return lines

    true_col = "y_true"
    pred_col = "triage_decision"

    lines.append("| image_path | true_label | pred_label | p_urgent |")
    lines.append("|------------|------------|------------|----------|")
    for _, row in df.iterrows():
        true_name = URGENCY_CLASSES[int(row[true_col])] if int(row[true_col]) < len(URGENCY_CLASSES) else str(row[true_col])
        pred_name = URGENCY_CLASSES[int(row[pred_col])] if int(row[pred_col]) < len(URGENCY_CLASSES) else str(row[pred_col])
        p = f"{row['p_urgent']:.4f}"
        lines.append(f"| {row['image_path']} | {true_name} | {pred_name} | {p} |")

    return lines


def write_run_summary(
    cfg: AppConfig,
    metrics: dict[str, Any],
    output_path: Path,
    git_hash: str,
) -> None:
    """Write a human-readable run summary to markdown.

    Args:
        cfg: Full application config for this run.
        metrics: Flat metrics dictionary from compute_metrics().
        output_path: Path to write run_summary.md.
        git_hash: Git commit hash for traceability.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Run Summary",
        "",
        f"Run name: {cfg.run_name}",
        f"Timestamp: {datetime.now(tz=timezone.utc).isoformat()}",
        f"Git hash: {git_hash}",
        "",
        "## Model",
        f"- Backbone: {cfg.model.backbone}",
        f"- Pretrained: {cfg.model.pretrained}",
        f"- Num classes: {cfg.model.num_classes}",
        "",
        "## Training",
        f"- Max epochs: {cfg.train.max_epochs}",
        f"- LR: {cfg.train.lr}",
        f"- Optimizer: {cfg.train.optimizer}",
        f"- Scheduler: {cfg.train.scheduler}",
        f"- Loss: {cfg.train.loss}",
        "",
        "## Triage Thresholds",
        f"- t_high: {cfg.triage.t_high}",
        f"- t_low: {cfg.triage.t_low}",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]

    key_metrics = [
        "accuracy", "macro_f1", "urgent_sensitivity", "urgent_specificity",
        "roc_auc", "pr_auc", "ece", "abstain_rate",
        "macro_f1_ci_lower", "macro_f1_ci_upper",
        "urgent_sensitivity_ci_lower", "urgent_sensitivity_ci_upper",
    ]
    for k in key_metrics:
        if k in metrics:
            v = metrics[k]
            if isinstance(v, float):
                lines.append(f"| {k} | {v:.4f} |")
            else:
                lines.append(f"| {k} | {v} |")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved run summary to %s", output_path)
