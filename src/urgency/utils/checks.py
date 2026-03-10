"""Precondition checks for data paths, run directories, and environment."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from urgency.utils.logging import get_logger

logger = get_logger(__name__)

REQUIRED_LABEL_COLUMNS = {"image_path", "diagnosis"}


def check_labels_csv(path: Path) -> None:
    """Validate that labels.csv exists and has required columns.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if required columns are missing or contain NaNs.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"labels.csv not found: {path}")

    df = pd.read_csv(path, nrows=5)
    missing = REQUIRED_LABEL_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"labels.csv is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # Full read to check for NaNs in required columns
    df = pd.read_csv(path)
    for col in REQUIRED_LABEL_COLUMNS:
        n_null = df[col].isna().sum()
        if n_null > 0:
            raise ValueError(
                f"Column '{col}' has {n_null} NaN values in {path}. "
                "All rows must have valid image_path and diagnosis."
            )


def check_splits_exist(splits_dir: Path) -> None:
    """Assert that all three split CSV files exist.

    Raises:
        FileNotFoundError: if any split file is missing.
    """
    splits_dir = Path(splits_dir)
    for name in ("train.csv", "val.csv", "test.csv"):
        p = splits_dir / name
        if not p.exists():
            raise FileNotFoundError(
                f"Split file not found: {p}. "
                "Run: python -m urgency.cli make-splits first."
            )


def check_run_dir(run_dir: Path) -> None:
    """Validate that a run directory has the expected artifacts.

    Raises:
        FileNotFoundError: if required files are missing.
    """
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    config_file = run_dir / "config.yaml"
    if not config_file.exists():
        raise FileNotFoundError(f"config.yaml not found in {run_dir}")

    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists() or not any(ckpt_dir.glob("*.ckpt")):
        raise FileNotFoundError(
            f"No checkpoints found in {ckpt_dir}. "
            "Run training first or verify the run_dir path."
        )


def check_cuda_available() -> None:
    """Log a warning if CUDA is not available (does not raise)."""
    import torch

    if not torch.cuda.is_available():
        logger.warning(
            "CUDA is not available. Training will run on CPU, which is slow. "
            "Consider using a machine with a GPU."
        )
