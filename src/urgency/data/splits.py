"""Patient-level train/val/test split generation.

When patient_id is present, all images from a patient are assigned to the
same split to prevent data leakage. When absent, falls back to stratified
image-level splitting.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from urgency.config import UrgencyMappingConfig
from urgency.utils.logging import get_logger

logger = get_logger(__name__)

URGENCY_CLASSES = ["urgent", "monitor", "uncertain"]
LABEL_TO_INT: dict[str, int] = {c: i for i, c in enumerate(URGENCY_CLASSES)}


def map_diagnosis_to_urgency(
    diagnosis: str,
    mapping: UrgencyMappingConfig,
) -> str | None:
    """Map a diagnosis string to an urgency category.

    Args:
        diagnosis: Raw diagnosis string from labels.csv.
        mapping: Urgency mapping configuration.

    Returns:
        "urgent", "monitor", "uncertain", or None if excluded.
        None is only returned when unmapped_behavior == "exclude".
    """
    normalized = diagnosis.strip().lower()
    if normalized in [d.lower() for d in mapping.urgent]:
        return "urgent"
    if normalized in [d.lower() for d in mapping.monitor]:
        return "monitor"
    # Unmapped
    if mapping.unmapped_behavior == "uncertain":
        return "uncertain"
    return None  # exclude


def apply_urgency_mapping(
    df: pd.DataFrame,
    mapping: UrgencyMappingConfig,
) -> pd.DataFrame:
    """Add 'urgency_label' column and optionally drop excluded rows.

    Logs the diagnosis-to-urgency mapping used and any dropped rows.
    """
    df = df.copy()
    df["urgency_label"] = df["diagnosis"].apply(
        lambda d: map_diagnosis_to_urgency(str(d), mapping)
    )

    excluded = df["urgency_label"].isna()
    n_excluded = excluded.sum()
    if n_excluded > 0:
        logger.info(
            "Excluded %d rows with unmapped diagnoses (unmapped_behavior='exclude').",
            n_excluded,
        )
        df = df[~excluded].copy()

    df["label"] = df["urgency_label"].map(LABEL_TO_INT)
    return df


def make_splits(
    labels_df: pd.DataFrame,
    splits_dir: Path,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Create train/val/test splits from a labeled DataFrame.

    Uses patient-level splitting when 'patient_id' column exists.
    Falls back to stratified image-level splitting otherwise.

    Saves train.csv, val.csv, test.csv to splits_dir.

    Args:
        labels_df: DataFrame with at least image_path, diagnosis, label columns.
        splits_dir: Directory to save split CSV files.
        val_frac: Fraction of data for validation.
        test_frac: Fraction of data for test.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with keys "train", "val", "test".
    """
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    has_patient_id = "patient_id" in labels_df.columns and labels_df["patient_id"].notna().any()

    if has_patient_id:
        splits = _patient_level_split(labels_df, val_frac, test_frac, seed)
        logger.info("Created patient-level splits.")
    else:
        splits = _image_level_split(labels_df, val_frac, test_frac, seed)
        logger.info("Created image-level stratified splits (no patient_id found).")

    for name, df in splits.items():
        df.to_csv(splits_dir / f"{name}.csv", index=False)
        dist = df["urgency_label"].value_counts().to_dict()
        logger.info("Split '%s': %d samples | distribution: %s", name, len(df), dist)

    verify_no_leakage(splits)
    return splits


def _patient_level_split(
    df: pd.DataFrame,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Split patients into train/val/test groups, then assign their images."""
    rng = np.random.default_rng(seed)

    # One row per patient: use majority urgency class for stratification
    patient_df = (
        df.groupby("patient_id")["urgency_label"]
        .agg(lambda x: x.mode().iloc[0])
        .reset_index()
        .rename(columns={"urgency_label": "majority_label"})
    )

    patients = patient_df["patient_id"].values
    strat = patient_df["majority_label"].values

    # Split patients: train vs (val + test)
    train_frac = 1.0 - val_frac - test_frac
    try:
        train_patients, temp_patients, _, temp_strat = train_test_split(
            patients,
            strat,
            test_size=(val_frac + test_frac),
            stratify=strat,
            random_state=int(rng.integers(0, 2**31)),
        )
        val_patients, test_patients = train_test_split(
            temp_patients,
            test_size=test_frac / (val_frac + test_frac),
            stratify=temp_strat,
            random_state=int(rng.integers(0, 2**31)),
        )
    except ValueError:
        # Fallback: unstratified when classes are too rare
        logger.warning(
            "Stratified patient split failed (likely too few samples per class). "
            "Falling back to unstratified patient split."
        )
        train_patients, temp_patients = train_test_split(
            patients,
            test_size=(val_frac + test_frac),
            random_state=int(rng.integers(0, 2**31)),
        )
        val_patients, test_patients = train_test_split(
            temp_patients,
            test_size=test_frac / (val_frac + test_frac),
            random_state=int(rng.integers(0, 2**31)),
        )

    train_set = set(train_patients)
    val_set = set(val_patients)
    test_set = set(test_patients)

    return {
        "train": df[df["patient_id"].isin(train_set)].reset_index(drop=True),
        "val": df[df["patient_id"].isin(val_set)].reset_index(drop=True),
        "test": df[df["patient_id"].isin(test_set)].reset_index(drop=True),
    }


def _image_level_split(
    df: pd.DataFrame,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Stratified image-level split (no patient_id)."""
    rng = np.random.default_rng(seed)
    strat = df["urgency_label"].values

    try:
        train_df, temp_df = train_test_split(
            df,
            test_size=(val_frac + test_frac),
            stratify=strat,
            random_state=int(rng.integers(0, 2**31)),
        )
        val_df, test_df = train_test_split(
            temp_df,
            test_size=test_frac / (val_frac + test_frac),
            stratify=temp_df["urgency_label"].values,
            random_state=int(rng.integers(0, 2**31)),
        )
    except ValueError:
        logger.warning(
            "Stratified image-level split failed. Falling back to unstratified split."
        )
        train_df, temp_df = train_test_split(
            df,
            test_size=(val_frac + test_frac),
            random_state=int(rng.integers(0, 2**31)),
        )
        val_df, test_df = train_test_split(
            temp_df,
            test_size=test_frac / (val_frac + test_frac),
            random_state=int(rng.integers(0, 2**31)),
        )

    return {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


def load_splits(splits_dir: Path) -> dict[str, pd.DataFrame]:
    """Load saved split CSV files from splits_dir.

    Also calls verify_no_leakage as a runtime guard.
    """
    splits_dir = Path(splits_dir)
    splits = {}
    for name in ("train", "val", "test"):
        path = splits_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Split file not found: {path}")
        splits[name] = pd.read_csv(path)

    if "patient_id" in splits["train"].columns:
        verify_no_leakage(splits)

    return splits


def verify_no_leakage(splits: dict[str, pd.DataFrame]) -> None:
    """Assert no patient_id appears in more than one split.

    Args:
        splits: Dictionary with "train", "val", "test" DataFrames.

    Raises:
        ValueError: If any patient_id appears in multiple splits.
    """
    if "patient_id" not in splits.get("train", pd.DataFrame()).columns:
        return  # Nothing to check

    sets: dict[str, set] = {}
    for name, df in splits.items():
        valid_ids = df["patient_id"].dropna()
        sets[name] = set(valid_ids)

    names = list(sets.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            overlap = sets[a] & sets[b]
            if overlap:
                raise ValueError(
                    f"Data leakage detected: {len(overlap)} patient(s) appear in both "
                    f"'{a}' and '{b}' splits. Example IDs: {list(overlap)[:5]}. "
                    "Re-run make-splits to regenerate clean splits."
                )
