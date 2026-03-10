"""Tests for patient-level and image-level split generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from urgency.data.splits import (
    LABEL_TO_INT,
    load_splits,
    make_splits,
    verify_no_leakage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_patient_df(
    n_patients: int = 50,
    images_per_patient: int = 3,
    seed: int = 0,
) -> pd.DataFrame:
    """Create a synthetic DataFrame with patient_id column."""
    import numpy as np

    rng = np.random.default_rng(seed)
    urgency_labels = ["urgent", "monitor", "uncertain"]
    rows = []
    for i in range(n_patients):
        pid = f"patient_{i:04d}"
        for j in range(images_per_patient):
            label = urgency_labels[rng.integers(0, 3)]
            rows.append(
                {
                    "image_path": f"images/{pid}_{j}.jpg",
                    "diagnosis": "melanoma" if label == "urgent" else "nevus",
                    "patient_id": pid,
                    "urgency_label": label,
                    "label": LABEL_TO_INT[label],
                }
            )
    return pd.DataFrame(rows)


def _make_no_patient_df(n_samples: int = 60, seed: int = 0) -> pd.DataFrame:
    """Create a synthetic DataFrame without patient_id."""
    import numpy as np

    rng = np.random.default_rng(seed)
    urgency_labels = ["urgent", "monitor", "uncertain"]
    rows = []
    for i in range(n_samples):
        label = urgency_labels[rng.integers(0, 3)]
        rows.append(
            {
                "image_path": f"images/img_{i:04d}.jpg",
                "diagnosis": "melanoma" if label == "urgent" else "nevus",
                "urgency_label": label,
                "label": LABEL_TO_INT[label],
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_patient_level_split_no_leakage(tmp_path: Path) -> None:
    """No patient_id should appear in more than one split."""
    df = _make_patient_df(n_patients=50, images_per_patient=3)
    splits = make_splits(df, splits_dir=tmp_path / "splits", seed=42)
    # verify_no_leakage is called inside make_splits; also call explicitly
    verify_no_leakage(splits)

    train_ids = set(splits["train"]["patient_id"].dropna())
    val_ids = set(splits["val"]["patient_id"].dropna())
    test_ids = set(splits["test"]["patient_id"].dropna())

    assert train_ids.isdisjoint(val_ids), "train/val overlap"
    assert train_ids.isdisjoint(test_ids), "train/test overlap"
    assert val_ids.isdisjoint(test_ids), "val/test overlap"


def test_patient_level_split_covers_all_data(tmp_path: Path) -> None:
    """All rows should appear in exactly one split."""
    df = _make_patient_df(n_patients=50, images_per_patient=3)
    splits = make_splits(df, splits_dir=tmp_path / "splits", seed=42)

    total = sum(len(s) for s in splits.values())
    assert total == len(df), f"Expected {len(df)} rows, got {total}"


def test_image_level_split_no_patient_id(tmp_path: Path) -> None:
    """Fallback to image-level split when patient_id is absent."""
    df = _make_no_patient_df(n_samples=60)
    splits = make_splits(df, splits_dir=tmp_path / "splits", seed=42)

    total = sum(len(s) for s in splits.values())
    assert total == len(df)

    # Sizes should be approximately correct
    assert len(splits["train"]) > len(splits["val"])
    assert len(splits["train"]) > len(splits["test"])


def test_split_files_saved_and_loaded(tmp_path: Path) -> None:
    """make_splits saves files; load_splits correctly reloads them."""
    df = _make_patient_df(n_patients=30)
    splits_dir = tmp_path / "splits"
    original = make_splits(df, splits_dir=splits_dir, seed=42)
    loaded = load_splits(splits_dir)

    for name in ("train", "val", "test"):
        assert len(original[name]) == len(loaded[name]), f"Size mismatch for '{name}'"


def test_leakage_detection_raises() -> None:
    """verify_no_leakage should raise ValueError when patients overlap splits."""
    df_train = pd.DataFrame(
        {
            "image_path": ["a.jpg", "b.jpg"],
            "patient_id": ["P001", "P001"],
            "urgency_label": ["urgent", "urgent"],
            "label": [0, 0],
        }
    )
    df_val = pd.DataFrame(
        {
            "image_path": ["c.jpg"],
            "patient_id": ["P001"],  # same patient as train -> leakage
            "urgency_label": ["monitor"],
            "label": [1],
        }
    )
    df_test = pd.DataFrame(
        {
            "image_path": ["d.jpg"],
            "patient_id": ["P002"],
            "urgency_label": ["monitor"],
            "label": [1],
        }
    )
    with pytest.raises(ValueError, match="leakage"):
        verify_no_leakage({"train": df_train, "val": df_val, "test": df_test})


def test_verify_no_leakage_passes_clean_splits() -> None:
    """verify_no_leakage should pass silently for non-overlapping splits."""
    df_train = pd.DataFrame({"patient_id": ["P001", "P002"], "image_path": ["a.jpg", "b.jpg"]})
    df_val = pd.DataFrame({"patient_id": ["P003"], "image_path": ["c.jpg"]})
    df_test = pd.DataFrame({"patient_id": ["P004"], "image_path": ["d.jpg"]})
    # Should not raise
    verify_no_leakage({"train": df_train, "val": df_val, "test": df_test})
