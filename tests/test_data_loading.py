"""Tests for ISICDataset and validation logic."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from urgency.config import DataConfig, UrgencyMappingConfig
from urgency.data.dataset import ISICDataset, ValidationReport
from urgency.data.splits import apply_urgency_mapping


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MAPPING = UrgencyMappingConfig(
    urgent=["melanoma"],
    monitor=["nevus"],
    unmapped_behavior="uncertain",
)

LABEL_MAP = {"melanoma": "urgent", "nevus": "monitor"}


def _make_dummy_image(path: Path, size: tuple[int, int] = (64, 64)) -> None:
    """Create a tiny RGB JPEG at path."""
    arr = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path, format="JPEG")


def _make_dataset(
    tmp_path: Path,
    n_urgent: int = 4,
    n_monitor: int = 4,
    n_uncertain: int = 2,
    add_missing: bool = False,
    add_invalid_label: bool = False,
    add_duplicate: bool = False,
) -> tuple[pd.DataFrame, Path]:
    """Create a synthetic dataset and return (DataFrame, images_dir)."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    rows = []
    for i in range(n_urgent):
        fname = f"urgent_{i}.jpg"
        _make_dummy_image(images_dir / fname)
        rows.append({"image_path": fname, "diagnosis": "melanoma", "urgency_label": "urgent", "label": 0})

    for i in range(n_monitor):
        fname = f"monitor_{i}.jpg"
        _make_dummy_image(images_dir / fname)
        rows.append({"image_path": fname, "diagnosis": "nevus", "urgency_label": "monitor", "label": 1})

    for i in range(n_uncertain):
        fname = f"uncertain_{i}.jpg"
        _make_dummy_image(images_dir / fname)
        rows.append({"image_path": fname, "diagnosis": "unknown", "urgency_label": "uncertain", "label": 2})

    if add_missing:
        rows.append({"image_path": "does_not_exist.jpg", "diagnosis": "melanoma", "urgency_label": "urgent", "label": 0})

    if add_invalid_label:
        fname = "bad_label.jpg"
        _make_dummy_image(images_dir / fname)
        rows.append({"image_path": fname, "diagnosis": "melanoma", "urgency_label": "bad_class", "label": 0})

    if add_duplicate:
        rows.append(rows[0].copy())  # duplicate the first row

    df = pd.DataFrame(rows)
    return df, images_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_dataset_loads_without_error(tmp_path: Path) -> None:
    """A valid dataset should load and iterate without errors."""
    df, images_dir = _make_dataset(tmp_path)
    ds = ISICDataset(df, images_dir, transform=None, validate=True)
    assert len(ds) == 10

    item = ds[0]
    assert "image" in item
    assert "label" in item
    assert item["label"] in (0, 1, 2)


def test_missing_file_detected(tmp_path: Path) -> None:
    """ValidationReport should flag files that do not exist on disk."""
    df, images_dir = _make_dataset(tmp_path, add_missing=True)
    ds = ISICDataset(df, images_dir, validate=False)
    report = ds.validate_dataset()

    assert len(report.missing_files) == 1
    assert "does_not_exist.jpg" in report.missing_files[0]
    assert report.is_valid is False


def test_invalid_label_detected(tmp_path: Path) -> None:
    """ValidationReport should flag labels outside the expected set."""
    df, images_dir = _make_dataset(tmp_path, add_invalid_label=True)
    ds = ISICDataset(df, images_dir, validate=False)
    report = ds.validate_dataset()

    assert len(report.invalid_labels) == 1
    assert "bad_class" in report.invalid_labels[0]
    assert report.is_valid is False


def test_duplicate_path_detected(tmp_path: Path) -> None:
    """ValidationReport should flag duplicate image_path entries."""
    df, images_dir = _make_dataset(tmp_path, add_duplicate=True)
    ds = ISICDataset(df, images_dir, validate=False)
    report = ds.validate_dataset()

    assert len(report.duplicate_paths) >= 1
    assert report.is_valid is False


def test_class_distribution_reported(tmp_path: Path) -> None:
    """ValidationReport should contain the correct class counts."""
    df, images_dir = _make_dataset(tmp_path, n_urgent=3, n_monitor=5, n_uncertain=2)
    ds = ISICDataset(df, images_dir, validate=False)
    report = ds.validate_dataset()

    assert report.class_distribution.get("urgent") == 3
    assert report.class_distribution.get("monitor") == 5
    assert report.class_distribution.get("uncertain") == 2


def test_valid_dataset_passes(tmp_path: Path) -> None:
    """A clean dataset should produce a valid ValidationReport."""
    df, images_dir = _make_dataset(tmp_path)
    ds = ISICDataset(df, images_dir, validate=False)
    report = ds.validate_dataset()

    assert report.is_valid is True
    assert report.missing_files == []
    assert report.invalid_labels == []
    assert report.duplicate_paths == []


def test_apply_urgency_mapping_excludes_unmapped(tmp_path: Path) -> None:
    """Rows with unmapped diagnoses should be dropped when behavior is 'exclude'."""
    mapping = UrgencyMappingConfig(urgent=["melanoma"], monitor=["nevus"], unmapped_behavior="exclude")
    df = pd.DataFrame(
        [
            {"image_path": "a.jpg", "diagnosis": "melanoma"},
            {"image_path": "b.jpg", "diagnosis": "nevus"},
            {"image_path": "c.jpg", "diagnosis": "totally_unknown"},
        ]
    )
    result = apply_urgency_mapping(df, mapping)
    assert len(result) == 2
    assert "totally_unknown" not in result["diagnosis"].values


def test_apply_urgency_mapping_assigns_uncertain(tmp_path: Path) -> None:
    """Rows with unmapped diagnoses should be assigned 'uncertain' by default."""
    mapping = UrgencyMappingConfig(urgent=["melanoma"], monitor=["nevus"], unmapped_behavior="uncertain")
    df = pd.DataFrame(
        [
            {"image_path": "a.jpg", "diagnosis": "unknown_lesion"},
        ]
    )
    result = apply_urgency_mapping(df, mapping)
    assert result.iloc[0]["urgency_label"] == "uncertain"
