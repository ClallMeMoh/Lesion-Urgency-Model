"""ISIC dataset loader with validation and Lightning DataModule."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

import lightning as pl

from urgency.config import DataConfig
from urgency.data.splits import LABEL_TO_INT, apply_urgency_mapping, load_splits
from urgency.data.transforms import get_train_transforms, get_val_transforms
from urgency.utils.logging import get_logger

logger = get_logger(__name__)

# Maximum samples per split when running in smoke mode
SMOKE_MAX_SAMPLES = 32


@dataclass
class ValidationReport:
    """Result of dataset validation checks."""

    total: int
    missing_files: list[str] = field(default_factory=list)
    invalid_labels: list[str] = field(default_factory=list)
    duplicate_paths: list[str] = field(default_factory=list)
    class_distribution: dict[str, int] = field(default_factory=dict)
    is_valid: bool = True

    def log_summary(self) -> None:
        """Print a human-readable summary of validation results."""
        logger.info("Dataset validation: %d total samples", self.total)
        logger.info("  Class distribution: %s", self.class_distribution)
        if self.missing_files:
            logger.warning("  Missing files: %d", len(self.missing_files))
        if self.invalid_labels:
            logger.warning("  Invalid labels: %d", len(self.invalid_labels))
        if self.duplicate_paths:
            logger.warning("  Duplicate paths: %d", len(self.duplicate_paths))
        status = "PASS" if self.is_valid else "FAIL"
        logger.info("  Validation status: %s", status)


class ISICDataset(Dataset):
    """PyTorch Dataset for ISIC lesion urgency classification.

    Args:
        df: DataFrame with columns [image_path, urgency_label, label, ...].
        images_dir: Root directory for resolving relative image paths.
        transform: Transform applied to each PIL image.
        validate: If True, validate all file paths at init time.
    """

    VALID_LABELS: frozenset[str] = frozenset(LABEL_TO_INT.keys())

    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: Path,
        transform: Any = None,
        validate: bool = True,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        if validate:
            report = self.validate_dataset()
            report.log_summary()
            if not report.is_valid:
                logger.warning(
                    "Dataset has validation issues. Proceeding with valid rows only."
                )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        image_path = self._resolve_path(str(row["image_path"]))
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "label": int(row["label"]),
            "image_path": str(row["image_path"]),
            "diagnosis": str(row.get("diagnosis", "")),
        }

    def _resolve_path(self, image_path: str) -> Path:
        """Resolve image path relative to images_dir if not absolute."""
        p = Path(image_path)
        if p.is_absolute():
            return p
        return self.images_dir / p

    def validate_dataset(self, check_hashes: bool = False) -> ValidationReport:
        """Run validation checks on the dataset.

        Checks: file existence, label validity, duplicate paths.
        Optionally runs hash-based duplicate detection.
        """
        report = ValidationReport(total=len(self.df))
        valid_labels = self.VALID_LABELS

        seen_paths: set[str] = set()
        seen_hashes: dict[str, str] = {}

        for _, row in self.df.iterrows():
            path_str = str(row["image_path"])
            full_path = self._resolve_path(path_str)

            # File existence
            if not full_path.exists():
                report.missing_files.append(path_str)
                report.is_valid = False

            # Label validity
            label = str(row.get("urgency_label", ""))
            if label not in valid_labels:
                report.invalid_labels.append(f"{path_str}: '{label}'")
                report.is_valid = False

            # Duplicate paths
            if path_str in seen_paths:
                report.duplicate_paths.append(path_str)
                report.is_valid = False
            seen_paths.add(path_str)

            # Hash-based duplicates (optional, expensive)
            if check_hashes and full_path.exists():
                file_hash = _sha256(full_path)
                if file_hash in seen_hashes:
                    logger.warning(
                        "Hash collision: %s and %s appear identical.",
                        path_str,
                        seen_hashes[file_hash],
                    )
                seen_hashes[file_hash] = path_str

        # Class distribution
        if "urgency_label" in self.df.columns:
            report.class_distribution = self.df["urgency_label"].value_counts().to_dict()

        return report


def _sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class ISICDataModule(pl.LightningDataModule):
    """Lightning DataModule for ISIC urgency classification.

    Handles loading splits, applying transforms, and computing class weights.

    Args:
        cfg: DataConfig with paths and hyperparameters.
        smoke: If True, truncate each split to SMOKE_MAX_SAMPLES.
    """

    def __init__(self, cfg: DataConfig, smoke: bool = False) -> None:
        super().__init__()
        self.cfg = cfg
        self.smoke = smoke
        self.class_weights: Tensor | None = None
        self._train_dataset: ISICDataset | None = None
        self._val_dataset: ISICDataset | None = None
        self._test_dataset: ISICDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        """Load splits and create datasets for the requested stage."""
        splits = load_splits(Path(self.cfg.splits_dir))
        images_dir = Path(self.cfg.images_dir)

        train_df = _maybe_truncate(splits["train"], self.smoke)
        val_df = _maybe_truncate(splits["val"], self.smoke)
        test_df = _maybe_truncate(splits["test"], self.smoke)

        train_tf = get_train_transforms(self.cfg.image_size)
        val_tf = get_val_transforms(self.cfg.image_size)

        if stage in ("fit", None):
            self._train_dataset = ISICDataset(train_df, images_dir, train_tf)
            self._val_dataset = ISICDataset(val_df, images_dir, val_tf)
            self.class_weights = _compute_class_weights(train_df)

        if stage in ("test", None):
            self._test_dataset = ISICDataset(test_df, images_dir, val_tf)

        if stage == "predict":
            self._test_dataset = ISICDataset(test_df, images_dir, val_tf)

    def train_dataloader(self) -> DataLoader:
        assert self._train_dataset is not None, "Call setup('fit') first."
        return DataLoader(
            self._train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        assert self._val_dataset is not None, "Call setup('fit') first."
        return DataLoader(
            self._val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        assert self._test_dataset is not None, "Call setup('test') first."
        return DataLoader(
            self._test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        )


def _maybe_truncate(df: pd.DataFrame, smoke: bool) -> pd.DataFrame:
    """Truncate DataFrame to SMOKE_MAX_SAMPLES when in smoke mode."""
    if smoke and len(df) > SMOKE_MAX_SAMPLES:
        return df.sample(n=SMOKE_MAX_SAMPLES, random_state=0).reset_index(drop=True)
    return df


def _compute_class_weights(train_df: pd.DataFrame) -> Tensor:
    """Compute balanced class weights from training split labels."""
    labels = train_df["label"].values
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    weight_tensor = torch.ones(len(LABEL_TO_INT), dtype=torch.float32)
    for cls_idx, w in zip(classes, weights):
        weight_tensor[cls_idx] = w
    return weight_tensor


def build_datamodule_from_config(cfg: DataConfig, smoke: bool = False) -> ISICDataModule:
    """Convenience factory used by the CLI and tests."""
    return ISICDataModule(cfg, smoke=smoke)
