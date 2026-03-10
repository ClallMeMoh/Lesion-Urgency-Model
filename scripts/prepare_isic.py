"""Convert an ISIC metadata export CSV to the standard labels.csv format.

Usage:
    python scripts/prepare_isic.py --input <isic_export.csv> --output data/

Supports both:
  - ISIC archive format: columns include 'diagnosis' (specific disease names)
  - ISIC 2024 challenge format: columns include 'diagnosis_1' (Benign/Malignant/Indeterminate)

Output:
    data/labels.csv with columns:
        image_path, diagnosis, patient_id, age, sex, anatom_site
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from urgency.utils.logging import get_logger

logger = get_logger(__name__)

OUTPUT_COLUMNS = ["image_path", "diagnosis", "patient_id", "age", "sex", "anatom_site"]


def convert_isic_export(input_path: Path, output_dir: Path) -> Path:
    """Convert an ISIC metadata CSV to standard labels.csv.

    Args:
        input_path: Path to the ISIC metadata CSV.
        output_dir: Directory to write labels.csv.

    Returns:
        Path to the created labels.csv file.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Reading ISIC export from %s", input_path)
    df = pd.read_csv(input_path, low_memory=False)

    logger.info("Input columns: %s", list(df.columns))
    logger.info("Input rows: %d", len(df))

    # Determine diagnosis column
    if "diagnosis_1" in df.columns:
        diag_col = "diagnosis_1"
        logger.info("Detected ISIC 2024 challenge format (diagnosis_1 column).")
    elif "diagnosis" in df.columns:
        diag_col = "diagnosis"
        logger.info("Detected ISIC archive format (diagnosis column).")
    else:
        raise ValueError(
            f"Input CSV must have 'diagnosis' or 'diagnosis_1' column. "
            f"Found: {list(df.columns)}"
        )

    if "isic_id" not in df.columns and "image_path" not in df.columns:
        raise ValueError(
            "Input CSV must have either 'isic_id' or 'image_path' column. "
            f"Found: {list(df.columns)}"
        )

    out = pd.DataFrame()

    # image_path: just the filename (resolved relative to images_dir at load time)
    if "image_path" in df.columns:
        out["image_path"] = df["image_path"]
    else:
        out["image_path"] = df["isic_id"].apply(lambda x: f"{x}.jpg")

    # diagnosis: normalize to lowercase, strip whitespace
    out["diagnosis"] = df[diag_col].str.strip().str.lower()

    # Optional columns
    out["patient_id"] = df.get("patient_id", pd.NA)
    out["age"] = df.get("age_approx", pd.NA)
    out["sex"] = df.get("sex", pd.NA)
    out["anatom_site"] = df.get("anatom_site_general", pd.NA)

    # Remove duplicate image paths (keep first occurrence)
    n_before = len(out)
    out = out.drop_duplicates(subset=["image_path"], keep="first").reset_index(drop=True)
    n_after = len(out)
    if n_before != n_after:
        logger.warning("Removed %d duplicate rows (by image_path).", n_before - n_after)

    output_path = output_dir / "labels.csv"
    out.to_csv(output_path, index=False)

    logger.info("Wrote %d rows to %s", len(out), output_path)
    _log_diagnosis_distribution(out)

    return output_path


def _log_diagnosis_distribution(df: pd.DataFrame) -> None:
    """Log the distribution of diagnosis values."""
    dist = df["diagnosis"].value_counts()
    logger.info("Diagnosis distribution:")
    for diag, count in dist.items():
        logger.info("  %-40s %d", diag, count)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ISIC dataset for urgency model.")
    parser.add_argument("--input", required=True, type=Path, help="ISIC metadata CSV path.")
    parser.add_argument("--output", default="data/", type=Path, help="Output directory.")
    args = parser.parse_args()
    convert_isic_export(args.input, args.output)


if __name__ == "__main__":
    main()
