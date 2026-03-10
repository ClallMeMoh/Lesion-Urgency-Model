"""Create a submission bundle zip from a training run.

Bundles the best checkpoint, configuration, calibration artifacts, and
evaluation reports into a single zip file for sharing or archiving.

Usage:
    python scripts/make_submission_bundle.py --run_dir runs/<run_id>
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from urgency.utils.logging import get_logger

logger = get_logger(__name__)

BUNDLE_FILES = [
    "checkpoints/best.ckpt",
    "config.yaml",
    "git_hash.txt",
    "temperature.pt",
    "thresholds.json",
]

REPORT_FILES = [
    "metrics.json",
    "leaderboard.csv",
    "run_summary.md",
    "error_analysis.md",
]


def make_bundle(run_dir: Path, reports_dir: Path, output_path: Path | None = None) -> Path:
    """Create a submission bundle zip.

    Args:
        run_dir: Path to a completed training run directory.
        reports_dir: Path to the reports directory.
        output_path: Output zip path. Defaults to <run_dir>_bundle.zip.

    Returns:
        Path to the created zip file.
    """
    run_dir = Path(run_dir)
    reports_dir = Path(reports_dir)

    if output_path is None:
        output_path = run_dir.parent / f"{run_dir.name}_bundle.zip"

    with zipfile.ZipFile(str(output_path), "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Run artifacts
        for rel_path in BUNDLE_FILES:
            src = run_dir / rel_path
            if src.exists():
                zf.write(str(src), arcname=f"run/{rel_path}")
                logger.info("Added: %s", rel_path)
            else:
                logger.warning("Skipped (not found): %s", rel_path)

        # Report files
        for fname in REPORT_FILES:
            src = reports_dir / fname
            if src.exists():
                zf.write(str(src), arcname=f"reports/{fname}")
                logger.info("Added: reports/%s", fname)
            else:
                logger.warning("Skipped (not found): reports/%s", fname)

    logger.info("Bundle saved to: %s", output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a submission bundle zip.")
    parser.add_argument("--run_dir", required=True, type=Path, help="Training run directory.")
    parser.add_argument("--reports_dir", default="reports/", type=Path, help="Reports directory.")
    parser.add_argument("--output", default=None, type=Path, help="Output zip path.")
    args = parser.parse_args()
    make_bundle(args.run_dir, args.reports_dir, args.output)


if __name__ == "__main__":
    main()
