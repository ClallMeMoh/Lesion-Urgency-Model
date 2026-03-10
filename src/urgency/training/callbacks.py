"""Lightning training callbacks."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import lightning as pl

from urgency.config import AppConfig
from urgency.utils.io import get_git_hash
from urgency.utils.logging import get_logger

logger = get_logger(__name__)


class BestModelCallback(pl.Callback):
    """Track and save the best checkpoint based on validation metrics.

    Primary metric: val/urgent_sensitivity (maximize)
    Secondary metric: val/macro_f1 (maximize, used to break ties)

    Saves best checkpoint to <run_dir>/checkpoints/best.ckpt and also
    copies it to <run_dir>/best_model.ckpt for easy access.
    """

    def __init__(self, run_dir: Path) -> None:
        super().__init__()
        self.run_dir = Path(run_dir)
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._best_sensitivity = -1.0
        self._best_f1 = -1.0
        self._best_epoch = -1

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        metrics = trainer.callback_metrics
        sensitivity = float(metrics.get("val/urgent_sensitivity", -1.0))
        f1 = float(metrics.get("val/macro_f1", -1.0))

        improved = (sensitivity > self._best_sensitivity) or (
            sensitivity == self._best_sensitivity and f1 > self._best_f1
        )

        if improved:
            self._best_sensitivity = sensitivity
            self._best_f1 = f1
            self._best_epoch = trainer.current_epoch
            trainer.save_checkpoint(str(self.ckpt_dir / "best.ckpt"))
            # Copy (not symlink) for Windows compatibility
            best_src = self.ckpt_dir / "best.ckpt"
            best_dst = self.run_dir / "best_model.ckpt"
            shutil.copy2(str(best_src), str(best_dst))
            logger.info(
                "New best model at epoch %d | sensitivity=%.4f | macro_f1=%.4f",
                self._best_epoch,
                self._best_sensitivity,
                self._best_f1,
            )


class MetricsLoggerCallback(pl.Callback):
    """Log per-epoch metrics to a JSONL file alongside TensorBoard."""

    def __init__(self, run_dir: Path) -> None:
        super().__init__()
        self.run_dir = Path(run_dir)
        self.jsonl_path = self.run_dir / "epoch_metrics.jsonl"

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
        metrics["epoch"] = trainer.current_epoch
        metrics["timestamp"] = datetime.now(tz=timezone.utc).isoformat()

        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")


class LeaderboardCallback(pl.Callback):
    """Append a single row to reports/leaderboard.csv at the end of training.

    Reads final val metrics from the Lightning trainer after fit completes.
    """

    COLUMNS = [
        "run_name", "timestamp", "backbone", "macro_f1", "urgent_sensitivity",
        "urgent_specificity", "roc_auc", "ece", "t_high", "t_low",
        "best_epoch", "git_hash", "config_hash",
    ]

    def __init__(self, cfg: AppConfig, run_dir: Path) -> None:
        super().__init__()
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.reports_dir = Path(cfg.reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self._leaderboard_path = self.reports_dir / "leaderboard.csv"

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        metrics = trainer.callback_metrics

        # Read best_epoch from BestModelCallback if present
        best_epoch = -1
        for cb in trainer.callbacks:
            if isinstance(cb, BestModelCallback):
                best_epoch = cb._best_epoch
                break

        row = {
            "run_name": self.cfg.run_name,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "backbone": self.cfg.model.backbone,
            "macro_f1": round(float(metrics.get("val/macro_f1", -1.0)), 4),
            "urgent_sensitivity": round(float(metrics.get("val/urgent_sensitivity", -1.0)), 4),
            "urgent_specificity": round(float(metrics.get("val/urgent_specificity", -1.0)), 4),
            "roc_auc": round(float(metrics.get("val/roc_auc", -1.0)), 4),
            "ece": round(float(metrics.get("val/ece", -1.0)), 4),
            "t_high": self.cfg.triage.t_high,
            "t_low": self.cfg.triage.t_low,
            "best_epoch": best_epoch,
            "git_hash": get_git_hash(),
            "config_hash": _config_hash(self.cfg),
        }

        write_header = not self._leaderboard_path.exists()
        with self._leaderboard_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        logger.info("Leaderboard updated: %s", self._leaderboard_path)


def _config_hash(cfg: AppConfig) -> str:
    """Compute a short hash of the config for traceability."""
    from omegaconf import OmegaConf

    try:
        cfg_str = OmegaConf.to_yaml(cfg)
    except Exception:
        cfg_str = str(cfg)
    return hashlib.md5(cfg_str.encode()).hexdigest()[:8]
