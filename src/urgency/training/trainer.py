"""Lightning module for urgency classification training."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

import lightning as pl

from urgency.config import AppConfig
from urgency.evaluation.metrics import apply_triage, compute_metrics
from urgency.training.losses import build_loss


class UrgencyLightningModule(pl.LightningModule):
    """Lightning module wrapping an urgency classifier.

    Handles:
        - Training and validation step logic
        - Per-epoch metric computation (accuracy, F1, sensitivity, ROC-AUC, etc.)
        - Optimizer and scheduler configuration
        - Accumulation of predictions for epoch-end evaluation

    Args:
        model: Instantiated nn.Module (UrgencyClassifier or Multimodal).
        cfg: Full application configuration.
        class_weights: Optional per-class weight tensor from training split.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: AppConfig,
        class_weights: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.loss_fn = build_loss(cfg.train, class_weights)

        # Accumulate per-batch predictions for epoch-end metric computation
        self._val_preds: list[dict[str, Any]] = []
        self._test_preds: list[dict[str, Any]] = []

        self.save_hyperparameters(ignore=["model"])

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor:
        logits = self.model(batch["image"])
        labels = batch["label"]
        loss = self.loss_fn(logits, labels)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        logits = self.model(batch["image"])
        labels = batch["label"]
        loss = self.loss_fn(logits, labels)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        probs = torch.softmax(logits, dim=1).detach().cpu()
        self._val_preds.append(
            {
                "labels": labels.cpu(),
                "probs": probs,
                "logits": logits.detach().cpu(),
            }
        )

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        metrics = self._compute_epoch_metrics(self._val_preds, prefix="val")
        for k, v in metrics.items():
            self.log(k, v, prog_bar=(k in ("val/macro_f1", "val/urgent_sensitivity")))
        self._val_preds.clear()

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        logits = self.model(batch["image"])
        labels = batch["label"]
        probs = torch.softmax(logits, dim=1).detach().cpu()
        self._test_preds.append(
            {
                "labels": labels.cpu(),
                "probs": probs,
                "logits": logits.detach().cpu(),
                "image_paths": batch.get("image_path", []),
                "diagnoses": batch.get("diagnosis", []),
            }
        )

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return
        metrics = self._compute_epoch_metrics(self._test_preds, prefix="test")
        for k, v in metrics.items():
            self.log(k, v)
        self._test_preds.clear()

    def _compute_epoch_metrics(
        self, preds: list[dict[str, Any]], prefix: str
    ) -> dict[str, float]:
        """Aggregate batch predictions and compute full metric suite."""
        all_labels = torch.cat([p["labels"] for p in preds]).numpy()
        all_probs = torch.cat([p["probs"] for p in preds]).numpy()

        p_urgent = all_probs[:, 0]
        y_pred = np.argmax(all_probs, axis=1)
        triage = apply_triage(p_urgent, self.cfg.triage.t_high, self.cfg.triage.t_low)

        # Use fewer bootstrap samples during training for speed
        results = compute_metrics(
            y_true=all_labels,
            y_pred=y_pred,
            p_urgent=p_urgent,
            triage_decisions=triage,
            n_bootstrap=0,  # no bootstrap during training
        )

        return {
            f"{prefix}/loss": float(
                torch.cat([p["logits"] for p in preds]).mean()
            ),  # rough proxy; actual loss already logged per step
            f"{prefix}/accuracy": results["accuracy"],
            f"{prefix}/macro_f1": results["macro_f1"],
            f"{prefix}/urgent_sensitivity": results["urgent_sensitivity"],
            f"{prefix}/urgent_specificity": results["urgent_specificity"],
            f"{prefix}/roc_auc": results["roc_auc"],
            f"{prefix}/ece": results["ece"],
            f"{prefix}/abstain_rate": results["abstain_rate"],
        }

    def configure_optimizers(self):  # type: ignore[override]
        """Configure optimizer and LR scheduler based on config."""
        params = self.model.parameters()
        train_cfg = self.cfg.train

        if train_cfg.optimizer == "adamw":
            optimizer = AdamW(
                params,
                lr=train_cfg.lr,
                weight_decay=train_cfg.weight_decay,
            )
        else:
            optimizer = SGD(
                params,
                lr=train_cfg.lr,
                weight_decay=train_cfg.weight_decay,
                momentum=0.9,
            )

        if train_cfg.scheduler == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=train_cfg.max_epochs,
                eta_min=train_cfg.lr * 1e-3,
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}

        if train_cfg.scheduler == "onecycle":
            # Requires total_steps; estimated from trainer steps_per_epoch
            total_steps = self.trainer.estimated_stepping_batches
            scheduler = OneCycleLR(
                optimizer,
                max_lr=train_cfg.lr,
                total_steps=total_steps,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }

        return optimizer
