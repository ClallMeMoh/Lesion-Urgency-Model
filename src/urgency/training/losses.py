"""Loss functions for urgency classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from urgency.config import TrainConfig


class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance.

    Reduces the relative loss for well-classified examples, focusing training
    on hard, misclassified examples.

    Args:
        gamma: Focusing parameter. Higher values focus more on hard examples.
        weight: Optional per-class weights of shape [num_classes].
        label_smoothing: Label smoothing factor applied before focal weighting.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)
        self.label_smoothing = label_smoothing

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute focal loss.

        Args:
            logits: Raw model output of shape [B, num_classes].
            targets: Integer class indices of shape [B].

        Returns:
            Scalar loss value.
        """
        # Standard CE with label smoothing
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        # Compute p_t for each sample
        probs = torch.softmax(logits, dim=1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - p_t) ** self.gamma
        return (focal_weight * ce_loss).mean()


def build_loss(cfg: TrainConfig, class_weights: Tensor | None = None) -> nn.Module:
    """Instantiate the loss function based on config.

    Args:
        cfg: Training configuration.
        class_weights: Optional per-class weight tensor from the training split.

    Returns:
        A loss module expecting (logits, targets) inputs.
    """
    weights = class_weights if cfg.class_weights else None

    if cfg.loss == "focal":
        return FocalLoss(
            gamma=2.0,
            weight=weights,
            label_smoothing=cfg.label_smoothing,
        )

    return nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=cfg.label_smoothing,
    )
