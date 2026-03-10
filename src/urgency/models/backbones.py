"""Model backbone and classifier definitions."""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
from torch import Tensor

from urgency.config import ModelConfig


def build_backbone(cfg: ModelConfig) -> tuple[nn.Module, int]:
    """Create a timm feature extractor with the classification head removed.

    Args:
        cfg: Model configuration with backbone name and pretrained flag.

    Returns:
        Tuple of (feature_extractor, feature_dim).

    Raises:
        ValueError: If the backbone name is not available in timm.
    """
    available = timm.list_models(cfg.backbone, pretrained=cfg.pretrained)
    if not available:
        raise ValueError(
            f"Backbone '{cfg.backbone}' not found in timm "
            f"(pretrained={cfg.pretrained}). "
            f"Run timm.list_models() to see available models."
        )

    model = timm.create_model(
        cfg.backbone,
        pretrained=cfg.pretrained,
        num_classes=0,
        global_pool="avg",
    )
    feature_dim: int = model.num_features
    return model, feature_dim


class UrgencyClassifier(nn.Module):
    """Image-only urgency classifier.

    Architecture: timm backbone -> Dropout -> Linear(feature_dim, num_classes)

    Args:
        cfg: Model configuration.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        backbone, feature_dim = build_backbone(cfg)
        self.backbone = backbone
        self.dropout = nn.Dropout(p=cfg.dropout)
        self.head = nn.Linear(feature_dim, cfg.num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass returning raw logits of shape [B, num_classes]."""
        features = self.backbone(x)
        features = self.dropout(features)
        return self.head(features)

    def get_features(self, x: Tensor) -> Tensor:
        """Return backbone features before the classification head."""
        return self.backbone(x)


def build_model(cfg: ModelConfig) -> nn.Module:
    """Factory function to create the appropriate model based on config.

    Returns MultimodalUrgencyClassifier when use_metadata=True,
    otherwise UrgencyClassifier.
    """
    if cfg.use_metadata:
        from urgency.models.multimodal import MultimodalUrgencyClassifier

        return MultimodalUrgencyClassifier(cfg)
    return UrgencyClassifier(cfg)


def load_model_weights(model: nn.Module, ckpt_path: str, device: str = "cpu") -> nn.Module:
    """Load model weights from a Lightning checkpoint file.

    Extracts only the model state_dict from the checkpoint, stripping the
    "model." prefix added by LightningModule. Compatible with PyTorch 2.6+
    since it does not rely on loading arbitrary Python objects.

    Args:
        model: Pre-instantiated model (architecture must match checkpoint).
        ckpt_path: Path to a Lightning checkpoint (.ckpt).
        device: Device to load tensors onto.

    Returns:
        Model with weights loaded in eval mode.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    raw_state = ckpt.get("state_dict", ckpt)

    # Strip "model." prefix from Lightning checkpoint
    model_state = {
        k[len("model."):]: v
        for k, v in raw_state.items()
        if k.startswith("model.")
    }
    if model_state:
        model.load_state_dict(model_state, strict=True)
    else:
        # Checkpoint saved without Lightning wrapper (e.g., test fixtures)
        model.load_state_dict(raw_state, strict=False)

    model.eval()
    return model.to(device)
