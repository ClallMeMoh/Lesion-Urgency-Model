"""Optional multimodal classifier fusing image and metadata features."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from urgency.config import ModelConfig
from urgency.models.backbones import build_backbone

# Embedding dimensions for categorical metadata features
SEX_EMBED_DIM = 8
SITE_EMBED_DIM = 16
METADATA_OUTPUT_DIM = 64

# Known categories (can be extended via config in future iterations)
SEX_CATEGORIES = ["male", "female", "unknown"]
SITE_CATEGORIES = [
    "head/neck", "upper extremity", "lower extremity", "torso",
    "palms/soles", "oral/genital", "unknown",
]


class MetadataEncoder(nn.Module):
    """Simple MLP encoder for tabular metadata features.

    Handles:
        - age: scalar, normalized by dividing by 100
        - sex: categorical embedding
        - anatomic_site: categorical embedding

    All inputs must be pre-encoded as integers (see encode_metadata helper).
    Output: METADATA_OUTPUT_DIM-dimensional embedding.
    """

    def __init__(self) -> None:
        super().__init__()
        self.sex_embed = nn.Embedding(len(SEX_CATEGORIES), SEX_EMBED_DIM, padding_idx=0)
        self.site_embed = nn.Embedding(len(SITE_CATEGORIES), SITE_EMBED_DIM, padding_idx=0)

        input_dim = 1 + SEX_EMBED_DIM + SITE_EMBED_DIM  # age + sex + site
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, METADATA_OUTPUT_DIM),
            nn.ReLU(),
        )

    def forward(self, age: Tensor, sex_idx: Tensor, site_idx: Tensor) -> Tensor:
        """
        Args:
            age: Float tensor [B, 1], normalized age (age / 100).
            sex_idx: Long tensor [B], index into SEX_CATEGORIES.
            site_idx: Long tensor [B], index into SITE_CATEGORIES.

        Returns:
            Embedding tensor [B, METADATA_OUTPUT_DIM].
        """
        sex_emb = self.sex_embed(sex_idx)       # [B, SEX_EMBED_DIM]
        site_emb = self.site_embed(site_idx)    # [B, SITE_EMBED_DIM]
        combined = torch.cat([age, sex_emb, site_emb], dim=1)
        return self.mlp(combined)


class MultimodalUrgencyClassifier(nn.Module):
    """Urgency classifier fusing image backbone features with metadata.

    Architecture:
        image -> backbone -> img_features [B, feature_dim]
        metadata -> MetadataEncoder -> meta_features [B, METADATA_OUTPUT_DIM]
        concat -> Linear(feature_dim + METADATA_OUTPUT_DIM, num_classes)

    Args:
        cfg: Model configuration.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        backbone, feature_dim = build_backbone(cfg)
        self.backbone = backbone
        self.dropout = nn.Dropout(p=cfg.dropout)
        self.metadata_encoder = MetadataEncoder()
        fused_dim = feature_dim + METADATA_OUTPUT_DIM
        self.head = nn.Linear(fused_dim, cfg.num_classes)

    def forward(
        self,
        x: Tensor,
        age: Tensor,
        sex_idx: Tensor,
        site_idx: Tensor,
    ) -> Tensor:
        """Forward pass returning raw logits [B, num_classes]."""
        img_features = self.backbone(x)
        img_features = self.dropout(img_features)
        meta_features = self.metadata_encoder(age, sex_idx, site_idx)
        fused = torch.cat([img_features, meta_features], dim=1)
        return self.head(fused)


def encode_sex(sex: str) -> int:
    """Map sex string to integer index."""
    return SEX_CATEGORIES.index(sex.lower()) if sex.lower() in SEX_CATEGORIES else 0


def encode_site(site: str) -> int:
    """Map anatomic site string to integer index."""
    site_lower = site.lower()
    for i, s in enumerate(SITE_CATEGORIES):
        if s in site_lower or site_lower in s:
            return i
    return SITE_CATEGORIES.index("unknown")
