"""Structured configuration dataclasses for the urgency model.

All configuration is defined here as dataclasses and registered with Hydra's
ConfigStore so that YAML overrides are type-checked at composition time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore


@dataclass
class UrgencyMappingConfig:
    """Maps diagnosis strings to urgency categories."""

    urgent: list[str] = field(
        default_factory=lambda: [
            "malignant",
        ]
    )
    monitor: list[str] = field(
        default_factory=lambda: [
            "benign",
        ]
    )
    # "uncertain": assign unmapped diagnoses to the uncertain class
    # "exclude": drop rows with unmapped diagnoses
    unmapped_behavior: str = "uncertain"


@dataclass
class DataConfig:
    """Dataset and dataloader settings."""

    labels_csv: str = "data/labels.csv"
    images_dir: str = "images/isic images"
    splits_dir: str = "data/splits"
    image_size: int = 224
    num_workers: int = 0
    batch_size: int = 32
    pin_memory: bool = False
    urgency_mapping: UrgencyMappingConfig = field(default_factory=UrgencyMappingConfig)


@dataclass
class ModelConfig:
    """Model architecture settings."""

    backbone: str = "efficientnet_b3"
    pretrained: bool = True
    dropout: float = 0.3
    num_classes: int = 3
    use_metadata: bool = False


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    max_epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    loss: str = "cross_entropy"
    label_smoothing: float = 0.1
    class_weights: bool = True
    gradient_clip_val: float = 1.0
    precision: str = "16-mixed"
    seed: int = 42


@dataclass
class TriageConfig:
    """Triage decision thresholds."""

    t_high: float = 0.6
    t_low: float = 0.3
    tune_thresholds: bool = True
    target_sensitivity: float = 0.90


@dataclass
class AppConfig:
    """Top-level application configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    triage: TriageConfig = field(default_factory=TriageConfig)
    run_name: str = "run_001"
    output_dir: str = "runs"
    reports_dir: str = "reports"
    # Internal flag used by smoke config to truncate datasets
    _smoke_: bool = False


def register_configs() -> None:
    """Register all config dataclasses with Hydra ConfigStore."""
    cs = ConfigStore.instance()
    cs.store(name="app_config", node=AppConfig)
    cs.store(group="data", name="isic", node=DataConfig)
    cs.store(group="model", name="efficientnet", node=ModelConfig)
    cs.store(group="train", name="default", node=TrainConfig)
