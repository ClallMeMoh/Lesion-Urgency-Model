"""Smoke test: load a minimal checkpoint and run one image through the pipeline.

This test does not require CUDA, pretrained weights, or actual ISIC data.
It verifies that the full inference chain (model -> calibration -> thresholds)
works end-to-end with a randomly initialized model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from urgency.config import AppConfig, DataConfig, ModelConfig, TrainConfig, TriageConfig
from urgency.data.splits import URGENCY_CLASSES
from urgency.inference.infer import DISCLAIMER, UrgencyInferer
from urgency.models.backbones import UrgencyClassifier


def _create_minimal_run(tmp_path: Path) -> Path:
    """Create a minimal run directory with a saved checkpoint and config.

    Uses efficientnet_b0 (smallest variant) with pretrained=False to
    avoid any network requests during testing.
    """
    from omegaconf import OmegaConf

    run_dir = tmp_path / "run_smoke"
    run_dir.mkdir()
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir()

    # Config: smallest backbone, no pretrained, CPU-friendly
    cfg = AppConfig(
        model=ModelConfig(
            backbone="efficientnet_b0",
            pretrained=False,
            dropout=0.0,
            num_classes=3,
        ),
        data=DataConfig(image_size=64),  # tiny images for speed
        train=TrainConfig(precision="32"),
        triage=TriageConfig(t_high=0.6, t_low=0.3),
        run_name="run_smoke",
    )

    # Save config as YAML
    omega_cfg = OmegaConf.structured(cfg)
    with (run_dir / "config.yaml").open("w") as f:
        f.write(OmegaConf.to_yaml(omega_cfg))

    # Instantiate model and save as a Lightning checkpoint
    from urgency.training.trainer import UrgencyLightningModule

    model = UrgencyClassifier(cfg.model)
    pl_module = UrgencyLightningModule(model=model, cfg=cfg, class_weights=None)

    # Save checkpoint using Lightning's save_checkpoint method
    trainer_mock = _minimal_trainer()
    ckpt_path = ckpt_dir / "best.ckpt"
    torch.save(pl_module.state_dict(), str(ckpt_path))

    # Also save as a proper Lightning checkpoint format
    checkpoint = {
        "state_dict": pl_module.state_dict(),
        "hyper_parameters": {"cfg": cfg},
        "pytorch-lightning_version": "2.2.0",
    }
    torch.save(checkpoint, str(ckpt_path))

    # Save thresholds
    with (run_dir / "thresholds.json").open("w") as f:
        json.dump({"t_high": 0.6, "t_low": 0.3}, f)

    return run_dir


def _minimal_trainer():
    """Return a mock object (unused but documents intent)."""
    return None


def _create_random_image(path: Path, size: int = 64) -> None:
    """Save a random RGB JPEG to path."""
    arr = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    Image.fromarray(arr).save(str(path), format="JPEG")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_infer_smoke_from_checkpoint(tmp_path: Path) -> None:
    """Full inference chain works with a randomly initialized model."""
    run_dir = _create_minimal_run(tmp_path)

    # Create a tiny test image
    img_path = tmp_path / "test_image.jpg"
    _create_random_image(img_path, size=64)

    # Load inferer and predict
    inferer = UrgencyInferer(run_dir, device="cpu")
    result = inferer.predict(img_path)

    # Triage decision must be a valid class
    assert result.triage_decision in URGENCY_CLASSES, (
        f"triage_decision '{result.triage_decision}' not in {URGENCY_CLASSES}"
    )

    # Probabilities must be valid
    assert 0.0 <= result.p_urgent <= 1.0, f"p_urgent={result.p_urgent} out of range"
    assert 0.0 <= result.p_monitor <= 1.0
    assert 0.0 <= result.p_uncertain <= 1.0

    # Probabilities should sum to ~1.0
    total = result.p_urgent + result.p_monitor + result.p_uncertain
    assert abs(total - 1.0) < 1e-4, f"Probabilities sum to {total}, expected ~1.0"

    # Disclaimer must mention "research"
    assert "research" in result.disclaimer.lower(), (
        f"Disclaimer does not mention 'research': {result.disclaimer}"
    )


def test_infer_batch_returns_correct_count(tmp_path: Path) -> None:
    """predict_batch returns one result per input image."""
    run_dir = _create_minimal_run(tmp_path)
    img_paths = []
    for i in range(3):
        p = tmp_path / f"img_{i}.jpg"
        _create_random_image(p)
        img_paths.append(p)

    inferer = UrgencyInferer(run_dir, device="cpu")
    results = inferer.predict_batch(img_paths)

    assert len(results) == 3
    for r in results:
        assert r.triage_decision in URGENCY_CLASSES


def test_inferer_uses_default_thresholds_when_missing(tmp_path: Path) -> None:
    """UrgencyInferer falls back to default thresholds when thresholds.json is absent."""
    run_dir = _create_minimal_run(tmp_path)
    # Remove thresholds.json
    thresh_file = run_dir / "thresholds.json"
    if thresh_file.exists():
        thresh_file.unlink()

    inferer = UrgencyInferer(run_dir, device="cpu")
    assert inferer.t_high == 0.6
    assert inferer.t_low == 0.3
