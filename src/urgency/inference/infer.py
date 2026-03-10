"""Single-image and batch inference pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

from urgency.config import AppConfig
from urgency.data.splits import URGENCY_CLASSES
from urgency.data.transforms import get_val_transforms
from urgency.evaluation.calibration import TemperatureScaler
from urgency.evaluation.metrics import apply_triage
from urgency.models.backbones import build_model
from urgency.utils.logging import get_logger

logger = get_logger(__name__)

DISCLAIMER = (
    "This is a research tool, not a diagnostic device. "
    "Results must not be used as a substitute for professional medical advice. "
    "Always consult a qualified dermatologist for any skin concerns."
)

DEFAULT_THRESHOLDS = {"t_high": 0.6, "t_low": 0.3}


@dataclass
class InferenceResult:
    """Output of a single-image urgency inference."""

    image_path: str
    p_urgent: float
    p_monitor: float
    p_uncertain: float
    triage_decision: str
    confidence: float
    disclaimer: str = field(default=DISCLAIMER)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_path": self.image_path,
            "p_urgent": round(self.p_urgent, 4),
            "p_monitor": round(self.p_monitor, 4),
            "p_uncertain": round(self.p_uncertain, 4),
            "triage_decision": self.triage_decision,
            "confidence": round(self.confidence, 4),
            "disclaimer": self.disclaimer,
        }


class UrgencyInferer:
    """Load a trained run and perform urgency inference on images.

    Initialization sequence:
        1. Load config from run_dir/config.yaml
        2. Load checkpoint from run_dir/checkpoints/best.ckpt
        3. Load TemperatureScaler from run_dir/temperature.pt (T=1 if absent)
        4. Load thresholds from run_dir/thresholds.json (defaults if absent)

    Args:
        run_dir: Path to a completed training run directory.
        device: Torch device string. Defaults to "cuda" if available, else "cpu".
    """

    def __init__(self, run_dir: Path, device: str | None = None) -> None:
        self.run_dir = Path(run_dir)
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.cfg = self._load_config()
        self.model = self._load_model()
        self.scaler = self._load_scaler()
        self.t_high, self.t_low = self._load_thresholds()
        self.transform = get_val_transforms(self.cfg.data.image_size)

        logger.info(
            "UrgencyInferer ready | device=%s | t_high=%.2f | t_low=%.2f",
            self.device,
            self.t_high,
            self.t_low,
        )

    def _load_config(self) -> AppConfig:
        """Load OmegaConf config and return as AppConfig dataclass."""
        from urgency.utils.io import load_app_config

        config_path = self.run_dir / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"config.yaml not found in {self.run_dir}")
        return load_app_config(config_path)

    def _load_model(self) -> torch.nn.Module:
        """Load model weights from best checkpoint.

        Loads only state_dict (weights_only=True compatible) and rebuilds the
        model from config. This avoids PyTorch 2.6's restriction on loading
        arbitrary Python objects from checkpoints.
        """
        ckpt_path = self.run_dir / "checkpoints" / "best.ckpt"
        if not ckpt_path.exists():
            ckpt_path = self.run_dir / "checkpoints" / "last.ckpt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"No checkpoint found in {self.run_dir / 'checkpoints'}")

        model = build_model(self.cfg.model)

        # Lightning checkpoints store model weights under "state_dict" with
        # a "model." prefix. Load with weights_only=False because the checkpoint
        # may contain optimizer state, but we only use the model weights.
        ckpt = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)

        # Strip the "model." prefix added by LightningModule
        model_state = {
            k[len("model."):]: v
            for k, v in state_dict.items()
            if k.startswith("model.")
        }
        if model_state:
            model.load_state_dict(model_state, strict=True)
        else:
            # Fallback: checkpoint saved without Lightning wrapper (test fixtures)
            model.load_state_dict(state_dict, strict=False)

        model.eval()
        return model.to(self.device)

    def _load_scaler(self) -> TemperatureScaler | None:
        """Load temperature scaler if it exists."""
        temp_path = self.run_dir / "temperature.pt"
        if temp_path.exists():
            scaler = TemperatureScaler.load(temp_path)
            return scaler.to(self.device)
        logger.debug("No temperature.pt found; using T=1.0 (uncalibrated).")
        return None

    def _load_thresholds(self) -> tuple[float, float]:
        """Load triage thresholds from thresholds.json."""
        thresh_path = self.run_dir / "thresholds.json"
        if thresh_path.exists():
            with thresh_path.open() as f:
                data = json.load(f)
            return float(data.get("t_high", DEFAULT_THRESHOLDS["t_high"])), float(
                data.get("t_low", DEFAULT_THRESHOLDS["t_low"])
            )
        logger.debug("No thresholds.json found; using defaults.")
        return DEFAULT_THRESHOLDS["t_high"], DEFAULT_THRESHOLDS["t_low"]

    @torch.no_grad()
    def predict(self, image_path: Path) -> InferenceResult:
        """Run inference on a single image.

        Args:
            image_path: Path to the input image file.

        Returns:
            InferenceResult with probabilities and triage decision.
        """
        image_path = Path(image_path)
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        logits = self.model(tensor)

        if self.scaler is not None:
            logits = self.scaler(logits)

        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        p_urgent = float(probs[0])
        p_monitor = float(probs[1])
        p_uncertain = float(probs[2]) if len(probs) > 2 else 0.0

        import numpy as np

        decision_idx = int(apply_triage(np.array([p_urgent]), self.t_high, self.t_low)[0])
        triage_decision = URGENCY_CLASSES[decision_idx]
        confidence = float(max(probs))

        return InferenceResult(
            image_path=str(image_path),
            p_urgent=p_urgent,
            p_monitor=p_monitor,
            p_uncertain=p_uncertain,
            triage_decision=triage_decision,
            confidence=confidence,
        )

    def predict_batch(self, image_paths: list[Path]) -> list[InferenceResult]:
        """Run inference on a list of images sequentially."""
        return [self.predict(p) for p in image_paths]
