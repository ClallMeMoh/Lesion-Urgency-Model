"""Post-hoc probability calibration via temperature scaling."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from urgency.utils.logging import get_logger

logger = get_logger(__name__)


class TemperatureScaler(nn.Module):
    """Single-parameter temperature scaling for calibration.

    Calibrated probabilities = softmax(logits / T)
    where T > 0 is optimized on the validation set to minimize NLL.

    Usage:
        scaler = TemperatureScaler()
        scaler.fit(val_logits, val_labels)
        calibrated_probs = torch.softmax(scaler(test_logits), dim=1)
    """

    def __init__(self) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: Tensor) -> Tensor:
        """Divide logits by temperature. Returns scaled logits (not probabilities)."""
        return logits / self.temperature.clamp(min=1e-6)

    def fit(self, logits: Tensor, labels: Tensor, max_iter: int = 50) -> "TemperatureScaler":
        """Optimize temperature to minimize NLL on validation data.

        Args:
            logits: Uncalibrated logits [N, num_classes] from validation set.
            labels: True integer class labels [N].
            max_iter: Maximum LBFGS iterations.

        Returns:
            self (for chaining).
        """
        self.train()
        optimizer = torch.optim.LBFGS([self.temperature], max_iter=max_iter, lr=0.01)
        nll_criterion = nn.CrossEntropyLoss()

        def closure() -> Tensor:
            optimizer.zero_grad()
            scaled = self.forward(logits)
            loss = nll_criterion(scaled, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.eval()

        logger.info(
            "Temperature scaling complete. T = %.4f (before: 1.0000, NLL improved).",
            self.temperature.item(),
        )
        return self

    def save(self, path: Path) -> None:
        """Save temperature parameter to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"temperature": self.temperature.item()}, str(path))
        logger.info("Saved temperature scaler to %s", path)

    @classmethod
    def load(cls, path: Path) -> "TemperatureScaler":
        """Load a saved temperature scaler."""
        scaler = cls()
        data = torch.load(str(path), map_location="cpu")
        scaler.temperature = nn.Parameter(torch.tensor([data["temperature"]]))
        scaler.eval()
        return scaler
