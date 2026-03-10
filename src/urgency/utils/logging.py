"""Logging setup utilities."""

from __future__ import annotations

import logging
from pathlib import Path

from rich.logging import RichHandler


def get_logger(name: str) -> logging.Logger:
    """Return a logger configured with a Rich console handler.

    Format: timestamp | level | name | message
    The root logger is only configured once; subsequent calls return cached loggers.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )
    return logging.getLogger(name)


def setup_tensorboard(log_dir: Path):  # type: ignore[return]
    """Create and return a TensorBoard SummaryWriter.

    Returns None if tensorboard is not installed (graceful degradation).
    """
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(log_dir=str(log_dir))
    except ImportError:
        logger = get_logger(__name__)
        logger.warning("TensorBoard not available; skipping SummaryWriter setup.")
        return None
