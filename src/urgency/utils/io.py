"""File I/O utilities."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def save_json(data: dict[str, Any], path: Path) -> None:
    """Write data to JSON with pretty printing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: Path) -> dict[str, Any]:
    """Load JSON file."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_config(cfg: DictConfig, path: Path) -> None:
    """Serialize an OmegaConf config to YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg))


def get_git_hash() -> str:
    """Return the current git commit hash, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it does not exist. Return path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_app_config(config_path: Path) -> "AppConfig":
    """Load a saved config YAML and return a typed AppConfig dataclass.

    Merges the loaded YAML with the structured AppConfig schema so that
    OmegaConf.to_object() returns a proper dataclass rather than a plain dict.

    Args:
        config_path: Path to config.yaml saved during training.

    Returns:
        AppConfig dataclass instance.
    """
    from urgency.config import AppConfig

    loaded = OmegaConf.load(str(Path(config_path)))
    schema = OmegaConf.structured(AppConfig)
    merged = OmegaConf.merge(schema, loaded)
    result = OmegaConf.to_object(merged)
    assert isinstance(result, AppConfig), f"Expected AppConfig, got {type(result)}"
    return result
