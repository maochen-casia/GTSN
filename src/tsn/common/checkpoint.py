"""Portable state-dict checkpoints; no serialized policy objects."""

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Write a complete checkpoint atomically to prevent interrupted partial files."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load tensor/primitive-only checkpoints onto CPU before constructing a model."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError(f"Unsupported checkpoint format: {path}")
    return payload

