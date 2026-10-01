"""JSON configuration and paths shared by command-line entry points."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path("/run/user/1016/experiments")


def read_json(path: str | Path) -> Any:
    """Read a UTF-8 JSON document without modifying its source."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    """Atomically write strict JSON; reject NaN/Infinity in experiment results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def experiment_path(path: str | Path) -> Path:
    """Require writable experiment artifacts to remain under the requested root."""
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(EXPERIMENT_ROOT.resolve()):
        raise ValueError(f"Experiment output must be under {EXPERIMENT_ROOT}: {resolved}")
    return resolved


def default_config(kind: str, filename: str) -> Path:
    return PROJECT_ROOT / "configs" / kind / filename

