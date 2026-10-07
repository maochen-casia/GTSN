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


def output_path(path: str | Path) -> Path:
    """Keep new runs in the project or the user's experiment storage."""
    resolved = Path(path).resolve()
    roots = (PROJECT_ROOT / 'runs', Path('/home/chenmao/GTSN/runs'), EXPERIMENT_ROOT)
    if not any(resolved.is_relative_to(root) and resolved != root for root in roots):
        raise ValueError('Output must be under project runs/ or /run/user/1016/experiments/')
    return resolved


def create_output(path: Path) -> None:
    """Allow a new directory or an empty Docker bind mount; never overwrite a run."""
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f'Output directory is not empty: {path}')
    path.mkdir(parents=True, exist_ok=True)
