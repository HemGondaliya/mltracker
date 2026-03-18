"""Reproducibility helpers – snapshot code, restore seeds, and re-run."""

from __future__ import annotations

import json
import os
import random
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mltracker.config import Config, get_config


def set_all_seeds(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch (if available)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np  # type: ignore
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except (ImportError, AttributeError):
        pass

    try:
        import tensorflow as tf  # type: ignore
        tf.random.set_seed(seed)
    except (ImportError, AttributeError):
        pass


def snapshot_code(
    experiment_id: str,
    source_dir: str = ".",
    *,
    config: Config | None = None,
    exclude: list[str] | None = None,
) -> str:
    """Create a tar.gz snapshot of the source directory for reproducibility.

    Returns the path to the created archive.
    """
    cfg = config or get_config()
    dest_dir = cfg.artifacts_dir(experiment_id)
    archive_path = dest_dir / "code_snapshot.tar.gz"

    _exclude = set(exclude or [
        "__pycache__", ".git", "node_modules", ".venv", "venv",
        "*.pyc", ".eggs", "*.egg-info", "dist", "build",
    ])

    def _filter(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
        name = Path(tarinfo.name).name
        for pat in _exclude:
            if pat.startswith("*"):
                if name.endswith(pat[1:]):
                    return None
            elif name == pat:
                return None
        return tarinfo

    with tarfile.open(str(archive_path), "w:gz") as tar:
        tar.add(source_dir, arcname="code", filter=_filter)

    return str(archive_path)


def export_reproducibility_config(
    experiment_data: dict[str, Any],
    output_path: str | None = None,
) -> dict[str, Any]:
    """Export everything needed to reproduce an experiment as a JSON config."""
    repro = {
        "experiment_id": experiment_data.get("experiment_id"),
        "name": experiment_data.get("name"),
        "hyperparameters": experiment_data.get("hyperparameters", {}),
        "dataset": experiment_data.get("dataset", {}),
        "model": experiment_data.get("model", {}),
        "environment": {
            "python_version": experiment_data.get("environment", {}).get("python_version"),
            "packages": experiment_data.get("environment", {}).get("packages", []),
        },
        "git": experiment_data.get("git", {}),
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }
    if output_path:
        Path(output_path).write_text(json.dumps(repro, indent=2))
    return repro
