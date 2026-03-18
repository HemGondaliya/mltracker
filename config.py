"""Global configuration for mltracker."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class Config:
    """Library-wide configuration.

    Attributes:
        base_dir: Root directory for all experiment data.
        storage_backend: One of 'sqlite', 'json', or 'postgresql'.
        db_url: Connection URL for PostgreSQL (ignored for other backends).
        auto_log_env: Capture environment info automatically on experiment start.
        auto_log_git: Capture git commit hash automatically on experiment start.
        default_user: Username used when no explicit login is performed.
    """

    base_dir: str = field(default_factory=lambda: os.environ.get(
        "MLTRACKER_DIR", str(Path.home() / ".mltracker")
    ))
    storage_backend: Literal["sqlite", "json", "postgresql"] = "sqlite"
    db_url: str | None = None
    auto_log_env: bool = True
    auto_log_git: bool = True
    default_user: str = "default"

    def experiments_dir(self) -> Path:
        p = Path(self.base_dir) / "experiments"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def artifacts_dir(self, experiment_id: str) -> Path:
        p = Path(self.base_dir) / "experiments" / experiment_id / "artifacts"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def db_path(self) -> Path:
        p = Path(self.base_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p / "mltracker.db"


# Global singleton – importable from anywhere.
_global_config = Config()


def get_config() -> Config:
    return _global_config


def set_config(cfg: Config) -> None:
    global _global_config
    _global_config = cfg
