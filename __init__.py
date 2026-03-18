"""Storage backends for persisting experiment data."""

from __future__ import annotations

import abc
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ── Abstract Base ────────────────────────────────────────────────────────────

class StorageBackend(abc.ABC):
    """Interface every storage backend must implement."""

    @abc.abstractmethod
    def save_experiment(self, experiment_id: str, data: dict[str, Any]) -> None: ...

    @abc.abstractmethod
    def load_experiment(self, experiment_id: str) -> dict[str, Any] | None: ...

    @abc.abstractmethod
    def update_experiment(self, experiment_id: str, data: dict[str, Any]) -> None: ...

    @abc.abstractmethod
    def list_experiments(self, user_id: str | None = None) -> list[dict[str, Any]]: ...

    @abc.abstractmethod
    def search_experiments(self, **filters: Any) -> list[dict[str, Any]]: ...

    @abc.abstractmethod
    def delete_experiment(self, experiment_id: str) -> bool: ...

    # Metrics (append-only time-series)
    @abc.abstractmethod
    def append_metric(self, experiment_id: str, name: str, value: float,
                      step: int | None = None, timestamp: str | None = None) -> None: ...

    @abc.abstractmethod
    def get_metrics(self, experiment_id: str, name: str | None = None) -> list[dict[str, Any]]: ...

    # Artifacts
    @abc.abstractmethod
    def register_artifact(self, experiment_id: str, filename: str, path: str,
                          artifact_type: str | None = None) -> None: ...

    @abc.abstractmethod
    def list_artifacts(self, experiment_id: str) -> list[dict[str, Any]]: ...

    # Users
    @abc.abstractmethod
    def save_user(self, user_data: dict[str, Any]) -> None: ...

    @abc.abstractmethod
    def load_user(self, user_id: str) -> dict[str, Any] | None: ...

    @abc.abstractmethod
    def load_user_by_username(self, username: str) -> dict[str, Any] | None: ...


# ── SQLite Backend ───────────────────────────────────────────────────────────

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    email TEXT,
    created_at TEXT NOT NULL,
    data_json TEXT
);

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    user_id TEXT,
    name TEXT,
    status TEXT DEFAULT 'running',
    created_at TEXT NOT NULL,
    ended_at TEXT,
    data_json TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    name TEXT NOT NULL,
    value REAL NOT NULL,
    step INTEGER,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);
CREATE INDEX IF NOT EXISTS idx_metrics_exp ON metrics(experiment_id, name);

CREATE TABLE IF NOT EXISTS artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    path TEXT NOT NULL,
    artifact_type TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);
"""


class SQLiteBackend(StorageBackend):
    """SQLite-based persistent storage."""

    def __init__(self, db_path: str | Path):
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.commit()

    # -- helpers --
    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    # -- experiments --
    def save_experiment(self, experiment_id: str, data: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO experiments (experiment_id, user_id, name, status, created_at, data_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (experiment_id, data.get("user_id"), data.get("name"), data.get("status", "running"),
             data.get("created_at", self._now()), json.dumps(data)),
        )
        self._conn.commit()

    def load_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT data_json FROM experiments WHERE experiment_id=?", (experiment_id,)
        ).fetchone()
        return json.loads(row["data_json"]) if row else None

    def update_experiment(self, experiment_id: str, data: dict[str, Any]) -> None:
        self._conn.execute(
            "UPDATE experiments SET data_json=?, status=?, ended_at=?, name=? WHERE experiment_id=?",
            (json.dumps(data), data.get("status", "running"), data.get("ended_at"), data.get("name"),
             experiment_id),
        )
        self._conn.commit()

    def list_experiments(self, user_id: str | None = None) -> list[dict[str, Any]]:
        if user_id:
            rows = self._conn.execute(
                "SELECT data_json FROM experiments WHERE user_id=? ORDER BY created_at DESC",
                (user_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT data_json FROM experiments ORDER BY created_at DESC"
            ).fetchall()
        return [json.loads(r["data_json"]) for r in rows]

    def search_experiments(self, **filters: Any) -> list[dict[str, Any]]:
        all_exps = self.list_experiments(user_id=filters.pop("user_id", None))
        results = []
        for exp in all_exps:
            match = True
            for key, val in filters.items():
                # Support dotted keys like "hyperparameters.optimizer"
                parts = key.split(".")
                obj = exp
                for p in parts:
                    if isinstance(obj, dict):
                        obj = obj.get(p)
                    else:
                        obj = None
                        break
                if obj != val:
                    match = False
                    break
            if match:
                results.append(exp)
        return results

    def delete_experiment(self, experiment_id: str) -> bool:
        cur = self._conn.execute("DELETE FROM experiments WHERE experiment_id=?", (experiment_id,))
        self._conn.execute("DELETE FROM metrics WHERE experiment_id=?", (experiment_id,))
        self._conn.execute("DELETE FROM artifacts WHERE experiment_id=?", (experiment_id,))
        self._conn.commit()
        return cur.rowcount > 0

    # -- metrics --
    def append_metric(self, experiment_id: str, name: str, value: float,
                      step: int | None = None, timestamp: str | None = None) -> None:
        self._conn.execute(
            "INSERT INTO metrics (experiment_id, name, value, step, timestamp) VALUES (?,?,?,?,?)",
            (experiment_id, name, value, step, timestamp or self._now()),
        )
        self._conn.commit()

    def get_metrics(self, experiment_id: str, name: str | None = None) -> list[dict[str, Any]]:
        if name:
            rows = self._conn.execute(
                "SELECT name, value, step, timestamp FROM metrics "
                "WHERE experiment_id=? AND name=? ORDER BY step, timestamp",
                (experiment_id, name),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT name, value, step, timestamp FROM metrics "
                "WHERE experiment_id=? ORDER BY name, step, timestamp",
                (experiment_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # -- artifacts --
    def register_artifact(self, experiment_id: str, filename: str, path: str,
                          artifact_type: str | None = None) -> None:
        self._conn.execute(
            "INSERT INTO artifacts (experiment_id, filename, path, artifact_type, created_at) "
            "VALUES (?,?,?,?,?)",
            (experiment_id, filename, path, artifact_type, self._now()),
        )
        self._conn.commit()

    def list_artifacts(self, experiment_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT filename, path, artifact_type, created_at FROM artifacts WHERE experiment_id=?",
            (experiment_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # -- users --
    def save_user(self, user_data: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO users (user_id, username, email, created_at, data_json) "
            "VALUES (?,?,?,?,?)",
            (user_data["user_id"], user_data["username"], user_data.get("email"),
             user_data.get("created_at", self._now()), json.dumps(user_data)),
        )
        self._conn.commit()

    def load_user(self, user_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT data_json FROM users WHERE user_id=?", (user_id,)
        ).fetchone()
        return json.loads(row["data_json"]) if row else None

    def load_user_by_username(self, username: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT data_json FROM users WHERE username=?", (username,)
        ).fetchone()
        return json.loads(row["data_json"]) if row else None


# ── JSON File Backend ────────────────────────────────────────────────────────

class JSONBackend(StorageBackend):
    """Flat-file JSON storage with folder-per-experiment layout."""

    def __init__(self, base_dir: str | Path):
        self._base = Path(base_dir)
        self._exp_dir = self._base / "experiments"
        self._users_dir = self._base / "users"
        self._exp_dir.mkdir(parents=True, exist_ok=True)
        self._users_dir.mkdir(parents=True, exist_ok=True)

    def _exp_path(self, eid: str) -> Path:
        p = self._exp_dir / eid
        p.mkdir(parents=True, exist_ok=True)
        return p

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    # -- experiments --
    def save_experiment(self, experiment_id: str, data: dict[str, Any]) -> None:
        p = self._exp_path(experiment_id) / "metadata.json"
        p.write_text(json.dumps(data, indent=2, default=str))

    def load_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        p = self._exp_dir / experiment_id / "metadata.json"
        if p.exists():
            return json.loads(p.read_text())
        return None

    def update_experiment(self, experiment_id: str, data: dict[str, Any]) -> None:
        self.save_experiment(experiment_id, data)

    def list_experiments(self, user_id: str | None = None) -> list[dict[str, Any]]:
        results = []
        for d in sorted(self._exp_dir.iterdir(), reverse=True):
            meta = d / "metadata.json"
            if meta.exists():
                exp = json.loads(meta.read_text())
                if user_id is None or exp.get("user_id") == user_id:
                    results.append(exp)
        return results

    def search_experiments(self, **filters: Any) -> list[dict[str, Any]]:
        all_exps = self.list_experiments(user_id=filters.pop("user_id", None))
        results = []
        for exp in all_exps:
            match = True
            for key, val in filters.items():
                parts = key.split(".")
                obj = exp
                for p in parts:
                    if isinstance(obj, dict):
                        obj = obj.get(p)
                    else:
                        obj = None
                        break
                if obj != val:
                    match = False
                    break
            if match:
                results.append(exp)
        return results

    def delete_experiment(self, experiment_id: str) -> bool:
        p = self._exp_dir / experiment_id
        if p.exists():
            shutil.rmtree(p)
            return True
        return False

    # -- metrics --
    def append_metric(self, experiment_id: str, name: str, value: float,
                      step: int | None = None, timestamp: str | None = None) -> None:
        p = self._exp_path(experiment_id) / "metrics.json"
        metrics: list[dict] = json.loads(p.read_text()) if p.exists() else []
        metrics.append({"name": name, "value": value, "step": step,
                        "timestamp": timestamp or self._now()})
        p.write_text(json.dumps(metrics, indent=2))

    def get_metrics(self, experiment_id: str, name: str | None = None) -> list[dict[str, Any]]:
        p = self._exp_dir / experiment_id / "metrics.json"
        if not p.exists():
            return []
        metrics = json.loads(p.read_text())
        if name:
            metrics = [m for m in metrics if m["name"] == name]
        return metrics

    # -- artifacts --
    def register_artifact(self, experiment_id: str, filename: str, path: str,
                          artifact_type: str | None = None) -> None:
        p = self._exp_path(experiment_id) / "artifacts.json"
        arts: list[dict] = json.loads(p.read_text()) if p.exists() else []
        arts.append({"filename": filename, "path": path,
                     "artifact_type": artifact_type, "created_at": self._now()})
        p.write_text(json.dumps(arts, indent=2))

    def list_artifacts(self, experiment_id: str) -> list[dict[str, Any]]:
        p = self._exp_dir / experiment_id / "artifacts.json"
        if p.exists():
            return json.loads(p.read_text())
        return []

    # -- users --
    def save_user(self, user_data: dict[str, Any]) -> None:
        p = self._users_dir / f"{user_data['user_id']}.json"
        p.write_text(json.dumps(user_data, indent=2))

    def load_user(self, user_id: str) -> dict[str, Any] | None:
        p = self._users_dir / f"{user_id}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def load_user_by_username(self, username: str) -> dict[str, Any] | None:
        for f in self._users_dir.glob("*.json"):
            data = json.loads(f.read_text())
            if data.get("username") == username:
                return data
        return None


# ── Factory ──────────────────────────────────────────────────────────────────

def create_backend(config) -> StorageBackend:
    """Instantiate the appropriate backend from a Config object."""
    if config.storage_backend == "sqlite":
        return SQLiteBackend(config.db_path())
    elif config.storage_backend == "json":
        return JSONBackend(config.base_dir)
    elif config.storage_backend == "postgresql":
        raise NotImplementedError(
            "PostgreSQL backend requires asyncpg or psycopg2. "
            "Implement PostgreSQLBackend following the StorageBackend interface."
        )
    else:
        raise ValueError(f"Unknown storage backend: {config.storage_backend}")
