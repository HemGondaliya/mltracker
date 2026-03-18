"""Core Experiment class – the primary user-facing API."""

from __future__ import annotations

import json
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mltracker.config import Config, get_config
from mltracker.storage import StorageBackend, create_backend
from mltracker.utils import capture_full_environment, get_git_info


class Experiment:
    """Represents a single ML experiment.

    Usage::

        exp = Experiment(name="ResNet50_CIFAR10", dataset="CIFAR10", model="ResNet50")
        exp.log_hyperparameters({"learning_rate": 0.001})
        for epoch in range(50):
            exp.log_metric("train_loss", loss, step=epoch)
        exp.save_artifact("model.pt")
        exp.end()
    """

    def __init__(
        self,
        name: str,
        dataset: str | None = None,
        model: str | None = None,
        *,
        tags: list[str] | None = None,
        description: str | None = None,
        config: Config | None = None,
        user_id: str | None = None,
        backend: StorageBackend | None = None,
    ):
        self._config = config or get_config()
        self._backend = backend or create_backend(self._config)

        self.experiment_id: str = self._generate_id()
        self.name: str = name
        self.status: str = "running"
        self.tags: list[str] = tags or []
        self.description: str = description or ""
        self.user_id: str = user_id or self._config.default_user

        # Timestamps
        self.created_at: str = datetime.now(timezone.utc).isoformat()
        self.ended_at: str | None = None
        self._start_wall: float = time.time()

        # Sections
        self._dataset: dict[str, Any] = {}
        self._model: dict[str, Any] = {}
        self._hyperparameters: dict[str, Any] = {}
        self._test_results: dict[str, Any] = {}
        self._environment: dict[str, Any] = {}
        self._git: dict[str, str | None] = {}
        self._notes: list[str] = []

        # Pre-fill convenience fields
        if dataset:
            self._dataset["name"] = dataset
        if model:
            self._model["name"] = model

        # Auto-capture environment
        if self._config.auto_log_env:
            self._environment = capture_full_environment()
        if self._config.auto_log_git:
            self._git = get_git_info()

        # Persist initial snapshot
        self._save()

    # ── ID generation ────────────────────────────────────────────────────

    @staticmethod
    def _generate_id() -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        short = uuid.uuid4().hex[:8]
        return f"exp_{ts}_{short}"

    # ── Serialisation ────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "status": self.status,
            "user_id": self.user_id,
            "tags": self.tags,
            "description": self.description,
            "created_at": self.created_at,
            "ended_at": self.ended_at,
            "duration_seconds": self.duration,
            "dataset": self._dataset,
            "model": self._model,
            "hyperparameters": self._hyperparameters,
            "test_results": self._test_results,
            "environment": self._environment,
            "git": self._git,
            "notes": self._notes,
        }

    def _save(self) -> None:
        data = self.to_dict()
        if self.status == "running":
            self._backend.save_experiment(self.experiment_id, data)
        else:
            self._backend.update_experiment(self.experiment_id, data)

    # ── Dataset logging ──────────────────────────────────────────────────

    def log_dataset(
        self,
        name: str | None = None,
        samples: int | None = None,
        features: Any = None,
        data_type: str | None = None,
        train_test_split: str | None = None,
        preprocessing: list[str] | None = None,
        source: str | None = None,
        **extra: Any,
    ) -> None:
        """Log dataset metadata."""
        if name:
            self._dataset["name"] = name
        if samples is not None:
            self._dataset["samples"] = samples
        if features is not None:
            self._dataset["features"] = features if isinstance(features, (list, str)) else list(features)
        if data_type:
            self._dataset["data_type"] = data_type
        if train_test_split:
            self._dataset["train_test_split"] = train_test_split
        if preprocessing:
            self._dataset["preprocessing"] = preprocessing
        if source:
            self._dataset["source"] = source
        self._dataset.update(extra)
        self._save()

    # ── Model logging ────────────────────────────────────────────────────

    def log_model(
        self,
        framework: str | None = None,
        architecture: str | None = None,
        parameters: int | None = None,
        layers: int | None = None,
        **extra: Any,
    ) -> None:
        """Log model architecture information."""
        if framework:
            self._model["framework"] = framework
        if architecture:
            self._model["architecture"] = architecture
        if parameters is not None:
            self._model["parameters"] = parameters
        if layers is not None:
            self._model["layers"] = layers
        self._model.update(extra)
        self._save()

    def log_model_summary(self, model_obj: Any) -> None:
        """Attempt to auto-extract info from a PyTorch / TF / sklearn model object."""
        cls_name = type(model_obj).__name__
        module = type(model_obj).__module__ or ""

        if "torch" in module:
            self._model["framework"] = "PyTorch"
            self._model["architecture"] = cls_name
            try:
                total = sum(p.numel() for p in model_obj.parameters())
                trainable = sum(p.numel() for p in model_obj.parameters() if p.requires_grad)
                self._model["parameters"] = total
                self._model["trainable_parameters"] = trainable
            except Exception:
                pass
        elif "tensorflow" in module or "keras" in module:
            self._model["framework"] = "TensorFlow/Keras"
            self._model["architecture"] = cls_name
            try:
                self._model["parameters"] = model_obj.count_params()
            except Exception:
                pass
        elif "sklearn" in module:
            self._model["framework"] = "scikit-learn"
            self._model["architecture"] = cls_name
            try:
                self._model["sklearn_params"] = model_obj.get_params()
            except Exception:
                pass
        else:
            self._model["architecture"] = cls_name

        self._save()

    # ── Hyperparameters ──────────────────────────────────────────────────

    def log_hyperparameters(self, params: dict[str, Any]) -> None:
        """Log (or update) hyperparameters."""
        self._hyperparameters.update(params)
        self._save()

    def log_hyperparameter(self, key: str, value: Any) -> None:
        self._hyperparameters[key] = value
        self._save()

    # ── Metrics ──────────────────────────────────────────────────────────

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        """Append a single metric data-point."""
        self._backend.append_metric(
            self.experiment_id, name, value, step=step,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log multiple metrics at the same step."""
        for k, v in metrics.items():
            self.log_metric(k, v, step=step)

    def get_metrics(self, name: str | None = None) -> list[dict[str, Any]]:
        """Retrieve stored metrics for this experiment."""
        return self._backend.get_metrics(self.experiment_id, name=name)

    # ── Test results ─────────────────────────────────────────────────────

    def log_test_results(self, results: dict[str, Any]) -> None:
        """Store final test / evaluation results."""
        self._test_results.update(results)
        self._save()

    # ── Artifacts ────────────────────────────────────────────────────────

    def save_artifact(self, filepath: str, artifact_type: str | None = None) -> str:
        """Copy a file into the experiment's artifact store and register it.

        Returns the destination path.
        """
        src = Path(filepath)
        if not src.exists():
            raise FileNotFoundError(f"Artifact source not found: {filepath}")

        dest_dir = self._config.artifacts_dir(self.experiment_id)
        dest = dest_dir / src.name
        shutil.copy2(str(src), str(dest))
        self._backend.register_artifact(
            self.experiment_id, src.name, str(dest), artifact_type=artifact_type
        )
        return str(dest)

    def list_artifacts(self) -> list[dict[str, Any]]:
        return self._backend.list_artifacts(self.experiment_id)

    # ── Notes ────────────────────────────────────────────────────────────

    def add_note(self, text: str) -> None:
        self._notes.append(text)
        self._save()

    # ── Timing ───────────────────────────────────────────────────────────

    @property
    def duration(self) -> float:
        """Wall-clock seconds since experiment start."""
        return round(time.time() - self._start_wall, 2)

    # ── Lifecycle ────────────────────────────────────────────────────────

    def end(self, status: str = "completed") -> None:
        """Finalise the experiment."""
        self.status = status
        self.ended_at = datetime.now(timezone.utc).isoformat()
        self._save()

    def fail(self, error: str | None = None) -> None:
        """Mark experiment as failed."""
        if error:
            self.add_note(f"FAILURE: {error}")
        self.end(status="failed")

    # ── Context manager ──────────────────────────────────────────────────

    def __enter__(self) -> "Experiment":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            self.fail(str(exc_val))
        elif self.status == "running":
            self.end()

    # ── Repr ─────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"Experiment(id={self.experiment_id!r}, name={self.name!r}, "
            f"status={self.status!r})"
        )
