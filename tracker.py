"""Tracker – the query / management layer over all experiments."""

from __future__ import annotations

import operator
import re
from typing import Any

from mltracker.config import Config, get_config
from mltracker.storage import StorageBackend, create_backend


class Tracker:
    """High-level interface for searching, comparing, and managing experiments.

    Usage::

        tracker = Tracker()
        results = tracker.search(model="ResNet50")
        tracker.compare(["exp_001", "exp_002"], metric="val_accuracy")
    """

    def __init__(self, config: Config | None = None, backend: StorageBackend | None = None):
        self._config = config or get_config()
        self._backend = backend or create_backend(self._config)

    # ── List & search ────────────────────────────────────────────────────

    def list_experiments(self, user_id: str | None = None) -> list[dict[str, Any]]:
        """Return all experiments, optionally filtered by user."""
        return self._backend.list_experiments(user_id=user_id)

    def get_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        return self._backend.load_experiment(experiment_id)

    def search(self, *, metric: str | None = None, **filters: Any) -> list[dict[str, Any]]:
        """Search experiments by metadata fields **or** metric expressions.

        ``filters`` match exact values on nested keys (use dots):

            tracker.search(model="ResNet50")              # top-level alias
            tracker.search(**{"hyperparameters.optimizer": "Adam"})

        ``metric`` supports simple comparisons::

            tracker.search(metric="accuracy > 0.9")
        """
        # Translate convenience aliases
        alias_map = {"model": "model.name", "dataset": "dataset.name", "name": "name"}
        resolved: dict[str, Any] = {}
        for k, v in filters.items():
            resolved[alias_map.get(k, k)] = v

        results = self._backend.search_experiments(**resolved)

        # Post-filter by metric expression if provided
        if metric:
            results = self._filter_by_metric(results, metric)

        return results

    def delete(self, experiment_id: str) -> bool:
        return self._backend.delete_experiment(experiment_id)

    # ── Comparison ───────────────────────────────────────────────────────

    def compare(
        self,
        experiment_ids: list[str],
        metric: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return a comparison table for the given experiments.

        Each item contains experiment metadata plus the last value of
        *metric* (if supplied).
        """
        rows: list[dict[str, Any]] = []
        for eid in experiment_ids:
            exp = self._backend.load_experiment(eid)
            if exp is None:
                continue
            row: dict[str, Any] = {
                "experiment_id": eid,
                "name": exp.get("name"),
                "status": exp.get("status"),
                "hyperparameters": exp.get("hyperparameters", {}),
                "test_results": exp.get("test_results", {}),
                "duration_seconds": exp.get("duration_seconds"),
            }
            if metric:
                metric_data = self._backend.get_metrics(eid, name=metric)
                if metric_data:
                    row[f"last_{metric}"] = metric_data[-1]["value"]
                    row[f"best_{metric}"] = max(m["value"] for m in metric_data)
            rows.append(row)
        return rows

    def leaderboard(
        self,
        metric: str,
        higher_is_better: bool = True,
        top_n: int = 10,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Rank experiments by a metric's best value."""
        experiments = self.list_experiments(user_id=user_id)
        scored: list[tuple[float, dict]] = []
        for exp in experiments:
            eid = exp["experiment_id"]
            ms = self._backend.get_metrics(eid, name=metric)
            if ms:
                best = max(m["value"] for m in ms) if higher_is_better else min(m["value"] for m in ms)
                scored.append((best, {
                    "experiment_id": eid,
                    "name": exp.get("name"),
                    f"best_{metric}": best,
                    "hyperparameters": exp.get("hyperparameters", {}),
                }))
        scored.sort(key=lambda t: t[0], reverse=higher_is_better)
        return [row for _, row in scored[:top_n]]

    # ── Helpers ──────────────────────────────────────────────────────────

    _OP_MAP = {
        ">": operator.gt, ">=": operator.ge,
        "<": operator.lt, "<=": operator.le,
        "==": operator.eq, "!=": operator.ne,
    }

    def _filter_by_metric(self, experiments: list[dict], expr: str) -> list[dict]:
        """Parse ``'accuracy > 0.9'`` and post-filter."""
        m = re.match(r"(\w+)\s*(>=|<=|!=|==|>|<)\s*([\d.]+)", expr)
        if not m:
            return experiments
        metric_name, op_str, threshold_str = m.groups()
        op_fn = self._OP_MAP[op_str]
        threshold = float(threshold_str)

        filtered = []
        for exp in experiments:
            eid = exp["experiment_id"]
            ms = self._backend.get_metrics(eid, name=metric_name)
            if ms:
                last_val = ms[-1]["value"]
                if op_fn(last_val, threshold):
                    filtered.append(exp)
            # Also check test_results
            tr = exp.get("test_results", {})
            if metric_name in tr and op_fn(float(tr[metric_name]), threshold):
                if exp not in filtered:
                    filtered.append(exp)
        return filtered
