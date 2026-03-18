"""Test suite for mltracker library."""

import json
import os
import sys
import tempfile
import time

# Add parent to path for testing without install
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mltracker.config import Config, set_config, get_config
from mltracker.experiment import Experiment
from mltracker.tracker import Tracker
from mltracker.user import login, register
from mltracker.storage import SQLiteBackend, JSONBackend, create_backend
from mltracker.reproducibility import set_all_seeds, export_reproducibility_config


def test_sqlite_experiment_lifecycle():
    """Full lifecycle: create → log → query → compare → delete."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(base_dir=tmp, storage_backend="sqlite")
        backend = create_backend(cfg)

        # Create experiment
        exp = Experiment("TestExp", dataset="MNIST", model="MLP", config=cfg, backend=backend)
        eid = exp.experiment_id
        assert eid.startswith("exp_")
        assert exp.status == "running"

        # Log dataset
        exp.log_dataset(name="MNIST", samples=70000, features=(28, 28, 1),
                        preprocessing=["normalize"])

        # Log model
        exp.log_model(framework="PyTorch", architecture="MLP", parameters=100_000, layers=3)

        # Log hyperparameters
        exp.log_hyperparameters({
            "learning_rate": 0.001, "batch_size": 128,
            "optimizer": "SGD", "epochs": 10,
        })

        # Log metrics
        for step in range(10):
            exp.log_metric("train_loss", 1.0 - step * 0.08, step=step)
            exp.log_metric("val_accuracy", 0.5 + step * 0.05, step=step)

        # Test results
        exp.log_test_results({"accuracy": 0.95, "f1_score": 0.94})

        # End
        exp.end()
        assert exp.status == "completed"
        assert exp.ended_at is not None
        assert exp.duration > 0

        # Verify persistence
        loaded = backend.load_experiment(eid)
        assert loaded is not None
        assert loaded["name"] == "TestExp"
        assert loaded["hyperparameters"]["optimizer"] == "SGD"
        assert loaded["test_results"]["accuracy"] == 0.95

        # Verify metrics
        metrics = backend.get_metrics(eid, name="train_loss")
        assert len(metrics) == 10
        assert metrics[0]["value"] == 1.0

        # Search
        tracker = Tracker(config=cfg, backend=backend)
        results = tracker.search(model="MLP")
        assert len(results) >= 1

        results = tracker.search(metric="accuracy > 0.9")
        assert len(results) >= 1

        # Leaderboard
        lb = tracker.leaderboard("val_accuracy", top_n=5)
        assert len(lb) >= 1

        # Compare
        cmp = tracker.compare([eid], metric="val_accuracy")
        assert len(cmp) == 1
        assert "last_val_accuracy" in cmp[0]

        # Delete
        assert tracker.delete(eid)
        assert backend.load_experiment(eid) is None

        print("  [PASS] SQLite experiment lifecycle")


def test_json_backend():
    """Verify JSON backend works identically."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(base_dir=tmp, storage_backend="json")
        backend = create_backend(cfg)

        exp = Experiment("JSONTest", dataset="CIFAR10", config=cfg, backend=backend)
        exp.log_hyperparameters({"lr": 0.01})
        exp.log_metric("loss", 0.5, step=0)
        exp.log_metric("loss", 0.3, step=1)
        exp.end()

        loaded = backend.load_experiment(exp.experiment_id)
        assert loaded is not None
        assert loaded["hyperparameters"]["lr"] == 0.01

        metrics = backend.get_metrics(exp.experiment_id, name="loss")
        assert len(metrics) == 2

        # List
        all_exp = backend.list_experiments()
        assert len(all_exp) >= 1

        print("  [PASS] JSON backend")


def test_user_accounts():
    """User registration, login, and experiment association."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(base_dir=tmp, storage_backend="sqlite")

        user = register("alice", email="alice@lab.org", config=cfg)
        assert user.username == "alice"

        user2 = login("alice", config=cfg)
        assert user2.user_id == user.user_id

        # Auto-create
        user3 = login("bob", config=cfg)
        assert user3.username == "bob"

        # Duplicate detection
        try:
            register("alice", config=cfg)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

        print("  [PASS] User accounts")


def test_context_manager():
    """Experiment as context manager with success and failure."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(base_dir=tmp, storage_backend="sqlite")
        backend = create_backend(cfg)

        # Success path
        with Experiment("CtxSuccess", config=cfg, backend=backend) as exp:
            exp.log_metric("x", 1.0)
        assert exp.status == "completed"

        # Failure path
        try:
            with Experiment("CtxFail", config=cfg, backend=backend) as exp2:
                exp2.log_metric("x", 1.0)
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert exp2.status == "failed"

        print("  [PASS] Context manager")


def test_reproducibility():
    """Seeds and export."""
    set_all_seeds(42)

    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(base_dir=tmp, storage_backend="sqlite")
        backend = create_backend(cfg)
        exp = Experiment("ReproTest", config=cfg, backend=backend)
        exp.log_hyperparameters({"seed": 42, "lr": 0.001})
        exp.end()

        repro = export_reproducibility_config(exp.to_dict())
        assert repro["hyperparameters"]["seed"] == 42
        assert repro["experiment_id"] == exp.experiment_id

        # Export to file
        out_path = os.path.join(tmp, "repro.json")
        export_reproducibility_config(exp.to_dict(), output_path=out_path)
        assert os.path.exists(out_path)
        data = json.loads(open(out_path).read())
        assert data["name"] == "ReproTest"

        print("  [PASS] Reproducibility")


def test_artifacts():
    """Artifact save and list."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(base_dir=tmp, storage_backend="sqlite")
        backend = create_backend(cfg)

        exp = Experiment("ArtifactTest", config=cfg, backend=backend)

        # Create a fake artifact file
        fake_model = os.path.join(tmp, "model.pt")
        with open(fake_model, "w") as f:
            f.write("fake model data")

        dest = exp.save_artifact(fake_model, artifact_type="model")
        assert os.path.exists(dest)

        arts = exp.list_artifacts()
        assert len(arts) == 1
        assert arts[0]["filename"] == "model.pt"

        exp.end()
        print("  [PASS] Artifacts")


def test_environment_capture():
    """Verify environment data is captured."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(base_dir=tmp, storage_backend="sqlite", auto_log_env=True)
        backend = create_backend(cfg)

        exp = Experiment("EnvTest", config=cfg, backend=backend)
        data = exp.to_dict()

        assert "python_version" in data["environment"]
        assert "os" in data["environment"]
        assert "ram" in data["environment"]

        exp.end()
        print("  [PASS] Environment capture")


def test_search_advanced():
    """Test nested key search and metric filtering."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(base_dir=tmp, storage_backend="sqlite")
        backend = create_backend(cfg)

        # Create several experiments
        for i, (opt, lr) in enumerate([("Adam", 0.001), ("SGD", 0.01), ("Adam", 0.0001)]):
            exp = Experiment(f"SearchTest_{i}", config=cfg, backend=backend)
            exp.log_hyperparameters({"optimizer": opt, "learning_rate": lr})
            exp.log_metric("val_accuracy", 0.8 + i * 0.05, step=0)
            exp.log_test_results({"accuracy": 0.8 + i * 0.05})
            exp.end()

        tracker = Tracker(config=cfg, backend=backend)

        # Search by nested key
        results = tracker.search(**{"hyperparameters.optimizer": "Adam"})
        assert len(results) == 2

        # Search by metric expression
        results = tracker.search(metric="accuracy > 0.85")
        assert len(results) >= 1

        print("  [PASS] Advanced search")


# ── Run all tests ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running mltracker tests...\n")
    test_sqlite_experiment_lifecycle()
    test_json_backend()
    test_user_accounts()
    test_context_manager()
    test_reproducibility()
    test_artifacts()
    test_environment_capture()
    test_search_advanced()
    print("\n✓ All tests passed!")
