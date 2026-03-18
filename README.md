# mltracker

A production-ready Python library for ML experiment tracking, storage, and reproducibility.

## Features

- **Zero dependencies** — runs on the Python standard library alone (optional deps for visualization)
- **Automatic environment capture** — Python version, OS, GPU, RAM, installed packages, git commit
- **Pluggable storage** — SQLite (default), JSON flat-files, or PostgreSQL
- **Multi-user accounts** — isolated experiment histories per user
- **Rich query system** — search by model, dataset, hyperparameters, or metric expressions
- **Visualization** — training curves, metric comparisons, leaderboards
- **PyTorch hooks** — automatic gradient, activation, and parameter statistics
- **Reproducibility** — seed management, code snapshots, exportable configs
- **Context-manager API** — experiments auto-close on success or record failures

---

## Quick Start

```python
from mltracker import Experiment

# Create and run an experiment
with Experiment(name="ResNet50_CIFAR10", dataset="CIFAR10", model="ResNet50") as exp:
    exp.log_dataset(name="CIFAR10", samples=60000, features=(32, 32, 3),
                    preprocessing=["normalize", "augmentation"])
    exp.log_model(framework="PyTorch", architecture="ResNet50",
                  parameters=23_500_000, layers=50)
    exp.log_hyperparameters({
        "learning_rate": 0.001, "batch_size": 64,
        "optimizer": "Adam", "epochs": 50,
    })

    for epoch in range(50):
        # ... training code ...
        exp.log_metric("train_loss", 0.5 - epoch * 0.01, step=epoch)
        exp.log_metric("val_accuracy", 0.6 + epoch * 0.006, step=epoch)

    exp.log_test_results({"accuracy": 0.93, "f1_score": 0.91})
    exp.save_artifact("model_weights.pth")
```

---

## Installation

```bash
pip install -e .             # core (zero deps)
pip install -e ".[viz]"      # + matplotlib
pip install -e ".[all]"      # + matplotlib, psutil, plotly
```

---

## API Reference

### Experiment

```python
from mltracker import Experiment

exp = Experiment(
    name="my_experiment",       # required
    dataset="CIFAR10",          # optional shortcut
    model="ResNet50",           # optional shortcut
    tags=["baseline"],          # optional tags
    description="First run",    # optional
)

# Dataset
exp.log_dataset(name="CIFAR10", samples=60000, features=(32,32,3),
                preprocessing=["normalize"])

# Model
exp.log_model(framework="PyTorch", architecture="ResNet50",
              parameters=23_500_000, layers=50)
exp.log_model_summary(pytorch_model)  # auto-extract from model object

# Hyperparameters
exp.log_hyperparameters({"learning_rate": 0.001, "batch_size": 64})

# Metrics (append-only time series)
exp.log_metric("train_loss", 0.34, step=1)
exp.log_metrics({"loss": 0.34, "accuracy": 0.89}, step=1)

# Test results
exp.log_test_results({"accuracy": 0.93, "f1_score": 0.91})

# Artifacts
exp.save_artifact("model.pth")
exp.save_artifact("confusion_matrix.png", artifact_type="image")

# Notes
exp.add_note("Switched to cosine annealing at epoch 20")

# Finish
exp.end()          # or exp.fail("OOM error")
```

### Tracker (query & compare)

```python
from mltracker import Tracker

tracker = Tracker()

# List all experiments
tracker.list_experiments()

# Search
tracker.search(model="ResNet50")
tracker.search(dataset="ImageNet")
tracker.search(metric="accuracy > 0.9")
tracker.search(**{"hyperparameters.optimizer": "Adam"})

# Compare
tracker.compare(["exp_001", "exp_002"], metric="val_accuracy")

# Leaderboard
tracker.leaderboard("val_accuracy", higher_is_better=True, top_n=10)
```

### User Accounts

```python
from mltracker import login, register

user = login("alice")                        # auto-creates if needed
user = register("bob", email="b@lab.org")    # explicit registration
exps = user.experiments()                    # user's experiment history
```

### Visualization

```python
from mltracker.visualization import (
    plot_training_curve, plot_comparison,
    plot_test_results, print_leaderboard,
)

plot_training_curve("exp_001", save_path="curves.png")
plot_comparison(["exp_001", "exp_002"], "val_accuracy", save_path="cmp.png")
plot_test_results(["exp_001", "exp_002"], save_path="tests.png")
print_leaderboard("val_accuracy")
```

### PyTorch Hooks

```python
from mltracker.hooks import PyTorchHooks

hooks = PyTorchHooks(exp)
hooks.register(model, log_gradients=True, log_activations=False)

for epoch in range(num_epochs):
    # ... training step ...
    hooks.log_gradient_stats(step=epoch)
    hooks.log_parameter_stats(model, step=epoch)

hooks.remove()
```

### Reproducibility

```python
from mltracker.reproducibility import set_all_seeds, snapshot_code

set_all_seeds(42)
snapshot_code(exp.experiment_id, source_dir="./src")
```

### Configuration

```python
from mltracker import Config
from mltracker.config import set_config

set_config(Config(
    base_dir="/data/experiments",
    storage_backend="json",      # "sqlite" | "json" | "postgresql"
))
```

Or via environment variable:

```bash
export MLTRACKER_DIR=/data/experiments
```

---

## Storage Layout

### SQLite (default)

```
~/.mltracker/
    mltracker.db          # single database file
    experiments/
        exp_20240101_.../
            artifacts/
```

### JSON flat-file

```
~/.mltracker/
    experiments/
        exp_20240101_.../
            metadata.json
            metrics.json
            artifacts.json
            artifacts/
    users/
        abc123.json
```

---

## Database Schema

```sql
CREATE TABLE users (
    user_id   TEXT PRIMARY KEY,
    username  TEXT UNIQUE NOT NULL,
    email     TEXT,
    created_at TEXT NOT NULL,
    data_json  TEXT
);

CREATE TABLE experiments (
    experiment_id TEXT PRIMARY KEY,
    user_id       TEXT REFERENCES users(user_id),
    name          TEXT,
    status        TEXT DEFAULT 'running',
    created_at    TEXT NOT NULL,
    ended_at      TEXT,
    data_json     TEXT NOT NULL
);

CREATE TABLE metrics (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
    name          TEXT NOT NULL,
    value         REAL NOT NULL,
    step          INTEGER,
    timestamp     TEXT NOT NULL
);

CREATE TABLE artifacts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
    filename      TEXT NOT NULL,
    path          TEXT NOT NULL,
    artifact_type TEXT,
    created_at    TEXT NOT NULL
);
```

---

## License

MIT
