"""
Example: Full mltracker workflow
================================
Demonstrates every major feature of the library.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mltracker import Experiment, Tracker, login, Config
from mltracker.config import set_config
from mltracker.reproducibility import set_all_seeds, export_reproducibility_config

# ── 1. Configure storage ────────────────────────────────────────────────────

tmpdir = tempfile.mkdtemp(prefix="mltracker_demo_")
set_config(Config(base_dir=tmpdir, storage_backend="sqlite"))
print(f"Storage directory: {tmpdir}\n")

# ── 2. User login ───────────────────────────────────────────────────────────

user = login("researcher_alice")
print(f"Logged in as: {user}\n")

# ── 3. Run Experiment 1: ResNet on CIFAR-10 ─────────────────────────────────

set_all_seeds(42)

with Experiment("ResNet50_CIFAR10", dataset="CIFAR10", model="ResNet50",
                user_id=user.user_id, tags=["baseline", "vision"]) as exp1:

    exp1.log_dataset(
        name="CIFAR10", samples=60000, features=(32, 32, 3),
        data_type="image", train_test_split="50000/10000",
        preprocessing=["normalize", "random_horizontal_flip", "random_crop"],
        source="torchvision",
    )

    exp1.log_model(
        framework="PyTorch", architecture="ResNet50",
        parameters=23_500_000, layers=50,
    )

    exp1.log_hyperparameters({
        "learning_rate": 0.001,
        "batch_size": 64,
        "optimizer": "Adam",
        "epochs": 50,
        "weight_decay": 1e-4,
        "scheduler": "CosineAnnealingLR",
    })

    # Simulate training loop
    for epoch in range(50):
        train_loss = 2.0 * (0.95 ** epoch)
        val_acc = min(0.93, 0.4 + epoch * 0.012)
        exp1.log_metrics({
            "train_loss": round(train_loss, 4),
            "val_accuracy": round(val_acc, 4),
        }, step=epoch)

    exp1.log_test_results({
        "accuracy": 0.932,
        "f1_score": 0.928,
        "precision": 0.935,
        "recall": 0.921,
    })

    exp1.add_note("Baseline run with default hyperparameters.")

print(f"Experiment 1 completed: {exp1}")
print(f"  Duration: {exp1.duration:.1f}s\n")

# ── 4. Run Experiment 2: Transformer on same data ───────────────────────────

with Experiment("ViT_CIFAR10", dataset="CIFAR10", model="ViT-B/16",
                user_id=user.user_id, tags=["transformer", "vision"]) as exp2:

    exp2.log_dataset(name="CIFAR10", samples=60000, features=(32, 32, 3))
    exp2.log_model(framework="PyTorch", architecture="ViT-B/16",
                   parameters=86_000_000, layers=12)
    exp2.log_hyperparameters({
        "learning_rate": 3e-4, "batch_size": 128,
        "optimizer": "AdamW", "epochs": 100,
    })

    for epoch in range(100):
        exp2.log_metric("train_loss", round(2.5 * (0.97 ** epoch), 4), step=epoch)
        exp2.log_metric("val_accuracy", round(min(0.96, 0.3 + epoch * 0.007), 4), step=epoch)

    exp2.log_test_results({"accuracy": 0.958, "f1_score": 0.955})

print(f"Experiment 2 completed: {exp2}\n")

# ── 5. Query & Compare ──────────────────────────────────────────────────────

tracker = Tracker()

print("=== All experiments ===")
for e in tracker.list_experiments():
    print(f"  {e['experiment_id']}  {e['name']}  status={e['status']}")

print("\n=== Search: dataset=CIFAR10 ===")
for e in tracker.search(dataset="CIFAR10"):
    print(f"  {e['name']}  acc={e.get('test_results', {}).get('accuracy')}")

print("\n=== Search: accuracy > 0.95 ===")
for e in tracker.search(metric="accuracy > 0.95"):
    print(f"  {e['name']}  acc={e['test_results']['accuracy']}")

print("\n=== Comparison ===")
cmp = tracker.compare([exp1.experiment_id, exp2.experiment_id], metric="val_accuracy")
for row in cmp:
    print(f"  {row['name']}: best_val_accuracy={row.get('best_val_accuracy')}")

print("\n=== Leaderboard (val_accuracy) ===")
from mltracker.visualization import print_leaderboard
print_leaderboard("val_accuracy")

# ── 6. Reproducibility export ───────────────────────────────────────────────

repro = export_reproducibility_config(exp2.to_dict())
print("\n=== Reproducibility config (excerpt) ===")
print(f"  Experiment: {repro['name']}")
print(f"  Hyperparameters: {repro['hyperparameters']}")
print(f"  Git: {repro['git']}")

print("\n✓ Demo complete.")
