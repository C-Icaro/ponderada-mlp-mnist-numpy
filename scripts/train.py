from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlp import MLPClassifier
from mlp.data import load_mnist


EXPERIMENTS = [
    {
        "name": "compact_relu_64_32",
        "hidden_layers": [64, 32],
        "activation": "relu",
        "learning_rate": 0.08,
        "learning_rate_decay": 0.96,
        "momentum": 0.0,
        "epochs": 8,
        "batch_size": 128,
        "l2": 0.0,
        "seed": 21,
    },
    {
        "name": "final_relu_128_64",
        "hidden_layers": [128, 64],
        "activation": "relu",
        "learning_rate": 0.12,
        "learning_rate_decay": 0.96,
        "momentum": 0.0,
        "epochs": 10,
        "batch_size": 128,
        "l2": 0.0,
        "seed": 42,
    },
    {
        "name": "improved_relu_256_128_momentum",
        "hidden_layers": [256, 128],
        "activation": "relu",
        "learning_rate": 0.05,
        "learning_rate_decay": 0.97,
        "momentum": 0.9,
        "epochs": 15,
        "batch_size": 128,
        "l2": 1e-4,
        "seed": 84,
    },
    {
        "name": "resolved_relu_256_128_l2_earlystop",
        "hidden_layers": [256, 128],
        "activation": "relu",
        "learning_rate": 0.05,
        "learning_rate_decay": 0.97,
        "momentum": 0.9,
        "epochs": 25,
        "batch_size": 128,
        "l2": 1e-3,
        "seed": 84,
        "early_stopping": True,
        "patience": 3,
        "monitor": "val_data_loss",
        "mode": "min",
        "min_delta": 0.0,
        "restore_best_weights": True,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NumPy MLP experiments on MNIST.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--quick", action="store_true", help="Use a tiny subset for smoke testing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    train_limit = 5_000 if args.quick else args.train_limit
    test_limit = 2_000 if args.quick else args.test_limit
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist(
        data_dir=args.data_dir,
        train_limit=train_limit,
        test_limit=test_limit,
    )

    experiments = _quick_experiments() if args.quick else EXPERIMENTS
    summary = []
    histories: dict[str, list[dict[str, float]]] = {}
    best_model: MLPClassifier | None = None
    best_accuracy = -1.0
    best_name = ""
    report_model: MLPClassifier | None = None
    report_name = ""

    for config in experiments:
        layer_sizes = [X_train.shape[1], *config["hidden_layers"], 10]
        model = MLPClassifier(layer_sizes, activation=config["activation"], seed=config["seed"])
        print(f"\n== {config['name']} ==")
        history = model.fit(
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
            learning_rate_decay=config["learning_rate_decay"],
            momentum=config["momentum"],
            l2=config["l2"],
            train_metric_sample_size=min(10_000, X_train.shape[0]),
            early_stopping=config.get("early_stopping", False),
            patience=config.get("patience", 3),
            monitor=config.get("monitor", "val_data_loss"),
            mode=config.get("mode", "min"),
            min_delta=config.get("min_delta", 0.0),
            restore_best_weights=config.get("restore_best_weights", True),
            verbose=True,
        )
        metric_sample_size = min(10_000, X_train.shape[0])
        train_metrics = model.evaluate(X_train[:metric_sample_size], y_train[:metric_sample_size], l2=config["l2"])
        val_metrics = model.evaluate(X_val, y_val, l2=config["l2"])
        test_metrics = model.evaluate(X_test, y_test, l2=config["l2"])
        row = {
            **config,
            "layer_sizes": layer_sizes,
            "test_loss": test_metrics["loss"],
            "test_data_loss": test_metrics["data_loss"],
            "test_regularization_loss": test_metrics["regularization_loss"],
            "test_accuracy": test_metrics["accuracy"],
            "final_val_accuracy": val_metrics["accuracy"],
            "final_val_loss": val_metrics["loss"],
            "final_val_data_loss": val_metrics["data_loss"],
            "final_train_data_loss": train_metrics["data_loss"],
            "final_train_accuracy": train_metrics["accuracy"],
            "epochs_run": len(history),
            "selected_epoch": model.best_epoch_ if model.best_epoch_ is not None else len(history),
            "stopped_epoch": model.stopped_epoch_,
            "restored_best_weights": model.restored_best_weights_,
            "best_monitor_value": model.best_monitor_value_,
        }
        summary.append(row)
        histories[config["name"]] = history
        _write_history(results_dir / f"history_{config['name']}.csv", history)

        if test_metrics["accuracy"] > best_accuracy:
            best_accuracy = test_metrics["accuracy"]
            best_model = model
            best_name = config["name"]
        if config["name"].startswith("resolved_"):
            report_model = model
            report_name = config["name"]

    _write_summary(results_dir, summary)
    _plot_histories(histories, results_dir / "loss_accuracy.png")
    final_model = report_model if report_model is not None else best_model
    final_name = report_name if report_model is not None else best_name
    if final_model is not None:
        _plot_confusion_matrix(
            y_test,
            final_model.predict(X_test),
            results_dir / "confusion_matrix_final.png",
            title=f"Matriz de confusao - {final_name}",
        )
    print("\nResumo salvo em", results_dir / "summary.json")


def _quick_experiments() -> list[dict[str, object]]:
    quick = []
    for config in EXPERIMENTS:
        item = dict(config)
        item["epochs"] = 2
        quick.append(item)
    return quick


def _write_history(path: Path, history: list[dict[str, float]]) -> None:
    fieldnames = sorted({key for row in history for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def _write_summary(results_dir: Path, summary: list[dict[str, object]]) -> None:
    with (results_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    fieldnames = sorted({key for row in summary for key in row})
    with (results_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def _plot_histories(histories: dict[str, list[dict[str, float]]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, history in histories.items():
        epochs = [row["epoch"] for row in history]
        axes[0].plot(epochs, [row.get("train_data_loss", row["train_loss"]) for row in history], marker="o", label=f"{name} treino")
        axes[0].plot(epochs, [row.get("val_data_loss", row.get("val_loss", np.nan)) for row in history], marker="x", label=f"{name} val")
        axes[1].plot(epochs, [row["train_accuracy"] for row in history], marker="o", label=f"{name} treino")
        axes[1].plot(epochs, [row.get("val_accuracy", np.nan) for row in history], marker="x", label=f"{name} val")

    axes[0].set_title("Cross-entropy por epoca")
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("Cross-entropy")
    axes[1].set_title("Acuracia por epoca")
    axes[1].set_xlabel("Epoca")
    axes[1].set_ylabel("Acuracia")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, path: Path, title: str) -> None:
    matrix = np.zeros((10, 10), dtype=np.int64)
    for actual, predicted in zip(y_true, y_pred):
        matrix[int(actual), int(predicted)] += 1

    fig, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_title(title)
    axis.set_xlabel("Predito")
    axis.set_ylabel("Real")
    axis.set_xticks(range(10))
    axis.set_yticks(range(10))
    for i in range(10):
        for j in range(10):
            axis.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
