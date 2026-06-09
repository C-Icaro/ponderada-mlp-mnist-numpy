"""Dataset loading helpers.

The assignment allows using Keras or Torchvision only to load MNIST. The neural
network itself receives plain NumPy arrays and all learning math stays in NumPy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from torchvision.datasets import MNIST


Array = np.ndarray


def load_mnist(
    data_dir: str | Path = "data",
    val_size: int = 10_000,
    train_limit: int | None = None,
    test_limit: int | None = None,
    seed: int = 42,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    data_dir = Path(data_dir)
    train_dataset = MNIST(root=data_dir, train=True, download=True)
    test_dataset = MNIST(root=data_dir, train=False, download=True)

    X_train_full = _images_to_matrix(train_dataset.data.numpy())
    y_train_full = train_dataset.targets.numpy().astype(np.int64)
    X_test = _images_to_matrix(test_dataset.data.numpy())
    y_test = test_dataset.targets.numpy().astype(np.int64)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(X_train_full.shape[0])
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    if train_limit is not None:
        train_indices = train_indices[:train_limit]
    if test_limit is not None:
        X_test = X_test[:test_limit]
        y_test = y_test[:test_limit]

    return (
        X_train_full[train_indices],
        y_train_full[train_indices],
        X_train_full[val_indices],
        y_train_full[val_indices],
        X_test,
        y_test,
    )


def _images_to_matrix(images: Array) -> Array:
    return images.reshape(images.shape[0], -1).astype(np.float32) / 255.0
