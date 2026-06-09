"""A small but complete MLP classifier written with NumPy only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .activations import Activation, get_activation
from .losses import accuracy, one_hot, softmax, softmax_cross_entropy
from .optimizers import SGD


Array = np.ndarray


@dataclass
class LayerCache:
    a_prev: Array
    z: Array


class MLPClassifier:
    """Multi-layer perceptron for multiclass classification.

    Parameters are stored as ``W1``, ``b1``, ..., ``WL``, ``bL``. Hidden layers
    use the configured activation; the output layer is linear and later passed
    through softmax.
    """

    def __init__(
        self,
        layer_sizes: list[int] | tuple[int, ...],
        activation: str = "relu",
        seed: int = 42,
    ) -> None:
        if len(layer_sizes) < 3:
            raise ValueError("Use at least input, one hidden, and output layer.")
        if any(size <= 0 for size in layer_sizes):
            raise ValueError("All layer sizes must be positive integers.")

        self.layer_sizes = tuple(int(size) for size in layer_sizes)
        self.activation: Activation = get_activation(activation)
        self.rng = np.random.default_rng(seed)
        self.params: dict[str, Array] = {}
        self._init_parameters()

    @property
    def num_layers(self) -> int:
        return len(self.layer_sizes) - 1

    def _init_parameters(self) -> None:
        for layer in range(1, len(self.layer_sizes)):
            fan_in = self.layer_sizes[layer - 1]
            fan_out = self.layer_sizes[layer]
            scale = np.sqrt(2.0 / fan_in) if layer < self.num_layers else np.sqrt(1.0 / fan_in)
            self.params[f"W{layer}"] = self.rng.normal(0.0, scale, (fan_in, fan_out)).astype(np.float32)
            self.params[f"b{layer}"] = np.zeros((1, fan_out), dtype=np.float32)

    def forward(self, X: Array, return_cache: bool = False) -> Array | tuple[Array, list[LayerCache]]:
        a = X.astype(np.float32, copy=False)
        caches: list[LayerCache] = []

        for layer in range(1, self.num_layers):
            z = a @ self.params[f"W{layer}"] + self.params[f"b{layer}"]
            caches.append(LayerCache(a_prev=a, z=z))
            a = self.activation.forward(z)

        logits = a @ self.params[f"W{self.num_layers}"] + self.params[f"b{self.num_layers}"]
        caches.append(LayerCache(a_prev=a, z=logits))

        if return_cache:
            return logits, caches
        return logits

    def predict_proba(self, X: Array) -> Array:
        logits = self.forward(X)
        return softmax(logits)

    def predict(self, X: Array) -> Array:
        return np.argmax(self.predict_proba(X), axis=1)

    def loss(self, X: Array, y: Array, l2: float = 0.0) -> float:
        logits = self.forward(X)
        data_loss, _ = softmax_cross_entropy(logits, y)
        return data_loss + self._l2_penalty(l2)

    def loss_and_gradients(self, X: Array, y: Array, l2: float = 0.0) -> tuple[float, dict[str, Array]]:
        logits, caches = self.forward(X, return_cache=True)
        loss, probs = softmax_cross_entropy(logits, y)
        loss += self._l2_penalty(l2)

        y_encoded = one_hot(y, self.layer_sizes[-1])
        batch_size = X.shape[0]
        dz = (probs - y_encoded) / batch_size
        grads: dict[str, Array] = {}

        for layer in range(self.num_layers, 0, -1):
            cache = caches[layer - 1]
            weight_name = f"W{layer}"
            bias_name = f"b{layer}"

            grads[weight_name] = cache.a_prev.T @ dz + l2 * self.params[weight_name]
            grads[bias_name] = np.sum(dz, axis=0, keepdims=True)

            if layer > 1:
                da_prev = dz @ self.params[weight_name].T
                dz = da_prev * self.activation.derivative(caches[layer - 2].z)

        return loss, grads

    def fit(
        self,
        X_train: Array,
        y_train: Array,
        X_val: Array | None = None,
        y_val: Array | None = None,
        epochs: int = 10,
        batch_size: int = 128,
        learning_rate: float = 0.1,
        learning_rate_decay: float = 1.0,
        l2: float = 0.0,
        momentum: float = 0.0,
        shuffle: bool = True,
        seed: int = 42,
        train_metric_sample_size: int | None = None,
        verbose: bool = True,
    ) -> list[dict[str, float]]:
        if epochs <= 0:
            raise ValueError("epochs must be positive.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        optimizer = SGD(learning_rate=learning_rate, momentum=momentum)
        rng = np.random.default_rng(seed)
        history: list[dict[str, float]] = []
        n_samples = X_train.shape[0]

        for epoch in range(1, epochs + 1):
            optimizer.learning_rate = learning_rate * (learning_rate_decay ** (epoch - 1))
            indices = np.arange(n_samples)
            if shuffle:
                rng.shuffle(indices)

            epoch_losses = []
            for start in range(0, n_samples, batch_size):
                batch_indices = indices[start : start + batch_size]
                X_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]
                batch_loss, grads = self.loss_and_gradients(X_batch, y_batch, l2=l2)
                optimizer.step(self.params, grads)
                epoch_losses.append(batch_loss)

            metric_indices = indices
            if train_metric_sample_size is not None and train_metric_sample_size < n_samples:
                metric_indices = indices[:train_metric_sample_size]
            train_metrics = self.evaluate(X_train[metric_indices], y_train[metric_indices], l2=l2)
            row = {
                "epoch": float(epoch),
                "learning_rate": float(optimizer.learning_rate),
                "batch_loss": float(np.mean(epoch_losses)),
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
            }

            if X_val is not None and y_val is not None:
                val_metrics = self.evaluate(X_val, y_val, l2=l2)
                row["val_loss"] = val_metrics["loss"]
                row["val_accuracy"] = val_metrics["accuracy"]

            history.append(row)
            if verbose:
                msg = (
                    f"epoch {epoch:02d}/{epochs} "
                    f"loss={row['train_loss']:.4f} acc={row['train_accuracy']:.4f}"
                )
                if "val_accuracy" in row:
                    msg += f" val_acc={row['val_accuracy']:.4f}"
                print(msg)

        return history

    def evaluate(self, X: Array, y: Array, l2: float = 0.0) -> dict[str, float]:
        probs = self.predict_proba(X)
        return {
            "loss": self.loss(X, y, l2=l2),
            "accuracy": accuracy(probs, y),
        }

    def _l2_penalty(self, l2: float) -> float:
        if l2 <= 0:
            return 0.0
        return float(0.5 * l2 * sum(np.sum(self.params[f"W{layer}"] ** 2) for layer in range(1, self.num_layers + 1)))
