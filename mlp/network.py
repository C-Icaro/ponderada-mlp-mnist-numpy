"""A small but complete MLP classifier written with NumPy only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .activations import Activation, get_activation
from .losses import softmax


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
