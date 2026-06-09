"""Optimizers for manual parameter updates."""

from __future__ import annotations

import numpy as np


Array = np.ndarray


class SGD:
    """Mini-batch stochastic gradient descent with optional momentum."""

    def __init__(self, learning_rate: float, momentum: float = 0.0) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if momentum < 0:
            raise ValueError("momentum cannot be negative.")
        self.learning_rate = float(learning_rate)
        self.momentum = float(momentum)
        self.velocity: dict[str, Array] = {}

    def step(self, params: dict[str, Array], grads: dict[str, Array]) -> None:
        for name, param in params.items():
            grad = grads[name]
            if self.momentum:
                if name not in self.velocity:
                    self.velocity[name] = np.zeros_like(param)
                self.velocity[name] = self.momentum * self.velocity[name] - self.learning_rate * grad
                param += self.velocity[name]
            else:
                param -= self.learning_rate * grad
