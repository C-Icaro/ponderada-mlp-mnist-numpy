"""Activation functions used by the manual MLP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class Activation:
    name: str
    forward: Callable[[Array], Array]
    derivative: Callable[[Array], Array]


def relu(z: Array) -> Array:
    return np.maximum(z, 0)


def relu_derivative(z: Array) -> Array:
    return (z > 0).astype(z.dtype)


def tanh(z: Array) -> Array:
    return np.tanh(z)


def tanh_derivative(z: Array) -> Array:
    value = np.tanh(z)
    return 1 - value * value


def sigmoid(z: Array) -> Array:
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z: Array) -> Array:
    value = sigmoid(z)
    return value * (1 - value)


def get_activation(name: str) -> Activation:
    activations = {
        "relu": Activation("relu", relu, relu_derivative),
        "tanh": Activation("tanh", tanh, tanh_derivative),
        "sigmoid": Activation("sigmoid", sigmoid, sigmoid_derivative),
    }

    key = name.lower()
    if key not in activations:
        choices = ", ".join(sorted(activations))
        raise ValueError(f"Unsupported activation {name!r}. Choose one of: {choices}.")
    return activations[key]
