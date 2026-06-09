"""Loss and metric helpers for classification."""

from __future__ import annotations

import numpy as np


Array = np.ndarray


def softmax(logits: Array) -> Array:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def one_hot(y: Array, num_classes: int) -> Array:
    y = y.astype(np.int64).ravel()
    encoded = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    encoded[np.arange(y.shape[0]), y] = 1.0
    return encoded


def cross_entropy(probs: Array, y: Array, eps: float = 1e-12) -> float:
    y = y.astype(np.int64).ravel()
    clipped = np.clip(probs[np.arange(y.shape[0]), y], eps, 1.0)
    return float(-np.mean(np.log(clipped)))


def accuracy(probs: Array, y: Array) -> float:
    predictions = np.argmax(probs, axis=1)
    return float(np.mean(predictions == y.ravel()))


def softmax_cross_entropy(logits: Array, y: Array) -> tuple[float, Array]:
    probs = softmax(logits)
    return cross_entropy(probs, y), probs
