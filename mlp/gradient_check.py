"""Reusable numerical gradient checking for the manual MLP."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from .network import MLPClassifier


Array = np.ndarray
Index = tuple[int, ...]
CheckSpec = tuple[str, Index]


@dataclass(frozen=True)
class GradientCheckResult:
    parameter: str
    index: Index
    analytical: float
    numerical: float
    absolute_error: float
    relative_error: float
    passed: bool

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["index"] = list(self.index)
        return data


def gradient_check(
    model: MLPClassifier,
    X: Array,
    y: Array,
    checks: Iterable[CheckSpec] | None = None,
    epsilon: float = 1e-3,
    tolerance: float = 1e-2,
    max_checks_per_parameter: int = 2,
    seed: int = 0,
    l2: float = 0.0,
) -> list[GradientCheckResult]:
    """Compare analytical gradients with finite-difference estimates.

    The model is restored to its original parameter values after every
    perturbation, so this can be run before training without changing the
    experiment.
    """

    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")

    _, grads = model.loss_and_gradients(X, y, l2=l2)
    selected_checks = list(checks) if checks is not None else _sample_checks(model, max_checks_per_parameter, seed)
    results: list[GradientCheckResult] = []

    for parameter, index in selected_checks:
        original = model.params[parameter][index]

        model.params[parameter][index] = original + epsilon
        loss_plus = model.loss(X, y, l2=l2)

        model.params[parameter][index] = original - epsilon
        loss_minus = model.loss(X, y, l2=l2)

        model.params[parameter][index] = original

        numerical = (loss_plus - loss_minus) / (2 * epsilon)
        analytical = float(grads[parameter][index])
        absolute_error = abs(analytical - numerical)
        denominator = max(1.0, abs(analytical) + abs(numerical))
        relative_error = absolute_error / denominator

        results.append(
            GradientCheckResult(
                parameter=parameter,
                index=index,
                analytical=analytical,
                numerical=float(numerical),
                absolute_error=float(absolute_error),
                relative_error=float(relative_error),
                passed=relative_error <= tolerance,
            )
        )

    return results


def summarize_gradient_check(results: list[GradientCheckResult]) -> dict[str, object]:
    return {
        "passed": all(result.passed for result in results),
        "num_checks": len(results),
        "max_absolute_error": max((result.absolute_error for result in results), default=0.0),
        "max_relative_error": max((result.relative_error for result in results), default=0.0),
        "checks": [result.to_dict() for result in results],
    }


def _sample_checks(model: MLPClassifier, max_checks_per_parameter: int, seed: int) -> list[CheckSpec]:
    if max_checks_per_parameter <= 0:
        raise ValueError("max_checks_per_parameter must be positive.")

    rng = np.random.default_rng(seed)
    checks: list[CheckSpec] = []
    for parameter in sorted(model.params):
        values = model.params[parameter]
        sample_size = min(max_checks_per_parameter, values.size)
        flat_indices = rng.choice(values.size, size=sample_size, replace=False)
        for flat_index in np.atleast_1d(flat_indices):
            index = tuple(int(part) for part in np.unravel_index(flat_index, values.shape))
            checks.append((parameter, index))
    return checks
