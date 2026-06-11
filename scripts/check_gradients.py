from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlp import MLPClassifier
from mlp.gradient_check import gradient_check, summarize_gradient_check


def main() -> None:
    rng = np.random.default_rng(13)
    X = rng.normal(size=(5, 4)).astype(np.float32)
    y = np.array([0, 1, 2, 1, 0])
    model = MLPClassifier([4, 6, 5, 3], activation="tanh", seed=3)

    results = gradient_check(
        model,
        X,
        y,
        epsilon=1e-3,
        tolerance=1e-2,
        max_checks_per_parameter=3,
        seed=9,
    )
    summary = summarize_gradient_check(results)

    output = ROOT / "results" / "gradient_check.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps({key: value for key, value in summary.items() if key != "checks"}, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
