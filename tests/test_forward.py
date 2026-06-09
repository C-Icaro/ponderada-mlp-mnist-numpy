import numpy as np

from mlp import MLPClassifier
from mlp.losses import softmax


def test_forward_accepts_multiple_hidden_layers():
    model = MLPClassifier([4, 5, 3, 2], activation="relu", seed=7)
    X = np.ones((6, 4), dtype=np.float32)

    logits = model.forward(X)
    probs = softmax(logits)

    assert logits.shape == (6, 2)
    assert probs.shape == (6, 2)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(6), atol=1e-6)
