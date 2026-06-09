import numpy as np

from mlp import MLPClassifier


def test_gradient_check_small_network():
    rng = np.random.default_rng(13)
    X = rng.normal(size=(5, 4)).astype(np.float32)
    y = np.array([0, 1, 2, 1, 0])
    model = MLPClassifier([4, 6, 5, 3], activation="tanh", seed=3)

    _, grads = model.loss_and_gradients(X, y)
    epsilon = 1e-3
    checks = [
        ("W1", (0, 0)),
        ("b1", (0, 2)),
        ("W2", (1, 3)),
        ("b2", (0, 4)),
        ("W3", (2, 1)),
        ("b3", (0, 2)),
    ]

    for name, index in checks:
        original = model.params[name][index]
        model.params[name][index] = original + epsilon
        loss_plus = model.loss(X, y)
        model.params[name][index] = original - epsilon
        loss_minus = model.loss(X, y)
        model.params[name][index] = original

        numerical = (loss_plus - loss_minus) / (2 * epsilon)
        analytical = grads[name][index]
        np.testing.assert_allclose(analytical, numerical, rtol=1e-2, atol=1e-3)


def test_training_reduces_toy_loss():
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    y = np.array([0, 1, 1, 0])
    model = MLPClassifier([2, 8, 4, 2], activation="tanh", seed=11)

    initial_loss = model.loss(X, y)
    model.fit(X, y, epochs=250, batch_size=4, learning_rate=0.3, verbose=False)
    final_loss = model.loss(X, y)

    assert final_loss < initial_loss * 0.8
