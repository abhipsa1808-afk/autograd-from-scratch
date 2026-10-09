"""Tests for minigrad.nn: layers, loss functions, and a real training run."""
import numpy as np

from minigrad.nn import Linear, ReLU, Sequential, Tanh, cross_entropy, mse_loss
from minigrad.tensor import Tensor


def numerical_grad(f, x, eps=1e-6):
    """Estimate df/dx by nudging each entry of x up and down a tiny bit."""
    grad = np.zeros_like(x)
    for i in np.ndindex(*x.shape):
        saved = x[i]
        x[i] = saved + eps
        up = float(f().data)
        x[i] = saved - eps
        down = float(f().data)
        x[i] = saved
        grad[i] = (up - down) / (2 * eps)
    return grad


def test_linear_output_shape_and_parameters():
    layer = Linear(3, 5, np.random.default_rng(0))
    out = layer(Tensor(np.ones((7, 3))))
    assert out.shape == (7, 5)
    assert [p.shape for p in layer.parameters()] == [(3, 5), (5,)]


def test_sequential_collects_all_parameters():
    rng = np.random.default_rng(0)
    model = Sequential(Linear(2, 4, rng), ReLU(), Linear(4, 3, rng), Tanh())
    assert len(model.parameters()) == 4
    assert model(Tensor(np.ones((6, 2)))).shape == (6, 3)


def test_zero_grad_clears_old_gradients():
    model = Sequential(Linear(2, 3, np.random.default_rng(0)), Tanh())
    model(Tensor(np.ones((4, 2)))).sum().backward()
    assert any(np.abs(p.grad).sum() > 0 for p in model.parameters())
    model.zero_grad()
    assert all(np.abs(p.grad).sum() == 0 for p in model.parameters())


def test_mse_loss_gradient():
    rng = np.random.default_rng(0)
    prediction = Tensor(rng.normal(size=(4, 2)))
    target = rng.normal(size=(4, 2))
    mse_loss(prediction, target).backward()
    expected = numerical_grad(lambda: mse_loss(prediction, target), prediction.data)
    np.testing.assert_allclose(prediction.grad, expected, rtol=1e-5, atol=1e-7)


def test_cross_entropy_value_for_equal_logits():
    loss = cross_entropy(Tensor(np.zeros((4, 3))), [0, 1, 2, 0])
    np.testing.assert_allclose(loss.data, np.log(3))


def test_cross_entropy_gradient():
    logits = Tensor(np.random.default_rng(0).normal(size=(5, 3)) * 3)
    labels = [0, 2, 1, 1, 0]
    cross_entropy(logits, labels).backward()
    expected = numerical_grad(lambda: cross_entropy(logits, labels), logits.data)
    np.testing.assert_allclose(logits.grad, expected, rtol=1e-5, atol=1e-7)


def test_cross_entropy_is_stable_for_huge_logits():
    loss = cross_entropy(Tensor([[1000.0, 0.0], [0.0, 1000.0]]), [0, 1])
    assert np.isfinite(loss.data)
    assert loss.data < 1e-6


def test_network_learns_xor():
    rng = np.random.default_rng(0)
    x = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0.0], [1.0], [1.0], [0.0]])
    model = Sequential(Linear(2, 8, rng), Tanh(), Linear(8, 1, rng))

    for _ in range(300):
        loss = mse_loss(model(x), y)
        model.zero_grad()
        loss.backward()
        for p in model.parameters():
            p.data -= 0.1 * p.grad

    assert loss.data < 0.01
    np.testing.assert_allclose(model(x).data, y, atol=0.1)
