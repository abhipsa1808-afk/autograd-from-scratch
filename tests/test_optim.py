"""Tests for minigrad.optim: update rules checked by hand, then real convergence."""
import numpy as np

from minigrad.nn import Linear, Sequential, Tanh, mse_loss
from minigrad.optim import SGD, Adam
from minigrad.tensor import Tensor


def test_sgd_step_matches_formula():
    p = Tensor([1.0, 2.0])
    p.grad = np.array([0.5, -1.0])
    SGD([p], lr=0.1).step()
    np.testing.assert_allclose(p.data, [0.95, 2.1])


def test_sgd_momentum_builds_up_speed():
    p = Tensor([1.0])
    optimizer = SGD([p], lr=0.1, momentum=0.9)
    p.grad = np.array([1.0])
    optimizer.step()  # velocity = 1.0, moves 0.1
    optimizer.step()  # velocity = 1.9, moves 0.19
    np.testing.assert_allclose(p.data, [1.0 - 0.1 - 0.19])


def test_adam_first_step_has_size_lr():
    p = Tensor([1.0, 1.0])
    p.grad = np.array([0.001, -500.0])
    Adam([p], lr=0.01).step()
    np.testing.assert_allclose(p.data, [0.99, 1.01], atol=1e-6)


def test_zero_grad_clears_gradients():
    p = Tensor([1.0, 2.0])
    p.grad = np.array([3.0, 4.0])
    SGD([p]).zero_grad()
    np.testing.assert_allclose(p.grad, [0.0, 0.0])


def test_every_optimizer_finds_the_minimum():
    makers = [
        lambda p: SGD(p, lr=0.1),
        lambda p: SGD(p, lr=0.05, momentum=0.9),
        lambda p: Adam(p, lr=0.1),
    ]
    for make_optimizer in makers:
        x = Tensor([0.0, 10.0])
        optimizer = make_optimizer([x])
        for _ in range(500):
            loss = ((x - 3.0) ** 2).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        np.testing.assert_allclose(x.data, [3.0, 3.0], atol=1e-3)


def test_adam_trains_xor():
    rng = np.random.default_rng(0)
    x = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0.0], [1.0], [1.0], [0.0]])
    model = Sequential(Linear(2, 8, rng), Tanh(), Linear(8, 1, rng))
    optimizer = Adam(model.parameters(), lr=0.05)

    for _ in range(100):
        loss = mse_loss(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert loss.data < 0.001
