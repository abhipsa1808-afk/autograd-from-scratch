"""Gradient checks: compare minigrad's gradients with numerical estimates."""
import numpy as np

from minigrad.tensor import Tensor


def numerical_grad(f, x, eps=1e-6):
    """Estimate df/dx by nudging each entry of x up and down a tiny bit."""
    grad = np.zeros_like(x)
    for i in np.ndindex(*x.shape):
        saved = x[i]
        x[i] = saved + eps
        up = f().data.sum()
        x[i] = saved - eps
        down = f().data.sum()
        x[i] = saved
        grad[i] = (up - down) / (2 * eps)
    return grad


def check(build, *shapes, positive=False):
    """Build a graph from random tensors and check every gradient numerically."""
    rng = np.random.default_rng(0)
    tensors = []
    for shape in shapes:
        values = rng.uniform(0.5, 2.0, shape) if positive else rng.normal(size=shape)
        tensors.append(Tensor(values))

    build(*tensors).sum().backward()

    for t in tensors:
        expected = numerical_grad(lambda: build(*tensors), t.data)
        np.testing.assert_allclose(t.grad, expected, rtol=1e-5, atol=1e-7)


def test_add():
    check(lambda a, b: a + b, (3, 4), (3, 4))


def test_mul():
    check(lambda a, b: a * b, (3, 4), (3, 4))


def test_sub():
    check(lambda a, b: a - b, (3, 4), (3, 4))


def test_div():
    check(lambda a, b: a / b, (3, 4), (3, 4), positive=True)


def test_pow():
    check(lambda a: a**3, (3, 4))


def test_neg():
    check(lambda a: -a, (3, 4))


def test_numbers_on_the_left():
    check(lambda a: 2.0 * a + 1.0 - a, (3, 4))
    check(lambda a: 1.0 / a, (3, 4), positive=True)


def test_matmul():
    check(lambda a, b: a @ b, (3, 4), (4, 2))


def test_broadcast_add():
    check(lambda a, b: a + b, (3, 4), (4,))


def test_broadcast_mul():
    check(lambda a, b: a * b, (3, 4), (3, 1))


def test_sum_axis():
    check(lambda a: a.sum(axis=0) * a.sum(axis=1, keepdims=True), (3, 4))


def test_mean():
    check(lambda a: a.mean(), (3, 4))
    check(lambda a: a.mean(axis=1), (3, 4))


def test_relu():
    check(lambda a: a.relu(), (3, 4))


def test_tanh():
    check(lambda a: a.tanh(), (3, 4))


def test_exp():
    check(lambda a: a.exp(), (3, 4))


def test_log():
    check(lambda a: a.log(), (3, 4), positive=True)


def test_tensor_used_twice():
    check(lambda a: a * a + a, (3, 4))


def test_tiny_network():
    def network(x, w1, b1, w2, b2):
        hidden = (x @ w1 + b1).relu()
        return ((hidden @ w2 + b2).tanh() ** 2).mean()

    check(network, (5, 3), (3, 4), (4,), (4, 1), (1,))


def test_long_chain_does_not_hit_recursion_limit():
    x = Tensor(1.0)
    y = x
    for _ in range(5000):
        y = y + x
    y.backward()
    assert x.grad == 5001.0
