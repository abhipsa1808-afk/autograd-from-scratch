"""minigrad.nn: neural network layers and loss functions built on Tensor."""
import numpy as np

from minigrad.tensor import Tensor


class Module:
    """Base class for layers: something you can call that may hold parameters."""

    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

    def __call__(self, x):
        return self.forward(x)


class Linear(Module):
    """Fully connected layer: y = x @ W + b."""

    def __init__(self, in_features, out_features, rng=None):
        rng = rng or np.random.default_rng()
        scale = np.sqrt(2.0 / in_features)  # He initialisation
        self.weight = Tensor(rng.normal(0.0, scale, (in_features, out_features)))
        self.bias = Tensor(np.zeros(out_features))

    def forward(self, x):
        return x @ self.weight + self.bias

    def parameters(self):
        return [self.weight, self.bias]


class ReLU(Module):
    def forward(self, x):
        return x.relu()


class Tanh(Module):
    def forward(self, x):
        return x.tanh()


class Sequential(Module):
    """Runs layers one after another."""

    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


def mse_loss(prediction, target):
    """Mean squared error, for predicting numbers."""
    return ((prediction - target) ** 2).mean()


def cross_entropy(logits, labels):
    """Softmax + negative log-likelihood, for predicting one class out of many."""
    labels = np.asarray(labels, dtype=np.int64)
    rows = np.arange(logits.shape[0])
    shifted = logits.data - logits.data.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    out = Tensor(-log_probs[rows, labels].mean(), (logits,), "cross_entropy")

    def _backward():
        grad = np.exp(log_probs)
        grad[rows, labels] -= 1.0
        logits.grad += grad / len(rows) * out.grad

    out._backward = _backward
    return out
