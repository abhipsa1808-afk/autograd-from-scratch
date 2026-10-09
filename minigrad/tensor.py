"""minigrad.tensor: a NumPy-backed Tensor with reverse-mode automatic differentiation."""
import numpy as np


class Tensor:
    def __init__(self, data, _parents=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._parents = _parents
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Tensor({self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def backward(self):
        order, seen = [], set()

        def visit(t):
            if t not in seen:
                seen.add(t)
                for p in t._parents:
                    visit(p)
                order.append(t)

        visit(self)
        self.grad = np.ones_like(self.data)
        for t in reversed(order):
            t._backward()
