"""minigrad.optim: optimisers that update parameters using their gradients."""
import numpy as np


class Optimizer:
    def __init__(self, parameters):
        self.parameters = list(parameters)

    def zero_grad(self):
        for p in self.parameters:
            p.grad = np.zeros_like(p.data)


class SGD(Optimizer):
    """Stochastic gradient descent, with optional momentum."""

    def __init__(self, parameters, lr=0.01, momentum=0.0):
        super().__init__(parameters)
        self.lr = lr
        self.momentum = momentum
        self.velocity = [np.zeros_like(p.data) for p in self.parameters]

    def step(self):
        for p, v in zip(self.parameters, self.velocity):
            v *= self.momentum
            v += p.grad
            p.data -= self.lr * v


class Adam(Optimizer):
    """Adam: adapts the step size for every parameter separately."""

    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.parameters]
        self.v = [np.zeros_like(p.data) for p in self.parameters]

    def step(self):
        self.t += 1
        for p, m, v in zip(self.parameters, self.m, self.v):
            m *= self.beta1
            m += (1 - self.beta1) * p.grad
            v *= self.beta2
            v += (1 - self.beta2) * p.grad**2
            m_hat = m / (1 - self.beta1**self.t)
            v_hat = v / (1 - self.beta2**self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
