"""Compare minigrad with PyTorch: same network, same data, same starting weights."""
import json
import time
from pathlib import Path

import numpy as np
import torch

from minigrad.datasets import load_fashion_mnist
from minigrad.nn import cross_entropy
from minigrad.optim import Adam
from minigrad.tensor import Tensor
from train import BATCH_SIZE, EPOCHS, LEARNING_RATE, accuracy, build_model


def train_minigrad(x, y, orders):
    model = build_model(np.random.default_rng(0))
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    losses = []
    started = time.time()
    for order in orders:
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start : start + BATCH_SIZE]
            loss = cross_entropy(model(Tensor(x[batch])), y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.data))
    return model, np.array(losses), time.time() - started


def build_pytorch_model():
    """The same network in PyTorch, starting from the same weights as minigrad."""
    start = [p.data for p in build_model(np.random.default_rng(0)).parameters()]
    model = torch.nn.Sequential(
        torch.nn.Linear(784, 128), torch.nn.ReLU(), torch.nn.Linear(128, 10)
    ).double()
    with torch.no_grad():
        # PyTorch stores Linear weights as (outputs, inputs), so transpose ours.
        model[0].weight.copy_(torch.from_numpy(np.ascontiguousarray(start[0].T)))
        model[0].bias.copy_(torch.from_numpy(start[1].copy()))
        model[2].weight.copy_(torch.from_numpy(np.ascontiguousarray(start[2].T)))
        model[2].bias.copy_(torch.from_numpy(start[3].copy()))
    return model


def train_pytorch(x, y, orders):
    model = build_pytorch_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    losses = []
    started = time.time()
    for order in orders:
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start : start + BATCH_SIZE]
            inputs = torch.from_numpy(x[batch].astype(np.float64))
            targets = torch.from_numpy(y[batch])
            loss = torch.nn.functional.cross_entropy(model(inputs), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    return model, np.array(losses), time.time() - started


def pytorch_accuracy(model, images, labels):
    with torch.no_grad():
        logits = model(torch.from_numpy(images.astype(np.float64)))
    return float((logits.argmax(dim=1).numpy() == labels).mean())


def main():
    x_train, y_train, x_test, y_test = load_fashion_mnist()
    rng = np.random.default_rng(1)
    orders = [rng.permutation(len(x_train)) for _ in range(EPOCHS)]

    mini_model, mini_losses, mini_seconds = train_minigrad(x_train, y_train, orders)
    torch_model, torch_losses, torch_seconds = train_pytorch(x_train, y_train, orders)

    gap = np.abs(mini_losses - torch_losses)
    results = {
        "torch_version": torch.__version__,
        "steps": len(mini_losses),
        "minigrad_test_accuracy": round(accuracy(mini_model, x_test, y_test), 4),
        "pytorch_test_accuracy": round(pytorch_accuracy(torch_model, x_test, y_test), 4),
        "minigrad_seconds": round(mini_seconds, 2),
        "pytorch_seconds": round(torch_seconds, 2),
        "max_loss_gap_first_100_steps": float(gap[:100].max()),
        "max_loss_gap_all_steps": float(gap.max()),
    }

    print(f"PyTorch {results['torch_version']} | {results['steps']} training steps each")
    print(f"{'':<16}{'minigrad':>10}{'PyTorch':>10}")
    print(
        f"{'test accuracy':<16}{results['minigrad_test_accuracy']:>10.2%}"
        f"{results['pytorch_test_accuracy']:>10.2%}"
    )
    print(
        f"{'training time':<16}{results['minigrad_seconds']:>9.2f}s"
        f"{results['pytorch_seconds']:>9.2f}s"
    )
    print(f"largest loss gap, first 100 steps: {results['max_loss_gap_first_100_steps']:.2e}")
    print(f"largest loss gap, all steps      : {results['max_loss_gap_all_steps']:.2e}")

    Path("weights").mkdir(exist_ok=True)
    Path("weights/comparison.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
