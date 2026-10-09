"""Train a neural network on Fashion-MNIST using only minigrad (no PyTorch)."""
import json
import time
from pathlib import Path

import numpy as np

from minigrad.datasets import load_fashion_mnist
from minigrad.nn import Linear, ReLU, Sequential, cross_entropy
from minigrad.optim import Adam
from minigrad.tensor import Tensor

EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.001


def build_model(rng=None):
    return Sequential(Linear(784, 128, rng), ReLU(), Linear(128, 10, rng))


def accuracy(model, images, labels, batch_size=1000):
    correct = 0
    for start in range(0, len(images), batch_size):
        logits = model(Tensor(images[start : start + batch_size]))
        correct += (logits.data.argmax(axis=1) == labels[start : start + batch_size]).sum()
    return float(correct / len(images))


def main():
    rng = np.random.default_rng(0)
    x_train, y_train, x_test, y_test = load_fashion_mnist()
    model = build_model(rng)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    history = []

    for epoch in range(1, EPOCHS + 1):
        started = time.time()
        order = rng.permutation(len(x_train))
        losses = []
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start : start + BATCH_SIZE]
            loss = cross_entropy(model(Tensor(x_train[batch])), y_train[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.data))

        row = {
            "epoch": epoch,
            "loss": round(float(np.mean(losses)), 4),
            "train_accuracy": round(accuracy(model, x_train, y_train), 4),
            "test_accuracy": round(accuracy(model, x_test, y_test), 4),
            "seconds": round(time.time() - started, 1),
        }
        history.append(row)
        print(
            f"epoch {epoch} | loss {row['loss']:.4f} | train {row['train_accuracy']:.2%}"
            f" | test {row['test_accuracy']:.2%} | {row['seconds']}s"
        )

    Path("weights").mkdir(exist_ok=True)
    weights = [p.data.astype(np.float32) for p in model.parameters()]
    np.savez("weights/fashion_mlp.npz", *weights)
    Path("weights/history.json").write_text(json.dumps(history, indent=2))
    print(f"final test accuracy: {history[-1]['test_accuracy']:.2%} (saved to weights/)")


if __name__ == "__main__":
    main()
