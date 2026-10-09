"""minigrad.datasets: load Fashion-MNIST from its original compressed files."""
import gzip
from pathlib import Path

import numpy as np

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def _read_images(path):
    """Each image is 28x28 grey pixels; return them as rows of 784 numbers in 0..1."""
    with gzip.open(path, "rb") as f:
        pixels = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    return pixels.reshape(-1, 784).astype(np.float32) / 255.0


def _read_labels(path):
    with gzip.open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.uint8, offset=8).astype(np.int64)


def load_fashion_mnist(folder="data"):
    """Return (train_images, train_labels, test_images, test_labels)."""
    folder = Path(folder)
    return (
        _read_images(folder / "train-images-idx3-ubyte.gz"),
        _read_labels(folder / "train-labels-idx1-ubyte.gz"),
        _read_images(folder / "t10k-images-idx3-ubyte.gz"),
        _read_labels(folder / "t10k-labels-idx1-ubyte.gz"),
    )
