"""Streamlit demo for minigrad, a neural network framework written from scratch."""
import json
from pathlib import Path

import numpy as np
import streamlit as st

from minigrad.datasets import CLASS_NAMES
from minigrad.nn import Linear, ReLU, Sequential, Tanh, cross_entropy
from minigrad.optim import SGD, Adam
from minigrad.tensor import Tensor

ROOT = Path(__file__).parent
WEIGHTS = ROOT / "weights"

EXPRESSIONS = {
    "a * b + a": lambda a, b: a * b + a,
    "(a + b) ** 2": lambda a, b: (a + b) ** 2,
    "a / b": lambda a, b: a / b,
    "tanh(a * b)": lambda a, b: (a * b).tanh(),
    "relu(a - b) * a": lambda a, b: (a - b).relu() * a,
}

OPTIMIZERS = {
    "Adam": lambda params, lr: Adam(params, lr=lr),
    "SGD + momentum": lambda params, lr: SGD(params, lr=lr, momentum=0.9),
    "SGD": lambda params, lr: SGD(params, lr=lr),
}


@st.cache_resource
def load_classifier():
    model = Sequential(Linear(784, 128), ReLU(), Linear(128, 10))
    saved = np.load(WEIGHTS / "fashion_mlp.npz")
    for i, p in enumerate(model.parameters()):
        p.data = saved[f"arr_{i}"].astype(np.float64)
    return model


@st.cache_data
def load_samples():
    saved = np.load(WEIGHTS / "demo_samples.npz")
    return saved["images"], saved["labels"]


def softmax(logits):
    shifted = logits - logits.max()
    return np.exp(shifted) / np.exp(shifted).sum()


def make_spirals(points_per_class=100, seed=0):
    rng = np.random.default_rng(seed)
    radius = np.linspace(0.1, 1.0, points_per_class)
    xs, ys = [], []
    for label in (0, 1):
        angle = radius * 4.0 + label * np.pi + rng.normal(0, 0.15, points_per_class)
        xs.append(np.column_stack([radius * np.cos(angle), radius * np.sin(angle)]))
        ys.append(np.full(points_per_class, label))
    return np.concatenate(xs), np.concatenate(ys)


def train_spirals(optimizer_name, lr, hidden, steps):
    x, y = make_spirals()
    rng = np.random.default_rng(0)
    model = Sequential(
        Linear(2, hidden, rng), Tanh(), Linear(hidden, hidden, rng), Tanh(), Linear(hidden, 2, rng)
    )
    optimizer = OPTIMIZERS[optimizer_name](model.parameters(), lr)
    losses = []
    for _ in range(steps):
        loss = cross_entropy(model(Tensor(x)), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.data))
    accuracy = float((model(Tensor(x)).data.argmax(axis=1) == y).mean())
    return model, x, y, losses, accuracy


def boundary_image(model, x, y, size=120, scale=3):
    """Colour every point of the plane by the class the network predicts there."""
    axis = np.linspace(-1.2, 1.2, size)
    grid = np.column_stack([g.ravel() for g in np.meshgrid(axis, axis)])
    predicted = model(Tensor(grid)).data.argmax(axis=1).reshape(size, size)
    image = np.where(predicted[..., None] == 0, [255, 214, 196], [196, 222, 255]).astype(np.uint8)
    cols = np.clip(((x[:, 0] + 1.2) / 2.4 * (size - 1)).round().astype(int), 1, size - 2)
    rows = np.clip(((x[:, 1] + 1.2) / 2.4 * (size - 1)).round().astype(int), 1, size - 2)
    for row, col, label in zip(rows, cols, y):
        image[row - 1 : row + 2, col - 1 : col + 2] = [200, 60, 30] if label == 0 else [30, 90, 200]
    return np.kron(image, np.ones((scale, scale, 1), dtype=np.uint8))


def gradients(expression, a_value, b_value, eps=1e-6):
    f = EXPRESSIONS[expression]
    a, b = Tensor(a_value), Tensor(b_value)
    out = f(a, b)
    out.backward()
    check_a = (f(Tensor(a_value + eps), b).data - f(Tensor(a_value - eps), b).data) / (2 * eps)
    check_b = (f(a, Tensor(b_value + eps)).data - f(a, Tensor(b_value - eps)).data) / (2 * eps)
    return float(out.data), float(a.grad), float(b.grad), float(check_a), float(check_b)


st.set_page_config(page_title="minigrad", layout="centered")
st.title("minigrad")
st.caption(
    "A neural network framework written from scratch in NumPy: automatic differentiation, "
    "layers, optimisers. No PyTorch or TensorFlow is used anywhere on this page."
)

tab_classify, tab_train, tab_autograd, tab_compare = st.tabs(
    ["Classify clothing", "Train a network live", "Autograd playground", "minigrad vs PyTorch"]
)

with tab_classify:
    st.write("A network trained with minigrad on Fashion-MNIST looks at a photo it has never seen.")
    images, labels = load_samples()
    index = st.slider("Pick a test image", 0, len(images) - 1, 0)
    probabilities = softmax(load_classifier()(Tensor(images[index][None] / 255.0)).data[0])
    guess = int(probabilities.argmax())
    left, right = st.columns(2)
    with left:
        picture = np.kron(images[index].reshape(28, 28), np.ones((8, 8), dtype=np.uint8))
        st.image(picture, width=224)
        st.write(f"True label: **{CLASS_NAMES[int(labels[index])]}**")
    with right:
        verdict = "correct" if guess == int(labels[index]) else "wrong"
        st.write(f"Prediction: **{CLASS_NAMES[guess]}** ({verdict})")
        for i in np.argsort(probabilities)[::-1][:3]:
            st.progress(float(probabilities[i]), text=f"{CLASS_NAMES[int(i)]}: {probabilities[i]:.1%}")

with tab_train:
    st.write("Train a small network on two interleaved spirals, right here, using minigrad.")
    optimizer_name = st.selectbox("Optimiser", list(OPTIMIZERS))
    lr = st.select_slider("Learning rate", options=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3], value=0.03)
    hidden = st.slider("Hidden units per layer", 4, 32, 16)
    steps = st.slider("Training steps", 50, 1000, 400, step=50)
    if st.button("Train"):
        model, x, y, losses, accuracy = train_spirals(optimizer_name, lr, hidden, steps)
        left, right = st.columns(2)
        with left:
            st.image(boundary_image(model, x, y), width=320)
        with right:
            st.metric("Accuracy", f"{accuracy:.1%}")
            st.metric("Final loss", f"{losses[-1]:.4f}")
        st.line_chart(losses)

with tab_autograd:
    st.write("minigrad computes derivatives automatically. A numerical estimate confirms each one.")
    expression = st.selectbox("Expression", list(EXPRESSIONS))
    a_value = st.number_input("a", value=2.0, step=0.5)
    b_value = st.number_input("b", value=3.0, step=0.5)
    if b_value == 0 and "/" in expression:
        st.warning("b must not be zero for this expression.")
    else:
        value, grad_a, grad_b, check_a, check_b = gradients(expression, a_value, b_value)
        st.metric("Value", f"{value:.6f}")
        st.markdown(
            "| | minigrad | numerical check |\n|---|---|---|\n"
            f"| d/da | {grad_a:.6f} | {check_a:.6f} |\n"
            f"| d/db | {grad_b:.6f} | {check_b:.6f} |"
        )

with tab_compare:
    st.write("The same network, data, starting weights and batch order, trained in both frameworks.")
    result = json.loads((WEIGHTS / "comparison.json").read_text())
    st.markdown(
        f"| | minigrad | PyTorch {result['torch_version']} |\n|---|---|---|\n"
        f"| Test accuracy | {result['minigrad_test_accuracy']:.2%} "
        f"| {result['pytorch_test_accuracy']:.2%} |\n"
        f"| Training time | {result['minigrad_seconds']} s | {result['pytorch_seconds']} s |"
    )
    st.write(
        f"Largest difference in loss across {result['steps']:,} training steps: "
        f"**{result['max_loss_gap_all_steps']:.1e}**, the rounding limit of 64-bit numbers."
    )
