# minigrad

A small deep learning framework written from scratch in NumPy. It has its own automatic differentiation engine, neural network layers, loss functions and optimisers. It trains a real image classifier and matches PyTorch to 15 decimal places at every training step.

No PyTorch or TensorFlow is used inside the framework. PyTorch appears in one script only, as the reference to compare against.

## Results

The same network (784 → 128 → 10) was trained in both frameworks on Fashion-MNIST, with the same starting weights, the same batches in the same order, and the same Adam settings.

| | minigrad | PyTorch 2.14.1 |
|---|---|---|
| Test accuracy (10,000 unseen images) | 87.76% | 87.76% |
| Training time, 5 epochs on CPU | 2.05 s | 1.55 s |

The largest difference in loss between the two frameworks, at any of the 4,690 training steps, was **6.66e-16**. That is the rounding limit of 64-bit floating point numbers, so the two frameworks compute the same thing.

Measured on a MacBook Air (Apple Silicon) with Python 3.12 and NumPy 2.5.3. Reproduce it with `python compare_pytorch.py`.

## Example

Gradients are computed automatically:

~~~python
from minigrad.tensor import Tensor

a, b = Tensor(2.0), Tensor(3.0)
c = a * b + a
c.backward()
print(a.grad, b.grad)  # 4.0 2.0
~~~

A full training loop looks like PyTorch:

~~~python
from minigrad.nn import Linear, ReLU, Sequential, cross_entropy
from minigrad.optim import Adam
from minigrad.tensor import Tensor

model = Sequential(Linear(784, 128), ReLU(), Linear(128, 10))
optimizer = Adam(model.parameters(), lr=0.001)

for images, labels in batches:
    loss = cross_entropy(model(Tensor(images)), labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
~~~

## What is inside

| File | What it does |
|---|---|
| `minigrad/tensor.py` | `Tensor` with reverse-mode automatic differentiation: `+ - * / ** @`, `sum`, `mean`, `relu`, `tanh`, `exp`, `log`, with NumPy broadcasting |
| `minigrad/nn.py` | `Linear`, `ReLU`, `Tanh`, `Sequential`, `mse_loss`, `cross_entropy` |
| `minigrad/optim.py` | `SGD` (with momentum) and `Adam` |
| `minigrad/datasets.py` | Loader for the original Fashion-MNIST files |
| `train.py` | Trains the classifier and saves its weights |
| `compare_pytorch.py` | Trains the same network in minigrad and PyTorch and compares them |
| `app.py` | Streamlit demo |
| `tests/` | 33 tests |

## How it works

Every `Tensor` remembers which tensors it was computed from, together with a small function that passes a gradient back to them. Calling `backward()` sorts this graph so that each tensor comes after everything that depends on it, then applies the chain rule once per tensor, from the output back to the inputs.

Three details took the most care:

- **Broadcasting.** When NumPy stretches a small array to match a larger one, for example a bias added to every row, the gradient has to be summed back down to the small shape.
- **Deep graphs.** The graph is sorted with a loop instead of recursion, so depth is not limited by Python's recursion limit. A test runs a chain of 5,000 operations.
- **Numerical stability.** `cross_entropy` computes log-softmax in a way that cannot overflow. A test feeds it logits of 1,000.

## How I know it is correct

1. **Gradient checks.** For every operation, the gradient from minigrad is compared with a numerical estimate obtained by nudging each input slightly and measuring the change in the output.
2. **Update rules checked by hand.** The SGD, momentum and Adam steps are tested against values worked out on paper.
3. **PyTorch, step by step.** The comparison above: identical losses for 4,690 steps in a row.

Run all of them with `python -m pytest -q`.

## Optimisers compared

The same small network on XOR, from the same starting weights, after 60 steps:

| Optimiser | Loss |
|---|---|
| SGD | 0.06174 |
| SGD with momentum | 0.00038 |
| Adam | 0.00016 |

## Demo

`streamlit run app.py` opens a page with four tabs:

- **Classify clothing**: the trained network labels test photos it has never seen.
- **Train a network live**: train on two interleaved spirals, choose the optimiser, and see the decision boundary.
- **Autograd playground**: pick an expression and compare minigrad's gradients with a numerical check.
- **minigrad vs PyTorch**: the results above.

## Run it yourself

~~~bash
git clone https://github.com/abhipsa1808-afk/autograd-from-scratch.git
cd autograd-from-scratch
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest -q
streamlit run app.py
~~~

The trained weights are included, so the demo works straight away. To train again, download the dataset first:

~~~bash
mkdir -p data
for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
  curl -L --fail -o data/$f.gz https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/$f.gz
done
python train.py
python compare_pytorch.py
~~~

## Limitations

- CPU only, and fully connected layers only. There are no convolutions.
- About 1.3 times slower than PyTorch on this small network. On larger models the gap would be much wider, because PyTorch has optimised kernels and GPU support.
- The accuracy is what a small fully connected network reaches on this dataset. Convolutional networks do better.
- Accuracy moves by about one percentage point with the order in which images are shuffled: `train.py` reaches 86.88% and the comparison run reaches 87.76%.

## Licence

MIT. See [LICENSE](LICENSE).
