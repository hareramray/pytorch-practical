"""Authoring source for the course. Regeneration overwrites notebooks, not learner copies."""
from pathlib import Path
import json
import textwrap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"
OUT.mkdir(exist_ok=True)
COURSE = []


def clean(s):
    return textwrap.dedent(s).strip() + "\n"


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": clean(s).splitlines(True)}


def code(s, solution=False):
    return {"cell_type": "code", "execution_count": None, "outputs": [],
            "metadata": {"tags": ["solution"], "jupyter": {"source_hidden": True}} if solution else {},
            "source": clean(s).splitlines(True)}


SETUP = '''
import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(2)  # Small examples often run faster with fewer CPU threads.
device = torch.device("cpu")  # Learn on CPU first; the performance lab covers CUDA.
plt.rcParams.update({"figure.figsize": (7, 4), "axes.grid": True})
print("PyTorch:", torch.__version__, "| Device:", device)
'''


def notebook(slug, title, level, mins, prereq, goals, steps, exercises, solution, takeaway, refs):
    idx = len(COURSE)
    cells = [md(f'''# {idx:02d} — {title}

**Level:** {level} · **Study time:** {mins} minutes (including experiments)

**Prerequisites:** {prereq}

## What you will be able to do
{goals}

## How to work through this lab
Run cells in order with **Shift+Enter**. Before each result, predict its shape or behavior.
Change one thing, rerun, and explain what changed. Use **Restart Kernel and Run All**
when you want a clean experiment. Lessons use local data and do not depend on earlier kernel state.
Record your observations in the experiment journal near the end.
'''), md("## Step 1 — Prepare a reproducible experiment\n\nA seed makes comparisons easier. Exact numerical results can still differ across devices and PyTorch versions."), code(SETUP)]
    for i, (heading, explanation, source) in enumerate(steps, 2):
        cells += [md(f"## Step {i} — {heading}\n\n{explanation}"), code(source)]
    cells += [md(f'''## Your turn — practice before opening the solutions

{exercises}

Use the empty cell below for your attempt. The worked examples above run as supplied;
you do not need to fill in exercises to use **Run All**. Solutions are below the journal
and their code is initially collapsed in JupyterLab.
'''), code("# Your experiment goes here. Add more cells if you need them.\n"),
        md('''## Experiment journal

| Question | Your notes |
|---|---|
| What did I predict? | |
| What single setting did I change? | |
| What result did I measure? | |
| Why might it have changed? | |
| What would I try next? | |

## Reference solutions
Try the exercises first. Expand the following cell to compare your reasoning.
'''), code(solution, solution=True), md(f'''## Before moving on

{takeaway}

**Explain it aloud:** What went into the model or operation, what came out, and how did you verify it?

## Official references and further reading
{refs}
''')]
    filename = f"{idx:02d}_{slug}.ipynb"
    COURSE.append(dict(filename=filename, title=title, level=level, mins=mins, cells=cells))


notebook("start_here", "Your first PyTorch experiment", "Beginner", "30–45", "Basic Python: variables, lists, functions, and loops.",
    "- Run and edit a notebook cell.\n- Recognize tensors, predictions, loss, and learning.\n- Check your environment before training.", [
    ("Check your tools", "A notebook connects a document to a running Python process called a **kernel**. Its variables stay in memory until you restart it. If an import fails, select the course kernel described in the README.", '''
import sys
print("Python executable:", sys.executable)
print("Python version:", sys.version.split()[0])
print("CUDA GPU available:", torch.cuda.is_available())
assert int(torch.__version__.split('.')[0]) >= 2
'''),
    ("Turn numbers into a tensor", "A tensor is a collection of numbers with a shape. Here each row is one observation. We use floating point numbers because learning updates usually involve fractions.", '''
hours = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
scores = 3 * hours + 2
print("Input shape:", hours.shape, "dtype:", hours.dtype)
print("Targets:", scores.squeeze().tolist())
assert hours.shape == (4, 1)
'''),
    ("Make a prediction and measure its error", "Our first model is a rule: prediction = weight × hours + bias. **Parameters** are the numbers we can adjust. **Loss** is one number summarizing the error; smaller is better for this metric.", '''
weight, bias = 1.0, 0.0
predictions = weight * hours + bias
loss = ((predictions - scores) ** 2).mean()
print("Predictions:", predictions.squeeze().tolist())
print("Mean squared error:", loss.item())
plt.scatter(hours[:, 0], scores[:, 0], label="target")
plt.plot(hours[:, 0], predictions[:, 0], label="our rule", color="orange")
plt.xlabel("Hours (toy example)"); plt.ylabel("Score"); plt.legend(); plt.show()
'''),
    ("Improve the rule manually", "Learning means adjusting parameters to reduce loss. Here you already know the generating rule, so you can do it manually. Later, autograd and an optimizer will do the arithmetic.", '''
better_predictions = 3.0 * hours + 2.0
better_loss = ((better_predictions - scores) ** 2).mean()
print("Before:", loss.item(), "After:", better_loss.item())
assert better_loss < loss
''')],
    "1. Predict the shape of a tensor containing three rows and two columns, then create it.\n2. Try weight = 2 and bias = 1. Is its loss smaller than our first rule?\n3. Write a function `mse(prediction, target)` and check it on the perfect predictions.",
    '''
matrix = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
assert matrix.shape == (3, 2)
trial_loss = ((2 * hours + 1 - scores) ** 2).mean()
assert trial_loss < loss
def mse(prediction, target):
    return ((prediction - target) ** 2).mean()
assert mse(better_predictions, scores).item() == 0
print("Trial MSE:", trial_loss.item())
''', "You should be able to distinguish **data**, **parameters**, **predictions**, and **loss**. No neural network is needed to understand these four pieces.",
    "- [PyTorch Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)\n- [JupyterLab notebooks](https://jupyterlab.readthedocs.io/en/stable/user/notebook.html)")

notebook("tensors", "Tensors, indexing, and devices", "Beginner", "45–60", "Notebook 00.",
    "- Create and index tensors.\n- Choose appropriate dtypes.\n- Understand copies, shared storage, and device placement.", [
    ("Create tensors and inspect them", "A scalar has zero dimensions, a vector has one, and a matrix has two. `shape` describes the axes; `dtype` describes the stored numbers; `device` describes where they live.", '''
scalar = torch.tensor(7.)
vector = torch.arange(5, dtype=torch.float32)
matrix = torch.arange(12, dtype=torch.float32).reshape(3, 4)
for name, t in [("scalar", scalar), ("vector", vector), ("matrix", matrix)]:
    print(name, "shape:", tuple(t.shape), "dtype:", t.dtype, "device:", t.device)
print(torch.zeros(2, 3))
print(torch.ones_like(matrix))
'''),
    ("Select, mask, and reduce", "Indexing selects values; reductions summarize them. For a matrix `[rows, columns]`, `mean(dim=0)` collapses the row axis and leaves one mean for each column.", '''
print("First row:", matrix[0])
print("Second column:", matrix[:, 1])
print("Last two rows:", matrix[-2:])
print("Values above 5:", matrix[matrix > 5])
print("Column means:", matrix.mean(dim=0))
print("Row means:", matrix.mean(dim=1))
assert matrix.mean(dim=0).shape == (4,)
'''),
    ("Distinguish a view from a copy", "Some operations share storage. Changing a view changes the original. Use `clone()` when you need independent data. `torch.from_numpy` also shares CPU memory with its NumPy array.", '''
original = torch.tensor([1., 2., 3.])
view = original[:2]
independent = original.clone()
view[0] = 99
print("Original:", original, "Copy:", independent)
array = np.array([1., 2.], dtype=np.float32)
shared = torch.from_numpy(array)
array[0] = 8
assert shared[0].item() == 8
assert independent[0].item() == 1
'''),
    ("Move data deliberately", "Operations generally require tensors on the same device. `.to(...)` returns the converted tensor, so keep the result. Class indices usually use `long`; learned numeric features usually use floating point.", '''
labels = torch.tensor([0, 1, 2], dtype=torch.long)
features = matrix.to(device=device, dtype=torch.float32)
numpy_copy = features.detach().cpu().numpy().copy()
print("Features:", features.dtype, "Labels:", labels.dtype)
assert np.allclose(numpy_copy, matrix.numpy())
''')],
    "1. Build a 4 × 3 tensor containing 0 to 11 and select its final column.\n2. Find each row's maximum value. Hint: `max(dim=...)` returns values and indices.\n3. Make an independent copy, set it to zero, and prove that the original is unchanged.",
    '''
exercise = torch.arange(12).reshape(4, 3)
assert exercise[:, -1].tolist() == [2, 5, 8, 11]
assert exercise.max(dim=1).values.tolist() == [2, 5, 8, 11]
copy = exercise.clone()
copy.zero_()
assert exercise.sum().item() == 66 and copy.sum().item() == 0
''', "Read shape, dtype, and device before debugging a tensor operation. An underscore suffix, such as `zero_()`, usually means the tensor is changed in place.",
    "- [Tensor tutorial](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)")

notebook("shapes_and_broadcasting", "Shapes, broadcasting, and matrix multiplication", "Beginner", "45–60", "Notebook 01.",
    "- Trace batch and feature axes.\n- Distinguish elementwise operations from matrix multiplication.\n- Prevent silent broadcasting errors.", [
    ("Write a layer as matrix arithmetic", "A batch `X` with shape `[B, F]` multiplied by weights `[F, H]` produces `[B, H]`. Adding a bias `[H]` repeats that bias across the batch without manually copying it.", '''
X = torch.randn(5, 3)
W = torch.randn(3, 2)
b = torch.tensor([1., -1.])
Y = X @ W + b
print("X:", X.shape, "W:", W.shape, "Y:", Y.shape)
assert Y.shape == (5, 2)
assert torch.allclose(Y[0], X[0] @ W + b)
'''),
    ("See the difference between multiply and matmul", "`*` multiplies corresponding elements, while `@` sums products along a shared axis. Broadcasting aligns dimensions from the right; dimensions must match or one must be 1.", '''
A = torch.tensor([[1., 2.], [3., 4.]])
print("Elementwise:", A * A)
print("Matrix multiply:", A @ A)
row = torch.tensor([[10., 20., 30.]])
column = torch.tensor([[1.], [2.]])
print("[1,3] + [2,1] -> [2,3]:", row + column)
'''),
    ("Catch a silent loss bug", "Subtracting predictions `[B,1]` and targets `[B]` creates `[B,B]`, not one error per example. Code can run and still solve the wrong problem. Check shapes before computing the loss.", '''
pred = torch.tensor([[1.], [2.], [3.]])
target = torch.tensor([1., 2., 3.])
wrong_error = pred - target
right_error = pred - target.unsqueeze(1)
print("Wrong shape:", wrong_error.shape, "Correct shape:", right_error.shape)
assert right_error.shape == pred.shape
assert right_error.square().mean() == 0
assert wrong_error.square().mean() > 0
'''),
    ("Reorder image axes", "PyTorch convolution layers expect `[batch, channels, height, width]`. `permute` reorders axes; `reshape` changes grouping. Reshaping cannot replace permuting because it does not express the same reordering.", '''
images_hwc = torch.arange(2 * 4 * 4 * 3).reshape(2, 4, 4, 3)
images_chw = images_hwc.permute(0, 3, 1, 2)
flat = images_chw.reshape(2, -1)
print("Image:", images_chw.shape, "Flat:", flat.shape)
assert images_chw[0, 2, 1, 3] == images_hwc[0, 1, 3, 2]
''')],
    "1. Standardize a `[10, 3]` batch separately for each feature. Hint: reduce over dim 0 with `keepdim=True`.\n2. Add a batch axis to a `[3, 8, 8]` image.\n3. Create batched matrices `[4, 2, 3]` and `[4, 3, 5]`; predict and verify the output shape.",
    '''
batch = torch.randn(10, 3)
z = (batch - batch.mean(0, keepdim=True)) / batch.std(0, keepdim=True).clamp_min(1e-6)
assert torch.allclose(z.mean(0), torch.zeros(3), atol=1e-6)
assert torch.randn(3, 8, 8).unsqueeze(0).shape == (1, 3, 8, 8)
assert (torch.randn(4, 2, 3) @ torch.randn(4, 3, 5)).shape == (4, 2, 5)
''', "Before executing a model, write the shape after every operation on paper. A successful operation is not proof that the shape is semantically correct.",
    "- [Broadcasting semantics](https://docs.pytorch.org/docs/stable/notes/broadcasting.html)")

notebook("autograd", "Autograd and gradient descent", "Beginner", "45–60", "Notebooks 01–02; slopes at an intuitive level.",
    "- Compute and check a derivative.\n- Explain gradient accumulation.\n- Minimize a simple function with gradient descent.", [
    ("Differentiate a scalar expression", "The derivative describes local sensitivity: how much the output changes when the input changes slightly. For `y = x² + 3x`, the derivative is `2x + 3`. Autograd records operations on tensors with `requires_grad=True`.", '''
x = torch.tensor(2., requires_grad=True)
y = x.square() + 3 * x
y.backward()
print("y:", y.item(), "dy/dx:", x.grad.item())
assert x.grad.item() == 7
'''),
    ("Check gradients numerically", "A centered finite difference estimates a derivative by evaluating nearby points. It is slower and approximate, but useful for checking your reasoning. We use double precision to reduce rounding error.", '''
def f(z):
    return z.square() + 3 * z
point = torch.tensor(2., dtype=torch.float64)
eps = 1e-5
numerical = (f(point + eps) - f(point - eps)) / (2 * eps)
assert abs(numerical.item() - x.grad.item()) < 1e-5
print("Finite difference:", numerical.item())
'''),
    ("Observe accumulation", "Calling backward adds to `.grad`. It does not replace the previous gradient. Recompute the forward pass to build a fresh graph; ordinary backward frees saved intermediate values.", '''
x.grad = None
f(x).backward()
first = x.grad.clone()
f(x).backward()  # A new forward graph, but the same leaf gradient buffer.
print("Once:", first.item(), "Twice:", x.grad.item())
assert torch.allclose(x.grad, 2 * first)
'''),
    ("Learn the minimum of a bowl", "Gradient descent moves opposite the derivative: `parameter -= learning_rate * gradient`. We update under `no_grad` so the update itself is not recorded as part of the next learning graph.", '''
parameter = torch.tensor(-4., requires_grad=True)
history = []
for step in range(60):
    loss = (parameter - 3).square()
    loss.backward()
    with torch.no_grad():
        parameter -= 0.1 * parameter.grad
    parameter.grad = None
    history.append(loss.item())
print("Learned minimum at:", parameter.item())
assert abs(parameter.item() - 3) < 1e-3
plt.plot(history); plt.yscale("log"); plt.xlabel("Step"); plt.ylabel("Loss"); plt.show()
''')],
    "1. Check the gradient of `x³` at x = 2.\n2. Minimize `(p + 2)²`, starting at 5.\n3. Predict what `.detach()` does to `requires_grad`, then verify it. Hint: a detached tensor shares storage but leaves the graph.",
    '''
q = torch.tensor(2., requires_grad=True)
q.pow(3).backward()
assert q.grad.item() == 12
p = torch.tensor(5., requires_grad=True)
for _ in range(80):
    (p + 2).square().backward()
    with torch.no_grad():
        p -= 0.1 * p.grad
    p.grad = None
assert abs(p.item() + 2) < 1e-4
assert not q.detach().requires_grad
''', "You should be able to explain why updates need gradients, why gradients must be reset, and why evaluation should avoid building a graph.",
    "- [Automatic differentiation](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)")

notebook("linear_regression", "Linear regression from scratch", "Beginner", "60–75", "Notebooks 00–03.",
    "- Train your first predictive model.\n- Separate training and test observations.\n- Compare learned parameters with a known data-generating rule.", [
    ("Generate noisy observations and split them", "We simulate `y = 3x + 2 + noise`. Noise prevents perfect prediction. Split before training: the test set estimates performance on observations that did not influence learning. Here random splitting fits the independent toy observations.", '''
X = torch.linspace(-2, 2, 240).unsqueeze(1)
y = 3 * X + 2 + 0.25 * torch.randn_like(X)
order = torch.randperm(len(X))
train_idx, test_idx = order[:180], order[180:]
X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]
plt.scatter(X_train[:, 0], y_train[:, 0], s=12, label="train")
plt.scatter(X_test[:, 0], y_test[:, 0], s=12, label="test")
plt.legend(); plt.show()
'''),
    ("Write every part of the training loop", "One iteration performs forward prediction, loss calculation, backward differentiation, and a parameter update. Loss is computed only on the training set. An **epoch** means one pass over all training examples; here every update uses the full training set.", '''
w = torch.randn(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
history = []
for epoch in range(180):
    prediction = X_train * w + b
    assert prediction.shape == y_train.shape
    loss = (prediction - y_train).square().mean()
    loss.backward()
    with torch.no_grad():
        w -= 0.08 * w.grad
        b -= 0.08 * b.grad
    w.grad = None
    b.grad = None
    history.append(loss.item())
print("Learned w, b:", w.item(), b.item())
assert history[-1] < history[0] * 0.1
'''),
    ("Evaluate against a baseline", "A model should improve over a simple alternative. Our baseline always predicts the training target mean. Compute it from training data only, then compare both methods on the same test set.", '''
with torch.no_grad():
    test_pred = X_test * w + b
    test_mse = (test_pred - y_test).square().mean().item()
    baseline_mse = (y_train.mean() - y_test).square().mean().item()
print({"test_MSE": test_mse, "mean_baseline_MSE": baseline_mse})
assert test_mse < baseline_mse
fig, ax = plt.subplots(1, 2, figsize=(10, 3))
ax[0].plot(history); ax[0].set(title="Learning curve", xlabel="Epoch", ylabel="Train MSE")
ax[1].scatter(X_test[:, 0], y_test[:, 0], s=12)
ax[1].plot(X[:, 0], (X * w + b).detach()[:, 0], color="orange")
ax[1].set(title="Held-out predictions", xlabel="x", ylabel="y")
plt.tight_layout(); plt.show()
''')],
    "1. Compute test mean absolute error (MAE).\n2. Fit a second model using `torch.optim.SGD` instead of manual updates.\n3. Compare both models' parameters within a small tolerance. Hint: initialize the second model at zero and train for 200 steps.",
    '''
with torch.no_grad():
    print("Test MAE:", (test_pred - y_test).abs().mean().item())
w2 = torch.zeros(1, requires_grad=True)
b2 = torch.zeros(1, requires_grad=True)
optimizer = torch.optim.SGD([w2, b2], lr=0.08)
for _ in range(200):
    optimizer.zero_grad(set_to_none=True)
    ((X_train * w2 + b2 - y_train) ** 2).mean().backward()
    optimizer.step()
assert torch.allclose(w2, w, atol=1e-3)
assert torch.allclose(b2, b, atol=1e-3)
''', "Training error measures fit to seen examples; test error estimates generalization. If you repeatedly choose settings using test error, that set has effectively become a validation set.",
    "- [Learning PyTorch with examples](https://docs.pytorch.org/tutorials/beginner/pytorch_with_examples.html)")

notebook("neural_networks", "Build neural networks with nn.Module", "Beginner", "60–90", "Notebook 04; Python classes at a basic level.",
    "- Define layers and a forward method.\n- Use a nonlinear activation.\n- Compare a linear model with a multilayer perceptron (MLP).", [
    ("Create a nonlinear prediction task", "A straight line cannot describe a sine wave. Multiple linear layers alone are still a linear transformation; a nonlinear activation such as Tanh or ReLU changes the set of functions a model can represent.", '''
X = torch.linspace(-3, 3, 240).unsqueeze(1)
y = torch.sin(2 * X) + 0.06 * torch.randn_like(X)
train_idx = torch.arange(0, len(X), 2)
val_idx = torch.arange(1, len(X), 2)
plt.scatter(X[:, 0], y[:, 0], s=8); plt.show()
'''),
    ("Define a module", "Assigning layers as attributes registers their parameters. `forward` describes how data flows through them. Calling `model(X)` runs forward through PyTorch's module machinery. `nn.Linear(in, out)` stores a weight shaped `[out, in]` and computes `X @ weight.T + bias`.", '''
class SineMLP(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(1, width), nn.Tanh(),
                                    nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1))
    def forward(self, x):
        return self.layers(x)

model = SineMLP()
print(model)
print("Trainable parameters:", sum(p.numel() for p in model.parameters()))
assert model(X[:4]).shape == (4, 1)
'''),
    ("Train a linear baseline and an MLP", "`MSELoss` replaces our hand-written squared error. Adam adapts update scales per parameter; it still depends on a useful learning rate. Both models see the same training observations.", '''
def fit(model, steps=500):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.015)
    losses = []
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(X[train_idx]), y[train_idx])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses

linear = nn.Linear(1, 1)
linear_history = fit(linear)
mlp_history = fit(model)
model.eval(); linear.eval()
with torch.no_grad():
    mlp_mse = nn.functional.mse_loss(model(X[val_idx]), y[val_idx]).item()
    linear_mse = nn.functional.mse_loss(linear(X[val_idx]), y[val_idx]).item()
print({"linear_validation_MSE": linear_mse, "MLP_validation_MSE": mlp_mse})
assert mlp_mse < linear_mse
plt.scatter(X[:, 0], y[:, 0], s=8, alpha=0.4)
plt.plot(X[:, 0], model(X).detach()[:, 0], label="MLP")
plt.plot(X[:, 0], linear(X).detach()[:, 0], label="linear")
plt.legend(); plt.show()
''')],
    "1. Count parameters in `nn.Linear(3, 5)` by hand, then check.\n2. Build an MLP with ReLU and verify its output shape.\n3. Compare your ReLU model on the same split. Keep the test set conceptually separate; this split is for development.",
    '''
layer = nn.Linear(3, 5)
assert sum(p.numel() for p in layer.parameters()) == 3 * 5 + 5
relu_model = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))
assert relu_model(X[:7]).shape == (7, 1)
_ = fit(relu_model)
relu_model.eval()
with torch.no_grad():
    print("ReLU validation MSE:", nn.functional.mse_loss(relu_model(X[val_idx]), y[val_idx]).item())
''', "Architecture controls which patterns the model can express; optimization determines which parameters it finds. A more complex model does not automatically generalize better.",
    "- [Build a neural network](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)")

notebook("datasets_and_dataloaders", "Datasets, minibatches, and preprocessing", "Beginner → intermediate", "60–75", "Notebooks 04–05.",
    "- Implement a Dataset.\n- Fit normalization on training data only.\n- Train with shuffled minibatches and a correctly weighted epoch loss.", [
    ("Split before estimating preprocessing statistics", "A **minibatch** is a small group of observations used for one update. Before batching, create train/validation splits and estimate normalization from training data only. Validation statistics must not leak into preprocessing.", '''
raw_X = torch.randn(500, 3) * torch.tensor([1., 10., 100.]) + torch.tensor([5., -4., 20.])
y = raw_X @ torch.tensor([[2.], [0.3], [-0.02]]) + 0.2 * torch.randn(500, 1)
indices = torch.randperm(500)
train_idx, val_idx = indices[:400], indices[400:]
mean = raw_X[train_idx].mean(0)
std = raw_X[train_idx].std(0).clamp_min(1e-6)
X = (raw_X - mean) / std
assert torch.allclose(X[train_idx].mean(0), torch.zeros(3), atol=1e-5)
'''),
    ("Implement the two Dataset methods", "A Dataset answers 'How many examples?' and 'What is example i?'. A DataLoader handles batching and shuffling. `num_workers=0` keeps these small examples portable in Windows notebooks.", '''
from torch.utils.data import Dataset

class RegressionDataset(Dataset):
    def __init__(self, features, targets):
        assert len(features) == len(targets)
        self.features, self.targets = features, targets
    def __len__(self):
        return len(self.features)
    def __getitem__(self, index):
        return self.features[index], self.targets[index]

train_ds = RegressionDataset(X[train_idx], y[train_idx])
val_ds = RegressionDataset(X[val_idx], y[val_idx])
loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0,
                    generator=torch.Generator().manual_seed(SEED))
xb, yb = next(iter(loader))
print("One batch:", xb.shape, yb.shape, "Batches:", len(loader))
assert xb.shape == (32, 3)
'''),
    ("Train and aggregate the epoch loss", "The last batch can be smaller. Multiply each batch's mean loss by its number of examples before summing; divide by the number of examples at the end. A plain mean of batch means would overweight a small final batch.", '''
model = nn.Linear(3, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
epoch_losses = []
for epoch in range(40):
    model.train()
    total, count = 0., 0
    for xb, yb in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(xb), yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(xb)
        count += len(xb)
    epoch_losses.append(total / count)
model.eval()
with torch.no_grad():
    val_mse = nn.functional.mse_loss(model(val_ds.features), val_ds.targets).item()
print("Final train MSE:", epoch_losses[-1], "Validation MSE:", val_mse)
assert val_mse < 0.2
''')],
    "1. With 400 training examples and batch size 64, predict how many batches and the final batch size.\n2. Create a validation loader without shuffling.\n3. Compute validation MSE by accumulating per-example squared errors, then compare with `val_mse`.",
    '''
alternative = DataLoader(train_ds, batch_size=64, shuffle=False)
sizes = [len(xb) for xb, yb in alternative]
assert len(sizes) == 7 and sizes[-1] == 16
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
total, count = 0., 0
with torch.no_grad():
    for xb, yb in val_loader:
        total += (model(xb) - yb).square().sum().item()
        count += len(xb)
assert abs(total / count - val_mse) < 1e-5
''', "Data preparation is part of the learned system. Save its training statistics with the model and apply the same transformation to new observations.",
    "- [Datasets and DataLoaders](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)")

notebook("classification", "Binary and multiclass classification", "Intermediate", "75–90", "Notebooks 05–06.",
    "- Train with logits and the appropriate classification loss.\n- Convert logits into probabilities and predictions.\n- Inspect a decision boundary and a confusion matrix.", [
    ("Solve a binary problem", "A **logit** is an unconstrained score. `BCEWithLogitsLoss` combines sigmoid with binary cross entropy stably. It expects floating targets with the same shape as logits. Do not apply sigmoid before this loss.", '''
X = torch.randn(500, 2)
y = (X[:, 0] * X[:, 1] > 0).float().unsqueeze(1)
train, val = torch.arange(400), torch.arange(400, 500)
binary = nn.Sequential(nn.Linear(2, 24), nn.ReLU(), nn.Linear(24, 1))
opt = torch.optim.Adam(binary.parameters(), lr=0.03)
loss_fn = nn.BCEWithLogitsLoss()
for _ in range(250):
    opt.zero_grad(set_to_none=True)
    loss = loss_fn(binary(X[train]), y[train])
    loss.backward(); opt.step()
binary.eval()
with torch.no_grad():
    probabilities = binary(X[val]).sigmoid()
    predicted = (probabilities >= 0.5).float()
    binary_accuracy = (predicted == y[val]).float().mean().item()
print("Binary validation accuracy:", binary_accuracy)
assert binary_accuracy > 0.85
'''),
    ("Visualize the boundary", "The network combines straight boundaries into a nonlinear decision region. Points near the boundary can be uncertain; a probability threshold controls the final yes/no decision.", '''
axis = torch.linspace(-3, 3, 100)
gx, gy = torch.meshgrid(axis, axis, indexing="ij")
grid = torch.stack([gx.flatten(), gy.flatten()], dim=1)
with torch.no_grad():
    field = binary(grid).sigmoid().reshape(100, 100)
plt.contourf(gx.numpy(), gy.numpy(), field.numpy(), levels=20, alpha=0.65)
plt.scatter(X[val, 0], X[val, 1], c=y[val, 0], edgecolors="white", s=25)
plt.xlabel("Feature 1"); plt.ylabel("Feature 2"); plt.show()
'''),
    ("Move to mutually exclusive classes", "For C classes, `CrossEntropyLoss` expects logits `[B,C]` and integer class indices `[B]` with dtype `long`. It includes log-softmax internally. Softmax is useful for displaying probabilities, but not before this loss.", '''
centers = torch.tensor([[-2., -1.], [2., -1.], [0., 2.]])
labels = torch.arange(3).repeat_interleave(180)
features = centers[labels] + 0.7 * torch.randn(540, 2)
order = torch.randperm(540)
tr, va = order[:420], order[420:]
multi = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 3))
optimizer = torch.optim.Adam(multi.parameters(), lr=0.03)
for _ in range(120):
    optimizer.zero_grad(set_to_none=True)
    loss = nn.functional.cross_entropy(multi(features[tr]), labels[tr])
    loss.backward(); optimizer.step()
multi.eval()
with torch.no_grad():
    logits = multi(features[va])
    predictions = logits.argmax(1)
    accuracy = (predictions == labels[va]).float().mean().item()
print("Multiclass validation accuracy:", accuracy)
confusion = torch.bincount(labels[va] * 3 + predictions, minlength=9).reshape(3, 3)
print("Rows = true class; columns = prediction:", confusion)
assert accuracy > 0.9
assert confusion.sum() == len(va)
''')],
    "1. Verify that each multiclass probability vector sums to 1.\n2. Compute binary precision and recall at thresholds 0.3 and 0.7. Hint: guard against zero denominators.\n3. Compute each class's recall from the confusion matrix.",
    '''
assert torch.allclose(logits.softmax(1).sum(1), torch.ones(len(va)), atol=1e-6)
truth = y[val].bool()
for threshold in [0.3, 0.7]:
    decision = probabilities >= threshold
    tp = (decision & truth).sum().item()
    precision = tp / max(decision.sum().item(), 1)
    recall = tp / max(truth.sum().item(), 1)
    print(threshold, "precision", precision, "recall", recall)
print("Per-class recall:", confusion.diag() / confusion.sum(1).clamp_min(1))
''', "Match the loss to the target meaning and shape. Multilabel classification differs from multiclass classification: several labels can be true at once, so independent binary logits are often appropriate.",
    "- [CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)\n- [BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)")

notebook("generalization", "Generalization, regularization, and early stopping", "Intermediate", "75–90", "Notebooks 05–07.",
    "- Use train, validation, and test sets for different purposes.\n- Compare regularized models.\n- Restore the best validation checkpoint without aliasing its tensors.", [
    ("Create a small noisy training set", "With little data, a flexible model may memorize noise. Training adjusts weights; validation selects settings and stopping time; the test set is reserved for the final estimate. Our inputs share a fixed, already bounded scale.", '''
X = torch.rand(240, 1) * 6 - 3
y = torch.sin(2 * X) + 0.3 * torch.randn_like(X)
tr, va, te = torch.arange(35), torch.arange(35, 130), torch.arange(130, 240)
'''),
    ("Train with validation-based early stopping", "Dropout randomly zeros activations during training. Weight decay discourages large weights. `eval()` changes dropout behavior; it does not disable gradients. Use `no_grad()` as well for validation. Deep-copy the best state because `state_dict()` alone holds references to live tensors.", '''
import copy

def run_experiment(dropout, weight_decay):
    torch.manual_seed(SEED)
    model = nn.Sequential(nn.Linear(1, 64), nn.Tanh(), nn.Dropout(dropout),
                          nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=weight_decay)
    best_loss, stale, best_epoch = float("inf"), 0, 0
    train_curve, val_curve = [], []
    for epoch in range(500):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(X[tr]), y[tr])
        loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = nn.functional.mse_loss(model(X[va]), y[va]).item()
        train_curve.append(loss.item()); val_curve.append(val_loss)
        if val_loss < best_loss - 1e-5:
            best_loss, stale, best_epoch = val_loss, 0, epoch
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= 50:
            break
    model.load_state_dict(best_state)
    return {"model": model, "best_val": best_loss, "best_epoch": best_epoch,
            "train_curve": train_curve, "val_curve": val_curve}

runs = {"plain": run_experiment(0., 0.), "regularized": run_experiment(0.1, 0.01)}
for name, run in runs.items():
    print(name, "best validation MSE:", round(run['best_val'], 4), "epoch:", run['best_epoch'])
    plt.plot(run["val_curve"], label=f"{name} validation")
plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.legend(); plt.show()
'''),
    ("Select using validation, then evaluate once", "Regularization is a hypothesis to test, not a guaranteed improvement. The winning model is chosen using validation results only. If you now tune based on test performance, obtain a fresh final test set for an unbiased assessment.", '''
winner = min(runs, key=lambda name: runs[name]["best_val"])
best_model = runs[winner]["model"]
best_model.eval()
with torch.no_grad():
    test_mse = nn.functional.mse_loss(best_model(X[te]), y[te]).item()
print("Selected:", winner, "Final test MSE:", test_mse)
assert np.isfinite(test_mse)
''')],
    "1. Plot the winning run's training and validation curves.\n2. Demonstrate that dropout is random in train mode and deterministic in eval mode.\n3. Verify the restored model's validation loss equals its recorded best loss. Hint: use eval and no_grad.",
    '''
selected = runs[winner]
plt.plot(selected['train_curve'], label="train (dropout active if configured)")
plt.plot(selected['val_curve'], label="validation")
plt.legend(); plt.show()
drop = nn.Dropout(0.5)
ones = torch.ones(100)
drop.train(); a, b = drop(ones), drop(ones)
assert not torch.equal(a, b)
drop.eval(); assert torch.equal(drop(ones), drop(ones))
with torch.no_grad():
    restored = nn.functional.mse_loss(best_model(X[va]), y[va]).item()
assert abs(restored - selected['best_val']) < 1e-6
''', "Explain why a low training loss can coexist with poor validation results. Compare multiple random seeds before making strong claims about small differences.",
    "- [Optimization tutorial](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)\n- [Saving model states](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)")

notebook("cnn_digits", "Computer vision with a convolutional network", "Intermediate", "75–100", "Notebooks 06–08.",
    "- Train on real handwritten digit images.\n- Trace convolution and pooling shapes.\n- Inspect held-out mistakes, not just average accuracy.", [
    ("Load a small real dataset without downloads", "Scikit-learn bundles 1,797 handwritten 8 × 8 digit images from the UCI optical recognition dataset. Pixel intensities range from 0 to 16. This fixed known range lets us scale by 16 without estimating statistics from validation data. Stratification preserves approximate class proportions.", '''
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
digits = load_digits()
X = torch.tensor(digits.images, dtype=torch.float32).unsqueeze(1) / 16.0
y = torch.tensor(digits.target, dtype=torch.long)
tr, va = train_test_split(np.arange(len(y)), test_size=0.25, random_state=SEED, stratify=y.numpy())
loader = DataLoader(TensorDataset(X[tr], y[tr]), batch_size=64, shuffle=True, num_workers=0)
fig, axes = plt.subplots(1, 8, figsize=(12, 2))
for i, ax in enumerate(axes):
    ax.imshow(X[i, 0], cmap="gray"); ax.set_title(str(y[i].item())); ax.axis("off")
plt.show()
print("[batch, channels, height, width]:", X.shape)
'''),
    ("Build local pattern detectors", "A convolution reuses a small filter across locations. Padding 1 with a 3 × 3 kernel preserves spatial size at stride 1. Pooling reduces 8 × 8 to 4 × 4. The final linear layer consumes the flattened feature maps.", '''
class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
                                      nn.MaxPool2d(2), nn.Conv2d(16, 32, 3, padding=1), nn.ReLU())
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(32 * 4 * 4, 10))
    def forward(self, x):
        return self.head(self.features(x))

model = DigitCNN()
probe = X[:4]
for layer in model.features:
    probe = layer(probe)
    print(type(layer).__name__, tuple(probe.shape))
assert model(X[:4]).shape == (4, 10)
'''),
    ("Train and track validation accuracy", "Use raw logits with cross entropy. This is a development experiment: the held-out split is validation because we inspect it during training. The capstone adds a separate final test set.", '''
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
history = []
for epoch in range(15):
    model.train()
    for xb, yb in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(xb), yb)
        loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad():
        accuracy = (model(X[va]).argmax(1) == y[va]).float().mean().item()
    history.append(accuracy)
print("Validation accuracy:", history[-1])
assert history[-1] > 0.85
plt.plot(range(1, len(history) + 1), history); plt.xlabel("Epoch"); plt.ylabel("Validation accuracy"); plt.show()
'''),
    ("Inspect mistakes", "A confusion matrix reveals which classes are confused. Looking at individual errors can suggest problems with data quality or model assumptions. Do not assume that confidence guarantees correctness.", '''
with torch.no_grad():
    probs = model(X[va]).softmax(1)
pred = probs.argmax(1)
confusion = torch.bincount(y[va] * 10 + pred, minlength=100).reshape(10, 10)
plt.imshow(confusion, cmap="Blues"); plt.xlabel("Predicted"); plt.ylabel("True"); plt.colorbar(); plt.show()
errors = (pred != y[va]).nonzero().flatten()[:6]
if len(errors):
    fig, axes = plt.subplots(1, len(errors), figsize=(12, 2), squeeze=False)
    for ax, i in zip(axes[0], errors.tolist()):
        ax.imshow(X[va[i], 0], cmap="gray")
        ax.set_title(f"True {y[va[i]].item()} / Pred {pred[i].item()}"); ax.axis("off")
    plt.show()
else:
    print("No validation errors in this run.")
''')],
    "1. Count the first convolution's parameters by hand.\n2. Visualize the first eight learned filters.\n3. Compute macro recall (the mean of class recalls) and compare it with accuracy.",
    '''
conv = model.features[0]
assert sum(p.numel() for p in conv.parameters()) == 16 * (1 * 3 * 3 + 1)
fig, axes = plt.subplots(1, 8, figsize=(12, 2))
for i, ax in enumerate(axes):
    ax.imshow(conv.weight[i, 0].detach(), cmap="coolwarm"); ax.axis("off")
plt.show()
macro_recall = (confusion.diag() / confusion.sum(1).clamp_min(1)).mean().item()
print("Macro recall:", macro_recall, "Accuracy:", history[-1])
''', "You have trained a real image classifier. The small bundled dataset is useful for learning, but it does not establish performance on photos or handwriting from a new source.",
    "- [Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)\n- [Bundled digits dataset and provenance](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)")

notebook("transfer_learning", "Transfer learning: freeze, replace, and fine-tune", "Intermediate", "75–100", "Notebooks 08–09.",
    "- Pretrain an encoder and reuse its features on a related task.\n- Verify frozen parameters stay unchanged.\n- Fine-tune with separate learning rates.", [
    ("Define related source and target tasks", "We first learn to recognize digits, then reuse the encoder to classify parity (even or odd) with fewer labels. All source pretraining uses the training partition: validation images are never seen in pretraining. This small local pretraining exercise demonstrates the mechanics; it is not an ImageNet-pretrained model or evidence of universal transfer gains.", '''
import copy
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
digits = load_digits()
X = torch.tensor(digits.data, dtype=torch.float32) / 16
y = torch.tensor(digits.target, dtype=torch.long)
tr, va = train_test_split(np.arange(len(y)), test_size=0.25, random_state=SEED, stratify=y.numpy())
source_loader = DataLoader(TensorDataset(X[tr], y[tr]), batch_size=64, shuffle=True)
encoder = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
source = nn.Sequential(encoder, nn.Linear(32, 10))
optimizer = torch.optim.Adam(source.parameters(), lr=0.003)
for epoch in range(12):
    for xb, yb in source_loader:
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(source(xb), yb)
        loss.backward(); optimizer.step()
source.eval()
with torch.no_grad():
    print("Source validation digit accuracy:", (source(X[va]).argmax(1) == y[va]).float().mean().item())
'''),
    ("Freeze an encoder and learn a new head", "`requires_grad_(False)` prevents parameter gradients. It is distinct from `eval()`, which changes layers such as dropout and batch normalization. Our encoder has neither, but we explicitly set it to eval to illustrate feature extraction practice.", '''
transferred_encoder = copy.deepcopy(encoder)
transferred_encoder.requires_grad_(False)
target_model = nn.Sequential(transferred_encoder, nn.Linear(32, 2))
small_tr = tr[:160]
target_y = y % 2
before = {name: p.detach().clone() for name, p in transferred_encoder.named_parameters()}
head_optimizer = torch.optim.Adam(target_model[1].parameters(), lr=0.03)
for epoch in range(160):
    target_model.train(); transferred_encoder.eval()
    head_optimizer.zero_grad(set_to_none=True)
    loss = nn.functional.cross_entropy(target_model(X[small_tr]), target_y[small_tr])
    loss.backward(); head_optimizer.step()
assert all(torch.equal(before[name], p) for name, p in transferred_encoder.named_parameters())
def parity_accuracy(model):
    model.eval()
    with torch.no_grad():
        return (model(X[va]).argmax(1) == target_y[va]).float().mean().item()
frozen_accuracy = parity_accuracy(target_model)
print("Frozen encoder validation parity accuracy:", frozen_accuracy)
'''),
    ("Unfreeze with a smaller encoder learning rate", "Fine-tuning allows the representation to adapt. A smaller encoder learning rate reduces abrupt changes to learned features. Build a new optimizer that includes the newly trainable parameters; simply unfreezing cannot add them to the old head-only optimizer.", '''
transferred_encoder.requires_grad_(True)
fine_optimizer = torch.optim.Adam([
    {"params": transferred_encoder.parameters(), "lr": 0.0003},
    {"params": target_model[1].parameters(), "lr": 0.003},
])
for epoch in range(70):
    target_model.train()
    fine_optimizer.zero_grad(set_to_none=True)
    loss = nn.functional.cross_entropy(target_model(X[small_tr]), target_y[small_tr])
    loss.backward(); fine_optimizer.step()
fine_accuracy = parity_accuracy(target_model)
print("Fine-tuned validation parity accuracy:", fine_accuracy)
assert any(not torch.equal(before[name], p) for name, p in transferred_encoder.named_parameters())
''')],
    "1. Train a parity classifier from scratch with the same 160 examples for 230 full-batch updates.\n2. Compare its validation accuracy with the frozen and fine-tuned models; explain why one seed is weak evidence.\n3. Count trainable parameters after freezing the transferred encoder again.",
    '''
torch.manual_seed(SEED)
scratch = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))
scratch_opt = torch.optim.Adam(scratch.parameters(), lr=0.003)
for _ in range(230):
    scratch_opt.zero_grad(set_to_none=True)
    loss = nn.functional.cross_entropy(scratch(X[small_tr]), target_y[small_tr])
    loss.backward(); scratch_opt.step()
print({"scratch": parity_accuracy(scratch), "frozen": frozen_accuracy, "fine_tuned": fine_accuracy})
transferred_encoder.requires_grad_(False)
assert sum(p.numel() for p in target_model.parameters() if p.requires_grad) == 32 * 2 + 2
''', "To extend this to a large pretrained vision model, use the official torchvision weight transforms, replace its head, and keep the target test data separate from every development step. Pretrained weights require an explicit download and a compatible torchvision installation.",
    "- [Transfer learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)")

notebook("embeddings_and_text", "Embeddings and a tiny text classifier", "Intermediate", "60–90", "Notebooks 06–07.",
    "- Build a vocabulary from training text only.\n- Pad variable-length sequences and mask padding.\n- Understand the limits of averaging word embeddings.", [
    ("Prepare a deliberately small text task", "Tokenization converts text into discrete symbols. An embedding learns a vector per token. This toy sentiment task teaches mechanics, not a deployable language model. The held-out sentences contain familiar sentiment words in new combinations; natural language has much more ambiguity.", '''
train_texts = ["good movie", "great story", "love this film", "excellent acting", "fun movie", "nice story",
               "bad movie", "awful story", "hate this film", "terrible acting", "boring movie", "poor story"]
train_labels = torch.tensor([1] * 6 + [0] * 6)
val_texts = ["good acting", "great film", "bad story", "boring film"]
val_labels = torch.tensor([1, 1, 0, 0])
vocab = {"<pad>": 0, "<unk>": 1}
for sentence in train_texts:
    for token in sentence.lower().split():
        if token not in vocab:
            vocab[token] = len(vocab)
def encode(sentence):
    return torch.tensor([vocab.get(word, 1) for word in sentence.lower().split()] or [1])
print(vocab)
assert encode("unseen").item() == vocab['<unk>']
'''),
    ("Collate variable-length examples", "A DataLoader normally stacks equal shapes. A custom collate function pads each batch to its longest sequence. Padding is storage, not meaningful text, so exclude it from the average.", '''
from torch.nn.utils.rnn import pad_sequence
def collate(batch):
    sequences, labels = zip(*batch)
    return pad_sequence(sequences, batch_first=True, padding_value=0), torch.stack(labels)
dataset = [(encode(text), label) for text, label in zip(train_texts, train_labels)]
loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate)
tokens, labels = next(iter(loader))
print("Token IDs:", tokens)

class TextClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(vocab), 12, padding_idx=0)
        self.head = nn.Linear(12, 2)
    def forward(self, tokens):
        mask = (tokens != 0).unsqueeze(-1)
        embedded = self.embedding(tokens)
        pooled = (embedded * mask).sum(1) / mask.sum(1).clamp_min(1)
        return self.head(pooled)

model = TextClassifier()
assert model(tokens).shape == (len(tokens), 2)
'''),
    ("Train, evaluate, and expose a limitation", "Averaging ignores word order. It can work as a baseline but cannot represent all language relationships. On 12 training sentences, metrics are extremely noisy and vocabulary coverage is limited.", '''
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
for epoch in range(70):
    for tokens, labels in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(tokens), labels)
        loss.backward(); optimizer.step()
model.eval()
val_tokens = pad_sequence([encode(text) for text in val_texts], batch_first=True)
with torch.no_grad():
    probabilities = model(val_tokens).softmax(1)[:, 1]
for text, p in zip(val_texts, probabilities):
    print(f"{text:15s} -> P(positive)={p.item():.3f}")
print("Toy validation accuracy:", ((probabilities >= 0.5).long() == val_labels).float().mean().item())
a, b = encode("good movie"), encode("movie good")
with torch.no_grad():
    assert torch.allclose(model(a[None]), model(b[None]), atol=1e-6)
''')],
    "1. Check that adding padding does not change a prediction.\n2. Count unknown tokens in 'wonderful new movie'.\n3. Compare 'not good' with 'good'. Explain why this training set cannot teach reliable negation.",
    '''
example = encode("good movie")[None]
padded = nn.functional.pad(example, (0, 3), value=0)
with torch.no_grad():
    assert torch.allclose(model(example), model(padded), atol=1e-6)
    for text in ["not good", "good"]:
        print(text, model(encode(text)[None]).softmax(1).tolist())
assert (encode("wonderful new movie") == 1).sum().item() == 2
# 'not' is unknown; no training sentence supplies supervision for negation.
''', "A tokenizer, vocabulary, padding convention, and model belong together. Save all of them for inference. This model is order-invariant; the next lessons introduce sequence-aware models.",
    "- [Word embeddings tutorial](https://docs.pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html)")

notebook("sequence_forecasting", "Sequence models and time-series forecasting", "Intermediate → advanced", "75–100", "Notebooks 05–08.",
    "- Build sliding input windows without future leakage.\n- Train a GRU for next-step prediction.\n- Compare with a persistence baseline.", [
    ("Create chronological windows", "For a sequence, a random window split can place almost identical neighboring observations on both sides. We use chronological boundaries. A target at time t may use only observations before t. Validation windows may use earlier training history: that is available at prediction time.", '''
t = torch.arange(900, dtype=torch.float32)
series = torch.sin(t * 0.08) + 0.35 * torch.sin(t * 0.021) + 0.04 * torch.randn(900)
window = 24
X = torch.stack([series[i-window:i] for i in range(window, len(series))]).unsqueeze(-1)
y = series[window:].unsqueeze(1)
target_times = torch.arange(window, len(series))
tr = target_times < 600
va = (target_times >= 600) & (target_times < 750)
te = target_times >= 750
mean, std = series[:600].mean(), series[:600].std()
Xn, yn = (X - mean) / std, (y - mean) / std
loader = DataLoader(TensorDataset(Xn[tr], yn[tr]), batch_size=64, shuffle=True)
print("[batch, time, features]:", Xn.shape)
assert target_times[tr].max() < target_times[va].min()
'''),
    ("Use a recurrent hidden state", "A GRU reads one time step at a time and updates a hidden representation. With `batch_first=True`, its input is `[B,T,F]`. We reset hidden state for each independent window by omitting an initial state. The final output summarizes the window.", '''
class Forecaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=24, batch_first=True)
        self.head = nn.Linear(24, 1)
    def forward(self, x):
        output, hidden = self.gru(x)
        return self.head(output[:, -1])
model = Forecaster()
assert model(Xn[:8]).shape == (8, 1)
'''),
    ("Select a checkpoint on the next time period", "Shuffling training windows is acceptable here because each window contains its own ordered history and hidden state is reset. Keep the order within each window. Choose the checkpoint using validation, then evaluate the final period.", '''
import copy
optimizer = torch.optim.Adam(model.parameters(), lr=0.008)
best_val = float('inf')
for epoch in range(25):
    model.train()
    for xb, yb in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        val_loss = nn.functional.mse_loss(model(Xn[va]), yn[va]).item()
    if val_loss < best_val:
        best_val, best_state = val_loss, copy.deepcopy(model.state_dict())
model.load_state_dict(best_state)
with torch.no_grad():
    predicted = model(Xn[te]) * std + mean
    mse = nn.functional.mse_loss(predicted, y[te]).item()
    persistence_mse = nn.functional.mse_loss(X[te, -1], y[te]).item()
print({"GRU_test_MSE": mse, "persistence_test_MSE": persistence_mse})
plt.plot(target_times[te], y[te, 0], label="actual")
plt.plot(target_times[te], predicted[:, 0], label="one-step forecast")
plt.legend(); plt.show()
assert np.isfinite(mse)
''')],
    "1. Compute MAE for both models.\n2. Verify a sample's last input corresponds to the time immediately before its target.\n3. Forecast 20 steps recursively from the beginning of the test period; compare this with one-step evaluation. Hint: feed each prediction back into the window.",
    '''
print("GRU MAE:", (predicted - y[te]).abs().mean().item())
print("Persistence MAE:", (X[te, -1] - y[te]).abs().mean().item())
i = 100
assert X[i, -1, 0] == series[target_times[i] - 1]
context = Xn[te][0:1].clone()
recursive = []
with torch.no_grad():
    for _ in range(20):
        next_value = model(context)
        recursive.append((next_value * std + mean).item())
        context = torch.cat([context[:, 1:], next_value.unsqueeze(1)], dim=1)
plt.plot(y[te][:20, 0], label="actual")
plt.plot(recursive, label="recursive forecast"); plt.legend(); plt.show()
# Recursive evaluation accumulates model errors instead of receiving each new true observation.
''', "One-step forecasts use newly observed true history each time; recursive forecasts do not. Report which setting you evaluated. A neural network need not beat persistence on every series.",
    "- [GRU](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html)")

notebook("attention_and_transformers", "Attention and a tiny causal transformer", "Advanced", "90–120", "Notebooks 02, 07, and 11–12.",
    "- Compute scaled dot-product attention by hand.\n- Apply a causal mask and verify no future information leaks.\n- Train and sample from a tiny next-token model.", [
    ("Inspect attention weights", "A query is compared with keys to decide how to mix values. Scores are divided by the square root of key width before softmax. A causal mask blocks future positions. Attention weights describe this mixing operation, not a complete explanation of model decisions.", '''
import math
Q, K, V = torch.randn(1, 4, 8), torch.randn(1, 4, 8), torch.randn(1, 4, 8)
scores = Q @ K.transpose(-2, -1) / math.sqrt(Q.shape[-1])
blocked = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
weights = scores.masked_fill(blocked, float('-inf')).softmax(-1)
attended = weights @ V
assert torch.allclose(weights.sum(-1), torch.ones(1, 4))
assert torch.equal(weights[0][blocked], torch.zeros(6))
plt.imshow(weights[0], vmin=0, vmax=1, cmap="Blues")
plt.xlabel("Key position"); plt.ylabel("Query position"); plt.colorbar(); plt.show()
'''),
    ("Create a miniature next-token task", "Our vocabulary is the digits 0 through 7. Each sequence follows a cyclic rule, starting at a random digit. Inputs are all tokens except the last; targets are shifted one position forward. This controlled task makes masking and optimization easy to verify, but is much simpler than natural language.", '''
vocab_size, context_length = 8, 12
def make_sequences(n, generator):
    start = torch.randint(vocab_size, (n, 1), generator=generator)
    return (start + torch.arange(context_length + 1)) % vocab_size
generator = torch.Generator().manual_seed(SEED)
sequences = make_sequences(320, generator)
train_seq, val_seq = sequences[:256], sequences[256:]
loader = DataLoader(TensorDataset(train_seq[:, :-1], train_seq[:, 1:]), batch_size=32, shuffle=True)
print("Input:", train_seq[0, :-1].tolist(), "Target:", train_seq[0, 1:].tolist())
'''),
    ("Build one transformer block", "Token embeddings encode identity; position embeddings encode order. Multiple attention heads learn separate mixing patterns. Residual additions preserve a direct information path, LayerNorm stabilizes scales, and a feed-forward network transforms each position. Here `MultiheadAttention` boolean True means **blocked**; mask conventions differ across APIs.", '''
class TinyTransformer(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.token = nn.Embedding(vocab_size, width)
        self.position = nn.Embedding(context_length, width)
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, num_heads=4, dropout=0., batch_first=True)
        self.ff = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width))
        self.output = nn.Linear(width, vocab_size)
    def forward(self, tokens):
        length = tokens.shape[1]
        x = self.token(tokens) + self.position(torch.arange(length, device=tokens.device))
        mask = torch.triu(torch.ones(length, length, device=tokens.device, dtype=torch.bool), diagonal=1)
        z = self.norm1(x)
        mixed, _ = self.attention(z, z, z, attn_mask=mask, need_weights=False)
        x = x + mixed
        x = x + self.ff(self.norm2(x))
        return self.output(x)

model = TinyTransformer()
assert model(train_seq[:2, :-1]).shape == (2, context_length, vocab_size)
'''),
    ("Train and verify causality", "Cross entropy needs one class dimension, so flatten batch and time together. Causality is a property we can test: changing a future token must leave earlier logits unchanged in evaluation mode.", '''
optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
for epoch in range(12):
    model.train()
    for xb, yb in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = nn.functional.cross_entropy(logits.reshape(-1, vocab_size), yb.reshape(-1))
        loss.backward(); optimizer.step()
model.eval()
with torch.no_grad():
    logits = model(val_seq[:, :-1])
    accuracy = (logits.argmax(-1) == val_seq[:, 1:]).float().mean().item()
    original = val_seq[:1, :-1].clone()
    changed = original.clone(); changed[:, -1] = (changed[:, -1] + 3) % vocab_size
    assert torch.allclose(model(original)[:, :-1], model(changed)[:, :-1], atol=1e-5)
print("Validation token accuracy on the cyclic task:", accuracy)
assert accuracy > 0.95
'''),
    ("Generate autoregressively", "At inference time the next true token is unavailable. Predict one token, append it to the context, then repeat. Greedy decoding takes the highest-scoring token; sampling introduces controlled randomness.", '''
generated = torch.tensor([[3]])
with torch.no_grad():
    for _ in range(18):
        context = generated[:, -context_length:]
        next_token = model(context)[:, -1].argmax(-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
print("Generated:", generated[0].tolist())
''')],
    "1. Compare manual attention with `scaled_dot_product_attention(..., is_causal=True)`.\n2. Count this transformer's trainable parameters.\n3. Sample a next token at temperatures 0.5 and 2.0. Explain how temperature affects probabilities.",
    '''
fast_attention = nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True, dropout_p=0.)
assert torch.allclose(attended, fast_attention, atol=1e-6)
print("Parameters:", sum(p.numel() for p in model.parameters()))
with torch.no_grad():
    last_logits = model(torch.tensor([[3, 4]]))[:, -1]
    for temperature in [0.5, 2.0]:
        distribution = (last_logits / temperature).softmax(-1)
        print(temperature, distribution.round(decimals=3), torch.multinomial(distribution, 1).item())
# Higher temperature flattens the distribution; lower temperature sharpens it.
''', "This notebook teaches transformer mechanics. Expert language-model work also requires realistic corpora, tokenizer design, stronger evaluation, positional strategies, and memory-aware training. The cyclic validation task contains repeated patterns and does not demonstrate language understanding.",
    "- [MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)\n- [Scaled dot-product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)")

notebook("autoencoders", "Denoising autoencoders and latent representations", "Advanced", "75–90", "Notebooks 08–09.",
    "- Train without class labels using reconstruction targets.\n- Encode an image into a small latent vector.\n- Compare denoising with the unprocessed noisy input.", [
    ("Corrupt inputs while keeping clean targets", "An autoencoder compresses and reconstructs data. A denoising autoencoder receives corrupted images but targets the clean originals. Generate fresh training noise so the model cannot memorize one fixed corruption. Validation noise is fixed for fair comparisons.", '''
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
digits = load_digits()
X = torch.tensor(digits.data, dtype=torch.float32) / 16
tr, va = train_test_split(np.arange(len(X)), test_size=0.25, random_state=SEED, stratify=digits.target)
loader = DataLoader(TensorDataset(X[tr]), batch_size=64, shuffle=True)
noise_scale = 0.35
noisy_val = (X[va] + noise_scale * torch.randn_like(X[va])).clamp(0, 1)
'''),
    ("Build a bottleneck", "The encoder reduces 64 pixel values to 12 latent values; the decoder reconstructs 64 pixels. A sigmoid output respects the scaled pixel range. MSE is a simple reconstruction objective; it may favor smooth averages.", '''
class Denoiser(nn.Module):
    def __init__(self, latent_dim=12):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(64, 48), nn.ReLU(), nn.Linear(48, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 48), nn.ReLU(), nn.Linear(48, 64), nn.Sigmoid())
    def forward(self, x):
        return self.decoder(self.encoder(x))
model = Denoiser()
assert model.encoder(X[:4]).shape == (4, 12)
'''),
    ("Train and inspect reconstructions", "Pixel error is useful but incomplete. Inspect reconstructions for lost strokes and changed identity. We do not use the digit class labels in optimization.", '''
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
for epoch in range(30):
    model.train()
    for (clean_batch,) in loader:
        noisy_batch = (clean_batch + noise_scale * torch.randn_like(clean_batch)).clamp(0, 1)
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(noisy_batch), clean_batch)
        loss.backward(); optimizer.step()
model.eval()
with torch.no_grad():
    restored = model(noisy_val)
    baseline_mse = nn.functional.mse_loss(noisy_val, X[va]).item()
    denoised_mse = nn.functional.mse_loss(restored, X[va]).item()
print({"noisy_input_MSE": baseline_mse, "denoised_MSE": denoised_mse})
assert denoised_mse < baseline_mse
fig, axes = plt.subplots(3, 6, figsize=(10, 5))
for row, (data, name) in enumerate([(X[va], "clean"), (noisy_val, "noisy"), (restored, "restored")]):
    for col in range(6):
        axes[row, col].imshow(data[col].reshape(8, 8), cmap="gray", vmin=0, vmax=1)
        axes[row, col].axis("off")
        if col == 0:
            axes[row, col].set_title(name)
plt.tight_layout(); plt.show()
''')],
    "1. Compute one reconstruction MSE per validation image and find the worst example.\n2. Interpolate between the latent vectors of two images, then decode the path.\n3. Measure reconstruction error for pure random noise. Why is that not enough to validate an anomaly detector?",
    '''
errors = (restored - X[va]).square().mean(1)
print("Worst validation index:", errors.argmax().item(), "MSE:", errors.max().item())
with torch.no_grad():
    z = model.encoder(X[va[:2]])
    alpha = torch.linspace(0, 1, 6).unsqueeze(1)
    interpolated = model.decoder((1 - alpha) * z[0] + alpha * z[1])
    noise = torch.rand(100, 64)
    print("Random-noise reconstruction MSE:", (model(noise) - noise).square().mean().item())
fig, axes = plt.subplots(1, 6, figsize=(10, 2))
for ax, pixels in zip(axes, interpolated):
    ax.imshow(pixels.reshape(8, 8), cmap="gray"); ax.axis("off")
plt.show()
# Real anomalies can reconstruct well; validate thresholds on labeled, representative examples.
''', "An ordinary autoencoder is not automatically a calibrated anomaly detector or a probabilistic generator. Latent interpolation is a useful probe, but random latent sampling need not produce meaningful images.",
    "- [Unsupervised learning with autoencoders (PyTorch examples)](https://github.com/pytorch/examples/tree/main/vae)")

notebook("checkpoints_and_inference", "Checkpoints, reproducibility, and inference", "Advanced", "60–90", "Notebooks 06–08.",
    "- Save weights and preprocessing together.\n- Resume optimization with its internal state.\n- Verify that a loaded model reproduces predictions.", [
    ("Train a model with learned preprocessing", "An inference pipeline includes preprocessing as well as weights. We create fresh local data so this notebook does not require a previous notebook's files. The checkpoint directory is inside the course workspace.", '''
raw_X = torch.randn(160, 3) * 4 + 8
y = raw_X @ torch.tensor([[0.5], [-1.], [2.]])
mean, std = raw_X.mean(0), raw_X.std(0)
X = (raw_X - mean) / std
model = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 1))
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
def train_steps(model, optimizer, count):
    model.train()
    for _ in range(count):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(X), y)
        loss.backward(); optimizer.step()
train_steps(model, optimizer, 80)
root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'notebooks').is_dir() and (p / 'requirements.txt').exists())
artifact_dir = root / 'artifacts' / '15_checkpoints'
artifact_dir.mkdir(parents=True, exist_ok=True)
'''),
    ("Save a structured checkpoint", "A state dict stores parameter and buffer tensors, not the Python class implementation. Record the architecture settings and preprocessing. For resumable Adam training, save optimizer state too. This checkpoint stores simple types and tensors so it can be loaded using `weights_only=True`.", '''
checkpoint_path = artifact_dir / 'regressor.pt'
torch.save({
    'format_version': 1,
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'completed_steps': 80,
    'config': {'input_dim': 3, 'hidden_dim': 16},
    'mean': mean, 'std': std,
    'torch_rng_state': torch.get_rng_state(),
}, checkpoint_path)
print("Saved:", checkpoint_path)
'''),
    ("Load and verify inference parity", "Recreate the same architecture, load on CPU, and put it in eval mode. `inference_mode()` avoids autograd bookkeeping. Load only artifacts you trust; `weights_only=True` narrows deserialization but is not a reason to trust an arbitrary file.", '''
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
config = checkpoint['config']
restored = nn.Sequential(nn.Linear(config['input_dim'], config['hidden_dim']), nn.ReLU(),
                         nn.Linear(config['hidden_dim'], 1))
restored.load_state_dict(checkpoint['model'])
model.eval(); restored.eval()
new_raw = torch.tensor([[7., 8., 9.], [4., 5., 6.]])
def predict(raw_features):
    assert raw_features.ndim == 2 and raw_features.shape[1] == config['input_dim']
    normalized = (raw_features - checkpoint['mean']) / checkpoint['std'].clamp_min(1e-6)
    with torch.inference_mode():
        return restored(normalized)
with torch.inference_mode():
    expected = model((new_raw - mean) / std)
assert torch.allclose(predict(new_raw), expected)
print("Reloaded predictions:", predict(new_raw).flatten().tolist())
'''),
    ("Resume and compare with uninterrupted training", "This full-batch model has no stochastic layers, so its continuation can be checked tightly. Real runs may also need Python/NumPy/CUDA RNG states, scheduler and scaler states, sampler position, data-loader generator state, and deterministic kernels. Saving one CPU RNG state alone does not guarantee exact resumption of every workload.", '''
resumed_optimizer = torch.optim.Adam(restored.parameters(), lr=0.03)
resumed_optimizer.load_state_dict(checkpoint['optimizer'])
train_steps(model, optimizer, 10)
torch.set_rng_state(checkpoint['torch_rng_state'])
train_steps(restored, resumed_optimizer, 10)
for expected_parameter, resumed_parameter in zip(model.parameters(), restored.parameters()):
    assert torch.allclose(expected_parameter, resumed_parameter, atol=1e-6)
print("Resumed parameters match the uninterrupted run.")
''')],
    "1. Save a smaller inference-only artifact with model state, configuration, mean, and std.\n2. Load it and check its keys.\n3. Count how many optimizer state entries were necessary to resume Adam. Explain why weights alone are insufficient.",
    '''
inference_path = artifact_dir / 'inference_only.pt'
torch.save({'model': restored.state_dict(), 'config': config, 'mean': mean, 'std': std}, inference_path)
inference_payload = torch.load(inference_path, map_location='cpu', weights_only=True)
assert set(inference_payload) == {'model', 'config', 'mean', 'std'}
print("Adam parameter-state entries:", len(checkpoint['optimizer']['state']))
# Adam keeps per-parameter step counts and moving averages; resetting them changes future updates.
''', "A reliable save/load workflow is a tested contract: same preprocessing, compatible architecture, same outputs. For deployment, also define input validation, batching, monitoring, and model versioning.",
    "- [Saving and loading models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)\n- [Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)")

notebook("custom_autograd_and_debugging", "Custom gradients and systematic debugging", "Advanced / expert practice", "75–100", "Notebooks 03, 05, and 08.",
    "- Implement a custom autograd Function.\n- Validate first and second derivatives.\n- Investigate a broken gradient and overfit a tiny batch.", [
    ("Implement a derivative explicitly", "Most models should compose existing PyTorch operations; autograd already differentiates them. A custom Function is useful for unusual operations or specialized implementations. Backward applies the chain rule: multiply the local derivative by the incoming gradient.", '''
class Cube(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.pow(3)
    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output * 3 * x.square()

x = torch.tensor([1., 2., -3.], requires_grad=True)
Cube.apply(x).sum().backward()
assert torch.allclose(x.grad, 3 * x.detach().square())
print("Custom gradients:", x.grad)
'''),
    ("Check first and second derivatives", "A correct forward result does not establish a correct backward implementation. Gradcheck compares analytical gradients with finite differences. Use double precision and smooth points; finite differences are unreliable at nondifferentiable boundaries.", '''
probe = torch.randn(5, dtype=torch.float64, requires_grad=True)
assert torch.autograd.gradcheck(Cube.apply, (probe,))
assert torch.autograd.gradgradcheck(Cube.apply, (probe,))
print("First- and second-order gradient checks passed.")

class WrongCube(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.pow(3)
    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output * 2 * x  # Deliberate bug: derivative of x^2, not x^3.

wrong_passes = torch.autograd.gradcheck(WrongCube.apply, (probe,), raise_exception=False)
assert not wrong_passes
print("Deliberately wrong backward passed?", wrong_passes)
'''),
    ("Inspect gradient flow", "A missing gradient can indicate an unused parameter, a detached tensor, or a disabled graph. Check gradients immediately after backward. Do not infer gradient health from the loss alone.", '''
model = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 1))
xb, yb = torch.randn(8, 3), torch.randn(8, 1)
loss = nn.functional.mse_loss(model(xb), yb)
loss.backward()
for name, parameter in model.named_parameters():
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad).all()
    print(name, "gradient norm:", parameter.grad.norm().item())
detached_prediction = model(xb).detach()
assert not detached_prediction.requires_grad
'''),
    ("Overfit a tiny batch as a diagnostic", "Before a long training run, verify a sufficiently flexible model can memorize a handful of examples with regularization disabled. Failure suggests checking labels, loss/activation compatibility, learning rate, parameter registration, and gradient flow. Success is a plumbing check, not proof of generalization.", '''
torch.manual_seed(SEED)
tiny_x, tiny_y = torch.randn(8, 3), torch.randn(8, 1)
tiny = nn.Sequential(nn.Linear(3, 32), nn.Tanh(), nn.Linear(32, 1))
optimizer = torch.optim.Adam(tiny.parameters(), lr=0.03)
for step in range(400):
    optimizer.zero_grad(set_to_none=True)
    loss = nn.functional.mse_loss(tiny(tiny_x), tiny_y)
    assert torch.isfinite(loss)
    loss.backward(); optimizer.step()
with torch.no_grad():
    final_loss = nn.functional.mse_loss(tiny(tiny_x), tiny_y).item()
print("Tiny-batch MSE:", final_loss)
assert final_loss < 1e-3
''')],
    "1. Implement a custom square function and run gradcheck.\n2. Compute a Hessian for `sum(x³)` with `torch.autograd.functional.hessian`.\n3. Clip the tiny model's gradients and verify their total norm is at most 0.1. Hint: clip after backward and before the optimizer step.",
    '''
class Square(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.square()
    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output * 2 * x
assert torch.autograd.gradcheck(Square.apply, (probe,))
hessian = torch.autograd.functional.hessian(lambda z: z.pow(3).sum(), probe)
assert torch.allclose(hessian, torch.diag(6 * probe))
optimizer.zero_grad(set_to_none=True)
nn.functional.mse_loss(tiny(tiny_x), tiny_y + 10).backward()
nn.utils.clip_grad_norm_(tiny.parameters(), 0.1)
total_norm = torch.stack([p.grad.norm() for p in tiny.parameters()]).norm()
assert total_norm <= 0.10001
''', "Debug in this order: shapes and dtypes, input/target alignment, finite loss, gradient presence, tiny-batch fit, then generalization. Use anomaly detection around forward and backward when tracking a difficult NaN; it is slower and is a diagnostic tool.",
    "- [Extending autograd](https://docs.pytorch.org/docs/stable/notes/extending.html)\n- [Gradcheck mechanics](https://docs.pytorch.org/docs/stable/notes/gradcheck.html)")

notebook("performance", "Profiling, gradient accumulation, AMP, and compilation", "Advanced / expert practice", "90–120", "Notebooks 06, 15–16.",
    "- Prove microbatch accumulation matches a full-batch update.\n- Profile CPU operations and benchmark with warmup.\n- Try guarded CUDA mixed precision and understand compilation overhead.", [
    ("Accumulate gradients correctly", "To simulate a batch of N examples, accumulate each microbatch's mean loss multiplied by its size/N. This handles an uneven final microbatch. Do not clear gradients between microbatches. Equivalence requires operations whose results do not depend on batch composition; BatchNorm and stochastic layers can break this comparison.", '''
import copy
X, y = torch.randn(23, 16), torch.randn(23, 1)
full = nn.Sequential(nn.Linear(16, 32), nn.Tanh(), nn.Linear(32, 1))
micro = copy.deepcopy(full)
opt_full = torch.optim.SGD(full.parameters(), lr=0.01)
opt_micro = torch.optim.SGD(micro.parameters(), lr=0.01)
opt_full.zero_grad(set_to_none=True)
nn.functional.mse_loss(full(X), y).backward(); opt_full.step()
opt_micro.zero_grad(set_to_none=True)
for xb, yb in zip(X.split(6), y.split(6)):
    loss = nn.functional.mse_loss(micro(xb), yb) * len(xb) / len(X)
    loss.backward()
opt_micro.step()
for a, b in zip(full.parameters(), micro.parameters()):
    assert torch.allclose(a, b, atol=1e-6)
print("Full-batch and accumulated updates agree.")
'''),
    ("Measure before optimizing", "A profiler identifies time spent inside operations; a benchmark estimates repeated runtime. Use a warmup and repeated measurements. On CUDA, synchronize before reading wall time because launches are asynchronous. These tiny CPU measurements describe this machine, not a universal speed claim.", '''
from torch.profiler import profile, ProfilerActivity
from torch.utils.benchmark import Timer
full.eval()
with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with torch.inference_mode():
        for _ in range(10):
            full(X)
print(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=6))
@torch.inference_mode()
def infer():
    return full(X)
for _ in range(10):
    infer()
measurement = Timer(stmt='infer()', globals={'infer': infer}, num_threads=2).blocked_autorange(min_run_time=0.2)
print(measurement)
'''),
    ("Try mixed precision when CUDA is available", "Autocast chooses lower precision for suitable operations; GradScaler reduces float16 gradient underflow. Unscale before gradient clipping. This optional branch demonstrates one real GPU training step; without CUDA it reports a skip and the CPU lessons remain complete.", '''
import os
RUN_CUDA = os.environ.get('PYTORCH_COURSE_CUDA', '0') == '1'  # Set True to try your CUDA GPU.
if RUN_CUDA and torch.cuda.is_available():
    gpu_model = copy.deepcopy(full).to('cuda').train()
    gpu_x, gpu_y = X.to('cuda'), y.to('cuda')
    gpu_optimizer = torch.optim.Adam(gpu_model.parameters(), lr=0.001)
    scaler = torch.amp.GradScaler('cuda')
    gpu_optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        gpu_loss = nn.functional.mse_loss(gpu_model(gpu_x), gpu_y)
    scaler.scale(gpu_loss).backward()
    scaler.unscale_(gpu_optimizer)
    nn.utils.clip_grad_norm_(gpu_model.parameters(), 1.)
    scaler.step(gpu_optimizer); scaler.update()
    torch.cuda.synchronize()
    assert torch.isfinite(gpu_loss)
    print('CUDA AMP step completed. Loss:', gpu_loss.item())
else:
    print('CUDA AMP skipped. Set RUN_CUDA=True if a compatible CUDA GPU is available.')
'''),
    ("Separate graph capture from optimized compilation", "The eager backend demonstrates `torch.compile` graph capture without requiring a native compiler. It is not a speed optimization. The optional inductor backend generates optimized code and may need a platform-specific toolchain. Compile time can dominate tiny workloads; compare steady-state performance only after compilation and warmup.", '''
compiled_demo = torch.compile(full, backend='eager', fullgraph=True)
with torch.inference_mode():
    assert torch.allclose(compiled_demo(X), full(X), atol=1e-6)
print('Graph capture correctness check passed (eager backend).')
RUN_INDUCTOR = False  # Optional: requires a supported compiler/toolchain for your platform.
if RUN_INDUCTOR:
    optimized = torch.compile(full, backend='inductor')
    with torch.inference_mode():
        optimized(X)  # Compilation / warmup, excluded from timing.
        assert torch.allclose(optimized(X), full(X), atol=1e-5)
    @torch.inference_mode()
    def infer_optimized():
        return optimized(X)
    print(Timer(stmt='infer_optimized()', globals={'infer_optimized': infer_optimized},
                num_threads=2).blocked_autorange(min_run_time=0.2))
else:
    print('Optimized inductor compilation skipped; enable after installing its platform prerequisites.')
''')],
    "1. Repeat the accumulation comparison with microbatch size 5 and Adam.\n2. Print the batch size that resulted from each split to confirm the final one is smaller.\n3. Compute the maximum absolute prediction difference between the graph-captured and eager models.",
    '''
a, b = copy.deepcopy(full), copy.deepcopy(full)
oa, ob = torch.optim.Adam(a.parameters(), lr=0.001), torch.optim.Adam(b.parameters(), lr=0.001)
oa.zero_grad(set_to_none=True)
nn.functional.mse_loss(a(X), y).backward(); oa.step()
ob.zero_grad(set_to_none=True)
for xb, yb in zip(X.split(5), y.split(5)):
    (nn.functional.mse_loss(b(xb), yb) * len(xb) / len(X)).backward()
ob.step()
assert all(torch.allclose(pa, pb, atol=1e-6) for pa, pb in zip(a.parameters(), b.parameters()))
print('Microbatch sizes:', [len(xb) for xb in X.split(5)])
with torch.inference_mode():
    print('Maximum graph-capture difference:', (compiled_demo(X) - full(X)).abs().max().item())
''', "Measure throughput, peak memory, and accuracy together. Faster arithmetic is not useful if data loading dominates or numerical changes hurt the task. Use larger representative workloads before drawing conclusions about GPU or compilation speedups.",
    "- [AMP documentation](https://docs.pytorch.org/docs/stable/amp.html)\n- [torch.compile tutorial](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)\n- [Profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)")

notebook("distributed_training", "Distributed training on two local CPU processes", "Advanced / expert practice", "90–120", "Notebooks 06, 15, and 17; keep the companion examples folder with this notebook.",
    "- Explain ranks, data sharding, and gradient synchronization.\n- Launch an actual two-process DDP training job.\n- Verify that both replicas finish with identical parameters.", [
    ("Inspect how ranks receive different examples", "In data parallel training, each process (rank) holds a copy of the model and processes a shard of data. DDP averages gradients so replicas make the same update. Equal local batch sizes make that average correspond to a global example mean. Our dataset divides exactly across ranks, avoiding sampler padding or dropped examples.", '''
from torch.utils.data.distributed import DistributedSampler
toy_dataset = TensorDataset(torch.arange(12))
shards = []
for rank in range(2):
    sampler = DistributedSampler(toy_dataset, num_replicas=2, rank=rank, shuffle=True, seed=SEED)
    sampler.set_epoch(0)
    shard = list(sampler)
    shards.append(shard)
    print('Rank', rank, 'indices:', shard)
assert set(shards[0]).isdisjoint(shards[1])
assert set(shards[0] + shards[1]) == set(range(12))
'''),
    ("Read the real worker program", "Windows launches fresh Python interpreters for spawned workers. Put worker functions in a script and protect process creation with `if __name__ == '__main__'`; defining the worker only inside a notebook is unreliable. The companion script uses the Gloo backend, a file rendezvous, 2 CPU processes, and destroys the process group in a finally block. Inspect `sampler.set_epoch(epoch)`, `all_reduce`, and the rank-0-only output.", '''
from IPython.display import Code, display
root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'examples' / 'ddp_train.py').exists())
script = root / 'examples' / 'ddp_train.py'
display(Code(script.read_text(encoding='utf-8'), language='python'))
'''),
    ("Launch and verify the distributed run", "This launches two short-lived local workers and normally takes seconds to a minute. It demonstrates synchronization, not a speed benefit. Small CPU jobs usually run slower with DDP overhead. If Gloo is absent, the branch reports a skip; use a PyTorch build with Gloo to complete the process experiment.", '''
import json
import subprocess
import sys
import torch.distributed as dist
RUN_DDP = True
if RUN_DDP and dist.is_available() and dist.is_gloo_available():
    output_dir = root / 'artifacts' / '18_ddp'
    result = subprocess.run([sys.executable, str(script), '--output-dir', str(output_dir)],
                            cwd=str(root), text=True, capture_output=True, timeout=180)
    print(result.stdout)
    if result.returncode:
        raise RuntimeError(result.stderr)
    report = json.loads((output_dir / 'report.json').read_text())
    print(report)
    assert report['world_size'] == 2
    assert report['replica_max_difference'] < 1e-6
    assert report['final_mse'] < 0.03
else:
    print('Two-process training skipped: RUN_DDP=False or Gloo is unavailable.')
''')],
    "1. Demonstrate that calling `set_epoch(1)` changes a shuffled sampler's order.\n2. Calculate the global batch size when each of two ranks processes 16 examples.\n3. Simulate averaging two local gradients and verify it matches the gradient of a combined equally sized batch.",
    '''
sampler = DistributedSampler(toy_dataset, num_replicas=2, rank=0, shuffle=True, seed=SEED)
sampler.set_epoch(0); first = list(sampler)
sampler.set_epoch(1); second = list(sampler)
assert first != second
print('Global batch size:', 2 * 16)
w = torch.tensor(0., requires_grad=True)
local_targets = [torch.tensor([1., 2.]), torch.tensor([3., 4.])]
local_grads = [torch.autograd.grad((w - targets).square().mean(), w)[0] for targets in local_targets]
combined_grad = torch.autograd.grad((w - torch.cat(local_targets)).square().mean(), w)[0]
assert torch.allclose(torch.stack(local_grads).mean(), combined_grad)
''', "DDP does not shard the data for you: use a sampler. All ranks must participate in collectives in a compatible order. On multiple CUDA GPUs, the usual architecture is one process per GPU with an appropriate supported backend; this local Gloo example does not test multi-GPU or multi-node behavior.",
    "- [Getting started with DDP](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html)\n- [DistributedSampler](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler)")

notebook("capstone", "Capstone: a reproducible digit classification experiment", "Advanced / expert practice", "120–180", "Notebooks 00–17; notebook 18 is optional for this project.",
    "- Own a complete workflow from split to saved inference artifact.\n- Compare configurations on validation data and report a final test result.\n- Write an honest model report and design your next experiment.", [
    ("Define the experiment and reserve a test set", "Task: classify a clean 8 × 8 grayscale digit as 0–9. Metric: accuracy plus macro recall. Baseline: the training majority class. Split: 60% train, 20% validation, 20% test with stratification. We scale by the documented fixed pixel range, 0–16. Keep the test set out of configuration and epoch selection.", '''
import copy
import json
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
digits = load_digits()
raw = torch.tensor(digits.images, dtype=torch.float32)
X = (raw / 16).unsqueeze(1)
y = torch.tensor(digits.target, dtype=torch.long)
development, test_idx = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y.numpy(), random_state=SEED)
train_idx, val_idx = train_test_split(development, test_size=0.25, stratify=y[development].numpy(), random_state=SEED)
assert set(train_idx).isdisjoint(val_idx) and set(development).isdisjoint(test_idx)
print('Split sizes:', len(train_idx), len(val_idx), len(test_idx))
'''),
    ("Compare two configurations fairly", "Each run starts from the same seed and uses a freshly seeded shuffled loader. A scheduler lowers the learning rate after each epoch. Select the best checkpoint by validation cross entropy; report validation accuracy alongside it. We use a fixed small search budget, not a promise that either configuration is optimal.", '''
def make_model(width):
    return nn.Sequential(nn.Conv2d(1, width, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Flatten(), nn.Linear(width * 4 * 4, 10))

configs = [{'width': 8, 'lr': 0.003, 'weight_decay': 0.0},
           {'width': 16, 'lr': 0.003, 'weight_decay': 0.001}]
runs = []
for config in configs:
    torch.manual_seed(SEED)
    model = make_model(config['width'])
    loader = DataLoader(TensorDataset(X[train_idx], y[train_idx]), batch_size=64, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED), num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=18)
    best_val = float('inf')
    curve = []
    for epoch in range(18):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(xb), yb)
            loss.backward(); optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            logits = model(X[val_idx])
            val_loss = nn.functional.cross_entropy(logits, y[val_idx]).item()
            val_acc = (logits.argmax(1) == y[val_idx]).float().mean().item()
        curve.append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch, best_acc = epoch + 1, val_acc
    runs.append({'config': config, 'state': best_state, 'val_loss': best_val,
                 'val_accuracy': best_acc, 'epoch': best_epoch, 'curve': curve})
    print(config, 'validation accuracy:', best_acc, 'selected epoch:', best_epoch)
selected = min(runs, key=lambda run: run['val_loss'])
final_model = make_model(selected['config']['width'])
final_model.load_state_dict(selected['state']); final_model.eval()
for run in runs:
    plt.plot(run['curve'], label=f"width={run['config']['width']}")
plt.xlabel('Epoch'); plt.ylabel('Validation cross entropy'); plt.legend(); plt.show()
'''),
    ("Evaluate the locked model", "Now compute test metrics once. A confusion matrix and example errors help describe limitations. They do not authorize another round of tuning on the same test set. This is an educational estimate on one split; writer identity is not available for a writer-disjoint split here.", '''
with torch.inference_mode():
    test_probabilities = final_model(X[test_idx]).softmax(1)
test_predictions = test_probabilities.argmax(1)
truth = y[test_idx]
confusion = torch.bincount(truth * 10 + test_predictions, minlength=100).reshape(10, 10)
accuracy = (test_predictions == truth).float().mean().item()
macro_recall = (confusion.diag() / confusion.sum(1).clamp_min(1)).mean().item()
majority = torch.bincount(y[train_idx]).argmax()
baseline_accuracy = (truth == majority).float().mean().item()
metrics = {'test_accuracy': accuracy, 'test_macro_recall': macro_recall,
           'majority_baseline_accuracy': baseline_accuracy, 'test_examples': len(test_idx)}
print(metrics)
assert accuracy > baseline_accuracy + 0.5
plt.imshow(confusion, cmap='Blues'); plt.xlabel('Predicted class'); plt.ylabel('True class'); plt.colorbar(); plt.show()
mistakes = (test_predictions != truth).nonzero().flatten()[:6]
if len(mistakes):
    fig, axes = plt.subplots(1, len(mistakes), figsize=(12, 2), squeeze=False)
    for ax, i in zip(axes[0], mistakes.tolist()):
        ax.imshow(raw[test_idx[i]], cmap='gray')
        ax.set_title(f'{truth[i].item()} -> {test_predictions[i].item()}'); ax.axis('off')
    plt.show()
'''),
    ("Save an artifact and test the inference contract", "The prediction function accepts raw images `[B,8,8]` in the 0–16 range. It validates shape, finite values, and range, then applies the saved scale. Reload parity verifies the artifact is useful outside the training function. The checkpoint and JSON report are written under artifacts/19_capstone.", '''
root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'notebooks').is_dir() and (p / 'requirements.txt').exists())
output = root / 'artifacts' / '19_capstone'
output.mkdir(parents=True, exist_ok=True)
torch.save({'state': final_model.state_dict(), 'config': selected['config'],
            'pixel_scale': 16., 'classes': list(range(10)), 'seed': SEED}, output / 'digits.pt')
payload = torch.load(output / 'digits.pt', map_location='cpu', weights_only=True)
loaded = make_model(payload['config']['width'])
loaded.load_state_dict(payload['state']); loaded.eval()
def predict_digits(raw_images):
    images = torch.as_tensor(raw_images, dtype=torch.float32, device='cpu')
    if images.ndim != 3 or tuple(images.shape[1:]) != (8, 8):
        raise ValueError('Expected a batch with shape [B, 8, 8].')
    if len(images) == 0 or not torch.isfinite(images).all() or (images < 0).any() or (images > 16).any():
        raise ValueError('Provide a nonempty batch of finite pixels in [0, 16].')
    with torch.inference_mode():
        return loaded((images / payload['pixel_scale']).unsqueeze(1)).softmax(1)
assert torch.allclose(predict_digits(raw[test_idx[:10]]), test_probabilities[:10], atol=1e-6)
report = {'selected_config': selected['config'], 'selected_epoch': selected['epoch'],
          'validation_accuracy': selected['val_accuracy'], **metrics,
          'seed': SEED, 'torch_version': str(torch.__version__),
          'limitations': ['Small low-resolution dataset', 'Single split and seed',
                          'No writer-disjoint evaluation', 'No external or shifted image evaluation']}
(output / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print('Saved model and report to:', output)
''')],
    "1. Verify the inference function rejects an image batch with the wrong shape.\n2. Compute test confidence for correct and incorrect predictions, and explain why this is not a calibration test.\n3. Write a model report in the journal: task, split, baseline, selection method, metrics, failure modes, and one next experiment. Use the structured reference below as a checklist, not as a claim you have run that next experiment.",
    '''
try:
    predict_digits(torch.zeros(4, 64))
except ValueError as error:
    print('Expected validation error:', error)
else:
    raise AssertionError('Bad shape should be rejected.')
confidence = test_probabilities.max(1).values
correct = test_predictions == truth
for name, mask in [('correct', correct), ('incorrect', ~correct)]:
    print(name, 'mean confidence:', confidence[mask].mean().item() if mask.any() else 'no examples')
print(json.dumps(report, indent=2))
# Next experiment proposal: compare across three seeds using validation only,
# then evaluate once on a newly reserved or external writer-disjoint test set.
# Mean confidence by correctness alone does not measure probability calibration.
''', "Completion checklist: rebuild the training loop without copying; explain every tensor shape; beat a baseline; prevent leakage; justify validation selection; reload the artifact; and describe a failure case. Expert skill comes from repeating this process on unfamiliar data and diagnosing failures, not from finishing a fixed number of notebooks.",
    "- [PyTorch quickstart](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)\n- [Digits dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)")


def write_course():
    for idx, item in enumerate(COURSE):
        cells = item['cells']
        navigation = ['[Course guide](../README.md)']
        if idx:
            navigation.append(f"[Previous: {idx-1:02d}]({COURSE[idx-1]['filename']})")
        if idx + 1 < len(COURSE):
            navigation.append(f"[Next: {idx+1:02d}]({COURSE[idx+1]['filename']})")
        cells.insert(1, md(' · '.join(navigation)))
        cells.append(md(' · '.join(navigation)))
        for i, cell in enumerate(cells):
            cell['id'] = f"lab{idx:02d}-cell{i:03d}"
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), f"{item['filename']}:cell{i}", 'exec')
        document = {'cells': cells, 'metadata': {
            'kernelspec': {'display_name': 'Python (PyTorch Practical)', 'language': 'python', 'name': 'pytorch-practical'},
            'language_info': {'name': 'python', 'file_extension': '.py', 'pygments_lexer': 'ipython3'},
        }, 'nbformat': 4, 'nbformat_minor': 5}
        (OUT / item['filename']).write_text(json.dumps(document, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    manifest = [{k: v for k, v in item.items() if k != 'cells'} for item in COURSE]
    (OUT / 'course_index.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Wrote {len(COURSE)} notebooks to {OUT}')


if __name__ == '__main__':
    write_course()
