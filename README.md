# PyTorch Practical — learn by building

Twenty Jupyter notebooks take you from your first tensor to advanced training and debugging practices. They assume basic Python and introduce the PyTorch concepts as you need them. Every notebook includes numbered steps, runnable examples, checks, three exercises, reference solutions, an experiment journal, and links to official documentation.

Start with [00 — Your first PyTorch experiment](notebooks/00_start_here.ipynb). Work in order through notebook 09 before choosing later projects. The advanced lessons introduce expert practices; mastery comes from applying them to new problems and explaining the failures you encounter.

## Open the course on this computer

The workspace now has a `.venv` with the course tools. From PowerShell in this folder, run:

```powershell
.\.venv\Scripts\python.exe -m jupyterlab
```

In JupyterLab, open `notebooks/00_start_here.ipynb` and select **Python (PyTorch Practical)**. Use **Shift+Enter** to run each cell. The first cell prints the Python and PyTorch environment. No environment activation or PowerShell execution-policy change is needed.

Alternatively, open the folder in VS Code with its Python and Jupyter extensions, open a notebook, and select `.venv\Scripts\python.exe` as its kernel.

The local environment reuses the existing PyTorch installation through `--system-site-packages`; newly installed course tools live in `.venv`. For an independent setup on another machine, follow the steps below.

## Fresh setup

On Windows, with Python 3.11–3.13 installed:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --prefix .venv --name pytorch-practical --display-name "Python (PyTorch Practical)"
.\.venv\Scripts\python.exe -m jupyterlab
```

Use `py -3.11` or `py -3.12` in the first command if that is your installed version. On macOS/Linux, use `python3 -m venv .venv` and replace `.\.venv\Scripts\python.exe` with `.venv/bin/python` and `--prefix .venv` remains the same.

CPU is sufficient for all required lessons. For a specific CPU-only or CUDA PyTorch build, use the [official installation selector](https://pytorch.org/get-started/locally/) before installing the remaining requirements. The dependency ranges are compatibility targets; the exact versions actually tested are recorded in `validation/report.json` and `validation/tested_versions.txt`.

Installation requires internet access. Once installed, the lessons use generated data or the small digits dataset bundled with scikit-learn. They do not download datasets, pretrained weights, or require service credentials. Notebook 18 requires its companion `examples/ddp_train.py`; keep the repository folders together.

## Learning path

Times below are study estimates including experiments, not just execution time. Start with three sessions per week and adjust the pace when you can explain the results without reading the solution.

| Order | Notebook | What you build or investigate | Study time |
|---|---|---|---|
| 00 | [Start here](notebooks/00_start_here.ipynb) | Predictions, parameters, and loss | 30–45 min |
| 01 | [Tensors](notebooks/01_tensors.ipynb) | Indexing, dtypes, devices, copies | 45–60 min |
| 02 | [Shapes and broadcasting](notebooks/02_shapes_and_broadcasting.ipynb) | Matrix arithmetic and silent shape bugs | 45–60 min |
| 03 | [Autograd](notebooks/03_autograd.ipynb) | Derivatives and gradient descent | 45–60 min |
| 04 | [Linear regression](notebooks/04_linear_regression.ipynb) | A complete training loop from scratch | 60–75 min |
| 05 | [Neural networks](notebooks/05_neural_networks.ipynb) | Nonlinear regression with an MLP | 60–90 min |
| 06 | [Datasets and DataLoaders](notebooks/06_datasets_and_dataloaders.ipynb) | Preprocessing and minibatch learning | 60–75 min |
| 07 | [Classification](notebooks/07_classification.ipynb) | Binary/multiclass classifiers and metrics | 75–90 min |
| 08 | [Generalization](notebooks/08_generalization.ipynb) | Regularization and early stopping | 75–90 min |
| 09 | [CNN digits](notebooks/09_cnn_digits.ipynb) | Real handwritten-digit recognition | 75–100 min |
| 10 | [Transfer learning](notebooks/10_transfer_learning.ipynb) | Pretrain, freeze, replace a head, fine-tune | 75–100 min |
| 11 | [Embeddings and text](notebooks/11_embeddings_and_text.ipynb) | A tiny sentiment classifier | 60–90 min |
| 12 | [Sequence forecasting](notebooks/12_sequence_forecasting.ipynb) | GRU forecasts with chronological splits | 75–100 min |
| 13 | [Attention and transformers](notebooks/13_attention_and_transformers.ipynb) | A causal next-token transformer | 90–120 min |
| 14 | [Autoencoders](notebooks/14_autoencoders.ipynb) | Denoise digits and inspect latent space | 75–90 min |
| 15 | [Checkpoints and inference](notebooks/15_checkpoints_and_inference.ipynb) | Save, reload, resume, verify predictions | 60–90 min |
| 16 | [Custom autograd and debugging](notebooks/16_custom_autograd_and_debugging.ipynb) | Gradient checks and tiny-batch overfitting | 75–100 min |
| 17 | [Performance](notebooks/17_performance.ipynb) | Profiling, accumulation, AMP, graph capture | 90–120 min |
| 18 | [Distributed training](notebooks/18_distributed_training.ipynb) | Actual two-process CPU DDP | 90–120 min |
| 19 | [Capstone](notebooks/19_capstone.ipynb) | Compare models, test once, save an artifact | 120–180 min |

Suggested milestones:

- **Foundation, 00–04:** write a small training loop from memory and explain every gradient reset.
- **Independent training, 05–09:** prepare data, choose a loss, beat a baseline, and inspect validation errors.
- **Model families, 10–14:** choose an architecture that fits the structure of a task and explain its limits.
- **Engineering, 15–18:** reproduce predictions, diagnose training failures, measure performance, and synchronize replicas.
- **Capstone, 19:** complete an experiment with a reserved test set and an honest report.

## How to learn actively

1. Duplicate a notebook before making extensive changes, or keep your experiments in a new notebook. Do not run `tools/build_notebooks.py` after editing the supplied notebooks unless you intend to regenerate them.
2. Read a step and predict the output shape, trend, or metric before running it.
3. Change one setting: learning rate, batch size, model width, noise, or training duration. Record a measurement and an explanation.
4. Try all three exercises before expanding the reference solution. Some notebook viewers show collapsed source differently; solutions are clearly labeled at the bottom.
5. Restart the kernel and run all cells to catch accidental reliance on old variables.
6. Rebuild the main example in a blank notebook. Use the original only when you get stuck.

All supplied example and solution cells are executable. Blank exercise cells are intentional places for your attempts. Assertions check properties such as shapes, gradients, checkpoint parity, and learning progress; exact scores can vary across versions and hardware.

## Important experiment boundaries

- **Training data** updates parameters; **validation data** selects architecture and stopping time; **test data** provides a final estimate after decisions are locked.
- Fit data-dependent preprocessing on training observations only. Keep chronological splits for forecasting.
- The transfer lesson pretrains a small encoder locally. It does not download a large pretrained vision model.
- The tiny text and cyclic-token datasets teach mechanics. Their scores are not evidence of real-world language performance.
- The distributed lesson uses two CPU processes with Gloo. It does not require GPUs or test multi-node infrastructure.
- The performance lesson runs CPU profiling and `torch.compile` with the eager backend by default. This demonstrates graph capture, not a compiler speedup. CUDA AMP is opt-in (`RUN_CUDA=True`), and optimized Inductor compilation is separately opt-in because it requires a suitable toolchain.

## Files and generated results

All 20 notebooks, including their reference solutions, were executed successfully in fresh kernels on this Windows machine with Python 3.13 and PyTorch 2.11.0+cu128. Validation also exercised CUDA AMP and two-process CPU DDP. Optimized Inductor compilation remains an optional, untested extension. See [the validation report](validation/report.json) for package versions and individual results.

```text
notebooks/                  20 clean notebooks, plus a machine-readable course index
examples/ddp_train.py        Worker program for notebook 18
tools/build_notebooks.py     Authoring source; regenerates supplied notebooks
tools/validate_notebooks.py  Fresh-kernel execution and validation
requirements.txt            Installation requirements
validation/report.json      Notebook validation results and environment versions
validation/executed/         Executed copies with plots and reference outputs
artifacts/                  Checkpoints, DDP results, and the capstone report
```

Notebooks 15, 18, and 19 write only their own named files under `artifacts/`; rerunning overwrites those example results. Keep any results you want to compare under a new name. The notebooks do not depend on artifacts produced by previous lessons.

To rerun validation from the project folder:

```powershell
.\.venv\Scripts\python.exe tools\validate_notebooks.py
```

To include CUDA AMP when a compatible GPU is available:

```powershell
.\.venv\Scripts\python.exe tools\validate_notebooks.py --cuda
```

To check just one lesson:

```powershell
.\.venv\Scripts\python.exe tools\validate_notebooks.py --pattern "13_*.ipynb"
```

The validator executes examples and solutions in fresh kernels and saves outputs separately, keeping the learning notebooks clean. It registers a kernel named `pytorch-course-validation` for the interpreter running the tool. Optimized Inductor compilation is not part of automatic validation.

If validation reports failures after you make changes, rerun only those entries with `--retry-failed` (and the same `--cuda` setting as the original run). This updates the full report while preserving the successful checks.

## Troubleshooting

| Symptom | What to check |
|---|---|
| `python` opens the Microsoft Store | Use the explicit `.venv\Scripts\python.exe` command above; for setup use the `py` launcher. |
| `ModuleNotFoundError` | Check the notebook kernel's `sys.executable`; install requirements with that same interpreter. |
| Course kernel is missing | Rerun the kernel registration command, then restart JupyterLab. |
| Variables are missing or stale | Restart the kernel and run all cells in order. |
| An edited model raises a size mismatch | Print input, prediction, and target shapes; revisit notebook 02. |
| Loss becomes NaN | Restore the original settings; inspect inputs, gradients, and learning rate in notebook 16. |
| Distributed process errors | Run `examples/ddp_train.py` from the terminal for the full traceback; verify Gloo is available and local worker communication is permitted. |
| Notebook runs slowly | Start with supplied CPU settings, close other active kernels, and reduce exploratory epochs rather than increasing model size. |

## Small glossary

| Term | Meaning in these notebooks |
|---|---|
| Tensor | Numbers arranged along named-by-convention axes such as batch, feature, or time. |
| Parameter | A model value that training can adjust. |
| Gradient | Local sensitivity of a loss to a parameter. |
| Optimizer | An algorithm that uses gradients to update parameters. |
| Epoch | One pass through the training dataset. |
| Batch | A group of examples processed together. |
| Logit | An unconstrained class score before sigmoid or softmax. |
| Overfitting | Learning details of training observations that do not generalize well. |
| Checkpoint | Saved model state and, when resuming training, associated training state. |
| Inference | Using a trained model to make predictions. |

Continue with the [official PyTorch tutorials](https://docs.pytorch.org/tutorials/) when you want to extend a lesson. Each notebook links to the relevant API or tutorial.
