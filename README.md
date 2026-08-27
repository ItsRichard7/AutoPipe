# AutoPipe

**Task-agnostic Automatic Preprocessing Pipeline Selection**

AutoPipe is a Python library that automates the selection of preprocessing pipelines for machine learning. Given a dataset, a learning task and a base model, it searches over a configurable space of transformations, evaluates combinations in parallel, and returns the best-performing pipeline along with a reproducible report.

Two task families are supported today:

- **Anomaly detection** on image datasets (the original scope, preserved unchanged).
- **Classification**, binary or multiclass, on tabular datasets.

The library **does not train or implement the base model** — it consumes any model exposing a scikit-learn–style `fit`/`predict` interface (e.g. `KNeighborsClassifier`, `IsolationForest`, `OneClassSVM`) or a PyTorch model.

> Started as CE-1114 – Proyecto de Aplicación de la Ingeniería en Computadores
> (I Semester 2026) and currently being extended as a Graduation Project (TFG),
> Instituto Tecnológico de Costa Rica.

---

## Features

- Modular pipeline definition via `BaseTransform` subclasses, with optional `fit`/`transform` semantics for **stateful** transforms
- Built-in image transforms: `GaussianBlur`, `Normalize`, `Resize`, `CLAHETransform`, `MedianFilter`, `GammaCorrection`, `GrayscaleConversion`
- Built-in tabular transforms: `ImputeMissing`, `ScaleNumeric`, `EncodeCategorical`, `SelectFeatures`
- A **canonical, versioned tabular search space**: one space shared by every dataset, so a configuration found on one is executable on another, and a version identifier that makes a stale configuration fail loudly instead of silently designating the wrong transforms
- Pluggable **learning tasks**: metrics, optimization direction and prediction handling travel together, selected with a single constructor argument
- Automatic pipeline search with **Grid Search** or **Bayesian optimization** (Optuna, TPE + median pruning)
- Every step is also tried as **skipped** automatically, so the search can drop transforms that hurt performance (skipped steps simply don't appear in the winning pipeline or its saved config)
- Parallel evaluation with `joblib` (loky backend, Windows-safe), with measured speedup
- **No data leakage**: pipelines are fitted on the training subset only, and cloned per candidate so stateful transforms cannot contaminate each other
- Reproducible results: same seed ⇒ same winning pipeline; winning config saved as JSON/YAML and re-runnable
- Report generation: **HTML always**, **PDF optional** (pdfkit + wkhtmltopdf, tolerant if the binary is missing)

## Requirements

- Python **>= 3.10** (developed and tested on 3.12)
- For PDF export: the external [`wkhtmltopdf`](https://wkhtmltopdf.org/) binary must be on your `PATH`. If it is absent, HTML reports are still produced and a warning is logged.

## Installation

```bash
# Clone the repo
git clone https://github.com/ricardobm/AutoPipe.git
cd AutoPipe

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

## Quick Start — tabular classification

The whole high-level flow takes fewer than 20 statements and requires no knowledge of the internals. This snippet is a condensed version of [`examples/tabular_quickstart.py`](examples/tabular_quickstart.py), which is executed by the test suite:

```python
from sklearn.neighbors import KNeighborsClassifier

from autopipe import GridSearchEngine, TabularDataLoader, build_tabular_search_space

# 1. Load the dataset (a CSV path or an in-memory DataFrame; last column is the target)
features, labels = TabularDataLoader("path/to/dataset.csv").load()

# 2. Ask for the canonical search space: 4 steps, 144 pipelines counting the skip option.
#    Building one by hand also works — it is just a dict of step name to option list.
search_space = build_tabular_search_space()

# 3. Run the search. The task drives metrics, so macro-F1 is used by default.
engine = GridSearchEngine(search_space, random_state=42, task="classification")
best_pipeline, metrics, results = engine.run(
    features, labels, base_model=KNeighborsClassifier(n_neighbors=5),
)

print(f"Best pipeline: {best_pipeline}")
print(f"macro-F1: {metrics['macro_f1']:.4f}")
```

Transforms are fitted **on the training subset only**, so no statistic from the validation subset can leak into the training data.

## Quick Start — image anomaly detection

The original flow is unchanged:

```python
from sklearn.ensemble import IsolationForest

from autopipe import GaussianBlur, GridSearchEngine, ImageDataLoader, Normalize, Resize, ReportGenerator

# 1. Load the dataset (expects normal/ and anomaly/ subdirectories)
images, labels = ImageDataLoader(dataset_path="path/to/dataset").load()

# 2. Define the preprocessing search space (one option list per step)
search_space = {
    "resize": [Resize(width=64, height=64)],
    "blur": [GaussianBlur(kernel_size=3), GaussianBlur(kernel_size=5)],
    "normalize": [Normalize(mean=0.5, std=0.5)],
}

# 3. Run the automatic search against your base model
engine = GridSearchEngine(search_space, val_size=0.2, random_state=42)
best_pipeline, metrics, results = engine.run(
    images, labels, base_model=IsolationForest(random_state=0), metric="auc_roc",
)

# 4. Generate a reproducible report (HTML always; PDF if wkhtmltopdf is available).
#    By default it is written to a reports/ folder next to your script/notebook.
report = ReportGenerator().generate(results, best_pipeline, metrics)

print(f"Best pipeline: {best_pipeline}")
print(f"AUC-ROC: {metrics['auc_roc']:.4f}  ->  {report['html']}")
```

`engine.run(...)` returns a tuple `(best_pipeline, best_metrics, results)`:

- `best_pipeline` — the winning `Pipeline`
- `best_metrics` — `dict` of all metrics for the winner
- `results` — the full comparison table (`list[dict]`), one entry per evaluated pipeline

### Examples

Four runnable notebooks live in [`examples/`](examples/). Each runs end-to-end in
a clean environment — the dataset-based ones fall back to a small synthetic
dataset if the real data is not present:

- [`tabular_walkthrough.ipynb`](examples/tabular_walkthrough.ipynb) — the tabular
  path end to end: the canonical versioned space, why some candidates are
  unevaluable, the no-leakage guarantee demonstrated live, and reproducibility.
- [`basic_usage.ipynb`](examples/basic_usage.ipynb) — full walkthrough on a
  synthetic dataset (Grid + Optuna, parallel runner, report, re-runnable config).
- [`mvtec_anomaly_detection.ipynb`](examples/mvtec_anomaly_detection.ipynb) —
  [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) industrial
  anomalies, including an adapter from MVTec's native layout to `normal/`+`anomaly/`.
- [`cifar10_synthetic_anomalies.ipynb`](examples/cifar10_synthetic_anomalies.ipynb) —
  CIFAR-10 in a one-vs-rest anomaly setup: Optuna over a **complete search space
  with all 7 built-in transforms**, including their ordering constraints.

## Bayesian search with Optuna

`OptunaSearchEngine` is a drop-in alternative that accepts the **same search space** and exposes the same `run(...)` signature, using a `TPESampler` (seeded for reproducibility) and a `MedianPruner`:

```python
from autopipe import OptunaSearchEngine

engine = OptunaSearchEngine(search_space, n_trials=50, random_state=42)
best_pipeline, metrics, results = engine.run(images, labels, base_model, metric="auc_roc")
```

## Parallel evaluation

`ParallelRunner` evaluates a list of pipelines concurrently and can report the speedup against sequential execution. Use `n_jobs=1` for sequential mode (handy in Jupyter/Windows):

```python
from autopipe import ParallelRunner

runner = ParallelRunner(n_jobs=-1, random_state=42)
report = runner.benchmark([best_pipeline], images, labels, base_model, metric="auc_roc")
print(f"Speedup: {report['speedup']:.2f}x  (consistent: {report['consistent']})")
```

> On Windows, joblib uses the `loky` backend by default; protect any script entry point
> with `if __name__ == "__main__":` when running parallel jobs from a `.py` file.

## Reproducing the winning pipeline

The winning configuration is saved during report generation and can be reloaded and rebuilt:

```python
config = ReportGenerator.load_pipeline_config("reports/winning_pipeline.json")
pipeline = ReportGenerator.rebuild_pipeline(config)   # built-in transforms only
```

## Project Structure

```
AutoPipe/
├── autopipe/          # main package
│   ├── data_loader/     # ImageDataLoader, TabularDataLoader
│   ├── pipeline/        # BaseTransform, StatefulTransform, transforms, Pipeline
│   ├── tasks/           # Task registry: anomaly detection, classification
│   ├── search_engine/   # GridSearchEngine, OptunaSearchEngine
│   ├── search_space/    # canonical versioned tabular space
│   ├── evaluator/       # DataSplitter, metric calculators
│   ├── parallel_runner/ # ParallelRunner
│   └── reporter/        # ReportGenerator
├── tests/
├── examples/
├── scripts/           # reproducible utilities (corpus listing, regression check)
├── spikes/            # feasibility experiments
├── docs/
└── configs/
```

## Running Tests

```bash
pytest
```

Coverage (`pytest-cov`) is printed automatically. To generate an HTML coverage report:

```bash
pytest --cov-report=html
# open htmlcov/index.html
```

> Note: the `coverage` C-tracer is currently incompatible with NumPy on Python 3.14.
> Use Python 3.12 (or 3.10/3.11) for the test suite.

## Supported Tasks and Metrics

The task is selected with the `task` argument of `GridSearchEngine`, `OptunaSearchEngine` and `ParallelRunner`. It defaults to `"anomaly_detection"`, so existing code keeps its behaviour. If `metric` is omitted, the task's primary metric is used.

### `anomaly_detection` (default)

| Metric | Key |
|---|---|
| AUC-ROC (primary) | `auc_roc` |
| F1-score | `f1` |
| Precision | `precision` |
| Recall | `recall` |
| False Positive Rate | `false_positive_rate` |

All are maximized except `false_positive_rate`, which is minimized.

### `classification`

| Metric | Key |
|---|---|
| Macro F1-score (primary) | `macro_f1` |
| Accuracy | `accuracy` |
| Balanced accuracy | `balanced_accuracy` |
| Macro precision | `macro_precision` |
| Macro recall | `macro_recall` |

All are maximized. Macro averaging weights every class equally, so a majority-class predictor scores poorly even on imbalanced datasets.

## Documentation

- [`docs/GUIA-tecnica.md`](docs/GUIA-tecnica.md) — technical guide (Spanish): architecture, the transform contract, what the evaluation protocol guarantees, and common errors.
- [`docs/decisiones/`](docs/decisiones/) — architecture decision records and the coupling audit.
- [`docs/experimentos/`](docs/experimentos/) — experimental pre-registration, feasibility spikes and corpus selection.

## Out of Scope

- Text, time series, or graph data
- Semantic segmentation and object detection
- Training or architecture optimization of the user's base model
- Generative data augmentation (e.g. GANs)
- REST API or graphical user interface

## Author

Ricardo Antonio Borbón Mena — ricardobm@estudiantec.cr
Supervisor: Prof. Luis Alberto Chavarría Zamora — lachavarria@itcr.ac.cr

## License

MIT
