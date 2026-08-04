# AutoPipe

**Automatic Pipeline Selection for Anomaly Detection in Image Datasets**

AutoPipe is a Python library that automates the selection of preprocessing pipelines for image anomaly detection. Given a dataset and a base anomaly detection model, it searches over a configurable space of image transformations (noise reduction, contrast enhancement, normalization, resizing, etc.), evaluates combinations in parallel, and returns the best-performing pipeline along with a reproducible report.

The library **does not train or implement the base model** — it consumes any model exposing a scikit-learn–style `fit`/`predict` interface (e.g. `IsolationForest`, `OneClassSVM`) or a PyTorch model.

> Developed as part of CE-1114 – Proyecto de Aplicación de la Ingeniería en Computadores,
> Instituto Tecnológico de Costa Rica, I Semester 2026.

---

## Features

- Modular pipeline definition via `BaseTransform` subclasses
- Built-in transforms: `GaussianBlur`, `Normalize`, `Resize`, `CLAHETransform`, `MedianFilter`
- Automatic pipeline search with **Grid Search** or **Bayesian optimization** (Optuna, TPE + median pruning)
- Every step is also tried as **skipped** automatically, so the search can drop transforms that hurt performance (skipped steps simply don't appear in the winning pipeline or its saved config)
- Parallel evaluation with `joblib` (loky backend, Windows-safe), with measured speedup
- Configurable metrics: AUC-ROC, F1-score, precision, recall, false positive rate
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

## Quick Start

The whole high-level flow takes fewer than 20 lines and requires no knowledge of the internals:

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

Three runnable notebooks live in [`examples/`](examples/). Each runs end-to-end in
a clean environment — the dataset-based ones fall back to a small synthetic
dataset if the real data is not present:

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
│   ├── data_loader/     # ImageDataLoader
│   ├── pipeline/        # BaseTransform, transforms, Pipeline
│   ├── search_engine/   # GridSearchEngine, OptunaSearchEngine
│   ├── evaluator/       # DataSplitter, MetricCalculator
│   ├── parallel_runner/ # ParallelRunner
│   └── reporter/        # ReportGenerator
├── tests/
├── examples/
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

## Supported Metrics

| Metric | Key |
|---|---|
| AUC-ROC | `auc_roc` |
| F1-score | `f1` |
| Precision | `precision` |
| Recall | `recall` |
| False Positive Rate | `false_positive_rate` |

All metrics are maximized during search except `false_positive_rate`, which is minimized.

## Out of Scope

- Multi-class classification, semantic segmentation, or object detection
- Training or architecture optimization of the anomaly detection model itself
- REST API or graphical user interface

## Author

Ricardo Antonio Borbón Mena — ricardobm@estudiantec.cr
Supervisor: Prof. Luis Alberto Chavarría Zamora — lachavarria@itcr.ac.cr

## License

MIT
