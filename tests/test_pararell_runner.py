# tests/test_pararell_runner.py
"""Pruebas unitarias para autopipe.parallel_runner.runner.ParallelRunner."""

from __future__ import annotations

import numpy as np
import pytest

from autopipe.parallel_runner.runner import ParallelRunner
from autopipe.pipeline.base import BaseTransform
from autopipe.pipeline.pipeline import Pipeline


class Identity(BaseTransform):
    """Transformación que conserva la imagen sin cambios."""

    def apply(self, image: np.ndarray) -> np.ndarray:
        return image

    def get_params(self) -> dict:
        return {"identity": True}


class Zeroer(BaseTransform):
    """Transformación que destruye la señal devolviendo ceros."""

    def apply(self, image: np.ndarray) -> np.ndarray:
        return np.zeros_like(image)

    def get_params(self) -> dict:
        return {"zero": True}


class MeanThresholdModel:
    """Modelo mock que separa por brillo medio (interfaz scikit-learn)."""

    def __init__(self) -> None:
        self.threshold_ = 0.0

    def fit(self, features: np.ndarray, labels: np.ndarray | None = None) -> "MeanThresholdModel":
        self.threshold_ = float(np.mean(features))
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return (features.mean(axis=1) >= self.threshold_).astype(int)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        margin = features.mean(axis=1) - self.threshold_
        positive = 1.0 / (1.0 + np.exp(-margin))
        return np.column_stack([1.0 - positive, positive])


@pytest.fixture
def dataset() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    normal = rng.integers(0, 20, size=(10, 8, 8), dtype=np.int64)
    anomaly = rng.integers(200, 256, size=(10, 8, 8), dtype=np.int64)
    images = np.concatenate([normal, anomaly], axis=0)
    labels = np.array([0] * 10 + [1] * 10)
    return images, labels


@pytest.fixture
def pipelines() -> list[Pipeline]:
    return [Pipeline([Identity()]), Pipeline([Zeroer()])]


def test_run_returns_results_for_all_pipelines(dataset, pipelines) -> None:
    images, labels = dataset
    runner = ParallelRunner(n_jobs=2, val_size=0.5, random_state=42)
    results = runner.run(pipelines, images, labels, MeanThresholdModel())

    assert len(results) == 2
    assert [record["pipeline_index"] for record in results] == [0, 1]
    assert results[0]["params"] == {"0_Identity": {"identity": True}}


def test_sequential_mode_n_jobs_1(dataset, pipelines) -> None:
    images, labels = dataset
    runner = ParallelRunner(n_jobs=1, val_size=0.5, random_state=42)
    results = runner.run(pipelines, images, labels, MeanThresholdModel())

    assert len(results) == 2
    assert results[0]["metrics"]["auc_roc"] == 1.0


def test_run_records_runtime(dataset, pipelines) -> None:
    images, labels = dataset
    runner = ParallelRunner(n_jobs=1, val_size=0.5, random_state=42)
    assert runner.last_runtime_ is None
    runner.run(pipelines, images, labels, MeanThresholdModel())
    assert isinstance(runner.last_runtime_, float)
    assert runner.last_runtime_ >= 0.0


def test_parallel_matches_sequential(dataset, pipelines) -> None:
    images, labels = dataset
    runner = ParallelRunner(n_jobs=2, val_size=0.5, random_state=42)
    report = runner.benchmark(pipelines, images, labels, MeanThresholdModel())

    assert report["consistent"] is True


def test_benchmark_reports_speedup(dataset, pipelines) -> None:
    images, labels = dataset
    runner = ParallelRunner(n_jobs=2, val_size=0.5, random_state=42)
    report = runner.benchmark(pipelines, images, labels, MeanThresholdModel())

    assert set(report) == {
        "results",
        "sequential_time",
        "parallel_time",
        "speedup",
        "consistent",
    }
    assert isinstance(report["speedup"], float)
    assert report["speedup"] > 0.0


def test_reproducible_same_seed(dataset, pipelines) -> None:
    images, labels = dataset
    results_a = ParallelRunner(n_jobs=1, val_size=0.5, random_state=7).run(
        pipelines, images, labels, MeanThresholdModel()
    )
    results_b = ParallelRunner(n_jobs=1, val_size=0.5, random_state=7).run(
        pipelines, images, labels, MeanThresholdModel()
    )

    objectives_a = [record["objective"] for record in results_a]
    objectives_b = [record["objective"] for record in results_b]
    assert objectives_a == objectives_b


def test_results_match_detects_differences() -> None:
    base = [{"metrics": {"auc_roc": 1.0, "f1": 1.0}}]

    assert ParallelRunner._results_match(base, base) is True
    # Tamaño distinto
    assert ParallelRunner._results_match(base, base + base) is False
    # Claves distintas
    assert ParallelRunner._results_match(base, [{"metrics": {"auc_roc": 1.0}}]) is False
    # Valor distinto
    assert (
        ParallelRunner._results_match(base, [{"metrics": {"auc_roc": 0.5, "f1": 1.0}}])
        is False
    )
    # NaN se considera igual a NaN
    nan_record = [{"metrics": {"auc_roc": float("nan"), "f1": 1.0}}]
    assert ParallelRunner._results_match(nan_record, nan_record) is True


def test_invalid_metric_raises(dataset, pipelines) -> None:
    images, labels = dataset
    runner = ParallelRunner(n_jobs=1, val_size=0.5, random_state=0)
    with pytest.raises(ValueError):
        runner.run(pipelines, images, labels, MeanThresholdModel(), metric="accuracy")
