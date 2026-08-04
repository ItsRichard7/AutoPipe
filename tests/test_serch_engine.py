# tests/test_serch_engine.py
"""Pruebas unitarias para el módulo autopipe.search_engine."""

from __future__ import annotations

import math

import numpy as np
import pytest

from autopipe.pipeline.base import BaseTransform
from autopipe.pipeline.pipeline import Pipeline
from autopipe.pipeline.transforms import Resize
from autopipe.search_engine.grid_search import GridSearchEngine
from autopipe.search_engine.optuna_search import OptunaSearchEngine


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
def separable_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Dataset donde las anomalías son más brillantes que las normales."""
    rng = np.random.default_rng(0)
    normal = rng.integers(0, 20, size=(10, 8, 8), dtype=np.int64)
    anomaly = rng.integers(200, 256, size=(10, 8, 8), dtype=np.int64)
    images = np.concatenate([normal, anomaly], axis=0)
    labels = np.array([0] * 10 + [1] * 10)
    return images, labels


@pytest.fixture
def search_space() -> dict:
    return {"step": [Identity(), Zeroer()]}


# --------------------------------------------------------------------------- #
# GridSearchEngine
# --------------------------------------------------------------------------- #
def test_grid_evaluates_full_cartesian_product(separable_dataset, search_space) -> None:
    images, labels = separable_dataset
    engine = GridSearchEngine(search_space, val_size=0.5, random_state=42)
    best_pipeline, best_metrics, results = engine.run(images, labels, MeanThresholdModel())

    # 2 candidatos del usuario + la opción automática de omitir el paso.
    assert len(results) == 3
    assert isinstance(best_pipeline, Pipeline)
    assert set(best_metrics) >= {"auc_roc", "f1", "precision", "recall"}


def test_grid_adds_skip_option_per_step(separable_dataset) -> None:
    images, labels = separable_dataset
    space = {"a": [Identity(), Zeroer()], "b": [Identity()]}
    engine = GridSearchEngine(space, val_size=0.5, random_state=42)
    _, _, results = engine.run(images, labels, MeanThresholdModel())

    # (2 + omitir) x (1 + omitir) = 6 combinaciones.
    assert len(results) == 6
    # Una de las combinaciones es el pipeline completamente vacío.
    assert any(record["params"] == {} for record in results)


def test_grid_skip_wins_when_transform_is_harmful(separable_dataset) -> None:
    images, labels = separable_dataset
    engine = GridSearchEngine({"step": [Zeroer()]}, val_size=0.5, random_state=42)
    best_pipeline, best_metrics, results = engine.run(images, labels, MeanThresholdModel())

    assert len(results) == 2
    assert best_pipeline.steps == []
    assert best_metrics["auc_roc"] == 1.0


def test_grid_failing_skip_candidate_does_not_abort_search() -> None:
    # Imágenes heterogéneas: sin Resize no se puede armar la matriz de features.
    rng = np.random.default_rng(0)
    images = np.empty(20, dtype=object)
    for i in range(20):
        side = 8 if i < 10 else 16
        images[i] = rng.integers(0, 255, size=(side, side), dtype=np.uint8)
    labels = np.array([0] * 10 + [1] * 10)

    engine = GridSearchEngine(
        {"resize": [Resize(width=8, height=8)]}, val_size=0.5, random_state=42
    )
    best_pipeline, _, results = engine.run(images, labels, MeanThresholdModel())

    assert len(results) == 2
    # El candidato "omitir resize" falló y quedó registrado con NaN, sin ganar.
    assert math.isnan(results[1]["objective"])
    assert len(best_pipeline.steps) == 1


def test_grid_selects_better_pipeline(separable_dataset, search_space) -> None:
    images, labels = separable_dataset
    engine = GridSearchEngine(search_space, val_size=0.5, random_state=42)
    best_pipeline, best_metrics, _ = engine.run(
        images, labels, MeanThresholdModel(), metric="auc_roc"
    )

    assert isinstance(best_pipeline.steps[0], Identity)
    assert best_metrics["auc_roc"] == 1.0


def test_grid_minimizes_false_positive_rate(separable_dataset, search_space) -> None:
    images, labels = separable_dataset
    engine = GridSearchEngine(search_space, val_size=0.5, random_state=42)
    best_pipeline, best_metrics, _ = engine.run(
        images, labels, MeanThresholdModel(), metric="false_positive_rate"
    )

    assert isinstance(best_pipeline.steps[0], Identity)
    assert best_metrics["false_positive_rate"] == 0.0


def test_grid_is_reproducible(separable_dataset, search_space) -> None:
    images, labels = separable_dataset
    run_a = GridSearchEngine(search_space, val_size=0.5, random_state=7).run(
        images, labels, MeanThresholdModel()
    )
    run_b = GridSearchEngine(search_space, val_size=0.5, random_state=7).run(
        images, labels, MeanThresholdModel()
    )

    objectives_a = [record["objective"] for record in run_a[2]]
    objectives_b = [record["objective"] for record in run_b[2]]
    assert objectives_a == objectives_b
    assert run_a[0].get_params() == run_b[0].get_params()


def test_grid_invalid_metric_raises(separable_dataset, search_space) -> None:
    images, labels = separable_dataset
    engine = GridSearchEngine(search_space, val_size=0.5, random_state=0)
    with pytest.raises(ValueError):
        engine.run(images, labels, MeanThresholdModel(), metric="accuracy")


def test_grid_records_objective_matches_metric(separable_dataset, search_space) -> None:
    images, labels = separable_dataset
    engine = GridSearchEngine(search_space, val_size=0.5, random_state=1)
    _, _, results = engine.run(images, labels, MeanThresholdModel(), metric="f1")
    for record in results:
        assert record["objective"] == record["metrics"]["f1"]


# --------------------------------------------------------------------------- #
# OptunaSearchEngine
# --------------------------------------------------------------------------- #
def test_optuna_runs_requested_trials(separable_dataset, search_space) -> None:
    images, labels = separable_dataset
    engine = OptunaSearchEngine(search_space, n_trials=5, val_size=0.5, random_state=42)
    best_pipeline, best_metrics, results = engine.run(images, labels, MeanThresholdModel())

    assert len(results) == 5
    assert isinstance(best_pipeline, Pipeline)
    assert "auc_roc" in best_metrics


def test_optuna_finds_better_pipeline(separable_dataset, search_space) -> None:
    images, labels = separable_dataset
    engine = OptunaSearchEngine(search_space, n_trials=8, val_size=0.5, random_state=42)
    best_pipeline, best_metrics, _ = engine.run(
        images, labels, MeanThresholdModel(), metric="auc_roc"
    )

    # Identity y "omitir" son igualmente óptimos; Zeroer nunca debe ganar.
    assert not any(isinstance(step, Zeroer) for step in best_pipeline.steps)
    assert best_metrics["auc_roc"] == 1.0


def test_optuna_skip_wins_when_transform_is_harmful(separable_dataset) -> None:
    images, labels = separable_dataset
    engine = OptunaSearchEngine(
        {"step": [Zeroer()]}, n_trials=6, val_size=0.5, random_state=42
    )
    best_pipeline, best_metrics, _ = engine.run(images, labels, MeanThresholdModel())

    assert best_pipeline.steps == []
    assert best_metrics["auc_roc"] == 1.0


def test_optuna_is_reproducible(separable_dataset, search_space) -> None:
    images, labels = separable_dataset
    best_a = OptunaSearchEngine(search_space, n_trials=6, val_size=0.5, random_state=11).run(
        images, labels, MeanThresholdModel()
    )[0]
    best_b = OptunaSearchEngine(search_space, n_trials=6, val_size=0.5, random_state=11).run(
        images, labels, MeanThresholdModel()
    )[0]

    assert best_a.get_params() == best_b.get_params()


def test_optuna_invalid_metric_raises(separable_dataset, search_space) -> None:
    images, labels = separable_dataset
    engine = OptunaSearchEngine(search_space, n_trials=3, val_size=0.5, random_state=0)
    with pytest.raises(ValueError):
        engine.run(images, labels, MeanThresholdModel(), metric="accuracy")
