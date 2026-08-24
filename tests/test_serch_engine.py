# tests/test_serch_engine.py
"""Pruebas unitarias para el módulo autopipe.search_engine."""

from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from autopipe.evaluator.metrics import MetricCalculator
from autopipe.pipeline.base import BaseTransform, Dataset, StatefulTransform
from autopipe.pipeline.pipeline import Pipeline
from autopipe.pipeline.transforms import Resize
from autopipe.search_engine.grid_search import (
    GridSearchEngine,
    evaluate_pipeline,
    evaluate_pipeline_safe,
    is_better,
)
from autopipe.search_engine.optuna_search import OptunaSearchEngine
from autopipe.tasks import CLASSIFICATION


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


class RecordingCenterer(StatefulTransform):
    """Centra por columna y **recuerda** con qué datos fue ajustada.

    Sirve para verificar que el ajuste ocurre solo con el subconjunto de
    entrenamiento y que un candidato no contamina al siguiente.
    """

    def __init__(self) -> None:
        self.seen_: np.ndarray | None = None
        self.fit_count = 0

    def _fit(self, X: Dataset, y: np.ndarray | None = None) -> None:
        self.seen_ = np.asarray(X, dtype=float).copy()
        self.mean_ = np.asarray(X, dtype=float).mean(axis=0)
        self.fit_count += 1

    def _transform(self, X: Dataset) -> Dataset:
        return np.asarray(X, dtype=float) - self.mean_

    def get_params(self) -> dict:
        return {}


class MeanRegressorModel:
    """Modelo mock determinista que predice según la media de cada fila."""

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "MeanRegressorModel":
        self.threshold_ = float(np.mean(features))
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return (features.mean(axis=1) >= self.threshold_).astype(int)


# --------------------------------------------------------------------------- #
# evaluate_pipeline: ajuste solo con entrenamiento y aislamiento de candidatos
# --------------------------------------------------------------------------- #
def _tabular_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Dataset donde entrenamiento y validación tienen escalas muy distintas."""
    data = np.array([[0.0], [10.0], [1000.0], [1010.0]])
    labels = np.array([0, 1, 0, 1])
    return data, labels, np.array([0, 1]), np.array([2, 3])


def test_pipeline_is_fitted_only_with_the_training_subset() -> None:
    """Prueba anti-fuga a nivel de protocolo.

    La transformación con estado debe ver **solo** las filas de entrenamiento.
    Si el ajuste ocurriera sobre el dataset completo, ``seen_`` tendría cuatro
    filas y las estadísticas de validación se habrían filtrado.
    """
    data, labels, train_idx, val_idx = _tabular_dataset()
    centerer = RecordingCenterer()

    evaluate_pipeline(
        Pipeline([centerer]),
        data,
        labels,
        MeanRegressorModel(),
        "auc_roc",
        train_idx,
        val_idx,
        MetricCalculator(),
    )

    # El pipeline se clona, así que la instancia original nunca se ajusta.
    assert centerer.fit_count == 0
    assert centerer.seen_ is None


def test_the_fitted_clone_only_saw_training_rows() -> None:
    """Verifica sobre el clon que el ajuste usó únicamente el entrenamiento."""
    data, labels, train_idx, val_idx = _tabular_dataset()
    pipeline = Pipeline([RecordingCenterer()])
    fitted: list[RecordingCenterer] = []

    original_fit = RecordingCenterer._fit

    def spy(self: RecordingCenterer, X: Dataset, y: np.ndarray | None = None) -> None:
        original_fit(self, X, y)
        fitted.append(self)

    RecordingCenterer._fit = spy  # type: ignore[method-assign]
    try:
        evaluate_pipeline(
            pipeline, data, labels, MeanRegressorModel(), "auc_roc",
            train_idx, val_idx, MetricCalculator(),
        )
    finally:
        RecordingCenterer._fit = original_fit  # type: ignore[method-assign]

    assert len(fitted) == 1
    np.testing.assert_allclose(fitted[0].seen_, data[train_idx])


def test_candidates_sharing_an_instance_do_not_contaminate_each_other() -> None:
    """Prueba de contaminación entre candidatos.

    Los motores construyen cada candidato tomando instancias del espacio de
    búsqueda, de modo que la misma transformación aparece en varios pipelines.
    Evaluar dos candidatos que comparten una instancia debe dar el mismo
    resultado que evaluarlos por separado. Sin el clonado, el estado del primero
    se arrastraría al segundo.
    """
    data, labels, train_idx, val_idx = _tabular_dataset()
    compartida = RecordingCenterer()

    def evaluar(pipeline: Pipeline) -> float:
        valor, _ = evaluate_pipeline(
            pipeline, data, labels, MeanRegressorModel(), "auc_roc",
            train_idx, val_idx, MetricCalculator(),
        )
        return valor

    en_secuencia = [evaluar(Pipeline([compartida])), evaluar(Pipeline([compartida]))]
    aislados = [
        evaluar(Pipeline([RecordingCenterer()])),
        evaluar(Pipeline([RecordingCenterer()])),
    ]

    assert en_secuencia == aislados
    assert compartida.fit_count == 0


def test_evaluate_pipeline_accepts_a_list_of_samples() -> None:
    """El conjunto de entrada puede ser una lista y no solo un arreglo."""
    data = [np.array([0.0]), np.array([10.0]), np.array([1000.0]), np.array([1010.0])]
    labels = np.array([0, 1, 0, 1])

    valor, metrics = evaluate_pipeline(
        Pipeline([RecordingCenterer()]), data, labels, MeanRegressorModel(),
        "auc_roc", np.array([0, 1]), np.array([2, 3]), MetricCalculator(),
    )

    assert set(metrics) == {"auc_roc", "f1", "precision", "recall", "false_positive_rate"}


# --------------------------------------------------------------------------- #
# evaluate_pipeline con tarea de clasificación
# --------------------------------------------------------------------------- #
def test_evaluate_pipeline_uses_the_requested_task() -> None:
    """Con la tarea de clasificación, las métricas y las claves son las suyas."""
    rng = np.random.default_rng(0)
    data = rng.normal(size=(40, 3))
    labels = rng.integers(0, 4, 40)

    valor, metrics = evaluate_pipeline(
        Pipeline([]),
        data,
        labels,
        LogisticRegression(max_iter=200),
        "macro_f1",
        np.arange(0, 30),
        np.arange(30, 40),
        CLASSIFICATION.build_metric_calculator(0.5),
        task="classification",
    )

    assert set(metrics) == set(CLASSIFICATION.valid_metrics)
    assert valor == metrics["macro_f1"]


def test_failed_evaluation_returns_the_metric_keys_of_the_task() -> None:
    """Un candidato inevaluable devuelve NaN en las claves de **su** tarea."""
    imagenes = [np.zeros((4, 4)), np.zeros((6, 6)), np.zeros((4, 4)), np.zeros((6, 6))]
    labels = np.array([0, 1, 0, 1])

    valor, metrics = evaluate_pipeline_safe(
        Pipeline([Identity()]), imagenes, labels, MeanThresholdModel(), "macro_f1",
        np.array([0, 1]), np.array([2, 3]),
        CLASSIFICATION.build_metric_calculator(0.5), task="classification",
    )

    assert math.isnan(valor)
    assert set(metrics) == set(CLASSIFICATION.valid_metrics)


@pytest.mark.parametrize(
    ("metric", "task", "esperado"),
    [
        ("false_positive_rate", None, True),
        ("auc_roc", None, False),
        ("macro_f1", "classification", False),
    ],
)
def test_is_better_direction_depends_on_the_task(
    metric: str, task: str | None, esperado: bool
) -> None:
    assert is_better(0.1, 0.9, metric, task) is esperado


# --------------------------------------------------------------------------- #
# Los motores reciben la tarea en el constructor
# --------------------------------------------------------------------------- #
@pytest.fixture
def multiclass_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Dataset tabular de tres clases, separable por la primera columna."""
    rng = np.random.default_rng(0)
    data = np.repeat(np.arange(3.0), 15).reshape(-1, 1) + rng.normal(0, 0.1, size=(45, 1))
    labels = np.repeat(np.arange(3), 15)
    return data, labels


def test_default_engine_keeps_the_migrated_behaviour(separable_dataset, search_space) -> None:
    """Sin indicar tarea ni métrica, se comporta como la librería base."""
    images, labels = separable_dataset
    engine = GridSearchEngine(search_space, val_size=0.5, random_state=42)

    assert engine.task.name == "anomaly_detection"

    _, best_metrics, _ = engine.run(images, labels, MeanThresholdModel())
    assert set(best_metrics) == {
        "auc_roc",
        "f1",
        "precision",
        "recall",
        "false_positive_rate",
    }


@pytest.mark.parametrize("engine_class", [GridSearchEngine, OptunaSearchEngine])
def test_engines_accept_a_classification_task(
    engine_class: type, multiclass_dataset, search_space
) -> None:
    """Construido con la tarea de clasificación, el motor optimiza macro-F1."""
    data, labels = multiclass_dataset
    engine = engine_class(search_space, val_size=0.4, random_state=42, task="classification")

    _, best_metrics, results = engine.run(data, labels, LogisticRegression(max_iter=200))

    assert engine.task.name == "classification"
    assert set(best_metrics) == set(CLASSIFICATION.valid_metrics)
    assert all(set(record["metrics"]) == set(CLASSIFICATION.valid_metrics) for record in results)


def test_metric_defaults_to_the_task_default(multiclass_dataset, search_space) -> None:
    """No indicar métrica equivale a pedir la métrica primaria de la tarea."""
    data, labels = multiclass_dataset
    engine = GridSearchEngine(search_space, val_size=0.4, random_state=42, task="classification")

    sin_metrica = engine.run(data, labels, LogisticRegression(max_iter=200))[2]
    explicita = engine.run(data, labels, LogisticRegression(max_iter=200), metric="macro_f1")[2]

    assert [r["objective"] for r in sin_metrica] == [r["objective"] for r in explicita]


def test_engine_rejects_a_metric_of_another_task(multiclass_dataset, search_space) -> None:
    data, labels = multiclass_dataset
    engine = GridSearchEngine(search_space, val_size=0.4, random_state=42, task="classification")

    with pytest.raises(ValueError, match="Métrica no soportada"):
        engine.run(data, labels, LogisticRegression(max_iter=200), metric="auc_roc")


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
