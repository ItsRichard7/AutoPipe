# tests/test_tasks.py
"""Pruebas unitarias para el módulo autopipe.tasks (ADR-002)."""

from __future__ import annotations

import pickle
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from autopipe.evaluator.metrics import ClassificationMetricCalculator, MetricCalculator
from autopipe.tasks import (
    ANOMALY_DETECTION,
    CLASSIFICATION,
    TASKS,
    Task,
    get_task,
)


@pytest.fixture
def multiclass_model() -> tuple[LogisticRegression, np.ndarray]:
    """Modelo entrenado sobre cinco clases, con su matriz de características."""
    rng = np.random.default_rng(0)
    features = rng.normal(size=(60, 4))
    labels = rng.integers(0, 5, 60)
    model = LogisticRegression(max_iter=200).fit(features, labels)
    return model, features[:10]


# --------------------------------------------------------------------------- #
# Registro y resolución de tareas
# --------------------------------------------------------------------------- #
def test_registry_contains_both_builtin_tasks() -> None:
    assert set(TASKS) == {"anomaly_detection", "classification"}


@pytest.mark.parametrize(
    ("nombre", "esperada"),
    [("anomaly_detection", ANOMALY_DETECTION), ("classification", CLASSIFICATION)],
)
def test_get_task_by_name(nombre: str, esperada: Task) -> None:
    assert get_task(nombre) is esperada


def test_get_task_accepts_an_instance() -> None:
    """Permite registrar una tarea propia sin modificar la librería."""
    assert get_task(CLASSIFICATION) is CLASSIFICATION


def test_get_task_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Tarea no soportada"):
        get_task("regresion")


def test_get_task_rejects_wrong_type() -> None:
    with pytest.raises(TypeError, match="nombre o una instancia"):
        get_task(42)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Coherencia interna de una tarea
# --------------------------------------------------------------------------- #
def _tarea(default_metric: str = "a", minimize: tuple[str, ...] = ()) -> Task:
    """Construye una tarea de prueba con funciones inertes."""
    return Task(
        name="prueba",
        valid_metrics=("a", "b"),
        default_metric=default_metric,
        minimize_metrics=minimize,
        build_metric_calculator=ANOMALY_DETECTION.build_metric_calculator,
        normalize_predictions=ANOMALY_DETECTION.normalize_predictions,
        predict_scores=ANOMALY_DETECTION.predict_scores,
    )


def test_default_metric_must_be_valid() -> None:
    with pytest.raises(ValueError, match="no está entre sus métricas válidas"):
        _tarea(default_metric="inexistente")


def test_minimize_metrics_must_be_valid() -> None:
    with pytest.raises(ValueError, match="no son.*válidas"):
        _tarea(minimize=("c",))


def test_task_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        ANOMALY_DETECTION.name = "otra"  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Métricas por tarea
# --------------------------------------------------------------------------- #
def test_accuracy_is_invalid_for_anomaly_and_valid_for_classification() -> None:
    """Criterio 2 del ADR-002: la validación es relativa a la tarea."""
    with pytest.raises(ValueError, match="Métrica no soportada"):
        ANOMALY_DETECTION.validate_metric("accuracy")

    CLASSIFICATION.validate_metric("accuracy")  # no debe levantar


def test_default_metrics_follow_the_anteproyecto() -> None:
    assert ANOMALY_DETECTION.default_metric == "auc_roc"
    assert CLASSIFICATION.default_metric == "macro_f1"


def test_only_false_positive_rate_is_minimized() -> None:
    assert ANOMALY_DETECTION.is_minimized("false_positive_rate")
    assert not ANOMALY_DETECTION.is_minimized("auc_roc")
    assert not CLASSIFICATION.is_minimized("macro_f1")


def test_metric_calculators_match_their_task() -> None:
    assert isinstance(ANOMALY_DETECTION.build_metric_calculator(0.5), MetricCalculator)
    assert isinstance(
        CLASSIFICATION.build_metric_calculator(0.5), ClassificationMetricCalculator
    )


# --------------------------------------------------------------------------- #
# Normalización de predicciones
# --------------------------------------------------------------------------- #
def test_classification_does_not_collapse_multiclass_labels() -> None:
    """Criterio 3 del ADR-002.

    Es el fallo silencioso documentado en la auditoría: la normalización binaria
    reduce cinco clases a dos sin levantar ninguna excepción.
    """
    etiquetas = np.array([0, 1, 2, 3, 4])

    np.testing.assert_array_equal(
        CLASSIFICATION.normalize_predictions(etiquetas), etiquetas
    )
    np.testing.assert_array_equal(
        ANOMALY_DETECTION.normalize_predictions(etiquetas), np.array([0, 1, 1, 1, 1])
    )


@pytest.mark.parametrize(
    ("crudas", "esperadas"),
    [
        ([0, 1, 0, 1], [0, 1, 0, 1]),
        ([-1, 1, -1, 1], [1, 0, 1, 0]),
        ([0.2, 0.8, 0.4], [0, 1, 0]),
    ],
)
def test_anomaly_normalization_preserves_migrated_behaviour(
    crudas: list, esperadas: list
) -> None:
    obtenidas = ANOMALY_DETECTION.normalize_predictions(np.array(crudas))
    np.testing.assert_array_equal(obtenidas, np.array(esperadas))


# --------------------------------------------------------------------------- #
# Obtención de puntajes
# --------------------------------------------------------------------------- #
def test_classification_discards_no_information_from_predict_proba(
    multiclass_model: tuple[LogisticRegression, np.ndarray],
) -> None:
    """Criterio 4 del ADR-002.

    La tarea de anomalías toma una sola columna de una matriz de probabilidades
    de cinco columnas; la de clasificación no extrae puntajes en absoluto, de
    modo que no hay información que descartar.
    """
    model, features = multiclass_model

    assert model.predict_proba(features).shape == (10, 5)
    assert ANOMALY_DETECTION.predict_scores(model, features).shape == (10,)
    assert CLASSIFICATION.predict_scores(model, features) is None


class ScoreSamplesModel:
    """Doble de prueba con la interfaz de un detector de anomalías de sklearn."""

    def score_samples(self, features: np.ndarray) -> np.ndarray:
        return -np.arange(len(features), dtype=float)


class DecisionFunctionModel:
    """Doble de prueba que solo expone ``decision_function``."""

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        return -np.arange(len(features), dtype=float)


class OpaqueModel:
    """Doble de prueba sin ninguna interfaz de puntajes."""


@pytest.mark.parametrize("model", [ScoreSamplesModel(), DecisionFunctionModel()])
def test_anomaly_scores_invert_the_sign_of_detector_outputs(model: object) -> None:
    """Un valor mayor debe indicar mayor anomalía, de ahí la negación."""
    scores = ANOMALY_DETECTION.predict_scores(model, np.zeros((3, 2)))

    np.testing.assert_allclose(scores, np.array([0.0, 1.0, 2.0]))


def test_anomaly_scores_are_none_without_a_scoring_interface() -> None:
    assert ANOMALY_DETECTION.predict_scores(OpaqueModel(), np.zeros((3, 2))) is None


def test_classification_metrics_are_correct_on_five_classes(
    multiclass_model: tuple[LogisticRegression, np.ndarray],
) -> None:
    """Las métricas macro se calculan sobre las cinco clases, no sobre dos."""
    from sklearn.metrics import f1_score

    model, features = multiclass_model
    y_true = model.predict(features)
    y_pred = np.roll(y_true, 1)

    calculador = CLASSIFICATION.build_metric_calculator(0.5)
    metricas = calculador.calculate(y_true, CLASSIFICATION.normalize_predictions(y_pred))

    assert metricas["macro_f1"] == pytest.approx(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    )


# --------------------------------------------------------------------------- #
# Serialización
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("task", [ANOMALY_DETECTION, CLASSIFICATION], ids=lambda t: t.name)
def test_task_survives_pickling(task: Task) -> None:
    """Criterio 5 del ADR-002: ParallelRunner envía la tarea a procesos de loky."""
    recuperada = pickle.loads(pickle.dumps(task))

    assert recuperada == task
    np.testing.assert_array_equal(
        recuperada.normalize_predictions(np.array([0, 1])),
        task.normalize_predictions(np.array([0, 1])),
    )
