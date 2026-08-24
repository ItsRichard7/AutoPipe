# autopipe/tasks/anomaly.py
"""Tarea de detección de anomalías binaria.

Es el caso de continuidad de la librería base: conserva exactamente el
comportamiento y las métricas con que se migró el proyecto.
"""

from __future__ import annotations

import numpy as np

from autopipe.evaluator.metrics import MetricCalculator
from autopipe.tasks.base import Task

VALID_METRICS = ("auc_roc", "f1", "precision", "recall", "false_positive_rate")
MINIMIZE_METRICS = ("false_positive_rate",)


def normalize_predictions(raw: np.ndarray) -> np.ndarray:
    """Normaliza las predicciones del modelo a etiquetas binarias {0, 1}.

    Soporta la convención de scikit-learn para detección de anomalías
    (``-1`` = anomalía, ``1`` = normal) y etiquetas {0, 1}; cualquier otro
    valor se binariza con umbral 0.5.

    Args:
        raw: Predicciones crudas del modelo.

    Returns:
        Arreglo de enteros con valores en {0, 1} (1 = anomalía).
    """
    raw = np.asarray(raw)
    unique = set(np.unique(raw).tolist())
    if unique <= {0, 1}:
        return raw.astype(int)
    if unique <= {-1, 1}:
        return (raw == -1).astype(int)
    return (raw >= 0.5).astype(int)


def predict_scores(model: object, features: np.ndarray) -> np.ndarray | None:
    """Obtiene puntajes de anomalía continuos del modelo para el AUC-ROC.

    Prioriza ``predict_proba`` (clasificadores), luego ``score_samples`` y
    ``decision_function`` (detectores de anomalías, cuyo signo se invierte para
    que un valor mayor indique mayor anomalía).

    Args:
        model: Modelo base ya entrenado.
        features: Matriz de características de validación.

    Returns:
        Arreglo de puntajes, o ``None`` si el modelo no expone ninguno.
    """
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(features))[:, 1]
    if hasattr(model, "score_samples"):
        return -np.asarray(model.score_samples(features))
    if hasattr(model, "decision_function"):
        return -np.asarray(model.decision_function(features))
    return None


def build_metric_calculator(threshold: float) -> MetricCalculator:
    """Construye el calculador de métricas de detección de anomalías.

    Args:
        threshold: Umbral para binarizar predicciones continuas.

    Returns:
        Un :class:`~autopipe.evaluator.metrics.MetricCalculator`.
    """
    return MetricCalculator(threshold=threshold)


ANOMALY_DETECTION = Task(
    name="anomaly_detection",
    valid_metrics=VALID_METRICS,
    default_metric="auc_roc",
    minimize_metrics=MINIMIZE_METRICS,
    build_metric_calculator=build_metric_calculator,
    normalize_predictions=normalize_predictions,
    predict_scores=predict_scores,
)
