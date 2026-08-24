# autopipe/tasks/classification.py
"""Tarea de clasificación, binaria o multiclase."""

from __future__ import annotations

import numpy as np

from autopipe.evaluator.metrics import ClassificationMetricCalculator
from autopipe.tasks.base import Task

VALID_METRICS = (
    "macro_f1",
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
)
MINIMIZE_METRICS: tuple[str, ...] = ()


def normalize_predictions(raw: np.ndarray) -> np.ndarray:
    """Devuelve las etiquetas de clase predichas sin alterarlas.

    En clasificación la salida de ``model.predict`` **ya son** las etiquetas de
    clase, de modo que cualquier reinterpretación las corrompería. En particular,
    la normalización binaria de la tarea de detección de anomalías colapsaría un
    problema de cinco clases a dos sin emitir ningún error.

    Args:
        raw: Predicciones crudas del modelo.

    Returns:
        Las mismas etiquetas, como arreglo NumPy.
    """
    return np.asarray(raw)


def predict_scores(model: object, features: np.ndarray) -> np.ndarray | None:
    """Indica que la tarea no usa puntajes continuos.

    Ninguna de las métricas macro de clasificación se calcula sobre un ranking,
    así que no hay puntajes que extraer. Devolver ``None`` evita además el error
    de tomar una sola columna de ``predict_proba`` en un problema multiclase y
    descartar el resto en silencio.

    Args:
        model: Modelo base ya entrenado. No se usa.
        features: Matriz de características de validación. No se usa.

    Returns:
        Siempre ``None``.
    """
    del model, features
    return None


def build_metric_calculator(threshold: float) -> ClassificationMetricCalculator:
    """Construye el calculador de métricas de clasificación.

    Args:
        threshold: Ignorado. Las predicciones de clasificación son etiquetas de
            clase, no puntajes continuos, así que no hay nada que binarizar. El
            parámetro existe por uniformidad con las demás tareas.

    Returns:
        Un :class:`~autopipe.evaluator.metrics.ClassificationMetricCalculator`.
    """
    del threshold
    return ClassificationMetricCalculator()


CLASSIFICATION = Task(
    name="classification",
    valid_metrics=VALID_METRICS,
    default_metric="macro_f1",
    minimize_metrics=MINIMIZE_METRICS,
    build_metric_calculator=build_metric_calculator,
    normalize_predictions=normalize_predictions,
    predict_scores=predict_scores,
)
