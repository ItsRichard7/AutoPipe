# autopipe/search_engine/grid_search.py
"""Búsqueda exhaustiva de pipelines mediante producto cartesiano."""

from __future__ import annotations

import copy
import itertools
import logging
import math
from collections.abc import Sequence

import cv2
import numpy as np

from autopipe.evaluator.metrics import MetricCalculator
from autopipe.evaluator.splitter import DataSplitter
from autopipe.pipeline.base import BaseTransform
from autopipe.pipeline.pipeline import Pipeline
from autopipe.tasks.anomaly import (
    MINIMIZE_METRICS,
    VALID_METRICS,
    normalize_predictions,
    predict_scores,
)

logger = logging.getLogger(__name__)


def _transform_to_matrix(pipeline: Pipeline, images: Sequence[np.ndarray]) -> np.ndarray:
    """Aplica un pipeline a cada imagen y las aplana en una matriz de características.

    Args:
        pipeline: Pipeline de transformaciones a aplicar.
        images: Conjunto de imágenes de entrada.

    Returns:
        Matriz de forma ``(n_imágenes, n_características)``.
    """
    rows = [np.asarray(pipeline.apply(image)).ravel() for image in images]
    return np.vstack(rows)


def evaluate_pipeline(
    pipeline: Pipeline,
    images: Sequence[np.ndarray],
    labels: np.ndarray,
    base_model: object,
    metric: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    metric_calculator: MetricCalculator,
) -> tuple[float, dict]:
    """Evalúa un pipeline entrenando una copia del modelo base.

    El modelo base se clona con ``copy.deepcopy`` para que cada pipeline se
    evalúe sobre un modelo sin entrenar e idéntico, preservando la
    reproducibilidad.

    Args:
        pipeline: Pipeline a evaluar.
        images: Conjunto de imágenes de entrada.
        labels: Etiquetas binarias del dataset.
        base_model: Modelo base compatible con la interfaz ``fit``/``predict``.
        metric: Nombre de la métrica objetivo.
        train_idx: Índices de entrenamiento.
        val_idx: Índices de validación.
        metric_calculator: Calculador de métricas.

    Returns:
        Tupla ``(valor_objetivo, métricas)`` con el valor de la métrica objetivo
        y el diccionario completo de métricas.
    """
    features = _transform_to_matrix(pipeline, images)
    model = copy.deepcopy(base_model)
    model.fit(features[train_idx], labels[train_idx])
    predictions = normalize_predictions(model.predict(features[val_idx]))
    scores = predict_scores(model, features[val_idx])
    metrics = metric_calculator.calculate(labels[val_idx], predictions, scores)
    return metrics[metric], metrics


def evaluate_pipeline_safe(
    pipeline: Pipeline,
    images: Sequence[np.ndarray],
    labels: np.ndarray,
    base_model: object,
    metric: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    metric_calculator: MetricCalculator,
) -> tuple[float, dict]:
    """Evalúa un pipeline tolerando fallos de evaluación.

    Los motores de búsqueda generan automáticamente candidatos con pasos
    omitidos; algunos pueden ser inevaluables (por ejemplo, omitir ``Resize``
    con imágenes de dimensiones heterogéneas impide construir la matriz de
    características). En ese caso se registra una advertencia y se devuelven
    métricas ``NaN``, de modo que el candidato nunca resulte ganador pero la
    búsqueda continúe.

    Args:
        pipeline: Pipeline a evaluar.
        images: Conjunto de imágenes de entrada.
        labels: Etiquetas binarias del dataset.
        base_model: Modelo base compatible con ``fit``/``predict``.
        metric: Nombre de la métrica objetivo.
        train_idx: Índices de entrenamiento.
        val_idx: Índices de validación.
        metric_calculator: Calculador de métricas.

    Returns:
        Tupla ``(valor_objetivo, métricas)``; ambas con ``NaN`` si el pipeline
        no se pudo evaluar.
    """
    try:
        return evaluate_pipeline(
            pipeline, images, labels, base_model, metric, train_idx, val_idx, metric_calculator
        )
    except (ValueError, cv2.error) as exc:
        logger.warning("El pipeline %r no se pudo evaluar: %s", pipeline, exc)
        nan = float("nan")
        return nan, {name: nan for name in VALID_METRICS}


def is_better(candidate: float, current: float | None, metric: str) -> bool:
    """Indica si ``candidate`` mejora al mejor valor actual para la métrica dada.

    Para ``false_positive_rate`` un valor menor es mejor; para el resto un valor
    mayor es mejor. Los valores ``NaN`` nunca se consideran mejores.

    Args:
        candidate: Valor candidato de la métrica.
        current: Mejor valor actual, o ``None`` si aún no hay ninguno.
        metric: Nombre de la métrica objetivo.

    Returns:
        ``True`` si el candidato mejora al actual.
    """
    if math.isnan(candidate):
        return False
    if current is None or math.isnan(current):
        return True
    if metric in MINIMIZE_METRICS:
        return candidate < current
    return candidate > current


def validate_metric(metric: str) -> None:
    """Valida que la métrica objetivo esté soportada.

    Args:
        metric: Nombre de la métrica objetivo.

    Raises:
        ValueError: Si la métrica no está en :data:`VALID_METRICS`.
    """
    if metric not in VALID_METRICS:
        raise ValueError(
            f"Métrica no soportada: {metric!r}. Use una de {VALID_METRICS}."
        )


class GridSearchEngine:
    """Búsqueda exhaustiva sobre el producto cartesiano del espacio de búsqueda.

    Genera un pipeline por cada combinación posible (una transformación por
    clave del espacio de búsqueda), evalúa cada uno con el mismo split de
    validación y selecciona el de mejor métrica objetivo.

    Para cada paso se incluye automáticamente la opción de **omitirlo**, de
    modo que la búsqueda también evalúa pipelines sin esa transformación. Un
    paso omitido simplemente no aparece en el pipeline resultante ni en su
    configuración serializada.

    Attributes:
        search_space (dict): Mapeo de nombre de paso a lista de transformaciones.
        splitter (DataSplitter): Particionador train/validación.
        metric_calculator (MetricCalculator): Calculador de métricas.
    """

    def __init__(
        self,
        search_space: dict[str, Sequence[BaseTransform]],
        val_size: float = 0.2,
        random_state: int | None = None,
        threshold: float = 0.5,
    ) -> None:
        """Inicializa el motor de búsqueda en malla.

        Args:
            search_space: Diccionario de nombre de paso a lista de transformaciones.
            val_size: Proporción del dataset para validación.
            random_state: Semilla para la partición (reproducibilidad).
            threshold: Umbral para binarizar predicciones continuas.
        """
        self.search_space = search_space
        self.splitter = DataSplitter(val_size=val_size, random_state=random_state)
        self.metric_calculator = MetricCalculator(threshold=threshold)

    def run(
        self,
        images: Sequence[np.ndarray],
        labels: np.ndarray,
        base_model: object,
        metric: str = "auc_roc",
    ) -> tuple[Pipeline, dict, list[dict]]:
        """Ejecuta la búsqueda exhaustiva.

        Args:
            images: Conjunto de imágenes de entrada.
            labels: Etiquetas binarias del dataset.
            base_model: Modelo base compatible con ``fit``/``predict``.
            metric: Métrica objetivo a optimizar.

        Returns:
            Tupla ``(best_pipeline, best_metrics, results)`` con el pipeline
            ganador, sus métricas y la tabla completa de resultados.

        Raises:
            ValueError: Si la métrica no está soportada.
        """
        validate_metric(metric)
        labels = np.asarray(labels)
        train_idx, val_idx = self.splitter.split(labels)
        pipelines = self._build_pipelines()

        results: list[dict] = []
        best_pipeline: Pipeline | None = None
        best_metrics: dict = {}
        best_objective: float | None = None
        for index, pipeline in enumerate(pipelines):
            objective, metrics = evaluate_pipeline_safe(
                pipeline,
                images,
                labels,
                base_model,
                metric,
                train_idx,
                val_idx,
                self.metric_calculator,
            )
            results.append(
                {
                    "pipeline_index": index,
                    "params": pipeline.get_params(),
                    "metrics": metrics,
                    "objective": objective,
                }
            )
            if is_better(objective, best_objective, metric):
                best_pipeline, best_metrics, best_objective = pipeline, metrics, objective

        if best_pipeline is None:
            best_pipeline = pipelines[0]
            best_metrics = results[0]["metrics"]
        logger.info(
            "Búsqueda en malla completada: %d pipelines evaluados.", len(pipelines)
        )
        return best_pipeline, best_metrics, results

    def _build_pipelines(self) -> list[Pipeline]:
        """Genera un pipeline por cada combinación del espacio de búsqueda.

        A cada paso se le añade automáticamente la opción de omitirse
        (``None``); los pasos omitidos se filtran al construir el pipeline.

        Returns:
            Lista de pipelines correspondientes al producto cartesiano
            ampliado con la opción de omisión por paso.
        """
        keys = list(self.search_space)
        option_lists = [list(self.search_space[key]) + [None] for key in keys]
        return [
            Pipeline([step for step in combo if step is not None])
            for combo in itertools.product(*option_lists)
        ]
