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
from autopipe.pipeline.base import BaseTransform, Dataset
from autopipe.pipeline.pipeline import Pipeline
from autopipe.tasks import DEFAULT_TASK, Task, get_task

logger = logging.getLogger(__name__)


def _resolve_task(task: str | Task | None) -> Task:
    """Resuelve la tarea, usando la de continuidad cuando no se especifica.

    Args:
        task: Nombre de la tarea, una instancia, o ``None``.

    Returns:
        La tarea resuelta; :data:`~autopipe.tasks.DEFAULT_TASK` si es ``None``.
    """
    return DEFAULT_TASK if task is None else get_task(task)


def _es_tabla(data: object) -> bool:
    """Indica si el objeto es una tabla con indexación posicional tipo pandas.

    Se comprueba por capacidades y no por tipo para que este módulo no tenga que
    importar pandas: el soporte tabular es opcional y las transformaciones de
    imagen no lo necesitan.

    Args:
        data: Objeto a inspeccionar.

    Returns:
        ``True`` si expone ``iloc`` y ``to_numpy``.
    """
    return hasattr(data, "iloc") and hasattr(data, "to_numpy")


def _select(data: Dataset, indices: np.ndarray) -> Dataset:
    """Selecciona un subconjunto de muestras conservando el tipo del contenedor.

    Args:
        data: Conjunto completo de muestras.
        indices: Índices posicionales de las muestras a seleccionar.

    Returns:
        El subconjunto: filas de la tabla si la entrada era una tabla, un arreglo
        si era un arreglo, y una lista en cualquier otro caso.
    """
    if _es_tabla(data):
        return data.iloc[indices]
    if isinstance(data, np.ndarray):
        return data[indices]
    return [data[index] for index in indices]


def _to_feature_matrix(samples: Dataset) -> np.ndarray:
    """Convierte las muestras transformadas en una matriz numérica.

    Una tabla se convierte de una vez, porque recorrerla daría sus columnas y no
    sus filas. En los demás casos se aplana cada muestra y se apilan: para datos
    tabulares ya numéricos cada fila es unidimensional y el aplanado no tiene
    efecto, de modo que la misma función sirve a ambas modalidades.

    Args:
        samples: Muestras ya transformadas por el pipeline.

    Returns:
        Matriz de forma ``(n_muestras, n_características)``.

    Raises:
        ValueError: Si una tabla conserva columnas no numéricas, lo que indica
            que al pipeline le falta un paso de codificación.
    """
    if _es_tabla(samples):
        try:
            return samples.to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "La tabla transformada conserva columnas no numéricas; el pipeline "
                "necesita un paso de codificación antes de entrenar el modelo."
            ) from exc
    return np.vstack([np.asarray(sample).ravel() for sample in samples])


def evaluate_pipeline(
    pipeline: Pipeline,
    images: Sequence[np.ndarray],
    labels: np.ndarray,
    base_model: object,
    metric: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    metric_calculator: MetricCalculator,
    task: str | Task | None = None,
) -> tuple[float, dict]:
    """Evalúa un pipeline ajustándolo únicamente con el subconjunto de entrenamiento.

    El pipeline se **clona** antes de ajustarse. Los motores de búsqueda
    construyen cada candidato tomando instancias del espacio de búsqueda, de modo
    que la misma transformación aparece en muchos candidatos; sin el clonado, una
    transformación con estado contaminaría a los candidatos siguientes y habría
    además una condición de carrera bajo el backend ``loky``.

    El modelo base se clona por la misma razón: cada pipeline se evalúa sobre un
    modelo sin entrenar e idéntico, lo que preserva la reproducibilidad.

    El ajuste del pipeline se realiza **solo con ``train_idx``**. Ajustarlo con
    el dataset completo filtraría hacia el entrenamiento las estadísticas del
    subconjunto de validación (véase ``docs/adr/ADR-001-contrato-fit-transform.md``).

    Args:
        pipeline: Pipeline a evaluar. No se modifica: se clona internamente.
        images: Conjunto de muestras de entrada.
        labels: Etiquetas del dataset.
        base_model: Modelo base compatible con la interfaz ``fit``/``predict``.
        metric: Nombre de la métrica objetivo.
        train_idx: Índices de entrenamiento.
        val_idx: Índices de validación.
        metric_calculator: Calculador de métricas, que debe provenir de la misma
            tarea que se pasa en ``task``.
        task: Tarea de aprendizaje, por nombre o instancia. Si es ``None`` se usa
            la detección de anomalías, por continuidad con la librería base.

    Returns:
        Tupla ``(valor_objetivo, métricas)`` con el valor de la métrica objetivo
        y el diccionario completo de métricas.
    """
    resolved = _resolve_task(task)
    fitted = copy.deepcopy(pipeline)

    train_features = _to_feature_matrix(
        fitted.fit_transform(_select(images, train_idx), labels[train_idx])
    )
    val_features = _to_feature_matrix(fitted.transform(_select(images, val_idx)))

    model = copy.deepcopy(base_model)
    model.fit(train_features, labels[train_idx])
    predictions = resolved.normalize_predictions(model.predict(val_features))
    scores = resolved.predict_scores(model, val_features)
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
    task: str | Task | None = None,
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
        images: Conjunto de muestras de entrada.
        labels: Etiquetas del dataset.
        base_model: Modelo base compatible con ``fit``/``predict``.
        metric: Nombre de la métrica objetivo.
        train_idx: Índices de entrenamiento.
        val_idx: Índices de validación.
        metric_calculator: Calculador de métricas.
        task: Tarea de aprendizaje, por nombre o instancia.

    Returns:
        Tupla ``(valor_objetivo, métricas)``; ambas con ``NaN`` si el pipeline
        no se pudo evaluar. Las claves del diccionario son las de la tarea.
    """
    try:
        return evaluate_pipeline(
            pipeline,
            images,
            labels,
            base_model,
            metric,
            train_idx,
            val_idx,
            metric_calculator,
            task,
        )
    except (ValueError, cv2.error) as exc:
        logger.warning("El pipeline %r no se pudo evaluar: %s", pipeline, exc)
        nan = float("nan")
        return nan, {name: nan for name in _resolve_task(task).valid_metrics}


def is_better(
    candidate: float,
    current: float | None,
    metric: str,
    task: str | Task | None = None,
) -> bool:
    """Indica si ``candidate`` mejora al mejor valor actual para la métrica dada.

    La dirección de la mejora la decide la tarea: en detección de anomalías un
    valor menor es mejor para ``false_positive_rate`` y mayor para el resto. Los
    valores ``NaN`` nunca se consideran mejores.

    Args:
        candidate: Valor candidato de la métrica.
        current: Mejor valor actual, o ``None`` si aún no hay ninguno.
        metric: Nombre de la métrica objetivo.
        task: Tarea de aprendizaje, por nombre o instancia.

    Returns:
        ``True`` si el candidato mejora al actual.
    """
    if math.isnan(candidate):
        return False
    if current is None or math.isnan(current):
        return True
    if _resolve_task(task).is_minimized(metric):
        return candidate < current
    return candidate > current


def validate_metric(metric: str, task: str | Task | None = None) -> None:
    """Valida que la métrica objetivo esté soportada por la tarea.

    Args:
        metric: Nombre de la métrica objetivo.
        task: Tarea de aprendizaje, por nombre o instancia. Si es ``None`` se
            valida contra las métricas de detección de anomalías.

    Raises:
        ValueError: Si la métrica no pertenece a las de la tarea.
    """
    _resolve_task(task).validate_metric(metric)


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
        task (Task): Tarea de aprendizaje que gobierna métricas y predicciones.
        splitter (DataSplitter): Particionador train/validación.
        metric_calculator: Calculador de métricas de la tarea.
    """

    def __init__(
        self,
        search_space: dict[str, Sequence[BaseTransform]],
        val_size: float = 0.2,
        random_state: int | None = None,
        threshold: float = 0.5,
        task: str | Task = DEFAULT_TASK,
    ) -> None:
        """Inicializa el motor de búsqueda en malla.

        Args:
            search_space: Diccionario de nombre de paso a lista de transformaciones.
            val_size: Proporción del dataset para validación.
            random_state: Semilla para la partición (reproducibilidad).
            threshold: Umbral para binarizar predicciones continuas.
            task: Tarea de aprendizaje, por nombre o instancia. Por defecto es la
                detección de anomalías, por continuidad con la librería base.
        """
        self.search_space = search_space
        self.task = get_task(task)
        self.splitter = DataSplitter(val_size=val_size, random_state=random_state)
        self.metric_calculator = self.task.build_metric_calculator(threshold)

    def run(
        self,
        images: Sequence[np.ndarray],
        labels: np.ndarray,
        base_model: object,
        metric: str | None = None,
    ) -> tuple[Pipeline, dict, list[dict]]:
        """Ejecuta la búsqueda exhaustiva.

        Args:
            images: Conjunto de muestras de entrada.
            labels: Etiquetas del dataset.
            base_model: Modelo base compatible con ``fit``/``predict``.
            metric: Métrica objetivo a optimizar. Si es ``None`` se usa la
                métrica por defecto de la tarea (``auc_roc`` en detección de
                anomalías, ``macro_f1`` en clasificación).

        Returns:
            Tupla ``(best_pipeline, best_metrics, results)`` con el pipeline
            ganador, sus métricas y la tabla completa de resultados.

        Raises:
            ValueError: Si la métrica no está soportada por la tarea.
        """
        metric = self.task.default_metric if metric is None else metric
        validate_metric(metric, self.task)
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
                self.task,
            )
            results.append(
                {
                    "pipeline_index": index,
                    "params": pipeline.get_params(),
                    "metrics": metrics,
                    "objective": objective,
                }
            )
            if is_better(objective, best_objective, metric, self.task):
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
