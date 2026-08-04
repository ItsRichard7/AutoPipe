# autopipe/parallel_runner/runner.py
"""Evaluación concurrente de múltiples pipelines con joblib."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence

import numpy as np
from joblib import Parallel, delayed

from autopipe.evaluator.metrics import MetricCalculator
from autopipe.evaluator.splitter import DataSplitter
from autopipe.pipeline.pipeline import Pipeline
from autopipe.search_engine.grid_search import evaluate_pipeline, validate_metric

logger = logging.getLogger(__name__)


class ParallelRunner:
    """Evalúa un conjunto de pipelines en paralelo con ``joblib.Parallel``.

    Usa el backend ``loky`` por defecto (seguro en Windows). Con ``n_jobs=1`` la
    ejecución es secuencial, apta para entornos Jupyter/Windows. Todos los
    pipelines comparten el mismo split de validación y se evalúan sobre copias
    independientes del modelo base, por lo que el resultado es idéntico al de la
    ejecución secuencial con la misma semilla.

    Attributes:
        n_jobs (int): Número de procesos (-1 = todos los núcleos disponibles).
        backend (str): Backend de joblib.
        splitter (DataSplitter): Particionador train/validación.
        metric_calculator (MetricCalculator): Calculador de métricas.
        last_runtime_ (float | None): Duración de la última llamada a :meth:`run`.
    """

    def __init__(
        self,
        n_jobs: int = -1,
        backend: str = "loky",
        val_size: float = 0.2,
        random_state: int | None = None,
        threshold: float = 0.5,
    ) -> None:
        """Inicializa el ejecutor paralelo.

        Args:
            n_jobs: Número de procesos; -1 usa todos los núcleos disponibles.
            backend: Backend de joblib (por defecto ``"loky"``).
            val_size: Proporción del dataset para validación.
            random_state: Semilla para la partición (reproducibilidad).
            threshold: Umbral para binarizar predicciones continuas.
        """
        self.n_jobs = n_jobs
        self.backend = backend
        self.splitter = DataSplitter(val_size=val_size, random_state=random_state)
        self.metric_calculator = MetricCalculator(threshold=threshold)
        self.last_runtime_: float | None = None

    def run(
        self,
        pipelines: Sequence[Pipeline],
        images: Sequence[np.ndarray],
        labels: np.ndarray,
        base_model: object,
        metric: str = "auc_roc",
    ) -> list[dict]:
        """Evalúa todos los pipelines en paralelo y devuelve la tabla de resultados.

        Args:
            pipelines: Pipelines a evaluar.
            images: Conjunto de imágenes de entrada.
            labels: Etiquetas binarias del dataset.
            base_model: Modelo base compatible con ``fit``/``predict``.
            metric: Métrica objetivo.

        Returns:
            Lista de diccionarios, uno por pipeline, en el mismo orden de entrada.

        Raises:
            ValueError: Si la métrica no está soportada.
        """
        validate_metric(metric)
        labels = np.asarray(labels)
        train_idx, val_idx = self.splitter.split(labels)
        start = time.perf_counter()
        results = self._evaluate(
            pipelines, images, labels, base_model, metric, train_idx, val_idx, self.n_jobs
        )
        self.last_runtime_ = time.perf_counter() - start
        logger.info(
            "Evaluados %d pipelines en %.3fs con n_jobs=%d.",
            len(results),
            self.last_runtime_,
            self.n_jobs,
        )
        return results

    def benchmark(
        self,
        pipelines: Sequence[Pipeline],
        images: Sequence[np.ndarray],
        labels: np.ndarray,
        base_model: object,
        metric: str = "auc_roc",
    ) -> dict:
        """Compara la ejecución secuencial y paralela y calcula el speedup.

        Args:
            pipelines: Pipelines a evaluar.
            images: Conjunto de imágenes de entrada.
            labels: Etiquetas binarias del dataset.
            base_model: Modelo base compatible con ``fit``/``predict``.
            metric: Métrica objetivo.

        Returns:
            Diccionario con ``results`` (de la ejecución paralela),
            ``sequential_time``, ``parallel_time``, ``speedup`` y ``consistent``
            (si los resultados paralelo y secuencial coinciden).

        Raises:
            ValueError: Si la métrica no está soportada.
        """
        validate_metric(metric)
        labels = np.asarray(labels)
        train_idx, val_idx = self.splitter.split(labels)

        start = time.perf_counter()
        sequential = self._evaluate(
            pipelines, images, labels, base_model, metric, train_idx, val_idx, 1
        )
        sequential_time = time.perf_counter() - start

        start = time.perf_counter()
        parallel = self._evaluate(
            pipelines, images, labels, base_model, metric, train_idx, val_idx, self.n_jobs
        )
        parallel_time = time.perf_counter() - start

        speedup = sequential_time / parallel_time if parallel_time > 0 else float("inf")
        consistent = self._results_match(sequential, parallel)
        logger.info(
            "Speedup: %.2fx (secuencial %.3fs, paralelo %.3fs, consistente=%s).",
            speedup,
            sequential_time,
            parallel_time,
            consistent,
        )
        return {
            "results": parallel,
            "sequential_time": sequential_time,
            "parallel_time": parallel_time,
            "speedup": speedup,
            "consistent": consistent,
        }

    def _evaluate(
        self,
        pipelines: Sequence[Pipeline],
        images: Sequence[np.ndarray],
        labels: np.ndarray,
        base_model: object,
        metric: str,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        n_jobs: int,
    ) -> list[dict]:
        """Evalúa los pipelines con joblib y arma la tabla de resultados.

        Args:
            pipelines: Pipelines a evaluar.
            images: Conjunto de imágenes de entrada.
            labels: Etiquetas binarias del dataset.
            base_model: Modelo base.
            metric: Métrica objetivo.
            train_idx: Índices de entrenamiento.
            val_idx: Índices de validación.
            n_jobs: Número de procesos a usar.

        Returns:
            Lista de diccionarios con los resultados, en orden de entrada.
        """
        outputs = Parallel(n_jobs=n_jobs, backend=self.backend)(
            delayed(evaluate_pipeline)(
                pipeline,
                images,
                labels,
                base_model,
                metric,
                train_idx,
                val_idx,
                self.metric_calculator,
            )
            for pipeline in pipelines
        )
        return [
            {
                "pipeline_index": index,
                "params": pipeline.get_params(),
                "metrics": metrics,
                "objective": objective,
            }
            for index, (pipeline, (objective, metrics)) in enumerate(zip(pipelines, outputs))
        ]

    @staticmethod
    def _results_match(sequential: list[dict], parallel: list[dict]) -> bool:
        """Comprueba que dos tablas de resultados sean numéricamente equivalentes.

        Args:
            sequential: Resultados de la ejecución secuencial.
            parallel: Resultados de la ejecución paralela.

        Returns:
            ``True`` si ambos tienen el mismo tamaño y métricas equivalentes
            (tratando ``NaN`` como iguales entre sí).
        """
        if len(sequential) != len(parallel):
            return False
        for left, right in zip(sequential, parallel):
            if set(left["metrics"]) != set(right["metrics"]):
                return False
            for key, value in left["metrics"].items():
                other = right["metrics"][key]
                if not ((math.isnan(value) and math.isnan(other)) or value == other):
                    return False
        return True
