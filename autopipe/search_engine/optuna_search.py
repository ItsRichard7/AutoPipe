# autopipe/search_engine/optuna_search.py
"""Búsqueda de pipelines mediante optimización bayesiana con Optuna."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from autopipe.evaluator.metrics import MetricCalculator
from autopipe.evaluator.splitter import DataSplitter
from autopipe.pipeline.base import BaseTransform
from autopipe.pipeline.pipeline import Pipeline
from autopipe.search_engine.grid_search import (
    evaluate_pipeline_safe,
    validate_metric,
)
from autopipe.tasks.anomaly import MINIMIZE_METRICS

logger = logging.getLogger(__name__)


class OptunaSearchEngine:
    """Búsqueda de pipelines con Optuna usando ``TPESampler`` y ``MedianPruner``.

    Es compatible con el mismo formato de espacio de búsqueda que
    :class:`~autopipe.search_engine.grid_search.GridSearchEngine`: un
    diccionario que mapea cada nombre de paso a una lista de transformaciones
    candidatas. Para cada paso se incluye automáticamente la opción de
    **omitirlo** (un índice categórico adicional); los pasos omitidos no
    aparecen en el pipeline resultante. La semilla del sampler garantiza
    reproducibilidad.

    Attributes:
        search_space (dict): Mapeo de nombre de paso a lista de transformaciones.
        n_trials (int): Número de pruebas de optimización.
        seed (int | None): Semilla del sampler.
        splitter (DataSplitter): Particionador train/validación.
        metric_calculator (MetricCalculator): Calculador de métricas.
    """

    def __init__(
        self,
        search_space: dict[str, Sequence[BaseTransform]],
        n_trials: int = 20,
        val_size: float = 0.2,
        random_state: int | None = None,
        threshold: float = 0.5,
    ) -> None:
        """Inicializa el motor de búsqueda con Optuna.

        Args:
            search_space: Diccionario de nombre de paso a lista de transformaciones.
            n_trials: Número de pruebas a ejecutar.
            val_size: Proporción del dataset para validación.
            random_state: Semilla del sampler y de la partición (reproducibilidad).
            threshold: Umbral para binarizar predicciones continuas.
        """
        self.search_space = search_space
        self.n_trials = n_trials
        self.seed = random_state
        self.splitter = DataSplitter(val_size=val_size, random_state=random_state)
        self.metric_calculator = MetricCalculator(threshold=threshold)

    def run(
        self,
        images: Sequence[np.ndarray],
        labels: np.ndarray,
        base_model: object,
        metric: str = "auc_roc",
    ) -> tuple[Pipeline, dict, list[dict]]:
        """Ejecuta la búsqueda con Optuna.

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
        keys = list(self.search_space)
        direction = "minimize" if metric in MINIMIZE_METRICS else "maximize"
        results: list[dict] = []

        def objective(trial: optuna.Trial) -> float:
            # El índice len(opciones) representa "omitir este paso".
            steps = []
            for key in keys:
                options = self.search_space[key]
                index = trial.suggest_categorical(key, list(range(len(options) + 1)))
                if index < len(options):
                    steps.append(options[index])
            pipeline = Pipeline(steps)
            value, metrics = evaluate_pipeline_safe(
                pipeline,
                images,
                labels,
                base_model,
                metric,
                train_idx,
                val_idx,
                self.metric_calculator,
            )
            trial.set_user_attr("metrics", metrics)
            results.append(
                {
                    "trial": trial.number,
                    "params": pipeline.get_params(),
                    "metrics": metrics,
                    "objective": value,
                }
            )
            return self._sanitize(value, direction)

        study = optuna.create_study(
            direction=direction,
            sampler=TPESampler(seed=self.seed),
            pruner=MedianPruner(),
        )
        study.optimize(objective, n_trials=self.n_trials)

        best_steps = []
        for key in keys:
            options = self.search_space[key]
            index = study.best_trial.params[key]
            if index < len(options):
                best_steps.append(options[index])
        best_pipeline = Pipeline(best_steps)
        best_metrics = study.best_trial.user_attrs["metrics"]
        logger.info("Búsqueda con Optuna completada: %d pruebas.", len(results))
        return best_pipeline, best_metrics, results

    @staticmethod
    def _sanitize(value: float, direction: str) -> float:
        """Reemplaza ``NaN`` por el peor valor posible según la dirección.

        Args:
            value: Valor de la métrica objetivo.
            direction: ``"maximize"`` o ``"minimize"``.

        Returns:
            El valor original, o ``±inf`` si era ``NaN``.
        """
        if math.isnan(value):
            return float("-inf") if direction == "maximize" else float("inf")
        return value
