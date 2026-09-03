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

from autopipe.evaluator.splitter import DataSplitter
from autopipe.pipeline.base import BaseTransform
from autopipe.pipeline.pipeline import Pipeline
from autopipe.protocols import BaseModel
from autopipe.search_engine.grid_search import (
    evaluate_pipeline_safe,
    validate_metric,
)
from autopipe.tasks import DEFAULT_TASK, Task, get_task

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
        n_startup_trials (int): Ensayos aleatorios antes de que TPE modele.
        seed (int | None): Semilla del sampler.
        task (Task): Tarea de aprendizaje que gobierna métricas y predicciones.
        splitter (DataSplitter): Particionador train/validación.
        metric_calculator: Calculador de métricas de la tarea.
        best_configuration_ (dict | None): Índices del pipeline ganador, uno por
            paso, disponibles tras :meth:`run`. Es la forma que consume
            ``optuna.enqueue_trial`` y la que se guarda en la base de
            meta-conocimiento; solo significa algo respecto de la versión del
            espacio de búsqueda con la que se obtuvo.
        tied_configurations_ (list | None): Todas las configuraciones distintas
            que alcanzaron el mejor valor, disponibles tras :meth:`run`.
            :attr:`best_configuration_` es una de ellas, la que el muestreador
            visitó primero, y esa elección es una propiedad de la corrida y no
            del dataset: si el empate tiene más de un elemento, quien vaya a
            **transferir** la configuración a otro dataset debe desempatar con un
            criterio explícito en lugar de heredar el orden de visita (véase
            :func:`~autopipe.search_space.canonical_configuration`).
    """

    def __init__(
        self,
        search_space: dict[str, Sequence[BaseTransform]],
        n_trials: int = 20,
        val_size: float = 0.2,
        random_state: int | None = None,
        threshold: float = 0.5,
        task: str | Task = DEFAULT_TASK,
        n_startup_trials: int = 10,
    ) -> None:
        """Inicializa el motor de búsqueda con Optuna.

        Args:
            search_space: Diccionario de nombre de paso a lista de transformaciones.
            n_trials: Número de pruebas a ejecutar.
            val_size: Proporción del dataset para validación.
            random_state: Semilla del sampler y de la partición (reproducibilidad).
            threshold: Umbral para binarizar predicciones continuas.
            task: Tarea de aprendizaje, por nombre o instancia. Por defecto es la
                detección de anomalías, por continuidad con la librería base.
            n_startup_trials: Ensayos muestreados al azar antes de que TPE empiece
                a modelar. Se declara de forma explícita, aunque coincida con el
                valor por defecto de Optuna, porque de él dependen los resultados:
                si una versión futura de la librería cambiara ese valor, los
                experimentos dejarían de ser reproducibles en silencio. El spike A
                midió además que interactúa con el arranque en caliente (véase
                ``docs/experimentos/SPIKE-A-enqueue-trial.md``).
        """
        self.search_space = search_space
        self.n_trials = n_trials
        self.n_startup_trials = n_startup_trials
        self.seed = random_state
        self.task = get_task(task)
        self.splitter = DataSplitter(val_size=val_size, random_state=random_state)
        self.metric_calculator = self.task.build_metric_calculator(threshold)
        self.best_configuration_: dict[str, int] | None = None
        self.tied_configurations_: list[dict[str, int]] | None = None

    def run(
        self,
        images: Sequence[np.ndarray],
        labels: np.ndarray,
        base_model: BaseModel,
        metric: str | None = None,
    ) -> tuple[Pipeline, dict, list[dict]]:
        """Ejecuta la búsqueda con Optuna.

        Args:
            images: Conjunto de muestras de entrada.
            labels: Etiquetas del dataset.
            base_model: Modelo base compatible con ``fit``/``predict``.
            metric: Métrica objetivo a optimizar. Si es ``None`` se usa la
                métrica por defecto de la tarea.

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
        keys = list(self.search_space)
        direction = "minimize" if self.task.is_minimized(metric) else "maximize"
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
                self.task,
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
            sampler=TPESampler(seed=self.seed, n_startup_trials=self.n_startup_trials),
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
        # Los índices del ganador, que es lo que ``optuna.enqueue_trial`` consume
        # al inyectar una configuración recomendada. El pipeline ya construido no
        # sirve para eso: hay que devolverle al muestreador los mismos enteros que
        # él registró.
        self.best_configuration_ = {key: study.best_trial.params[key] for key in keys}
        self.tied_configurations_ = self._tied(study, keys)
        best_metrics = study.best_trial.user_attrs["metrics"]
        logger.info("Búsqueda con Optuna completada: %d pruebas.", len(results))
        return best_pipeline, best_metrics, results

    @staticmethod
    def _tied(study: optuna.Study, keys: list[str]) -> list[dict[str, int]]:
        """Configuraciones distintas que alcanzaron el mejor valor del estudio.

        Se comparan los valores objetivo con una tolerancia absoluta pequeña:
        dos pipelines que producen la misma matriz de características dan
        exactamente la misma métrica, pero el orden de sumación en coma flotante
        puede separarlas en los últimos bits.

        Args:
            study: Estudio ya optimizado.
            keys: Pasos del espacio de búsqueda, en orden.

        Returns:
            Las configuraciones empatadas, en orden de primera aparición. La
            primera es siempre :attr:`best_configuration_`. Si ningún ensayo
            resultó evaluable, la lista contiene únicamente esa: un empate entre
            pipelines que no se pudieron evaluar no es un empate.
        """
        mejor = study.best_trial.value
        ganadora = {key: study.best_trial.params[key] for key in keys}
        if mejor is None or not math.isfinite(mejor):
            return [ganadora]
        empatadas: list[dict[str, int]] = []
        for trial in study.trials:
            if trial.value is None or not math.isclose(trial.value, mejor, abs_tol=1e-9):
                continue
            configuracion = {key: trial.params[key] for key in keys}
            if configuracion not in empatadas:
                empatadas.append(configuracion)
        return empatadas

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
