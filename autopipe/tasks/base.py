# autopipe/tasks/base.py
"""Definición de una tarea de aprendizaje automático."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from autopipe.protocols import BaseModel, MetricCalculatorLike


@dataclass(frozen=True)
class Task:
    """Agrupa todo lo que depende de la tarea de aprendizaje.

    Una tarea no tiene comportamiento propio que extender: es una colección de
    decisiones. Por eso se modela como una estructura de datos inmutable y no
    como una jerarquía de clases.

    Las **cuatro** responsabilidades viajan juntas por diseño. Un registro
    suelto —las métricas por un lado y la normalización de predicciones por
    otro— permitiría que una tarea nueva declarara sus métricas y heredara en
    silencio la normalización binaria, que es exactamente el modo de fallo
    documentado en ``docs/decisiones/AUDITORIA-acoplamiento.md``.

    Todas las funciones almacenadas deben ser **funciones de módulo**, no
    lambdas ni closures: las tareas se envían a procesos de trabajo mediante el
    backend ``loky`` de :class:`~autopipe.parallel_runner.runner.ParallelRunner`
    y deben ser serializables con ``pickle``.

    Attributes:
        name (str): Identificador de la tarea.
        valid_metrics (tuple): Métricas que la tarea calcula y admite como
            objetivo.
        default_metric (str): Métrica objetivo por defecto; debe pertenecer a
            ``valid_metrics``.
        minimize_metrics (tuple): Subconjunto de ``valid_metrics`` en el que un
            valor menor es mejor.
        build_metric_calculator (Callable): Recibe el umbral de binarización y
            devuelve un calculador con el método
            ``calculate(y_true, y_pred, y_score=None) -> dict``.
        normalize_predictions (Callable): Convierte la salida cruda de
            ``model.predict`` en las etiquetas que esperan las métricas.
        predict_scores (Callable): Obtiene los puntajes continuos que necesitan
            las métricas basadas en ranking, o ``None`` si la tarea no los usa.
    """

    name: str
    valid_metrics: tuple[str, ...]
    default_metric: str
    minimize_metrics: tuple[str, ...]
    build_metric_calculator: Callable[[float], MetricCalculatorLike]
    normalize_predictions: Callable[[np.ndarray], np.ndarray]
    predict_scores: Callable[[BaseModel, np.ndarray], np.ndarray | None]

    def __post_init__(self) -> None:
        """Valida la coherencia interna de la tarea.

        Raises:
            ValueError: Si ``default_metric`` no está entre las métricas
                válidas, o si ``minimize_metrics`` contiene alguna métrica
                desconocida.
        """
        if self.default_metric not in self.valid_metrics:
            raise ValueError(
                f"La métrica por defecto {self.default_metric!r} de la tarea "
                f"{self.name!r} no está entre sus métricas válidas {self.valid_metrics}."
            )
        desconocidas = tuple(sorted(set(self.minimize_metrics) - set(self.valid_metrics)))
        if desconocidas:
            raise ValueError(
                f"La tarea {self.name!r} declara métricas a minimizar que no son "
                f"válidas: {desconocidas}."
            )

    def is_minimized(self, metric: str) -> bool:
        """Indica si la métrica dada se optimiza minimizándola.

        Args:
            metric: Nombre de la métrica.

        Returns:
            ``True`` si un valor menor de la métrica es mejor.
        """
        return metric in self.minimize_metrics

    def validate_metric(self, metric: str) -> None:
        """Valida que la métrica esté soportada por la tarea.

        Args:
            metric: Nombre de la métrica objetivo.

        Raises:
            ValueError: Si la métrica no pertenece a ``valid_metrics``.
        """
        if metric not in self.valid_metrics:
            raise ValueError(
                f"Métrica no soportada para la tarea {self.name!r}: {metric!r}. "
                f"Use una de {self.valid_metrics}."
            )
