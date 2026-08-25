# autopipe/protocols.py
"""Contratos estructurales que la librería consume por *duck-typing*.

AutoPipe no exige heredar de ninguna clase para el modelo base, para el
calculador de métricas ni para la tabla de datos: le basta con que el objeto
exponga los métodos que necesita. Declararlo como :class:`typing.Protocol` hace
que ese requisito quede escrito y verificable por el analizador de tipos, en
lugar de vivir solo en la documentación.

Ninguno de estos protocolos importa pandas ni scikit-learn: describen la forma,
no la procedencia. Así el soporte tabular sigue siendo opcional y el modelo base
puede venir de scikit-learn, de PyTorch o de código propio.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class BaseModel(Protocol):
    """Modelo base con la interfaz mínima de scikit-learn.

    Es el contrato que la librería documenta para el modelo que aporta quien la
    usa. Solo se requieren ``fit`` y ``predict``; los métodos de puntajes
    (``predict_proba``, ``score_samples``, ``decision_function``) se consultan de
    forma opcional según la tarea.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> Any:
        """Entrena el modelo."""
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predice sobre un conjunto de características."""
        ...


class MetricCalculatorLike(Protocol):
    """Calculador de métricas de una tarea.

    Es la forma que comparten
    :class:`~autopipe.evaluator.metrics.MetricCalculator` y
    :class:`~autopipe.evaluator.metrics.ClassificationMetricCalculator`, de modo
    que el protocolo de evaluación no necesita distinguirlos.
    """

    def calculate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_score: np.ndarray | None = None,
    ) -> dict:
        """Calcula las métricas de la tarea."""
        ...


class TableLike(Protocol):
    """Tabla con indexación posicional y conversión a arreglo.

    Describe lo que la librería necesita de un ``DataFrame`` de pandas sin
    importarlo: seleccionar filas por posición y convertirse en una matriz
    numérica.
    """

    @property
    def iloc(self) -> Any:
        """Indexador posicional."""
        ...

    def to_numpy(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Convierte la tabla en un arreglo de NumPy."""
        ...
