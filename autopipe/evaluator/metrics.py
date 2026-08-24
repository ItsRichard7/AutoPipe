# autopipe/evaluator/metrics.py
"""Cálculo de métricas de evaluación por tarea de aprendizaje."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


class MetricCalculator:
    """Calcula métricas de clasificación binaria para detección de anomalías.

    Métricas calculadas: AUC-ROC, F1-score, precisión, recall y tasa de falsos
    positivos. La etiqueta positiva (anomalía) es 1.

    Attributes:
        threshold (float): Umbral para binarizar predicciones continuas.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        """Inicializa el calculador de métricas.

        Args:
            threshold: Umbral usado para binarizar ``y_pred`` cuando contiene
                valores continuos en lugar de etiquetas {0, 1}.
        """
        self.threshold = float(threshold)

    def calculate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_score: np.ndarray | None = None,
    ) -> dict:
        """Calcula todas las métricas a partir de etiquetas verdaderas y predichas.

        Args:
            y_true: Etiquetas verdaderas binarias (0 = normal, 1 = anomalía).
            y_pred: Predicciones del modelo, ya sea etiquetas {0, 1} o puntajes
                continuos (que se binarizan con ``threshold``).
            y_score: Puntajes continuos opcionales para el cálculo de AUC-ROC.
                Si es ``None`` se usa ``y_pred``.

        Returns:
            Diccionario con las claves ``auc_roc``, ``f1``, ``precision``,
            ``recall`` y ``false_positive_rate``.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        scores = y_pred if y_score is None else np.asarray(y_score)
        binary_pred = self._to_binary(y_pred)

        tn, fp, _, _ = confusion_matrix(y_true, binary_pred, labels=[0, 1]).ravel()
        denominator = fp + tn
        false_positive_rate = float(fp / denominator) if denominator > 0 else 0.0

        return {
            "auc_roc": self._safe_auc(y_true, scores),
            "f1": float(f1_score(y_true, binary_pred, zero_division=0)),
            "precision": float(precision_score(y_true, binary_pred, zero_division=0)),
            "recall": float(recall_score(y_true, binary_pred, zero_division=0)),
            "false_positive_rate": false_positive_rate,
        }

    def _to_binary(self, y_pred: np.ndarray) -> np.ndarray:
        """Convierte predicciones a etiquetas binarias {0, 1}.

        Si todos los valores ya pertenecen a {0, 1} se devuelven como enteros;
        de lo contrario se binarizan usando ``threshold``.

        Args:
            y_pred: Predicciones del modelo.

        Returns:
            Arreglo de enteros con valores en {0, 1}.
        """
        unique_values = np.unique(y_pred)
        if np.isin(unique_values, [0, 1]).all():
            return y_pred.astype(int)
        return (y_pred >= self.threshold).astype(int)

    def _safe_auc(self, y_true: np.ndarray, scores: np.ndarray) -> float:
        """Calcula el AUC-ROC manejando el caso de una sola clase presente.

        Args:
            y_true: Etiquetas verdaderas binarias.
            scores: Puntajes o predicciones para el cálculo de AUC.

        Returns:
            El valor de AUC-ROC, o ``float('nan')`` si ``y_true`` contiene una
            sola clase.
        """
        if len(np.unique(y_true)) < 2:
            logger.warning(
                "AUC-ROC no está definido con una sola clase en y_true; se devuelve NaN."
            )
            return float("nan")
        return float(roc_auc_score(y_true, scores))


class ClassificationMetricCalculator:
    """Calcula métricas macro de clasificación, binaria o multiclase.

    Expone la misma forma de llamada que :class:`MetricCalculator`
    (``calculate(y_true, y_pred, y_score=None) -> dict``), de modo que los
    motores de búsqueda no distinguen entre una y otra.

    A diferencia de la detección de anomalías, aquí las predicciones son
    **etiquetas de clase** y se usan tal cual: no se binarizan ni se reinterpretan.
    Las métricas se promedian con ``average="macro"``, que pesa todas las clases
    por igual con independencia de su frecuencia, conforme a la métrica primaria
    de clasificación fijada en el anteproyecto.
    """

    def calculate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_score: np.ndarray | None = None,
    ) -> dict:
        """Calcula las métricas de clasificación.

        Args:
            y_true: Etiquetas verdaderas de clase.
            y_pred: Etiquetas de clase predichas por el modelo.
            y_score: Ignorado. Existe por uniformidad con
                :meth:`MetricCalculator.calculate`; ninguna de las métricas
                macro requiere puntajes continuos.

        Returns:
            Diccionario con las claves ``macro_f1``, ``accuracy``,
            ``balanced_accuracy``, ``macro_precision`` y ``macro_recall``.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return {
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_precision": float(
                precision_score(y_true, y_pred, average="macro", zero_division=0)
            ),
            "macro_recall": float(
                recall_score(y_true, y_pred, average="macro", zero_division=0)
            ),
        }
