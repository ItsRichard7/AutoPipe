# autopipe/pipeline/pipeline.py
"""Composición secuencial de transformaciones de preprocesamiento."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from autopipe.pipeline.base import BaseTransform, Dataset
from autopipe.pipeline.transforms import CLAHETransform, Normalize

logger = logging.getLogger(__name__)


class Pipeline:
    """Aplica una secuencia ordenada de transformaciones a una imagen.

    Al construirse valida que ciertas combinaciones incompatibles (por ejemplo
    una :class:`~autopipe.pipeline.transforms.Normalize` antes de una
    :class:`~autopipe.pipeline.transforms.CLAHETransform`) emitan una
    advertencia mediante ``logging``.

    Attributes:
        steps (list[BaseTransform]): Transformaciones en orden de aplicación.
    """

    def __init__(self, steps: Sequence[BaseTransform]) -> None:
        """Inicializa el pipeline.

        Args:
            steps: Secuencia de transformaciones que heredan de
                :class:`~autopipe.pipeline.base.BaseTransform`.

        Raises:
            TypeError: Si algún paso no es una instancia de ``BaseTransform``.
        """
        for step in steps:
            if not isinstance(step, BaseTransform):
                raise TypeError(
                    f"Todos los pasos deben heredar de BaseTransform, se recibió "
                    f"{type(step).__name__}."
                )
        self.steps = list(steps)
        self._check_compatibility()

    def __repr__(self) -> str:
        """Representación legible que lista los pasos del pipeline."""
        steps = ", ".join(repr(step) for step in self.steps)
        return f"Pipeline([{steps}])"

    def add(self, transform: BaseTransform) -> Pipeline:
        """Añade una transformación al final del pipeline (API encadenable).

        Permite construir el pipeline paso a paso, p. ej.
        ``Pipeline([]).add(Resize(...)).add(GaussianBlur(...))``.

        Args:
            transform: Transformación que hereda de :class:`BaseTransform`.

        Returns:
            El mismo pipeline, para permitir el encadenamiento.

        Raises:
            TypeError: Si ``transform`` no es una instancia de ``BaseTransform``.
        """
        if not isinstance(transform, BaseTransform):
            raise TypeError(
                f"El paso debe heredar de BaseTransform, se recibió "
                f"{type(transform).__name__}."
            )
        self.steps.append(transform)
        self._check_compatibility()
        return self

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Aplica todas las transformaciones en orden.

        Args:
            image: Imagen de entrada como arreglo NumPy.

        Returns:
            La imagen resultante tras aplicar cada transformación.
        """
        result = image
        for step in self.steps:
            result = step.apply(result)
        return result

    def fit(self, X: Dataset, y: np.ndarray | None = None) -> Pipeline:
        """Ajusta el pipeline completo con el subconjunto de entrenamiento.

        El ajuste es **secuencial**: cada paso se ajusta sobre la salida del paso
        anterior, no sobre los datos originales. Es la única forma correcta de
        encadenar transformaciones con estado, porque lo que un imputador o un
        escalador deben aprender depende de lo que hicieron los pasos previos.

        Debe invocarse **solo con el subconjunto de entrenamiento**: ajustar con
        el dataset completo filtraría información de la validación.

        Como optimización, el último paso se ajusta pero no se transforma: su
        salida se descartaría, y sobre conjuntos de imágenes ese trabajo no es
        despreciable. Use :meth:`fit_transform` cuando necesite el resultado.

        Args:
            X: Conjunto de muestras de entrenamiento.
            y: Etiquetas de entrenamiento, si algún paso las necesita.

        Returns:
            El mismo pipeline, para permitir el encadenamiento.
        """
        current = X
        for step in self.steps[:-1]:
            current = step.fit(current, y).transform(current)
        if self.steps:
            self.steps[-1].fit(current, y)
        return self

    def transform(self, X: Dataset) -> Dataset:
        """Transforma un conjunto completo aplicando los pasos en orden.

        Args:
            X: Conjunto de muestras a transformar.

        Returns:
            El conjunto transformado. Un pipeline sin pasos devuelve la entrada
            sin modificar.

        Raises:
            NotFittedError: Si algún paso con estado aún no fue ajustado.
        """
        current = X
        for step in self.steps:
            current = step.transform(current)
        return current

    def fit_transform(self, X: Dataset, y: np.ndarray | None = None) -> Dataset:
        """Ajusta el pipeline con ``X`` y devuelve ``X`` transformado.

        Equivale a :meth:`fit` seguido de :meth:`transform`, pero recorre los
        pasos una sola vez.

        Args:
            X: Conjunto de muestras de entrenamiento.
            y: Etiquetas de entrenamiento, si algún paso las necesita.

        Returns:
            El conjunto de entrenamiento transformado.
        """
        current = X
        for step in self.steps:
            current = step.fit(current, y).transform(current)
        return current

    def get_params(self) -> dict:
        """Devuelve los parámetros de cada paso del pipeline.

        Returns:
            Diccionario que asocia un identificador ``"{índice}_{Clase}"`` con
            el diccionario de parámetros de cada transformación.
        """
        return {
            f"{index}_{type(step).__name__}": step.get_params()
            for index, step in enumerate(self.steps)
        }

    def _check_compatibility(self) -> None:
        """Advierte sobre combinaciones de pasos incompatibles.

        Actualmente detecta una ``Normalize`` aplicada antes de una
        ``CLAHETransform``, dado que CLAHE opera sobre valores enteros y la
        normalización produce imágenes de punto flotante.
        """
        seen_normalize = False
        for step in self.steps:
            if isinstance(step, Normalize):
                seen_normalize = True
            elif isinstance(step, CLAHETransform) and seen_normalize:
                logger.warning(
                    "Combinación potencialmente incompatible: Normalize antes de "
                    "CLAHETransform; CLAHE espera valores enteros."
                )
