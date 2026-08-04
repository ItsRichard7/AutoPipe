# autopipe/pipeline/base.py
"""Clase base abstracta para todas las transformaciones de preprocesamiento."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseTransform(ABC):
    """Interfaz común para todas las transformaciones de preprocesamiento.

    Toda transformación, tanto las predefinidas como las personalizadas del
    usuario, debe heredar de esta clase e implementar :meth:`apply` y
    :meth:`get_params`.
    """

    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Aplica la transformación a una imagen.

        Args:
            image: Imagen de entrada como arreglo NumPy.

        Returns:
            La imagen transformada como arreglo NumPy.
        """
        raise NotImplementedError

    @abstractmethod
    def get_params(self) -> dict:
        """Devuelve los parámetros de configuración de la transformación.

        Returns:
            Diccionario con los parámetros que definen la transformación.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        """Representación legible basada en los parámetros de la transformación."""
        params = ", ".join(f"{key}={value!r}" for key, value in self.get_params().items())
        return f"{type(self).__name__}({params})"
