# autopipe/pipeline/transforms.py
"""Transformaciones de preprocesamiento predefinidas para imágenes."""

from __future__ import annotations

import cv2
import numpy as np

from autopipe.pipeline.base import BaseTransform


def _ensure_uint8(image: np.ndarray) -> np.ndarray:
    """Convierte una imagen a ``uint8`` en el rango [0, 255].

    Las imágenes de punto flotante se asumen normalizadas en [0, 1] cuando su
    valor máximo no supera 1.0; en caso contrario se recortan al rango válido.

    Args:
        image: Imagen de entrada como arreglo NumPy.

    Returns:
        La imagen convertida a ``uint8``.
    """
    if image.dtype == np.uint8:
        return image
    array = image.astype(np.float64)
    if array.size > 0 and array.max() <= 1.0:
        array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


class GaussianBlur(BaseTransform):
    """Suavizado gaussiano mediante :func:`cv2.GaussianBlur`.

    Attributes:
        kernel_size (int): Tamaño del kernel cuadrado (entero impar positivo).
    """

    def __init__(self, kernel_size: int) -> None:
        """Inicializa la transformación.

        Args:
            kernel_size: Tamaño del kernel cuadrado, entero impar positivo.

        Raises:
            ValueError: Si ``kernel_size`` no es un entero impar positivo.
        """
        if not isinstance(kernel_size, int) or kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size debe ser un entero impar positivo, se recibió {kernel_size!r}."
            )
        self.kernel_size = kernel_size

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Aplica el suavizado gaussiano a la imagen."""
        return cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), 0)

    def get_params(self) -> dict:
        """Devuelve los parámetros de la transformación."""
        return {"kernel_size": self.kernel_size}


class Normalize(BaseTransform):
    """Normalización por media y desviación estándar.

    Las imágenes enteras se escalan previamente a [0, 1] dividiendo entre 255.
    La salida es de tipo ``float32`` y se calcula como ``(x - mean) / std``.

    Attributes:
        mean (float): Media a sustraer.
        std (float): Desviación estándar por la que dividir.
    """

    def __init__(self, mean: float, std: float) -> None:
        """Inicializa la transformación.

        Args:
            mean: Media a sustraer.
            std: Desviación estándar por la que dividir; debe ser distinta de 0.

        Raises:
            ValueError: Si ``std`` es igual a 0.
        """
        if std == 0:
            raise ValueError("std debe ser distinta de 0.")
        self.mean = float(mean)
        self.std = float(std)

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Aplica la normalización a la imagen."""
        array = image.astype(np.float32)
        if np.issubdtype(image.dtype, np.integer):
            array = array / 255.0
        return (array - self.mean) / self.std

    def get_params(self) -> dict:
        """Devuelve los parámetros de la transformación."""
        return {"mean": self.mean, "std": self.std}


class Resize(BaseTransform):
    """Redimensionamiento mediante :func:`cv2.resize`.

    Attributes:
        width (int): Ancho de salida en píxeles.
        height (int): Alto de salida en píxeles.
    """

    def __init__(self, width: int, height: int) -> None:
        """Inicializa la transformación.

        Args:
            width: Ancho de salida en píxeles (entero positivo).
            height: Alto de salida en píxeles (entero positivo).

        Raises:
            ValueError: Si ``width`` o ``height`` no son enteros positivos.
        """
        if not isinstance(width, int) or not isinstance(height, int):
            raise ValueError("width y height deben ser enteros.")
        if width <= 0 or height <= 0:
            raise ValueError(
                f"width y height deben ser positivos, se recibió ({width}, {height})."
            )
        self.width = width
        self.height = height

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Redimensiona la imagen al tamaño configurado."""
        return cv2.resize(image, (self.width, self.height))

    def get_params(self) -> dict:
        """Devuelve los parámetros de la transformación."""
        return {"width": self.width, "height": self.height}


class CLAHETransform(BaseTransform):
    """Ecualización adaptativa de histograma con contraste limitado (CLAHE).

    Para imágenes en color (3 canales) la ecualización se aplica al canal de
    luminancia en el espacio LAB; para imágenes en escala de grises se aplica
    directamente. La entrada se convierte a ``uint8`` antes de procesar.

    Attributes:
        clip_limit (float): Umbral de recorte de contraste.
        tile_grid_size (tuple): Tamaño de la cuadrícula de mosaicos (filas, columnas).
    """

    def __init__(self, clip_limit: float, tile_grid_size: tuple) -> None:
        """Inicializa la transformación.

        Args:
            clip_limit: Umbral de recorte de contraste; debe ser positivo.
            tile_grid_size: Tupla ``(filas, columnas)`` de enteros positivos.

        Raises:
            ValueError: Si ``clip_limit`` no es positivo o ``tile_grid_size`` no
                es una tupla de dos enteros positivos.
        """
        if clip_limit <= 0:
            raise ValueError(f"clip_limit debe ser positivo, se recibió {clip_limit!r}.")
        if len(tile_grid_size) != 2 or any(
            not isinstance(value, int) or value <= 0 for value in tile_grid_size
        ):
            raise ValueError(
                "tile_grid_size debe ser una tupla de dos enteros positivos, "
                f"se recibió {tile_grid_size!r}."
            )
        self.clip_limit = float(clip_limit)
        self.tile_grid_size = tuple(tile_grid_size)

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Aplica CLAHE a la imagen."""
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        prepared = _ensure_uint8(image)
        if prepared.ndim == 2:
            return clahe.apply(prepared)
        lab = cv2.cvtColor(prepared, cv2.COLOR_RGB2LAB)
        luminance, channel_a, channel_b = cv2.split(lab)
        luminance = clahe.apply(luminance)
        merged = cv2.merge((luminance, channel_a, channel_b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

    def get_params(self) -> dict:
        """Devuelve los parámetros de la transformación."""
        return {"clip_limit": self.clip_limit, "tile_grid_size": self.tile_grid_size}


class MedianFilter(BaseTransform):
    """Filtro de mediana mediante :func:`cv2.medianBlur`.

    Attributes:
        kernel_size (int): Tamaño del kernel (entero impar mayor o igual a 3).
    """

    def __init__(self, kernel_size: int) -> None:
        """Inicializa la transformación.

        Args:
            kernel_size: Tamaño del kernel, entero impar mayor o igual a 3.

        Raises:
            ValueError: Si ``kernel_size`` no es un entero impar >= 3.
        """
        if not isinstance(kernel_size, int) or kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size debe ser un entero impar >= 3, se recibió {kernel_size!r}."
            )
        self.kernel_size = kernel_size

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Aplica el filtro de mediana a la imagen."""
        return cv2.medianBlur(image, self.kernel_size)

    def get_params(self) -> dict:
        """Devuelve los parámetros de la transformación."""
        return {"kernel_size": self.kernel_size}


class GammaCorrection(BaseTransform):
    """Corrección gamma de la intensidad de la imagen.

    Aplica ``salida = (entrada / 255) ** gamma * 255`` sobre imágenes ``uint8``
    mediante una tabla de búsqueda; para imágenes de punto flotante aplica la
    potencia directamente sobre los valores no negativos. ``gamma < 1`` aclara la
    imagen y ``gamma > 1`` la oscurece.

    Attributes:
        gamma (float): Exponente de la corrección gamma.
    """

    def __init__(self, gamma: float) -> None:
        """Inicializa la transformación.

        Args:
            gamma: Exponente de la corrección gamma; debe ser positivo.

        Raises:
            ValueError: Si ``gamma`` no es positivo.
        """
        if gamma <= 0:
            raise ValueError(f"gamma debe ser positivo, se recibió {gamma!r}.")
        self.gamma = float(gamma)

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Aplica la corrección gamma a la imagen."""
        if image.dtype == np.uint8:
            table = (((np.arange(256) / 255.0) ** self.gamma) * 255.0).astype(np.uint8)
            return cv2.LUT(image, table)
        array = np.clip(image.astype(np.float64), 0.0, None)
        return np.power(array, self.gamma)

    def get_params(self) -> dict:
        """Devuelve los parámetros de la transformación."""
        return {"gamma": self.gamma}


class GrayscaleConversion(BaseTransform):
    """Conversión de una imagen RGB a escala de grises.

    Las imágenes de tres canales se convierten con :func:`cv2.cvtColor`; las
    imágenes que ya están en escala de grises (2D) se devuelven sin cambios.
    """

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Convierte la imagen a escala de grises."""
        if image.ndim == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    def get_params(self) -> dict:
        """Devuelve los parámetros de la transformación (ninguno)."""
        return {}
