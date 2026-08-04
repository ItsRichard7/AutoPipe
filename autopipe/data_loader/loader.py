# autopipe/data_loader/loader.py
"""Carga y validación de datasets de imágenes para detección de anomalías."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)


class ImageDataLoader:
    """Carga imágenes desde un directorio con subdirectorios ``normal`` y ``anomaly``.

    El dataset debe estar organizado de la siguiente forma::

        dataset_path/
        ├── normal/
        │   ├── img_00.png
        │   └── ...
        └── anomaly/
            ├── img_00.png
            └── ...

    Las imágenes de ``normal`` reciben la etiqueta 0 y las de ``anomaly`` la
    etiqueta 1.

    Attributes:
        dataset_path (Path): Ruta raíz del dataset.
    """

    NORMAL_DIR = "normal"
    ANOMALY_DIR = "anomaly"
    NORMAL_LABEL = 0
    ANOMALY_LABEL = 1
    SUPPORTED_EXTENSIONS = (".png", ".jpg", ".bmp")

    def __init__(self, dataset_path: str | Path) -> None:
        """Inicializa el cargador.

        Args:
            dataset_path: Ruta al directorio raíz del dataset.
        """
        self.dataset_path = Path(dataset_path)

    def load(self) -> tuple[np.ndarray, np.ndarray]:
        """Carga todas las imágenes válidas y sus etiquetas binarias.

        Returns:
            Una tupla ``(images, labels)`` donde ``images`` es un arreglo NumPy
            con las imágenes (apiladas si comparten dimensiones, de lo contrario
            un arreglo de objetos) y ``labels`` un arreglo de enteros con 0 para
            imágenes normales y 1 para anómalas.

        Raises:
            FileNotFoundError: Si la raíz o algún subdirectorio requerido no existe.
            ValueError: Si no se encuentra ninguna imagen válida o si algún
                archivo de imagen no se puede leer.
        """
        self._validate_structure()

        images: list[np.ndarray] = []
        labels: list[int] = []
        for subdir, label in (
            (self.NORMAL_DIR, self.NORMAL_LABEL),
            (self.ANOMALY_DIR, self.ANOMALY_LABEL),
        ):
            for image_path in self._list_image_files(self.dataset_path / subdir):
                images.append(self._read_image(image_path))
                labels.append(label)

        if not images:
            raise ValueError(
                f"No se encontraron imágenes con extensiones "
                f"{self.SUPPORTED_EXTENSIONS} en {self.dataset_path}."
            )

        logger.info(
            "Dataset cargado: %d imágenes (%d normales, %d anómalas).",
            len(images),
            labels.count(self.NORMAL_LABEL),
            labels.count(self.ANOMALY_LABEL),
        )
        return self._stack(images), np.asarray(labels, dtype=np.int64)

    def _validate_structure(self) -> None:
        """Verifica que la raíz y los subdirectorios requeridos existan.

        Raises:
            FileNotFoundError: Si la raíz o algún subdirectorio requerido falta.
        """
        if not self.dataset_path.is_dir():
            raise FileNotFoundError(
                f"El directorio raíz del dataset no existe: {self.dataset_path}."
            )
        for subdir in (self.NORMAL_DIR, self.ANOMALY_DIR):
            subdir_path = self.dataset_path / subdir
            if not subdir_path.is_dir():
                raise FileNotFoundError(
                    f"Falta el subdirectorio requerido: {subdir_path}."
                )

    def _list_image_files(self, directory: Path) -> list[Path]:
        """Lista los archivos con extensión soportada dentro de un directorio.

        Los archivos con extensión no soportada se omiten registrando una
        advertencia mediante ``logging`` (RF-002).

        Args:
            directory: Directorio a inspeccionar.

        Returns:
            Lista ordenada de rutas a archivos de imagen soportados.
        """
        files: list[Path] = []
        for path in directory.iterdir():
            if not path.is_file():
                continue
            if path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                files.append(path)
            else:
                logger.warning(
                    "Archivo con extensión no soportada omitido: %s.", path
                )
        return sorted(files)

    def _read_image(self, image_path: Path) -> np.ndarray:
        """Lee una imagen desde disco como arreglo NumPy.

        Args:
            image_path: Ruta al archivo de imagen.

        Returns:
            La imagen como arreglo NumPy.

        Raises:
            ValueError: Si el archivo no se puede decodificar como imagen.
        """
        try:
            with Image.open(image_path) as image:
                image.load()
                return np.asarray(image)
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(
                f"No se pudo leer el archivo de imagen: {image_path}."
            ) from exc

    def _stack(self, images: list[np.ndarray]) -> np.ndarray:
        """Apila las imágenes en un único arreglo NumPy.

        Si todas las imágenes comparten la misma forma se devuelve un arreglo
        denso de dimensión ``(n, ...)``; de lo contrario se devuelve un arreglo
        de objetos y se registra una advertencia.

        Args:
            images: Lista de imágenes como arreglos NumPy.

        Returns:
            Arreglo NumPy con todas las imágenes.
        """
        shapes = {image.shape for image in images}
        if len(shapes) == 1:
            return np.stack(images, axis=0)

        logger.warning(
            "Las imágenes tienen dimensiones heterogéneas (%d formas distintas); "
            "se devuelve un arreglo de objetos.",
            len(shapes),
        )
        object_array = np.empty(len(images), dtype=object)
        for index, image in enumerate(images):
            object_array[index] = image
        return object_array
