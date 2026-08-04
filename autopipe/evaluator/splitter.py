# autopipe/evaluator/splitter.py
"""Partición de datasets en subconjuntos de entrenamiento y validación."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


class DataSplitter:
    """Genera particiones train/validación reproducibles.

    Soporta una partición simple train/validación y validación cruzada k-fold,
    ambas con opción de estratificación por etiqueta. La aleatoriedad se controla
    mediante ``random_state`` para garantizar reproducibilidad.

    Attributes:
        val_size (float): Proporción del dataset destinada a validación.
        random_state (int | None): Semilla para la partición.
        stratify (bool): Si la partición debe estratificarse por etiqueta.
    """

    def __init__(
        self,
        val_size: float = 0.2,
        random_state: int | None = None,
        stratify: bool = True,
    ) -> None:
        """Inicializa el particionador.

        Args:
            val_size: Proporción del dataset para validación, en el rango (0, 1).
            random_state: Semilla para reproducibilidad.
            stratify: Si la partición debe estratificarse por etiqueta.

        Raises:
            ValueError: Si ``val_size`` no está en el intervalo abierto (0, 1).
        """
        if not 0.0 < val_size < 1.0:
            raise ValueError(
                f"val_size debe estar en el intervalo (0, 1), se recibió {val_size!r}."
            )
        self.val_size = float(val_size)
        self.random_state = random_state
        self.stratify = stratify

    def split(self, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Divide los índices del dataset en entrenamiento y validación.

        Args:
            labels: Arreglo de etiquetas binarias del dataset.

        Returns:
            Tupla ``(train_idx, val_idx)`` con los índices de cada subconjunto.
        """
        labels = np.asarray(labels)
        indices = np.arange(len(labels))
        stratify_target = labels if self.stratify else None
        train_idx, val_idx = train_test_split(
            indices,
            test_size=self.val_size,
            random_state=self.random_state,
            stratify=stratify_target,
        )
        return np.asarray(train_idx), np.asarray(val_idx)

    def k_fold_split(
        self, labels: np.ndarray, n_splits: int = 5
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Genera particiones de validación cruzada k-fold.

        Args:
            labels: Arreglo de etiquetas binarias del dataset.
            n_splits: Número de particiones; debe ser >= 2.

        Returns:
            Lista de tuplas ``(train_idx, val_idx)``, una por fold.

        Raises:
            ValueError: Si ``n_splits`` es menor que 2.
        """
        if n_splits < 2:
            raise ValueError(f"n_splits debe ser >= 2, se recibió {n_splits!r}.")
        labels = np.asarray(labels)
        if self.stratify:
            splitter: StratifiedKFold | KFold = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=self.random_state
            )
        else:
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        return [
            (np.asarray(train_idx), np.asarray(val_idx))
            for train_idx, val_idx in splitter.split(np.zeros(len(labels)), labels)
        ]

    def get_config(self) -> dict:
        """Devuelve la configuración del particionador.

        Returns:
            Diccionario con ``val_size``, ``random_state`` y ``stratify``.
        """
        return {
            "val_size": self.val_size,
            "random_state": self.random_state,
            "stratify": self.stratify,
        }

    def save_config(self, path: str | Path) -> None:
        """Guarda la configuración del particionador en JSON o YAML.

        El formato se determina por la extensión del archivo: ``.json`` para
        JSON y ``.yaml``/``.yml`` para YAML.

        Args:
            path: Ruta de salida.

        Raises:
            ValueError: Si la extensión no es ``.json``, ``.yaml`` o ``.yml``.
        """
        path = Path(path)
        config = self.get_config()
        suffix = path.suffix.lower()
        if suffix == ".json":
            path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        elif suffix in (".yaml", ".yml"):
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
        else:
            raise ValueError(
                f"Extensión no soportada para configuración: {path.suffix!r}. "
                "Use .json, .yaml o .yml."
            )

    @classmethod
    def load_config(cls, path: str | Path) -> DataSplitter:
        """Crea un particionador desde un archivo de configuración JSON o YAML.

        Args:
            path: Ruta del archivo de configuración.

        Returns:
            Una instancia de :class:`DataSplitter` con la configuración cargada.

        Raises:
            ValueError: Si la extensión no es ``.json``, ``.yaml`` o ``.yml``.
        """
        path = Path(path)
        suffix = path.suffix.lower()
        content = path.read_text(encoding="utf-8")
        if suffix == ".json":
            config = json.loads(content)
        elif suffix in (".yaml", ".yml"):
            config = yaml.safe_load(content)
        else:
            raise ValueError(
                f"Extensión no soportada para configuración: {path.suffix!r}. "
                "Use .json, .yaml o .yml."
            )
        return cls(**config)
