# autopipe/data_loader/tabular.py
"""Carga y validación de datasets tabulares para clasificación."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TabularDataLoader:
    """Carga un dataset tabular y separa las características de la etiqueta.

    Acepta una ruta a un archivo CSV o un ``DataFrame`` ya en memoria, de modo que
    el mismo cargador sirve para datos en disco y para datos obtenidos por otra
    vía (por ejemplo, la descarga del corpus desde OpenML).

    Devuelve las características como ``DataFrame`` y **no** como matriz numérica,
    a propósito: conservar los tipos de cada columna es lo que permite a las
    transformaciones tabulares distinguir por sí mismas qué columnas son
    categóricas y cuáles numéricas. Si esa distinción tuviera que pasarse como
    parámetro, el espacio de búsqueda dependería del dataset y las
    configuraciones dejarían de ser transferibles entre datasets, que es
    justamente lo que el arranque en caliente necesita.

    Las etiquetas se codifican como enteros ``0..k-1`` con un orden determinista
    (el de ``numpy.unique``), y los valores originales quedan en
    :attr:`classes_`. Para etiquetas que ya son ``{0, 1}`` la codificación es la
    identidad, de modo que se preserva la convención de detección de anomalías.

    Attributes:
        source (Path | pandas.DataFrame): Origen de los datos.
        target_column (str | int): Nombre o posición de la columna objetivo.
        classes_ (numpy.ndarray | None): Valores originales de las etiquetas, en
            el orden de la codificación. Disponible tras :meth:`load`.
        feature_names_ (list[str] | None): Nombres de las columnas de
            características, tras :meth:`load`.
        categorical_columns_ (list[str] | None): Columnas no numéricas, tras
            :meth:`load`.
        numeric_columns_ (list[str] | None): Columnas numéricas, tras :meth:`load`.
    """

    SUPPORTED_EXTENSIONS = (".csv",)

    def __init__(
        self,
        source: str | Path | pd.DataFrame,
        target_column: str | int = -1,
        na_values: list[str] | None = None,
    ) -> None:
        """Inicializa el cargador.

        Args:
            source: Ruta a un archivo CSV, o un ``DataFrame`` en memoria.
            target_column: Nombre de la columna objetivo, o su posición. El valor
                por defecto ``-1`` toma la última columna, que es la convención
                de los repositorios OpenML y UCI.
            na_values: Cadenas adicionales a interpretar como valor faltante al
                leer un CSV (por ejemplo ``["?"]``, habitual en UCI).
        """
        self.source = source if isinstance(source, pd.DataFrame) else Path(source)
        self.target_column = target_column
        self.na_values = na_values
        self.classes_: np.ndarray | None = None
        self.feature_names_: list[str] | None = None
        self.categorical_columns_: list[str] | None = None
        self.numeric_columns_: list[str] | None = None

    def load(self) -> tuple[pd.DataFrame, np.ndarray]:
        """Carga el dataset y devuelve las características y las etiquetas.

        Returns:
            Tupla ``(features, labels)`` donde ``features`` es un ``DataFrame``
            con el índice reiniciado y ``labels`` un arreglo de enteros
            ``0..k-1``.

        Raises:
            FileNotFoundError: Si la ruta indicada no existe.
            ValueError: Si la extensión no está soportada, si el dataset está
                vacío, si la columna objetivo no existe, si el objetivo tiene
                valores faltantes, si no queda ninguna columna de características
                o si hay menos de dos clases.
        """
        data = self._read()
        target_name = self._resolve_target_name(data)

        labels_raw = data[target_name]
        if labels_raw.isna().any():
            raise ValueError(
                f"La columna objetivo {target_name!r} tiene "
                f"{int(labels_raw.isna().sum())} valores faltantes."
            )

        features = data.drop(columns=[target_name]).reset_index(drop=True)
        if features.shape[1] == 0:
            raise ValueError("El dataset no tiene ninguna columna de características.")

        self.classes_, encoded = np.unique(labels_raw.to_numpy(), return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError(
                f"La columna objetivo {target_name!r} tiene una sola clase; se "
                f"requieren al menos dos para clasificación."
            )

        self.feature_names_ = list(features.columns)
        self.numeric_columns_ = list(features.select_dtypes(include="number").columns)
        self.categorical_columns_ = [
            name for name in self.feature_names_ if name not in self.numeric_columns_
        ]

        faltantes = int(features.isna().to_numpy().sum())
        if faltantes:
            logger.warning(
                "El dataset tiene %d valores faltantes en las características; el "
                "pipeline debe imputarlos antes de entrenar.",
                faltantes,
            )
        logger.info(
            "Dataset tabular cargado: %d instancias, %d características "
            "(%d numéricas, %d categóricas), %d clases.",
            features.shape[0],
            features.shape[1],
            len(self.numeric_columns_),
            len(self.categorical_columns_),
            len(self.classes_),
        )
        return features, encoded.astype(np.int64)

    def _read(self) -> pd.DataFrame:
        """Obtiene el ``DataFrame`` crudo desde el origen configurado.

        Returns:
            Copia del dataset como ``DataFrame``.

        Raises:
            FileNotFoundError: Si la ruta no existe.
            ValueError: Si la extensión no está soportada o el dataset está vacío.
        """
        if isinstance(self.source, pd.DataFrame):
            data = self.source.copy()
        else:
            if not self.source.is_file():
                raise FileNotFoundError(f"El archivo del dataset no existe: {self.source}.")
            if self.source.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                raise ValueError(
                    f"Extensión no soportada: {self.source.suffix!r}. "
                    f"Use una de {self.SUPPORTED_EXTENSIONS}."
                )
            data = pd.read_csv(self.source, na_values=self.na_values)

        if data.empty:
            raise ValueError("El dataset está vacío.")
        return data

    def _resolve_target_name(self, data: pd.DataFrame) -> str:
        """Traduce :attr:`target_column` al nombre real de la columna.

        Args:
            data: Dataset crudo.

        Returns:
            El nombre de la columna objetivo.

        Raises:
            ValueError: Si la columna indicada no existe.
        """
        if isinstance(self.target_column, int):
            try:
                return str(data.columns[self.target_column])
            except IndexError as exc:
                raise ValueError(
                    f"La posición {self.target_column} está fuera de rango: el "
                    f"dataset tiene {data.shape[1]} columnas."
                ) from exc
        if self.target_column not in data.columns:
            raise ValueError(
                f"La columna objetivo {self.target_column!r} no existe. "
                f"Columnas disponibles: {list(data.columns)}."
            )
        return str(self.target_column)
