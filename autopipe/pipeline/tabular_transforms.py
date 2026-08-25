# autopipe/pipeline/tabular_transforms.py
"""Transformaciones de preprocesamiento con estado para datos tabulares.

Todas heredan de :class:`~autopipe.pipeline.base.StatefulTransform`: aprenden del
subconjunto de entrenamiento y aplican lo aprendido a cualquier otro conjunto.

**Sus parámetros no dependen del dataset.** Ninguna recibe nombres ni posiciones
de columnas: cada una detecta por sí misma sobre qué columnas actuar a partir de
los tipos de la tabla. Es la condición para que una configuración hallada en un
dataset pueda inyectarse como arranque en caliente en otro.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from autopipe.pipeline.base import Dataset, StatefulTransform

CATEGORIA_DESCONOCIDA = -1
"""Código asignado por :class:`EncodeCategorical` en modo ordinal a una categoría
que no apareció en el subconjunto de entrenamiento."""


class TabularTransform(StatefulTransform):
    """Base de las transformaciones tabulares.

    Aporta lo que todas comparten: exigir una tabla, recordar las columnas vistas
    en el ajuste y resolver el contrato por muestra de
    :class:`~autopipe.pipeline.base.BaseTransform` construyendo una tabla de una
    fila.

    Attributes:
        columns_ (list[str] | None): Columnas vistas durante el ajuste.
    """

    columns_: list[str] | None = None

    @staticmethod
    def _as_frame(X: Dataset) -> pd.DataFrame:
        """Verifica que la entrada sea una tabla y la devuelve.

        Args:
            X: Conjunto de muestras.

        Returns:
            La tabla de entrada.

        Raises:
            TypeError: Si la entrada no es un ``DataFrame``.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"Las transformaciones tabulares operan sobre un DataFrame de pandas; "
                f"se recibió {type(X).__name__}. Use "
                f"autopipe.data_loader.TabularDataLoader para cargar el dataset."
            )
        return X

    def _recordar_columnas(self, frame: pd.DataFrame) -> None:
        """Guarda las columnas vistas en el ajuste.

        Args:
            frame: Tabla de entrenamiento.
        """
        self.columns_ = list(frame.columns)

    def _verificar_columnas(self, frame: pd.DataFrame) -> None:
        """Comprueba que estén presentes todas las columnas del ajuste.

        Transformar una tabla a la que le falta una columna produciría valores de
        relleno silenciosos, así que se rechaza de forma explícita.

        Args:
            frame: Tabla a transformar.

        Raises:
            ValueError: Si falta alguna columna vista durante el ajuste.
        """
        faltantes = [
            nombre for nombre in (self.columns_ or []) if nombre not in frame.columns
        ]
        if faltantes:
            raise ValueError(
                f"A la tabla le faltan columnas que sí estaban en el ajuste: "
                f"{faltantes}. El conjunto a transformar debe tener las mismas "
                f"columnas que el de entrenamiento."
            )

    def apply(self, sample: np.ndarray) -> np.ndarray:
        """Transforma una sola muestra construyendo una tabla de una fila.

        Args:
            sample: Valores de una muestra, en el orden de :attr:`columns_`.

        Returns:
            La muestra transformada, como arreglo.

        Raises:
            NotFittedError: Si la transformación aún no fue ajustada.
        """
        self._check_fitted()
        fila = pd.DataFrame([list(np.asarray(sample, dtype=object))], columns=self.columns_)
        return self._transform(fila).to_numpy()[0]


class ImputeMissing(TabularTransform):
    """Rellena los valores faltantes con un estadístico del entrenamiento.

    Las columnas numéricas se rellenan con la media o la mediana, según
    ``numeric_strategy``. Las categóricas se rellenan siempre con la moda, porque
    en ellas la media no está definida.

    Si una columna está enteramente vacía en el entrenamiento no hay estadístico
    que aprender: se rellena con ``0.0`` si es numérica y con la cadena
    ``"faltante"`` si es categórica, de modo que no se propaguen valores nulos
    hacia el modelo.

    Attributes:
        numeric_strategy (str): ``"mean"`` o ``"median"``.
        fill_values_ (dict | None): Valor de relleno aprendido por columna.
    """

    ESTRATEGIAS = ("mean", "median")
    RELLENO_NUMERICO = 0.0
    RELLENO_CATEGORICO = "faltante"

    def __init__(self, numeric_strategy: str = "mean") -> None:
        """Inicializa la imputación.

        Args:
            numeric_strategy: Estadístico para columnas numéricas.

        Raises:
            ValueError: Si la estrategia no está soportada.
        """
        if numeric_strategy not in self.ESTRATEGIAS:
            raise ValueError(
                f"numeric_strategy debe ser una de {self.ESTRATEGIAS}, se recibió "
                f"{numeric_strategy!r}."
            )
        self.numeric_strategy = numeric_strategy
        self.fill_values_: dict[str, object] | None = None

    def _fit(self, X: Dataset, y: np.ndarray | None = None) -> None:
        """Aprende el valor de relleno de cada columna."""
        frame = self._as_frame(X)
        self._recordar_columnas(frame)
        self.fill_values_ = {
            nombre: self._valor_de_relleno(frame[nombre]) for nombre in frame.columns
        }

    def _valor_de_relleno(self, columna: pd.Series) -> object:
        """Calcula el valor con el que se rellenará una columna.

        Args:
            columna: Columna del subconjunto de entrenamiento.

        Returns:
            El estadístico aprendido, o el relleno de reserva si la columna está
            enteramente vacía.
        """
        if pd.api.types.is_numeric_dtype(columna):
            estadistico = (
                columna.mean() if self.numeric_strategy == "mean" else columna.median()
            )
            return self.RELLENO_NUMERICO if pd.isna(estadistico) else float(estadistico)
        modas = columna.mode(dropna=True)
        return self.RELLENO_CATEGORICO if modas.empty else modas.iloc[0]

    def _transform(self, X: Dataset) -> pd.DataFrame:
        """Rellena los faltantes con los valores aprendidos."""
        frame = self._as_frame(X)
        self._verificar_columnas(frame)
        frame = frame.copy()
        for nombre, valor in (self.fill_values_ or {}).items():
            frame[nombre] = frame[nombre].fillna(valor)
        return frame

    def get_params(self) -> dict:
        """Devuelve los parámetros de la transformación."""
        return {"numeric_strategy": self.numeric_strategy}


class ScaleNumeric(TabularTransform):
    """Escala las columnas numéricas con estadísticos del entrenamiento.

    Las columnas no numéricas se dejan intactas, de modo que la transformación
    puede aplicarse antes o después de la codificación.

    Métodos disponibles:

    * ``"standard"``: ``(x - media) / desviación``.
    * ``"minmax"``: ``(x - mínimo) / (máximo - mínimo)``.
    * ``"robust"``: ``(x - mediana) / rango interquartílico``.

    Cuando la escala aprendida es cero —una columna constante en el
    entrenamiento— se sustituye por uno, de modo que la columna quede centrada en
    lugar de producir una división por cero.

    Attributes:
        method (str): Método de escalado.
        centers_ (dict | None): Desplazamiento aprendido por columna.
        scales_ (dict | None): Divisor aprendido por columna.
        numeric_columns_ (list[str] | None): Columnas numéricas del ajuste.
    """

    METODOS = ("standard", "minmax", "robust")

    def __init__(self, method: str = "standard") -> None:
        """Inicializa el escalado.

        Args:
            method: Método de escalado.

        Raises:
            ValueError: Si el método no está soportado.
        """
        if method not in self.METODOS:
            raise ValueError(
                f"method debe ser uno de {self.METODOS}, se recibió {method!r}."
            )
        self.method = method
        self.centers_: dict[str, float] | None = None
        self.scales_: dict[str, float] | None = None
        self.numeric_columns_: list[str] | None = None

    def _fit(self, X: Dataset, y: np.ndarray | None = None) -> None:
        """Aprende el desplazamiento y el divisor de cada columna numérica."""
        frame = self._as_frame(X)
        self._recordar_columnas(frame)
        self.numeric_columns_ = list(frame.select_dtypes(include="number").columns)
        self.centers_ = {}
        self.scales_ = {}
        for nombre in self.numeric_columns_:
            centro, escala = self._estadisticos(frame[nombre])
            self.centers_[nombre] = centro
            self.scales_[nombre] = escala if escala > 0 else 1.0

    def _estadisticos(self, columna: pd.Series) -> tuple[float, float]:
        """Calcula el desplazamiento y el divisor de una columna.

        Args:
            columna: Columna numérica del entrenamiento.

        Returns:
            Tupla ``(centro, escala)``. Los valores no finitos se sustituyen por
            ``0.0`` y ``1.0`` respectivamente.
        """
        valores = columna.to_numpy(dtype=float)
        if self.method == "standard":
            centro, escala = np.nanmean(valores), np.nanstd(valores)
        elif self.method == "minmax":
            centro = np.nanmin(valores)
            escala = np.nanmax(valores) - centro
        else:
            centro = np.nanmedian(valores)
            escala = np.nanpercentile(valores, 75) - np.nanpercentile(valores, 25)
        centro = 0.0 if not np.isfinite(centro) else float(centro)
        escala = 1.0 if not np.isfinite(escala) else float(escala)
        return centro, escala

    def _transform(self, X: Dataset) -> pd.DataFrame:
        """Aplica el escalado aprendido a las columnas numéricas."""
        frame = self._as_frame(X)
        self._verificar_columnas(frame)
        frame = frame.copy()
        for nombre in self.numeric_columns_ or []:
            centro = (self.centers_ or {})[nombre]
            escala = (self.scales_ or {})[nombre]
            frame[nombre] = (frame[nombre].astype(float) - centro) / escala
        return frame

    def get_params(self) -> dict:
        """Devuelve los parámetros de la transformación."""
        return {"method": self.method}


class EncodeCategorical(TabularTransform):
    """Convierte las columnas categóricas en columnas numéricas.

    Métodos disponibles:

    * ``"onehot"``: una columna indicadora por categoría vista en el
      entrenamiento, nombrada ``"{columna}_{categoría}"``.
    * ``"ordinal"``: un entero por categoría, según el orden de las categorías
      aprendidas.

    Las categorías se aprenden **solo del entrenamiento**. Una categoría que
    aparezca únicamente en validación no genera ninguna columna nueva: en modo
    ``"onehot"`` produce ceros en todas las indicadoras de esa columna, y en modo
    ``"ordinal"`` recibe el código :data:`CATEGORIA_DESCONOCIDA`. Así el número de
    columnas queda fijado por el ajuste y la validación no puede alterarlo.

    Attributes:
        method (str): Método de codificación.
        categories_ (dict | None): Categorías aprendidas por columna.
        categorical_columns_ (list[str] | None): Columnas codificadas.
        output_columns_ (list[str] | None): Columnas de la tabla resultante.
    """

    METODOS = ("onehot", "ordinal")

    def __init__(self, method: str = "onehot") -> None:
        """Inicializa la codificación.

        Args:
            method: Método de codificación.

        Raises:
            ValueError: Si el método no está soportado.
        """
        if method not in self.METODOS:
            raise ValueError(
                f"method debe ser uno de {self.METODOS}, se recibió {method!r}."
            )
        self.method = method
        self.categories_: dict[str, list] | None = None
        self.categorical_columns_: list[str] | None = None
        self.output_columns_: list[str] | None = None

    def _fit(self, X: Dataset, y: np.ndarray | None = None) -> None:
        """Aprende las categorías de cada columna no numérica."""
        frame = self._as_frame(X)
        self._recordar_columnas(frame)
        numericas = list(frame.select_dtypes(include="number").columns)
        self.categorical_columns_ = [
            nombre for nombre in frame.columns if nombre not in numericas
        ]
        self.categories_ = {
            nombre: sorted(frame[nombre].dropna().unique().tolist(), key=str)
            for nombre in self.categorical_columns_
        }

        salida = list(numericas)
        if self.method == "onehot":
            for nombre in self.categorical_columns_:
                salida.extend(
                    f"{nombre}_{categoria}" for categoria in self.categories_[nombre]
                )
        else:
            salida.extend(self.categorical_columns_)
        self.output_columns_ = salida

    def _transform(self, X: Dataset) -> pd.DataFrame:
        """Codifica las columnas categóricas con las categorías aprendidas."""
        frame = self._as_frame(X)
        self._verificar_columnas(frame)
        numericas = [
            nombre
            for nombre in frame.columns
            if nombre not in (self.categorical_columns_ or [])
        ]
        piezas = [frame[numericas].reset_index(drop=True)]

        for nombre in self.categorical_columns_ or []:
            columna = frame[nombre].reset_index(drop=True)
            categorias = (self.categories_ or {})[nombre]
            if self.method == "onehot":
                piezas.append(self._indicadoras(columna, nombre, categorias))
            else:
                piezas.append(self._ordinal(columna, nombre, categorias))

        resultado = pd.concat(piezas, axis=1)
        return resultado.reindex(columns=self.output_columns_, fill_value=0)

    @staticmethod
    def _indicadoras(columna: pd.Series, nombre: str, categorias: list) -> pd.DataFrame:
        """Construye las columnas indicadoras de una columna categórica.

        Args:
            columna: Valores de la columna.
            nombre: Nombre de la columna original.
            categorias: Categorías aprendidas en el ajuste.

        Returns:
            Tabla con una columna indicadora por categoría aprendida.
        """
        return pd.DataFrame(
            {
                f"{nombre}_{categoria}": (columna == categoria).astype(int)
                for categoria in categorias
            },
            index=columna.index,
        )

    @staticmethod
    def _ordinal(columna: pd.Series, nombre: str, categorias: list) -> pd.DataFrame:
        """Codifica una columna categórica como enteros.

        Args:
            columna: Valores de la columna.
            nombre: Nombre de la columna original.
            categorias: Categorías aprendidas en el ajuste.

        Returns:
            Tabla con una única columna de códigos enteros.
        """
        codigos = {categoria: indice for indice, categoria in enumerate(categorias)}
        return pd.DataFrame(
            {nombre: columna.map(codigos).fillna(CATEGORIA_DESCONOCIDA).astype(int)},
            index=columna.index,
        )

    def get_params(self) -> dict:
        """Devuelve los parámetros de la transformación."""
        return {"method": self.method}
