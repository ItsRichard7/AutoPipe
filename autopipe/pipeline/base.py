# autopipe/pipeline/base.py
"""Clase base abstracta para todas las transformaciones de preprocesamiento."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import cast

import numpy as np

from autopipe.protocols import TableLike

Dataset = Sequence[np.ndarray] | np.ndarray | TableLike
"""Conjunto de muestras: una secuencia de arreglos, un arreglo cuyas filas son
las muestras, o una tabla con indexación posicional (el caso de los datos
tabulares)."""


class NotFittedError(RuntimeError):
    """Se intentó transformar con una transformación que aún no fue ajustada.

    Las transformaciones con estado deben ajustarse con
    :meth:`StatefulTransform.fit` antes de transformar cualquier conjunto.
    """


class BaseTransform(ABC):
    """Interfaz común para todas las transformaciones de preprocesamiento.

    Toda transformación, tanto las predefinidas como las personalizadas del
    usuario, debe heredar de esta clase e implementar :meth:`apply` y
    :meth:`get_params`.

    El contrato expone dos niveles de granularidad:

    * :meth:`apply` opera **sobre una sola muestra** y es el método que toda
      subclase debe implementar.
    * :meth:`fit` y :meth:`transform` operan **sobre el conjunto completo** y
      traen implementación por defecto: ``fit`` no aprende nada y ``transform``
      aplica :meth:`apply` muestra por muestra. Ese comportamiento por defecto
      corresponde a una transformación **sin estado**, como son todas las
      transformaciones de imagen de la librería.

    Las transformaciones **con estado** (escalado, imputación, codificación,
    selección de características) deben sobrescribir :meth:`fit` y
    :meth:`transform` de forma explícita, para que el ajuste se realice
    únicamente con el subconjunto de entrenamiento y no se filtre información
    de la validación (véase ``docs/decisiones/ADR-001-contrato-fit-transform.md``).
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

    def fit(self, X: Dataset, y: np.ndarray | None = None) -> BaseTransform:
        """Ajusta la transformación al conjunto de entrenamiento.

        La implementación por defecto no aprende nada, porque corresponde a una
        transformación sin estado. Las subclases con estado deben sobrescribir
        este método y almacenar en la instancia lo que aprendan.

        Debe invocarse **solo con el subconjunto de entrenamiento**: ajustar con
        el dataset completo filtraría información de la validación.

        Args:
            X: Conjunto de muestras de entrenamiento.
            y: Etiquetas de entrenamiento, si la transformación las necesita.
                La implementación por defecto las ignora.

        Returns:
            La propia transformación, para permitir el encadenamiento.
        """
        return self

    def transform(self, X: Dataset) -> Dataset:
        """Transforma un conjunto completo de muestras.

        La implementación por defecto aplica :meth:`apply` a cada muestra, de
        modo que una transformación sin estado no necesita sobrescribirla.

        Args:
            X: Conjunto de muestras a transformar.

        Returns:
            Lista con las muestras transformadas, en el mismo orden.
        """
        # La implementación por defecto recorre las muestras una por una, de modo
        # que solo aplica a conjuntos iterables por muestra. Las transformaciones
        # que reciben tablas sobrescriben este método, porque recorrer una tabla
        # daría sus columnas y no sus filas.
        return [self.apply(sample) for sample in cast(Sequence[np.ndarray], X)]

    def fit_transform(self, X: Dataset, y: np.ndarray | None = None) -> Dataset:
        """Ajusta la transformación con ``X`` y devuelve ``X`` transformado.

        Args:
            X: Conjunto de muestras de entrenamiento.
            y: Etiquetas de entrenamiento, si la transformación las necesita.

        Returns:
            El conjunto transformado.
        """
        return self.fit(X, y).transform(X)

    def __repr__(self) -> str:
        """Representación legible basada en los parámetros de la transformación."""
        params = ", ".join(f"{key}={value!r}" for key, value in self.get_params().items())
        return f"{type(self).__name__}({params})"


class StatefulTransform(BaseTransform):
    """Contrato de las transformaciones **con estado**.

    Una transformación con estado aprende algo del conjunto de entrenamiento
    (medias, medianas, categorías, columnas seleccionadas) y lo reutiliza al
    transformar otros conjuntos. Es el caso del escalado, la imputación, la
    codificación y la selección de características para datos tabulares.

    La subclase implementa únicamente la lógica propia, en :meth:`_fit` y
    :meth:`_transform`. Los métodos públicos heredados de aquí se encargan de
    registrar que el ajuste ocurrió y de rechazar cualquier transformación
    previa al ajuste, de modo que una subclase no pueda olvidarse de ello.

    El ajuste debe realizarse **solo con el subconjunto de entrenamiento**:
    ajustar con el dataset completo filtraría información de la validación e
    invalidaría la evaluación (véase
    ``docs/decisiones/ADR-001-contrato-fit-transform.md``).

    Attributes:
        fitted (bool): ``True`` si la transformación ya fue ajustada.
    """

    # Atributo de clase para que exista aun si una subclase define su propio
    # ``__init__`` sin invocar al de la clase base.
    _fitted: bool = False

    @abstractmethod
    def _fit(self, X: Dataset, y: np.ndarray | None = None) -> None:
        """Aprende el estado necesario a partir del conjunto de entrenamiento.

        Args:
            X: Conjunto de muestras de entrenamiento.
            y: Etiquetas de entrenamiento, si la transformación las necesita.
        """
        raise NotImplementedError

    @abstractmethod
    def _transform(self, X: Dataset) -> Dataset:
        """Transforma un conjunto usando el estado ya aprendido.

        Args:
            X: Conjunto de muestras a transformar.

        Returns:
            El conjunto transformado.
        """
        raise NotImplementedError

    @property
    def fitted(self) -> bool:
        """Indica si la transformación ya fue ajustada."""
        return self._fitted

    def fit(self, X: Dataset, y: np.ndarray | None = None) -> StatefulTransform:
        """Ajusta la transformación con el subconjunto de entrenamiento.

        Args:
            X: Conjunto de muestras de entrenamiento.
            y: Etiquetas de entrenamiento, si la transformación las necesita.

        Returns:
            La propia transformación, para permitir el encadenamiento.
        """
        self._fitted = False
        self._fit(X, y)
        self._fitted = True
        return self

    def transform(self, X: Dataset) -> Dataset:
        """Transforma un conjunto completo con el estado aprendido.

        Args:
            X: Conjunto de muestras a transformar.

        Returns:
            El conjunto transformado.

        Raises:
            NotFittedError: Si la transformación aún no fue ajustada.
        """
        self._check_fitted()
        return self._transform(X)

    def apply(self, sample: np.ndarray) -> np.ndarray:
        """Transforma una sola muestra delegando en :meth:`transform`.

        Existe para satisfacer el contrato por muestra de
        :class:`BaseTransform`; la vía eficiente es :meth:`transform`.

        Args:
            sample: Muestra individual a transformar.

        Returns:
            La muestra transformada.

        Raises:
            NotFittedError: Si la transformación aún no fue ajustada.
        """
        self._check_fitted()
        batch = np.asarray(sample).reshape(1, -1)
        return np.asarray(self._transform(batch))[0]

    def _check_fitted(self) -> None:
        """Verifica que el ajuste ya haya ocurrido.

        Raises:
            NotFittedError: Si la transformación aún no fue ajustada.
        """
        if not self._fitted:
            raise NotFittedError(
                f"{type(self).__name__} no ha sido ajustada; invoque fit() con el "
                f"subconjunto de entrenamiento antes de transformar."
            )
