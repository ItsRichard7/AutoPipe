# autopipe/meta_features/extractor.py
"""Extracción de meta-características de un dataset tabular, sobre ``pymfe``.

El vector que produce este módulo es la entrada del recomendador de arranque en
caliente: describe un dataset con números comparables con los de otro dataset, de
modo que se pueda buscar el «más parecido» en la base de meta-conocimiento.

Todo el diseño sale de mediciones del spike B
(``docs/experimentos/SPIKE-B-pymfe.md``), no de suposiciones:

* **Los grupos se tratan distinto según su costo.** ``general``, ``statistical``
  e ``info-theory`` se extraen del dataset completo: cuestan menos de tres
  segundos incluso en el techo de 50 000 instancias, y son estimadores de
  cantidades poblacionales que mejoran con más datos. ``complexity`` se extrae de
  una submuestra fija, porque su costo crece de forma cuadrática (exponente
  medido 1,91) y en 50 000 instancias tardaría unos 21,6 minutos por dataset.
* **La submuestra es de tamaño fijo, no de fracción fija.** Las medidas de
  complejidad son propiedades geométricas de la muestra concreta y cambian con su
  tamaño, así que dos datasets solo son comparables si se extrajeron con el mismo
  tamaño. Los criterios de inclusión admiten datasets desde 500 instancias, de
  modo que 500 es el único tamaño que todo dataset elegible puede alcanzar.
* **Se imputa antes de extraer.** Con valores faltantes, ``complexity`` falla con
  un ``Exception`` de mensaje vacío desde ``precompute_adjacency_graph``. Imputar
  por la mediana lo resuelve por completo.
* **Los valores se redondean.** Dos extracciones sobre el mismo dataset difieren
  en ``hubs.mean`` y ``hubs.sd`` en torno a 3e-15, por orden de sumación en coma
  flotante. Redondear vuelve el vector estable bit a bit, que es lo que el
  criterio de reproducibilidad del anteproyecto necesita.
"""

from __future__ import annotations

import hashlib
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pymfe.mfe import MFE
from sklearn.model_selection import StratifiedShuffleSplit

from autopipe.pipeline.tabular_transforms import ImputeMissing

logger = logging.getLogger(__name__)


class MetaFeatureError(RuntimeError):
    """No se pudieron extraer las meta-características de un dataset.

    Existe para dar contexto a los fallos de ``pymfe``, que en el caso más
    frecuente lanza un ``Exception`` con **mensaje vacío**: sin envolverlo, un
    fallo sobre un corpus de decenas de datasets es indepurable.
    """


class MetaFeatureExtractor:
    """Extrae el vector de meta-características de un dataset tabular.

    Attributes:
        random_state (int): Semilla de la submuestra y de ``pymfe``.
        sample_size (int): Tamaño fijo de la submuestra del grupo de complejidad.
        decimals (int): Decimales a los que se redondea cada valor.
        cache_dir (Path | None): Carpeta de caché en disco, o ``None``.
    """

    GRUPOS_COMPLETOS = ("general", "statistical", "info-theory")
    """Grupos que se extraen del dataset completo."""

    GRUPO_SUBMUESTREADO = "complexity"
    """Grupo que se extrae de la submuestra de tamaño fijo."""

    TAMANO_MUESTRA = 500
    DECIMALES = 9

    def __init__(
        self,
        random_state: int = 42,
        sample_size: int = TAMANO_MUESTRA,
        decimals: int = DECIMALES,
        cache_dir: str | Path | None = None,
    ) -> None:
        """Inicializa el extractor.

        Args:
            random_state: Semilla de la submuestra estratificada y de ``pymfe``.
            sample_size: Tamaño fijo de la submuestra del grupo de complejidad.
                Debe ser el mismo para todos los datasets que se vayan a
                comparar entre sí.
            decimals: Decimales a los que se redondea cada meta-característica.
            cache_dir: Carpeta donde cachear los vectores ya extraídos. Si es
                ``None`` no se cachea.

        Raises:
            ValueError: Si ``sample_size`` o ``decimals`` no son positivos.
        """
        if sample_size <= 0:
            raise ValueError(
                f"sample_size debe ser un entero positivo, se recibió {sample_size!r}."
            )
        if decimals <= 0:
            raise ValueError(
                f"decimals debe ser un entero positivo, se recibió {decimals!r}."
            )
        self.random_state = random_state
        self.sample_size = sample_size
        self.decimals = decimals
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

    def extract(
        self,
        features: pd.DataFrame | np.ndarray,
        labels: np.ndarray,
        dataset_id: str | None = None,
    ) -> dict[str, float]:
        """Extrae el vector de meta-características de un dataset.

        Args:
            features: Tabla de características, con o sin valores faltantes.
            labels: Etiquetas de clase.
            dataset_id: Identificador del dataset. Solo se usa para nombrar la
                entrada de caché; si es ``None`` no se cachea aunque haya
                ``cache_dir``.

        Returns:
            Diccionario de nombre de meta-característica a valor. Las claves son
            las mismas para cualquier dataset, que es la condición para que el
            recomendador compare vectores alineados. Los valores no finitos se
            conservan tal cual: descartarlos es una decisión de corpus y no de
            dataset (véase :func:`columnas_no_finitas`).

        Raises:
            MetaFeatureError: Si ``pymfe`` no pudo extraer algún grupo.
        """
        cache = self._ruta_cache(dataset_id)
        if cache is not None and cache.is_file():
            logger.info("Meta-características leídas de la caché: %s", cache)
            return json.loads(cache.read_text(encoding="utf-8"))

        frame = self._como_tabla(features)
        etiquetas = np.asarray(labels)
        imputada = ImputeMissing("median").fit_transform(frame)

        vector = self._extraer(imputada, etiquetas, self.GRUPOS_COMPLETOS)
        muestra_X, muestra_y = self._submuestra(imputada, etiquetas)
        complejidad = self._extraer(muestra_X, muestra_y, (self.GRUPO_SUBMUESTREADO,))

        repetidas = set(vector) & set(complejidad)
        if repetidas:
            raise MetaFeatureError(
                f"Los grupos completos y el submuestreado devolvieron las mismas "
                f"meta-características {sorted(repetidas)}, calculadas sobre "
                f"conjuntos de datos distintos. Habría que decidir cuál conservar."
            )
        vector.update(complejidad)

        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(vector, indent=2, sort_keys=True), "utf-8")
        return vector

    @staticmethod
    def _como_tabla(features: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Devuelve la entrada como tabla, aceptando también una matriz.

        Args:
            features: Tabla o matriz de características.

        Returns:
            La tabla de características.
        """
        if isinstance(features, pd.DataFrame):
            return features
        matriz = np.asarray(features)
        return pd.DataFrame(matriz, columns=[f"c{i}" for i in range(matriz.shape[1])])

    def _submuestra(
        self, frame: pd.DataFrame, labels: np.ndarray
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Toma una submuestra estratificada de tamaño fijo.

        Si el dataset no supera :attr:`sample_size` se devuelve entero. Si la
        estratificación no es posible —alguna clase tiene menos miembros que los
        que exige el reparto— se cae a un muestreo aleatorio simple, dejando
        constancia en el registro: es preferible una muestra peor a no poder
        describir el dataset.

        Args:
            frame: Tabla ya imputada.
            labels: Etiquetas de clase.

        Returns:
            Tupla ``(submuestra, etiquetas)``.
        """
        if len(frame) <= self.sample_size:
            return frame, labels

        particion = StratifiedShuffleSplit(
            n_splits=1, train_size=self.sample_size, random_state=self.random_state
        )
        try:
            indices, _ = next(particion.split(frame, labels))
        except ValueError as exc:
            logger.warning(
                "No se pudo estratificar la submuestra (%s); se usa muestreo simple.",
                exc,
            )
            generador = np.random.default_rng(self.random_state)
            indices = generador.choice(len(frame), size=self.sample_size, replace=False)
        return frame.iloc[indices], labels[indices]

    def _extraer(
        self, frame: pd.DataFrame, labels: np.ndarray, grupos: tuple[str, ...]
    ) -> dict[str, float]:
        """Invoca a ``pymfe`` sobre un conjunto de grupos.

        Args:
            frame: Tabla ya imputada.
            labels: Etiquetas de clase.
            grupos: Grupos de ``pymfe`` a extraer.

        Returns:
            Diccionario de nombre a valor, ya redondeado.

        Raises:
            MetaFeatureError: Si ``pymfe`` falla. Se captura ``Exception`` a
                propósito y no una clase concreta: ``pymfe`` lanza la clase base
                con mensaje vacío cuando el grafo de adyacencia del grupo de
                complejidad no se puede precomputar.
        """
        matriz = frame.to_numpy()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                extractor = MFE(groups=list(grupos), random_state=self.random_state)
                extractor.fit(matriz, labels)
                nombres, valores = extractor.extract()
        except Exception as exc:  # noqa: BLE001 - pymfe lanza Exception sin mensaje
            # Se evalúa ``str(exc)`` y no ``exc``: un objeto excepción siempre es
            # verdadero, de modo que ``exc or ...`` nunca caería al texto de
            # reserva y el mensaje volvería a quedar vacío, que es justo lo que
            # este envoltorio existe para evitar.
            detalle = str(exc) or "sin mensaje"
            raise MetaFeatureError(
                f"pymfe no pudo extraer los grupos {list(grupos)} de un dataset de "
                f"{len(frame)} instancias y {frame.shape[1]} columnas. "
                f"Causa: {type(exc).__name__}: {detalle}."
            ) from exc
        return {
            str(nombre): self._redondear(valor)
            for nombre, valor in zip(nombres, valores)
        }

    def _redondear(self, valor: object) -> float:
        """Redondea un valor conservando los no finitos.

        Args:
            valor: Valor devuelto por ``pymfe``.

        Returns:
            El valor redondeado a :attr:`decimals` decimales, o el propio valor
            no finito si lo era.
        """
        numero = float(valor)  # type: ignore[arg-type]
        return numero if not np.isfinite(numero) else round(numero, self.decimals)

    def _ruta_cache(self, dataset_id: str | None) -> Path | None:
        """Calcula la ruta de caché de un dataset.

        La firma incluye la configuración que cambia el vector, de modo que
        cambiar la semilla o el tamaño de muestra no reutilice un vector viejo en
        silencio.

        Args:
            dataset_id: Identificador del dataset, o ``None``.

        Returns:
            La ruta del archivo de caché, o ``None`` si no corresponde cachear.
        """
        if self.cache_dir is None or dataset_id is None:
            return None
        firma = "|".join(
            [
                str(self.random_state),
                str(self.sample_size),
                str(self.decimals),
                ",".join(self.GRUPOS_COMPLETOS),
                self.GRUPO_SUBMUESTREADO,
            ]
        )
        resumen = hashlib.sha256(firma.encode("utf-8")).hexdigest()[:12]
        seguro = "".join(c if c.isalnum() or c in "-_" else "_" for c in dataset_id)
        return self.cache_dir / f"{seguro}__{resumen}.json"


def columnas_no_finitas(vector: dict[str, float]) -> set[str]:
    """Devuelve los nombres cuyo valor no es finito.

    Alrededor del 13 % del vector no es finito, y **qué posiciones lo son depende
    del dataset**: hay meta-características que existen para unos datasets y no
    para otros. Decidir qué columnas se descartan es una propiedad del corpus
    completo, no de un dataset suelto, así que el extractor las conserva y esta
    función permite acumularlas.

    Args:
        vector: Vector de meta-características.

    Returns:
        Conjunto de nombres con valor no finito.
    """
    return {nombre for nombre, valor in vector.items() if not np.isfinite(valor)}
