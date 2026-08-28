# autopipe/meta_knowledge/base.py
"""Base de meta-conocimiento: qué pipeline ganó en cada dataset del corpus.

Es lo que el recomendador de arranque en caliente consulta: dado un dataset
nuevo, busca en esta base los datasets más parecidos —comparando sus vectores de
meta-características— e inyecta sus configuraciones ganadoras como ensayos
iniciales.

Cada entrada guarda **las 107 meta-características completas**, incluidas las no
finitas, y no solo las que sobreviven a la regla de descarte. La regla depende del
corpus entero, de modo que guardar únicamente su resultado la volvería
irreversible: no se podría recalcular al cambiar el corpus, ni auditar por qué se
descartó una columna. :meth:`MetaKnowledgeBase.surviving_features` la aplica sobre
lo guardado.

La configuración ganadora se guarda de dos formas: como **índices por paso**, que
es lo que ``optuna.enqueue_trial`` necesita, y como los parámetros legibles del
pipeline. Los índices solo significan algo respecto de una versión concreta del
espacio de búsqueda, así que la versión viaja con cada entrada y se verifica al
cargar.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from autopipe.search_space import TABULAR_SPACE_VERSION, validate_space_version


@dataclass(frozen=True)
class MetaKnowledgeEntry:
    """Lo que se sabe de un dataset del corpus.

    Attributes:
        dataset_id (str): Identificador estable del dataset.
        dataset_name (str): Nombre legible.
        meta_features (dict): Vector completo, incluidas las no finitas.
        configuration (dict): Índice elegido por cada paso del espacio. El índice
            ``len(opciones)`` representa omitir el paso.
        pipeline_params (dict): Parámetros del pipeline ganador, para leerlos sin
            resolver los índices.
        metric (str): Métrica optimizada.
        metric_value (float): Valor alcanzado por el ganador.
        seed (int): Semilla de la búsqueda.
        n_trials (int): Ensayos ejecutados.
        n_failed_trials (int): Ensayos que resultaron inevaluables.
        space_version (str): Versión del espacio de búsqueda.
        seconds (float): Duración de la búsqueda, para medir el costo real.
    """

    dataset_id: str
    dataset_name: str
    meta_features: dict[str, float]
    configuration: dict[str, int]
    pipeline_params: dict
    metric: str
    metric_value: float
    seed: int
    n_trials: int
    n_failed_trials: int
    space_version: str = TABULAR_SPACE_VERSION
    seconds: float = 0.0

    def feature_names(self) -> list[str]:
        """Nombres de las meta-características, en orden estable.

        Returns:
            Los nombres ordenados alfabéticamente.
        """
        return sorted(self.meta_features)


@dataclass
class MetaKnowledgeBase:
    """Colección de entradas, una por dataset del corpus.

    Al construirse verifica dos invariantes de los que depende el recomendador:
    que todas las entradas describan sus datasets con **el mismo conjunto de
    meta-características**, y que todas provengan de **la misma versión del
    espacio de búsqueda**. Sin la primera, comparar dos vectores sería comparar
    cosas distintas; sin la segunda, una configuración inyectada designaría
    transformaciones que no son las que se creyó recomendar.

    Attributes:
        entries (list[MetaKnowledgeEntry]): Entradas de la base.
    """

    entries: list[MetaKnowledgeEntry] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Verifica los invariantes de la base.

        Raises:
            ValueError: Si las entradas no comparten el conjunto de nombres de
                meta-características, si mezclan versiones del espacio, o si hay
                identificadores repetidos.
        """
        if not self.entries:
            return
        nombres = self.entries[0].feature_names()
        discrepantes = [e.dataset_name for e in self.entries if e.feature_names() != nombres]
        if discrepantes:
            raise ValueError(
                f"Estas entradas no comparten el conjunto de meta-características del "
                f"resto: {discrepantes}. El recomendador compararía vectores "
                f"desalineados."
            )
        versiones = {e.space_version for e in self.entries}
        if len(versiones) > 1:
            raise ValueError(
                f"La base mezcla versiones del espacio de búsqueda {sorted(versiones)}. "
                f"Los índices de una configuración solo significan algo respecto de una "
                f"versión concreta."
            )
        identificadores = [e.dataset_id for e in self.entries]
        repetidos = {i for i in identificadores if identificadores.count(i) > 1}
        if repetidos:
            raise ValueError(f"Hay identificadores de dataset repetidos: {sorted(repetidos)}.")

    def __len__(self) -> int:
        """Número de entradas."""
        return len(self.entries)

    def __iter__(self):
        """Itera sobre las entradas."""
        return iter(self.entries)

    @property
    def dataset_ids(self) -> list[str]:
        """Identificadores de los datasets, en orden de inserción."""
        return [entrada.dataset_id for entrada in self.entries]

    def feature_names(self) -> list[str]:
        """Nombres de las meta-características, comunes a todas las entradas.

        Returns:
            Los nombres ordenados, o una lista vacía si la base está vacía.
        """
        return self.entries[0].feature_names() if self.entries else []

    def surviving_features(self) -> list[str]:
        """Meta-características que son finitas en **todas** las entradas.

        Es la regla del pre-registro: se descarta toda meta-característica no
        finita en algún dataset del corpus, en lugar de imputar al nivel meta.

        Returns:
            Los nombres supervivientes, en orden estable.
        """
        return [
            nombre
            for nombre in self.feature_names()
            if all(np.isfinite(e.meta_features[nombre]) for e in self.entries)
        ]

    def feature_matrix(self, names: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
        """Arma la matriz de meta-características de la base.

        Args:
            names: Columnas a incluir. Si es ``None`` se usan las supervivientes.

        Returns:
            Tupla ``(matriz, nombres)`` con una fila por entrada, en el orden de
            :attr:`entries`.
        """
        columnas = self.surviving_features() if names is None else list(names)
        matriz = np.array(
            [[e.meta_features[nombre] for nombre in columnas] for e in self.entries],
            dtype=float,
        )
        return matriz, columnas

    def entry_for(self, dataset_id: str) -> MetaKnowledgeEntry:
        """Devuelve la entrada de un dataset.

        Args:
            dataset_id: Identificador buscado.

        Returns:
            La entrada correspondiente.

        Raises:
            KeyError: Si el dataset no está en la base.
        """
        for entrada in self.entries:
            if entrada.dataset_id == dataset_id:
                return entrada
        raise KeyError(f"El dataset {dataset_id!r} no está en la base de meta-conocimiento.")

    def without(self, dataset_id: str) -> MetaKnowledgeBase:
        """Devuelve la base sin un dataset, para un pliegue del *leave-one-out*.

        Args:
            dataset_id: Identificador del dataset a excluir.

        Returns:
            Una base nueva sin esa entrada.

        Raises:
            KeyError: Si el dataset no está en la base.
        """
        self.entry_for(dataset_id)
        return MetaKnowledgeBase(
            [entrada for entrada in self.entries if entrada.dataset_id != dataset_id]
        )

    def save(self, path: str | Path) -> Path:
        """Escribe la base en disco como JSON.

        Args:
            path: Ruta del archivo.

        Returns:
            La ruta escrita.
        """
        destino = Path(path)
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(
            json.dumps(
                {
                    "space_version": self.entries[0].space_version
                    if self.entries
                    else TABULAR_SPACE_VERSION,
                    "n_entries": len(self.entries),
                    "entries": [asdict(entrada) for entrada in self.entries],
                },
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return destino

    @classmethod
    def load(cls, path: str | Path, check_version: bool = True) -> MetaKnowledgeBase:
        """Lee una base de disco.

        Args:
            path: Ruta del archivo JSON.
            check_version: Si se verifica que la versión del espacio sea la
                vigente. Desactivarlo solo tiene sentido para inspeccionar una
                base antigua, nunca para inyectar sus configuraciones.

        Returns:
            La base cargada.

        Raises:
            ValueError: Si la versión del espacio no es la vigente, o si las
                entradas violan algún invariante.
        """
        datos = json.loads(Path(path).read_text(encoding="utf-8"))
        entradas = [MetaKnowledgeEntry(**registro) for registro in datos["entries"]]
        if check_version and entradas:
            validate_space_version(entradas[0].space_version)
        return cls(entradas)
