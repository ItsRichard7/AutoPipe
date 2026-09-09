# scripts/reference_optima.py
"""Óptimos de referencia del experimento 2 (pre-registro v2.0, §4).

Por cada pareja (conjunto de datos, semilla) evalúa **las 144 configuraciones** del
espacio canónico sobre la misma partición que usarán las búsquedas de esa pareja, y
guarda el máximo. Ese máximo, $M(d,s)$, es la vara del experimento 2: el objetivo es
su 99\\ %, y el *regret* normalizado se mide contra él.

**Por qué exhaustivo y no estimado.** Es factible porque el espacio es pequeño, y es
lo que convierte una vara autorreferencial ---el 95\\ % de lo que el propio arranque
en frío logró--- en una externa e igual para todas las condiciones. En un espacio
con hiperparámetros continuos habría que conformarse con un proxy, y esa es una de
las razones por las que el ADR-004 descartó construirlo.

**La misma partición, o el número no significa nada.** La rejilla se evalúa con el
mismo ``val_size`` y el mismo ``random_state`` que la búsqueda, de modo que
$M(d,s)$ acota por arriba cualquier valor que una búsqueda pueda alcanzar en esa
pareja. Es un invariante comprobable, y ``tests/test_reference_optima.py`` lo
comprueba: si alguna curva superara su referencia, indicaría que las particiones no
coinciden.

De paso calcula la **anchura de la meseta al 99\\ %**, que es una de las dos
covariables declaradas. La del experimento 1 estaba medida al 95\\ % y no
corresponde al objetivo vigente.

**Reanudable.** Cada pareja se cachea por separado.

Ejecutar con:
    .\\.venv\\Scripts\\python.exe scripts\\reference_optima.py
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import time
import warnings
from pathlib import Path

import numpy as np
import openml
from build_meta_knowledge import leer_corpus
from download_corpus import descargar
from sklearn.neighbors import KNeighborsClassifier

from autopipe.search_engine import GridSearchEngine
from autopipe.search_space import TABULAR_SPACE_VERSION, build_tabular_search_space

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

CORPUS = Path("docs/experimentos/CORPUS-E2.md")
SALIDA = Path("docs/experimentos/REFERENCIA-E2.json")
CACHE = Path("datasets/.reference_e2")

SEMILLAS = (101, 202, 303, 404, 505, 606, 707, 808, 909, 1010)
"""Las diez del pre-registro v2.0, idénticas en todas las condiciones."""

FRACCION_OBJETIVO = 0.99
"""El objetivo es esta fracción del óptimo de referencia."""

VAL_SIZE = 0.2
METRICA = "macro_f1"


def _modelo() -> KNeighborsClassifier:
    """Devuelve el modelo base congelado, heredado del pre-registro v1.0."""
    return KNeighborsClassifier(
        n_neighbors=5, weights="uniform", metric="minkowski", p=2, algorithm="brute"
    )


def _cache(dataset_id: str, semilla: int) -> Path:
    """Ruta de caché de la rejilla de una pareja.

    La semilla viaja en el nombre porque cambia la partición y con ella todos los
    valores: reutilizar en silencio la rejilla de otra semilla daría una referencia
    que no acota las curvas de esta.
    """
    return CACHE / f"{dataset_id}__{TABULAR_SPACE_VERSION}__s{semilla}.json"


def rejilla(
    dataset_id: str, semilla: int, features: object, labels: np.ndarray
) -> list[float]:
    """Evalúa las 144 configuraciones sobre la partición de una pareja.

    Args:
        dataset_id: Identificador del conjunto de datos.
        semilla: Semilla de la partición.
        features: Características del conjunto.
        labels: Etiquetas.

    Returns:
        Valor de macro-F1 de cada configuración. Las inevaluables vienen como
        ``NaN``.
    """
    ruta = _cache(dataset_id, semilla)
    if ruta.is_file():
        return json.loads(ruta.read_text(encoding="utf-8"))["objetivos"]

    motor = GridSearchEngine(
        build_tabular_search_space(),
        val_size=VAL_SIZE,
        random_state=semilla,
        task="classification",
    )
    _, _, resultados = motor.run(features, labels, _modelo(), metric=METRICA)
    objetivos = [float(r["objective"]) for r in resultados]

    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(
        json.dumps({"dataset_id": dataset_id, "semilla": semilla, "objetivos": objetivos}),
        encoding="utf-8",
    )
    return objetivos


def resumir(objetivos: list[float]) -> dict:
    """Extrae de una rejilla el óptimo, la meseta y la región inevaluable.

    Args:
        objetivos: Valor de cada una de las 144 configuraciones.

    Returns:
        Diccionario con ``optimo``, ``n_en_meseta``, ``n_evaluables`` y
        ``fraccion_inevaluable``.

    Raises:
        ValueError: Si ninguna configuración resultó evaluable, en cuyo caso no hay
            óptimo contra el que medir y la pareja no se puede usar.
    """
    finitos = [v for v in objetivos if math.isfinite(v)]
    if not finitos:
        raise ValueError(
            "Ninguna de las 144 configuraciones resultó evaluable, así que no hay "
            "óptimo de referencia contra el que medir el regret."
        )
    # El óptimo y el objetivo van **sin redondear**. Redondearlos a seis decimales
    # los deja por debajo del valor alcanzable —0,8333333 pasaría a 0,833333— y
    # entonces una búsqueda que encuentra el óptimo lo supera: el regret saldría
    # negativo y se rompería el invariante que el pre-registro declara. Lo detectó
    # `test_no_search_can_beat_its_own_reference`.
    optimo = max(finitos)
    umbral = FRACCION_OBJETIVO * optimo
    return {
        "optimo": optimo,
        "objetivo": umbral,
        "n_en_meseta": sum(1 for v in finitos if v >= umbral),
        "n_evaluables": len(finitos),
        "fraccion_inevaluable": round(1 - len(finitos) / len(objetivos), 4),
    }


def calcular(corpus: list[tuple[int, str]], semillas: tuple[int, ...] = SEMILLAS) -> dict:
    """Calcula la referencia de todas las parejas del corpus.

    Args:
        corpus: Pares ``(identificador, nombre)``.
        semillas: Semillas del protocolo.

    Returns:
        Documento con una entrada por conjunto y, dentro, una por semilla.
    """
    inicio = time.perf_counter()
    datasets = []
    for numero, (did, nombre) in enumerate(corpus, 1):
        features, labels = None, None
        por_semilla = {}
        for semilla in semillas:
            if not _cache(str(did), semilla).is_file() and features is None:
                features, labels, _, _ = descargar(did)
            por_semilla[str(semilla)] = resumir(rejilla(str(did), semilla, features, labels))
        mesetas = [v["n_en_meseta"] for v in por_semilla.values()]
        datasets.append(
            {
                "dataset_id": str(did),
                "dataset_name": nombre,
                "semillas": por_semilla,
                "meseta_mediana": statistics.median(mesetas),
            }
        )
        logger.info(
            "[%2d/%d] %-42s optimo %.4f  meseta %3.0f/144",
            numero,
            len(corpus),
            nombre[:42],
            statistics.median(v["optimo"] for v in por_semilla.values()),
            statistics.median(mesetas),
        )
    return {
        "space_version": TABULAR_SPACE_VERSION,
        "fraccion_objetivo": FRACCION_OBJETIVO,
        "semillas": list(semillas),
        "n_datasets": len(datasets),
        "datasets": datasets,
        "minutos": round((time.perf_counter() - inicio) / 60, 2),
    }


def main() -> int:
    """Punto de entrada.

    Returns:
        ``0`` si la referencia se calculó.

    Raises:
        SystemExit: Si no existe el corpus del experimento 2.
    """
    parser = argparse.ArgumentParser(description="Óptimos de referencia del experimento 2.")
    parser.add_argument("--limit", type=int, help="Procesar solo los primeros N.")
    args = parser.parse_args()

    openml.config.set_console_log_level(logging.WARNING)
    logging.getLogger("autopipe").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    if not CORPUS.is_file():
        raise SystemExit(f"No existe {CORPUS}. Ejecute antes scripts/build_corpus_e2.py.")

    resultado = calcular(leer_corpus(CORPUS)[: args.limit])
    SALIDA.write_text(
        json.dumps(resultado, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    mesetas = [d["meseta_mediana"] for d in resultado["datasets"]]
    logger.info(
        "\n%d conjuntos x %d semillas en %.1f min. Meseta al %.0f %%: mediana %.0f de "
        "144, mínima %.0f, máxima %.0f. Referencia en %s",
        resultado["n_datasets"],
        len(resultado["semillas"]),
        resultado["minutos"],
        100 * FRACCION_OBJETIVO,
        statistics.median(mesetas),
        min(mesetas),
        max(mesetas),
        SALIDA,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
