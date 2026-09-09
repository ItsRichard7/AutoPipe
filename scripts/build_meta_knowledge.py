# scripts/build_meta_knowledge.py
"""Construye la base de meta-conocimiento sobre el corpus seleccionado (RE4).

Por cada dataset del corpus: lo descarga, extrae su vector de
meta-características y busca su mejor pipeline con 50 ensayos de Optuna sobre el
espacio canónico, con la semilla 42 que fija el pre-registro. El resultado es una
entrada por dataset, y el conjunto es lo que el recomendador de arranque en
caliente consultará.

**Reanudable.** Cada entrada se cachea en disco por separado y las ya construidas
se saltan. Una interrupción a mitad no obliga a repetir lo hecho.

**Secuencial a propósito.** La hoja de ruta contemplaba paralelizar entre
datasets, pero la medición lo desaconseja: los 50 ensayos cuestan 16,5 s en el
dataset más grande del corpus y 1 s en el más pequeño, de modo que el corpus
entero son un par de minutos. Un pool de procesos añadiría serialización y una
depuración más difícil para ahorrar minutos.

Ejecutar con:
    .\\.venv\\Scripts\\python.exe scripts\\build_meta_knowledge.py
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
import warnings
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
import openml
import optuna
from download_corpus import descargar
from sklearn.neighbors import KNeighborsClassifier

from autopipe.meta_features import MetaFeatureExtractor
from autopipe.meta_knowledge import MetaKnowledgeBase, MetaKnowledgeEntry
from autopipe.search_engine import OptunaSearchEngine
from autopipe.search_space import (
    TABULAR_SPACE_VERSION,
    build_tabular_search_space,
    canonical_configuration,
    pipeline_from_configuration,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

CORPUS = Path("docs/experimentos/CORPUS-seleccionado.md")
BASE = Path("docs/experimentos/BASE-meta-conocimiento.json")
INFORME = Path("docs/experimentos/BASE-meta-conocimiento.md")
CACHE_ENTRADAS = Path("datasets/.meta_knowledge")
CACHE_META = Path("datasets/.meta_cache")

SEMILLA = 42
"""Semilla de la construcción de la base, fijada en el pre-registro."""

ENSAYOS = 50
VAL_SIZE = 0.2
METRICA = "macro_f1"

DESEMPATE = "canonica"
"""Regla con la que se elige el ganador entre configuraciones que empatan.

Viaja en el nombre del archivo de caché junto con la versión del espacio y la
semilla, por la misma razón que ellas: cambiar la regla cambia qué configuración
guarda cada entrada, y reutilizar en silencio una caché construida con otra regla
mezclaría dos criterios dentro de la misma base.
"""

FILA = re.compile(r"^\|\s*\d+\s*\|\s*(\d+)\s*\|\s*`([^`]+)`\s*\|")


def leer_corpus(ruta: Path) -> list[tuple[int, str]]:
    """Lee el corpus seleccionado.

    Args:
        ruta: Documento generado por ``screen_corpus.py --seleccionar``.

    Returns:
        Lista de pares ``(identificador de OpenML, nombre)``.

    Raises:
        SystemExit: Si el archivo no existe o no tiene filas legibles.
    """
    if not ruta.is_file():
        raise SystemExit(
            f"No existe {ruta}. Ejecute antes el cribado con "
            f"scripts/screen_corpus.py --seleccionar N."
        )
    filas = [
        (int(m.group(1)), m.group(2))
        for m in (FILA.match(linea) for linea in ruta.read_text("utf-8").splitlines())
        if m
    ]
    if not filas:
        raise SystemExit(f"No se pudo leer ningún dataset de {ruta}.")
    return filas


def _modelo() -> KNeighborsClassifier:
    """Devuelve el modelo base congelado del pre-registro."""
    return KNeighborsClassifier(
        n_neighbors=5, weights="uniform", metric="minkowski", p=2, algorithm="brute"
    )


def construir_entrada(did: int, nombre: str, extractor: MetaFeatureExtractor) -> MetaKnowledgeEntry:
    """Construye la entrada de un dataset.

    Args:
        did: Identificador de OpenML.
        nombre: Nombre del dataset.
        extractor: Extractor de meta-características, con su caché.

    Returns:
        La entrada, lista para incorporarse a la base.

    Raises:
        RuntimeError: Si la búsqueda no dejó configuración ganadora, lo que
            indicaría que ningún ensayo fue evaluable.
    """
    features, labels, _, _ = descargar(did)
    vector = extractor.extract(features, labels, dataset_id=str(did))

    inicio = time.perf_counter()
    motor = OptunaSearchEngine(
        build_tabular_search_space(),
        n_trials=ENSAYOS,
        val_size=VAL_SIZE,
        random_state=SEMILLA,
        task="classification",
    )
    _, metricas, resultados = motor.run(features, labels, _modelo(), metric=METRICA)
    duracion = time.perf_counter() - inicio

    if not motor.tied_configurations_:
        raise RuntimeError(
            f"La búsqueda sobre {nombre} no dejó configuración ganadora."
        )
    # Entre las que empatan en el mejor valor, se guarda la canónica y no la que
    # el muestreador visitó primero. Esa elección es una propiedad de la corrida,
    # y transferirla como arranque en caliente transferiría la arbitrariedad:
    # omitir la codificación no cuesta nada en un dataset sin categóricas y
    # vuelve el pipeline inevaluable en uno que sí las tenga.
    configuracion = canonical_configuration(motor.tied_configurations_)
    ganador = pipeline_from_configuration(configuracion)
    fallidos = sum(1 for r in resultados if np.isnan(r["objective"]))
    return MetaKnowledgeEntry(
        dataset_id=str(did),
        dataset_name=nombre,
        meta_features=vector,
        configuration=configuracion,
        pipeline_params=ganador.get_params(),
        metric=METRICA,
        metric_value=round(float(metricas[METRICA]), 6),
        seed=SEMILLA,
        n_trials=ENSAYOS,
        n_failed_trials=fallidos,
        space_version=TABULAR_SPACE_VERSION,
        seconds=round(duracion, 2),
    )


def _cache(did: int) -> Path:
    """Ruta de caché de la entrada de un dataset."""
    return (
        CACHE_ENTRADAS / f"{did}__{TABULAR_SPACE_VERSION}__s{SEMILLA}__{DESEMPATE}.json"
    )


def _leer_cache(did: int) -> MetaKnowledgeEntry | None:
    """Lee una entrada cacheada, si existe.

    Args:
        did: Identificador de OpenML.

    Returns:
        La entrada, o ``None`` si no está cacheada.
    """
    ruta = _cache(did)
    if not ruta.is_file():
        return None
    return MetaKnowledgeEntry(**json.loads(ruta.read_text(encoding="utf-8")))


def _guardar_cache(entrada: MetaKnowledgeEntry) -> None:
    """Cachea una entrada en disco.

    Args:
        entrada: Entrada a cachear.
    """
    ruta = _cache(int(entrada.dataset_id))
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(
        json.dumps(asdict(entrada), indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )


def _informe(base: MetaKnowledgeBase) -> str:
    """Redacta el resumen legible de la base.

    Args:
        base: Base ya construida.

    Returns:
        El documento markdown.
    """
    descartadas = sorted(set(base.feature_names()) - set(base.surviving_features()))
    configuraciones = {
        tuple(sorted(e.configuration.items())): 0 for e in base
    }
    for entrada in base:
        configuraciones[tuple(sorted(entrada.configuration.items()))] += 1
    filas = "\n".join(
        f"| {e.dataset_id} | `{e.dataset_name}` | {e.metric_value:.4f} | "
        f"{e.n_failed_trials} | {e.seconds:.1f} | "
        f"`{' '.join(f'{k}={v}' for k, v in sorted(e.configuration.items()))}` |"
        for e in base
    )
    fallidos = sum(e.n_failed_trials for e in base)
    ensayos = sum(e.n_trials for e in base)
    return f"""# Base de meta-conocimiento (RE4)

> **Archivo generado.** No editar a mano: se regenera con
> `.\\.venv\\Scripts\\python.exe scripts\\build_meta_knowledge.py`.

- **Generado:** {date.today().isoformat()}
- **Datasets:** {len(base)}
- **Ensayos por dataset:** {ENSAYOS}, semilla {SEMILLA}
- **Versión del espacio:** `{TABULAR_SPACE_VERSION}`
- **Métrica:** {METRICA}
- **Datos completos:** `BASE-meta-conocimiento.json`

## Qué es

Lo que el recomendador de arranque en caliente consulta. Dado un dataset nuevo,
busca aquí los más parecidos comparando vectores de meta-características e
inyecta sus configuraciones ganadoras como ensayos iniciales.

## Meta-características

- Extraídas por dataset: **{len(base.feature_names())}**
- Sobreviven a la regla de descarte: **{len(base.surviving_features())}**
- Descartadas por resultar no finitas en algún dataset: **{len(descartadas)}**

La regla del pre-registro descarta toda meta-característica no finita en algún
dataset del corpus, sin imputar al nivel meta. Las descartadas son:

{", ".join(f"`{n}`" for n in descartadas)}

El vector completo se conserva en cada entrada: la regla depende del corpus, así
que guardar solo su resultado la volvería irreversible.

## Diversidad de los pipelines ganadores

**{len(configuraciones)} configuraciones distintas entre {len(base)} datasets.**

Es la condición para que el meta-aprendizaje tenga sentido. Si todos los datasets
compartieran el mismo ganador, el recomendador no tendría nada que aprender:
bastaría recomendar siempre esa configuración, y la comparación entre arranque
frío y caliente mediría otra cosa.

### Desempate entre configuraciones que empatan

El óptimo casi nunca es único. Sobre un dataset numérico y sin faltantes, la
imputación y la codificación son la identidad, de modo que varias combinaciones
de esos dos pasos producen **el mismo pipeline** y empatan exactamente; la
medición está en `DIVERSIDAD-ganadores.md`.

Entre las empatadas, esta base guarda **la que omite menos pasos y, a igualdad,
la de índices menores**, y no la que el muestreador visitó primero. Sin esa
regla, la configuración guardada sería una propiedad de la corrida y no del
dataset, y transferirla como arranque en caliente transferiría esa
arbitrariedad: un ganador que omite la codificación es **inevaluable** en un
dataset receptor con columnas categóricas, y ese ensayo inyectado se perdería sin
producir una métrica.

El desempate se calcula entre los empates observados dentro de los {ENSAYOS}
ensayos del presupuesto. Mirar el espacio completo equivaldría a construir la
base por búsqueda exhaustiva, que no es lo pre-registrado.

## Ensayos inevaluables

{fallidos} de {ensayos} ensayos ({100 * fallidos / ensayos:.1f} %) resultaron
inevaluables. Es muy inferior a la fracción de la región inevaluable del espacio,
que llega al 56 % en datasets con categóricas y faltantes: el muestreador aprende
a esquivarla después de los ensayos de arranque.

## Entradas

La configuración es el índice elegido por paso; el índice mayor de cada paso
significa omitirlo, y solo es interpretable con la versión del espacio indicada
arriba.

| OpenML ID | Nombre | {METRICA} | Fallidos | Segundos | Configuración ganadora |
|---:|---|---:|---:|---:|---|
{filas}
"""


def main() -> int:
    """Punto de entrada.

    Returns:
        ``0`` si la base se construyó completa.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Procesar solo los primeros N.")
    parser.add_argument(
        "--rehacer", action="store_true", help="Ignorar la caché y recalcular todo."
    )
    # El experimento 2 usa un corpus de 42 conjuntos y necesita su propia base. Las
    # rutas entran por parámetro en lugar de duplicar este guion: la construcción es
    # idéntica —misma semilla, mismos 50 ensayos, mismo desempate— y lo único que
    # cambia es sobre qué lista se aplica. La caché va por dataset, así que los 25 del
    # experimento 1 se reutilizan sin recalcularse.
    parser.add_argument("--corpus", type=Path, default=CORPUS, help="Lista de entrada.")
    parser.add_argument("--base", type=Path, default=BASE, help="Base de salida.")
    parser.add_argument("--informe", type=Path, default=INFORME, help="Informe de salida.")
    args = parser.parse_args()

    openml.config.set_console_log_level(logging.WARNING)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logging.getLogger("autopipe").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    corpus = leer_corpus(args.corpus)[: args.limit]
    extractor = MetaFeatureExtractor(random_state=SEMILLA, cache_dir=CACHE_META)
    entradas: list[MetaKnowledgeEntry] = []
    inicio = time.perf_counter()

    for numero, (did, nombre) in enumerate(corpus, 1):
        cacheada = None if args.rehacer else _leer_cache(did)
        if cacheada is not None:
            entradas.append(cacheada)
            logger.info("[%2d/%d] %-42s (en caché)", numero, len(corpus), nombre)
            continue
        entrada = construir_entrada(did, nombre, extractor)
        _guardar_cache(entrada)
        entradas.append(entrada)
        logger.info(
            "[%2d/%d] %-42s %s %.4f  fallidos %2d  %5.1f s",
            numero,
            len(corpus),
            nombre,
            METRICA,
            entrada.metric_value,
            entrada.n_failed_trials,
            entrada.seconds,
        )

    base = MetaKnowledgeBase(entradas)
    base.save(args.base)
    args.informe.write_text(_informe(base), encoding="utf-8")
    total = len(base.feature_names())
    sobreviven = len(base.surviving_features())
    logger.info(
        "\n%d entradas en %.1f min. Meta-características: %d, sobreviven %d "
        "(descartadas %d por no finitas en algún dataset). Base en %s",
        len(base),
        (time.perf_counter() - inicio) / 60,
        total,
        sobreviven,
        total - sobreviven,
        args.base,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
