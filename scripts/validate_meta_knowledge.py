# scripts/validate_meta_knowledge.py
"""Valida la base de meta-conocimiento ya construida (RE4).

La base es la entrada del recomendador, de modo que un defecto suyo no produce un
error visible: produce recomendaciones equivocadas. Este script comprueba cuatro
cosas sobre el artefacto guardado, no sobre el código que lo produjo.

1. **Cobertura.** Hay exactamente una entrada por dataset del corpus, sin faltar
   ninguno ni sobrar entradas de datasets que ya no están.
2. **Campos.** Cada entrada trae todo lo que el esquema promete, con valores en
   rangos verosímiles y con la semilla y el presupuesto que fija el pre-registro.
3. **Interpretabilidad de la configuración.** Cada índice cae dentro del rango de
   su paso y, al resolverlo contra el espacio vigente, **reproduce exactamente**
   los parámetros del pipeline guardado. Es la comprobación que detectaría que la
   base y el espacio se separaron.
4. **Reproducibilidad.** Reconstruye desde cero una muestra de datasets, sin
   caché, y verifica que salga la misma configuración ganadora y la misma métrica.

Ejecutar con:
    .\\.venv\\Scripts\\python.exe scripts\\validate_meta_knowledge.py
    .\\.venv\\Scripts\\python.exe scripts\\validate_meta_knowledge.py --reproducir 5
"""

from __future__ import annotations

import argparse
import logging
import math
import warnings
from pathlib import Path

import numpy as np
import openml
import optuna
from build_meta_knowledge import BASE, CORPUS, ENSAYOS, SEMILLA, construir_entrada, leer_corpus

from autopipe.meta_features import MetaFeatureExtractor
from autopipe.meta_knowledge import MetaKnowledgeBase, MetaKnowledgeEntry
from autopipe.search_space import (
    TABULAR_SPACE_STEPS,
    TABULAR_SPACE_VERSION,
    build_tabular_search_space,
    pipeline_from_configuration,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

CACHE_META = Path("datasets/.meta_cache")
MUESTRA_REPRODUCIBILIDAD = 3


def verificar_cobertura(base: MetaKnowledgeBase, corpus: list[tuple[int, str]]) -> list[str]:
    """Comprueba que la base cubra exactamente el corpus.

    Args:
        base: Base cargada.
        corpus: Datasets del corpus seleccionado.

    Returns:
        Lista de problemas; vacía si la cobertura es exacta.
    """
    esperados = {str(did) for did, _ in corpus}
    presentes = set(base.dataset_ids)
    problemas = []
    if faltan := sorted(esperados - presentes):
        problemas.append(f"faltan entradas para los datasets {faltan}")
    if sobran := sorted(presentes - esperados):
        problemas.append(f"hay entradas de datasets que no están en el corpus: {sobran}")
    return problemas


def verificar_campos(entrada: MetaKnowledgeEntry) -> list[str]:
    """Comprueba que una entrada tenga valores verosímiles.

    Args:
        entrada: Entrada a revisar.

    Returns:
        Lista de problemas; vacía si la entrada está sana.
    """
    problemas = []
    if not entrada.meta_features:
        problemas.append("no tiene meta-características")
    if not math.isfinite(entrada.metric_value) or not 0.0 <= entrada.metric_value <= 1.0:
        problemas.append(f"la métrica vale {entrada.metric_value!r}, fuera de [0, 1]")
    if entrada.seed != SEMILLA:
        problemas.append(f"la semilla es {entrada.seed} y el pre-registro fija {SEMILLA}")
    if entrada.n_trials != ENSAYOS:
        problemas.append(f"son {entrada.n_trials} ensayos y el anteproyecto fija {ENSAYOS}")
    if not 0 <= entrada.n_failed_trials <= entrada.n_trials:
        problemas.append(f"los ensayos fallidos ({entrada.n_failed_trials}) no son creíbles")
    if entrada.n_failed_trials == entrada.n_trials:
        problemas.append("todos los ensayos fallaron: el ganador no significa nada")
    if entrada.space_version != TABULAR_SPACE_VERSION:
        problemas.append(f"la versión del espacio es {entrada.space_version!r}")
    if not entrada.pipeline_params and any(
        indice < len(build_tabular_search_space()[paso])
        for paso, indice in entrada.configuration.items()
    ):
        problemas.append("la configuración conserva pasos pero no hay parámetros guardados")
    return problemas


def verificar_configuracion(entrada: MetaKnowledgeEntry) -> list[str]:
    """Comprueba que la configuración se resuelva al pipeline guardado.

    Es la verificación más fuerte del artefacto: si la base y el espacio de
    búsqueda se hubieran separado, los índices seguirían siendo enteros válidos
    pero designarían otras transformaciones. Comparar el pipeline reconstruido
    contra los parámetros guardados lo detecta.

    Args:
        entrada: Entrada a revisar.

    Returns:
        Lista de problemas; vacía si la configuración es interpretable.
    """
    if set(entrada.configuration) != set(TABULAR_SPACE_STEPS):
        return [
            f"los pasos de la configuración {sorted(entrada.configuration)} no son los "
            f"del espacio {list(TABULAR_SPACE_STEPS)}"
        ]
    try:
        reconstruido = pipeline_from_configuration(entrada.configuration)
    except ValueError as exc:
        return [f"la configuración no se pudo resolver: {exc}"]
    if reconstruido.get_params() != entrada.pipeline_params:
        return [
            "el pipeline reconstruido desde los índices no coincide con los parámetros "
            f"guardados: {reconstruido.get_params()} contra {entrada.pipeline_params}"
        ]
    return []


def verificar_alineacion(base: MetaKnowledgeBase) -> list[str]:
    """Comprueba que los vectores sean comparables entre sí.

    Args:
        base: Base cargada.

    Returns:
        Lista de problemas; vacía si los vectores están alineados.
    """
    problemas = []
    if not base.surviving_features():
        problemas.append(
            "ninguna meta-característica sobrevive a la regla de descarte: el "
            "recomendador no tendría con qué comparar datasets"
        )
    matriz, columnas = base.feature_matrix()
    if matriz.shape != (len(base), len(columnas)):
        problemas.append(f"la matriz tiene forma {matriz.shape}, incoherente con la base")
    if not np.isfinite(matriz).all():
        problemas.append("la matriz de supervivientes contiene valores no finitos")
    return problemas


def verificar_reproducibilidad(
    base: MetaKnowledgeBase, corpus: list[tuple[int, str]], cuantos: int
) -> list[str]:
    """Reconstruye una muestra de entradas y las compara con las guardadas.

    Args:
        base: Base cargada.
        corpus: Datasets del corpus.
        cuantos: Cuántos datasets reconstruir.

    Returns:
        Lista de problemas; vacía si todo lo reconstruido coincide.
    """
    generador = np.random.default_rng(SEMILLA)
    indices = generador.choice(len(corpus), size=min(cuantos, len(corpus)), replace=False)
    extractor = MetaFeatureExtractor(random_state=SEMILLA, cache_dir=CACHE_META)
    problemas = []
    for indice in sorted(indices):
        did, nombre = corpus[indice]
        guardada = base.entry_for(str(did))
        rehecha = construir_entrada(did, nombre, extractor)
        if rehecha.configuration != guardada.configuration:
            problemas.append(
                f"{nombre}: la configuración cambió al reconstruir "
                f"({guardada.configuration} contra {rehecha.configuration})"
            )
        elif rehecha.metric_value != guardada.metric_value:
            problemas.append(
                f"{nombre}: la métrica cambió al reconstruir "
                f"({guardada.metric_value} contra {rehecha.metric_value})"
            )
        else:
            logger.info("  %-42s reproducible", nombre)
    return problemas


def main() -> int:
    """Punto de entrada.

    Returns:
        ``0`` si la base es válida, ``1`` si se encontró algún problema.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reproducir",
        type=int,
        default=MUESTRA_REPRODUCIBILIDAD,
        help="Cuántos datasets reconstruir desde cero (0 para omitir la prueba).",
    )
    args = parser.parse_args()

    openml.config.set_console_log_level(logging.WARNING)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logging.getLogger("autopipe").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    corpus = leer_corpus(CORPUS)
    base = MetaKnowledgeBase.load(BASE)
    logger.info("Base: %d entradas contra %d datasets del corpus.\n", len(base), len(corpus))

    problemas: list[str] = [f"cobertura: {p}" for p in verificar_cobertura(base, corpus)]
    for entrada in base:
        for problema in verificar_campos(entrada) + verificar_configuracion(entrada):
            problemas.append(f"{entrada.dataset_name}: {problema}")
    problemas += [f"alineación: {p}" for p in verificar_alineacion(base)]

    logger.info(
        "Cobertura, campos, configuración y alineación: %s",
        "sin problemas" if not problemas else f"{len(problemas)} problemas",
    )
    logger.info(
        "Meta-características: %d extraídas, %d supervivientes.",
        len(base.feature_names()),
        len(base.surviving_features()),
    )

    if args.reproducir:
        logger.info("\nReproducibilidad sobre %d datasets:", args.reproducir)
        problemas += verificar_reproducibilidad(base, corpus, args.reproducir)

    if problemas:
        logger.error("\nLa base NO es válida:")
        for problema in problemas:
            logger.error("  - %s", problema)
        return 1
    logger.info("\nLa base es válida.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
