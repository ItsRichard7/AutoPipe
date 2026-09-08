# scripts/run_lodo.py
"""Arnés de validación *leave-one-dataset-out*: arranque en frío contra caliente (RE6).

Implementa el protocolo tal como lo congela el pre-registro v1.0, y **solo** lo
que ese documento dice. Cada regla no evidente lleva aquí la referencia a la
sección que la fija, para que se pueda auditar sin adivinar.

**Qué hace.** Por cada dataset del corpus, lo excluye, ajusta el recomendador con
los 24 restantes, y ejecuta sobre el excluido dos condiciones con las mismas cinco
semillas: arranque en frío y arranque en caliente. De cada corrida guarda la curva
completa de treinta ensayos, no solo el número resumen, porque de esa curva salen
la variable primaria, la medida descriptiva a quince ensayos y las reglas de
censura alternativas.

**La variable primaria.** «Ensayos hasta el objetivo» es la posición, empezando en
uno, del primer ensayo que alcanza el 95 % del mejor valor que el arranque en frío
logró con esa misma semilla sobre ese mismo dataset. Los ensayos inevaluables
consumen su turno y nunca alcanzan el objetivo. Los cinco inyectados cuentan.

**Una propiedad que se comprueba y no se supone.** El arranque en frío no puede
censurarse: el objetivo se define como el 95 % de su propio mejor valor, de modo
que si ese máximo ocurre en el ensayo *t*, el frío cruzó el umbral en algún ensayo
menor o igual que *t*. No es improbable, es imposible. El arnés lo verifica en
cada corrida y aborta si alguna vez ocurre, porque solo podría deberse a un error
de implementación.

**Este arnés no se ejecuta sobre el corpus hasta la semana 10.** La función
:func:`ejecutar_lodo` recibe el cargador de datasets como parámetro, de modo que
las pruebas lo recorren entero sobre un corpus sintético de juguete sin tocar la
red ni producir un solo número del experimento.

Ejecutar con:
    .\\.venv\\Scripts\\python.exe scripts\\run_lodo.py
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import time
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import openml
import optuna
from download_corpus import descargar
from scipy.stats import wilcoxon
from sklearn.neighbors import KNeighborsClassifier

from autopipe.meta_features import MetaFeatureExtractor
from autopipe.meta_knowledge import MetaKnowledgeBase
from autopipe.meta_recommender import MetaRecommender
from autopipe.search_engine import OptunaSearchEngine
from autopipe.search_space import build_tabular_search_space

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

BASE = Path("docs/experimentos/BASE-meta-conocimiento.json")
DIVERSIDAD = Path("docs/experimentos/DIVERSIDAD-valores.json")
RESULTADOS = Path("docs/experimentos/LODO-resultados.json")
CACHE_META = Path("datasets/.meta_cache")

# --- Constantes del protocolo, fijadas en el pre-registro v1.0 ---------------
SEMILLAS = (101, 202, 303, 404, 505)
"""Las cinco del pre-registro v1.0, idénticas en ambas condiciones."""

PRESUPUESTO = 30
"""Ensayos por condición, fijados por el anteproyecto en su Sección 4.3.3."""

CENSURA = PRESUPUESTO + 1
"""Valor de la variable cuando una condición no alcanza el objetivo."""

CENSURAS_ROBUSTEZ = (60, 300)
"""Castigos más severos con los que se recalcula el análisis primario."""

FRACCION_OBJETIVO = 0.95
"""El objetivo es esta fracción del mejor valor del arranque en frío."""

PRESUPUESTO_REDUCIDO = 15
"""Ensayos de la medida descriptiva de desempeño a presupuesto reducido."""

N_CONFIGURACIONES = 5
"""Configuraciones inyectadas como arranque en caliente."""

SEMILLA_ALEATORIA = 202
"""Semilla de la estrategia de recomendación aleatoria."""

SEMILLA_META = 42
"""Semilla de la extracción de meta-características, la de la base."""

ESTRATEGIAS_CALIENTE = ("nearest", "random", "frequent")
"""Estrategias que se ejecutan. ``nearest`` produce el contraste primario; las
otras dos son los controles secundarios declarados en el pre-registro."""

METRICA = "macro_f1"
VAL_SIZE = 0.2

CargadorDataset = Callable[[str], tuple[object, np.ndarray]]
"""Función que entrega ``(características, etiquetas)`` de un dataset."""


def _modelo() -> KNeighborsClassifier:
    """Devuelve el modelo base congelado del pre-registro."""
    return KNeighborsClassifier(
        n_neighbors=5, weights="uniform", metric="minkowski", p=2, algorithm="brute"
    )


def _cargar_de_openml(dataset_id: str) -> tuple[object, np.ndarray]:
    """Carga un dataset del corpus desde OpenML."""
    features, labels, _, _ = descargar(int(dataset_id))
    return features, labels


# --------------------------------------------------------------------------- #
# La variable medida
# --------------------------------------------------------------------------- #
def ensayos_hasta_el_objetivo(
    curva: Sequence[float], objetivo: float, censura: int = CENSURA
) -> int:
    """Posición del primer ensayo que alcanza el objetivo.

    Implementa las tres precisiones del pre-registro v1.0: el conteo empieza en
    uno, los ensayos inevaluables consumen su turno y nunca alcanzan el objetivo,
    y cuando ninguno lo alcanza se devuelve el valor de censura.

    Args:
        curva: Valor de la métrica en cada ensayo, en orden. Los inevaluables
            vienen como ``NaN``.
        objetivo: Umbral a alcanzar.
        censura: Valor a devolver si ningún ensayo lo alcanza.

    Returns:
        La posición del primer ensayo que alcanza el objetivo, empezando en uno,
        o ``censura``.
    """
    for posicion, valor in enumerate(curva, start=1):
        if math.isfinite(valor) and valor >= objetivo:
            return posicion
    return censura


def mejor_hasta(curva: Sequence[float], ensayos: int) -> float:
    """Mejor valor alcanzado en los primeros ensayos de una curva.

    Args:
        curva: Valor de la métrica en cada ensayo.
        ensayos: Cuántos ensayos considerar.

    Returns:
        El mejor valor finito, o ``NaN`` si no hubo ninguno.
    """
    finitos = [v for v in curva[:ensayos] if math.isfinite(v)]
    return max(finitos) if finitos else float("nan")


def fraccion_inevaluable(curva: Sequence[float]) -> float:
    """Fracción de ensayos de una corrida que no se pudieron evaluar.

    Es una de las dos covariables declaradas en el pre-registro v1.0.

    Args:
        curva: Valor de la métrica en cada ensayo.

    Returns:
        La fracción, entre 0 y 1.
    """
    return sum(1 for v in curva if not math.isfinite(v)) / len(curva) if curva else 0.0


# --------------------------------------------------------------------------- #
# Una corrida y un pliegue
# --------------------------------------------------------------------------- #
def ejecutar_condicion(
    features: object,
    labels: np.ndarray,
    semilla: int,
    warm_start: list[dict[str, int]] | None = None,
) -> list[float]:
    """Ejecuta una condición y devuelve su curva completa de ensayos.

    Args:
        features: Características del dataset.
        labels: Etiquetas.
        semilla: Semilla del muestreador y de la partición.
        warm_start: Configuraciones a inyectar, o ``None`` para el arranque en
            frío.

    Returns:
        El valor de la métrica en cada ensayo, en orden. Los inevaluables vienen
        como ``NaN``.
    """
    motor = OptunaSearchEngine(
        build_tabular_search_space(),
        n_trials=PRESUPUESTO,
        val_size=VAL_SIZE,
        random_state=semilla,
        task="classification",
        warm_start=warm_start,
    )
    _, _, resultados = motor.run(features, labels, _modelo(), metric=METRICA)
    return [float(r["objective"]) for r in resultados]


def ejecutar_pliegue(
    dataset_id: str,
    dataset_name: str,
    base: MetaKnowledgeBase,
    vector: dict[str, float],
    features: object,
    labels: np.ndarray,
) -> dict:
    """Ejecuta un pliegue completo: las dos condiciones con las cinco semillas.

    El recomendador se ajusta con ``base.without(dataset_id)``, de modo que ni la
    normalización ni los vecinos ven nunca el dataset excluido. Es la garantía de
    ausencia de fuga en el nivel meta que exige el pre-registro.

    Args:
        dataset_id: Identificador del dataset excluido.
        dataset_name: Su nombre, para los mensajes.
        base: Base de meta-conocimiento completa.
        vector: Meta-características del dataset excluido.
        features: Sus características.
        labels: Sus etiquetas.

    Returns:
        Diccionario con las curvas y las variables derivadas por semilla y
        condición.

    Raises:
        RuntimeError: Si el arranque en frío resulta censurado, lo que es
            imposible por construcción y solo puede indicar un error.
    """
    pliegue = base.without(dataset_id)
    recomendaciones = {}
    for estrategia in ESTRATEGIAS_CALIENTE:
        recomendador = MetaRecommender(
            n_configurations=N_CONFIGURACIONES,
            strategy=estrategia,
            random_state=SEMILLA_ALEATORIA,
        ).fit(pliegue)
        recomendaciones[estrategia] = {
            "configuraciones": recomendador.recommend(vector),
            "vecinos": [v["dataset_name"] for v in recomendador.neighbours_],
        }

    por_semilla = []
    for semilla in SEMILLAS:
        frio = ejecutar_condicion(features, labels, semilla)
        mejor_frio = mejor_hasta(frio, PRESUPUESTO)
        objetivo = FRACCION_OBJETIVO * mejor_frio

        condiciones = {
            "frio": {
                "curva": frio,
                "ensayos": ensayos_hasta_el_objetivo(frio, objetivo),
                "mejor": round(mejor_frio, 6),
                "a_presupuesto_reducido": round(
                    mejor_hasta(frio, PRESUPUESTO_REDUCIDO), 6
                ),
                "fraccion_inevaluable": round(fraccion_inevaluable(frio), 4),
            }
        }
        if condiciones["frio"]["ensayos"] >= CENSURA:
            raise RuntimeError(
                f"El arranque en frío quedó censurado en {dataset_name} con la "
                f"semilla {semilla}, lo que es imposible por construcción: el "
                f"objetivo es el 95 % de su propio mejor valor. Indica un error "
                f"en la implementación, no un resultado."
            )

        for estrategia in ESTRATEGIAS_CALIENTE:
            curva = ejecutar_condicion(
                features, labels, semilla, recomendaciones[estrategia]["configuraciones"]
            )
            condiciones[estrategia] = {
                "curva": curva,
                "ensayos": ensayos_hasta_el_objetivo(curva, objetivo),
                "mejor": round(mejor_hasta(curva, PRESUPUESTO), 6),
                "a_presupuesto_reducido": round(
                    mejor_hasta(curva, PRESUPUESTO_REDUCIDO), 6
                ),
                "fraccion_inevaluable": round(fraccion_inevaluable(curva), 4),
            }

        por_semilla.append({"semilla": semilla, "objetivo": round(objetivo, 6), **condiciones})

    return {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "recomendaciones": recomendaciones,
        "semillas": por_semilla,
    }


# --------------------------------------------------------------------------- #
# Agregación y contraste
# --------------------------------------------------------------------------- #
def observacion_pareada(pliegue: dict, condicion: str, censura: int = CENSURA) -> float:
    """Mediana de las cinco semillas para una condición de un dataset.

    Es la observación que entra en la prueba de Wilcoxon, una por dataset.

    Args:
        pliegue: Resultado de :func:`ejecutar_pliegue`.
        condicion: ``frio`` o el nombre de una estrategia.
        censura: Valor de censura con el que recalcular la variable.

    Returns:
        La mediana de los ensayos hasta el objetivo.
    """
    valores = [
        ensayos_hasta_el_objetivo(s[condicion]["curva"], s["objetivo"], censura)
        for s in pliegue["semillas"]
    ]
    return statistics.median(valores)


def contrastar(pliegues: list[dict], estrategia: str, censura: int = CENSURA) -> dict:
    """Aplica el análisis primario a una condición caliente contra el frío.

    La prueba es de Wilcoxon pareada por dataset, de una cola en la dirección de
    la hipótesis, y se reporta también el valor de dos colas para que la
    lateralidad no sea un grado de libertad oculto.

    Args:
        pliegues: Resultados de todos los pliegues.
        estrategia: Condición caliente a contrastar.
        censura: Valor de censura.

    Returns:
        Diccionario con las medianas, la reducción y los valores $p$.
    """
    frios = [observacion_pareada(p, "frio", censura) for p in pliegues]
    calientes = [observacion_pareada(p, estrategia, censura) for p in pliegues]
    mediana_frio = statistics.median(frios)
    mediana_caliente = statistics.median(calientes)
    reduccion = (
        (mediana_frio - mediana_caliente) / mediana_frio if mediana_frio else float("nan")
    )

    diferencias = [c - f for c, f in zip(calientes, frios)]
    if any(d != 0 for d in diferencias):
        una_cola = float(wilcoxon(calientes, frios, alternative="less").pvalue)
        dos_colas = float(wilcoxon(calientes, frios, alternative="two-sided").pvalue)
    else:
        una_cola = dos_colas = float("nan")

    return {
        "estrategia": estrategia,
        "censura": censura,
        "n_pares": len(pliegues),
        "mediana_frio": mediana_frio,
        "mediana_caliente": mediana_caliente,
        "reduccion": round(reduccion, 4) if math.isfinite(reduccion) else None,
        "p_una_cola": una_cola,
        "p_dos_colas": dos_colas,
        "confirma": bool(
            math.isfinite(reduccion) and reduccion >= 0.30 and una_cola < 0.05
        ),
        "censurados_frio": sum(
            1
            for p in pliegues
            for s in p["semillas"]
            if ensayos_hasta_el_objetivo(s["frio"]["curva"], s["objetivo"], censura)
            >= censura
        ),
        "censurados_caliente": sum(
            1
            for p in pliegues
            for s in p["semillas"]
            if ensayos_hasta_el_objetivo(s[estrategia]["curva"], s["objetivo"], censura)
            >= censura
        ),
    }


def analizar(pliegues: list[dict], meseta: dict[str, int] | None = None) -> dict:
    """Produce el análisis completo: primario, secundarios y robustez.

    Args:
        pliegues: Resultados de todos los pliegues.
        meseta: Anchura de la meseta por dataset, la segunda covariable
            declarada. Puede faltar.

    Returns:
        Diccionario con todos los contrastes y la tabla por dataset.
    """
    meseta = meseta or {}
    por_dataset = []
    for pliegue in pliegues:
        fila = {
            "dataset_id": pliegue["dataset_id"],
            "dataset_name": pliegue["dataset_name"],
            "meseta": meseta.get(pliegue["dataset_id"]),
        }
        for condicion in ("frio", *ESTRATEGIAS_CALIENTE):
            fila[condicion] = observacion_pareada(pliegue, condicion)
            fila[f"{condicion}_inevaluable"] = round(
                statistics.mean(
                    s[condicion]["fraccion_inevaluable"] for s in pliegue["semillas"]
                ),
                4,
            )
            fila[f"{condicion}_a_15"] = round(
                statistics.median(
                    s[condicion]["a_presupuesto_reducido"] for s in pliegue["semillas"]
                ),
                6,
            )
        por_dataset.append(fila)

    return {
        "primario": contrastar(pliegues, "nearest"),
        "secundarios": [
            contrastar(pliegues, e) for e in ESTRATEGIAS_CALIENTE if e != "nearest"
        ],
        "robustez": [
            contrastar(pliegues, "nearest", censura) for censura in CENSURAS_ROBUSTEZ
        ],
        "por_dataset": por_dataset,
    }


def ejecutar_lodo(
    base: MetaKnowledgeBase,
    cargar: CargadorDataset,
    vectores: dict[str, dict[str, float]] | None = None,
    meseta: dict[str, int] | None = None,
) -> dict:
    """Recorre el *leave-one-dataset-out* completo.

    El cargador se recibe como parámetro para que las pruebas puedan recorrer el
    protocolo entero sobre un corpus sintético, sin red y sin producir ningún
    número del experimento.

    Args:
        base: Base de meta-conocimiento del corpus.
        cargar: Función que entrega ``(características, etiquetas)`` de un
            dataset por su identificador.
        vectores: Meta-características por dataset. Si faltan, se toman de la
            propia base, que las guarda completas.
        meseta: Anchura de la meseta por dataset.

    Returns:
        Diccionario con los pliegues y el análisis.
    """
    vectores = vectores or {e.dataset_id: e.meta_features for e in base}
    pliegues = []
    inicio = time.perf_counter()
    for numero, entrada in enumerate(base, 1):
        features, labels = cargar(entrada.dataset_id)
        pliegue = ejecutar_pliegue(
            entrada.dataset_id,
            entrada.dataset_name,
            base,
            vectores[entrada.dataset_id],
            features,
            labels,
        )
        pliegues.append(pliegue)
        logger.info(
            "[%2d/%d] %-42s frio %4.1f  nearest %4.1f",
            numero,
            len(base),
            entrada.dataset_name[:42],
            observacion_pareada(pliegue, "frio"),
            observacion_pareada(pliegue, "nearest"),
        )
    return {
        "pliegues": pliegues,
        "analisis": analizar(pliegues, meseta),
        "minutos": round((time.perf_counter() - inicio) / 60, 2),
    }


def leer_meseta(ruta: Path) -> dict[str, int]:
    """Lee la anchura de la meseta por dataset del análisis de diversidad.

    Args:
        ruta: Documento ``DIVERSIDAD-valores.json``.

    Returns:
        Diccionario de identificador a número de configuraciones en meseta, o
        vacío si el archivo no existe.
    """
    if not ruta.is_file():
        logger.warning("No existe %s: la covariable de meseta quedará vacía.", ruta)
        return {}
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    return {d["dataset_id"]: d["n_en_meseta"] for d in datos["meseta"]["datasets"]}


def main() -> int:
    """Punto de entrada.

    Returns:
        ``0`` si el experimento se ejecutó.
    """
    parser = argparse.ArgumentParser(description="Leave-one-dataset-out del TFG.")
    parser.add_argument("--limit", type=int, help="Procesar solo los primeros N.")
    args = parser.parse_args()

    openml.config.set_console_log_level(logging.WARNING)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logging.getLogger("autopipe").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    if not BASE.is_file():
        raise SystemExit(f"No existe {BASE}. Ejecute antes scripts/build_meta_knowledge.py.")
    base = MetaKnowledgeBase.load(BASE)
    if args.limit:
        base = MetaKnowledgeBase(base.entries[: args.limit])

    extractor = MetaFeatureExtractor(random_state=SEMILLA_META, cache_dir=CACHE_META)
    logger.info("Extractor listo (%s). Pliegues: %d.", type(extractor).__name__, len(base))

    resultado = ejecutar_lodo(base, _cargar_de_openml, meseta=leer_meseta(DIVERSIDAD))
    RESULTADOS.write_text(
        json.dumps(resultado, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    primario = resultado["analisis"]["primario"]
    logger.info(
        "\nLODO completo en %.1f min. Mediana frío %.1f contra caliente %.1f, "
        "reducción %s, p (una cola) %.4f. Resultados en %s",
        resultado["minutos"],
        primario["mediana_frio"],
        primario["mediana_caliente"],
        primario["reduccion"],
        primario["p_una_cola"],
        RESULTADOS,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
