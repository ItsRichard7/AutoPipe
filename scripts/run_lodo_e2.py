# scripts/run_lodo_e2.py
"""Arnés del experimento 2 (pre-registro v2.0).

Implementa el protocolo tal como lo congela el pre-registro del experimento 2, y
**solo** lo que ese documento dice. Cada regla no evidente lleva aquí la referencia
a la sección que la fija.

**Qué cambia respecto del arnés del experimento 1**, que sigue existiendo y sigue
siendo el que gobierna aquellos resultados:

* **La vara es externa.** El objetivo es el 99\\ % del óptimo exhaustivo de la
  pareja (conjunto, semilla), que ``scripts/reference_optima.py`` calcula, y no el
  95\\ % del mejor valor del propio arranque en frío.
* **El arranque en frío puede censurarse.** La demostración de imposibilidad de la
  v1.0 dependía de que la vara fuera su propio máximo, y ya no lo es. La
  comprobación activa que abortaba la corrida **se retira aquí a propósito**, y no
  por descuido: lo dice el pre-registro v2.0 §4.2.
* **La variable primaria es el *regret* normalizado.** Los ensayos hasta el
  objetivo se conservan como secundaria declarada, con la definición del
  anteproyecto, para poder comparar los dos experimentos en los mismos términos.
* **Cinco condiciones y diez semillas**, sobre 42 conjuntos.
* **Dos contrastes confirmatorios** con corrección de Holm--Bonferroni.

Ejecutar con:
    .\\.venv\\Scripts\\python.exe scripts\\run_lodo_e2.py
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

from autopipe.meta_knowledge import MetaKnowledgeBase
from autopipe.meta_recommender import MetaRecommender
from autopipe.search_engine import OptunaSearchEngine
from autopipe.search_space import build_tabular_search_space

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

BASE = Path("docs/experimentos/BASE-E2.json")
REFERENCIA = Path("docs/experimentos/REFERENCIA-E2.json")
RESULTADOS = Path("docs/experimentos/LODO-E2-resultados.json")

# --- Constantes del protocolo, fijadas en el pre-registro v2.0 ---------------
SEMILLAS = (101, 202, 303, 404, 505, 606, 707, 808, 909, 1010)
"""Las diez del pre-registro v2.0, idénticas en todas las condiciones."""

PRESUPUESTO = 60
"""Ensayos por condición. Sube de 30 porque con el objetivo al 99 % la peor corrida
del experimento 1 necesitó 30 exactos, justo el presupuesto."""

CENSURA = PRESUPUESTO + 1
"""Valor de la variable secundaria cuando una condición no alcanza el objetivo."""

CENSURAS_ROBUSTEZ = (120, 600)
"""Castigos más severos con los que se recalcula la variable secundaria."""

PRESUPUESTOS_REDUCIDOS = (5, 15, 30)
"""Presupuestos de la medida descriptiva. El de 30 permite comparar con el
experimento 1 a igual presupuesto."""

N_CONFIGURACIONES = 5
SEMILLA_ALEATORIA = 202
"""Semilla de la estrategia de recomendación aleatoria, heredada de la v1.0."""

CONDICION_FRIA = "frio"
CONDICION_PRIMARIA = "nearest"
CONDICIONES_CALIENTES = ("nearest", "frequent", "random", "hybrid")
"""Las cuatro condiciones con arranque en caliente. ``nearest`` es idéntica a la del
experimento 1: es lo que permite atribuir al instrumento lo que mejore."""

CONFIRMATORIOS = (("nearest", "frio"), ("nearest", "frequent"))
"""Los dos contrastes confirmatorios, con corrección de Holm sobre la familia."""

REDUCCION_EXIGIDA = 0.30
ALFA = 0.05

METRICA = "macro_f1"
VAL_SIZE = 0.2

CargadorDataset = Callable[[str], tuple[object, np.ndarray]]


def _modelo() -> KNeighborsClassifier:
    """Devuelve el modelo base congelado, heredado del pre-registro v1.0."""
    return KNeighborsClassifier(
        n_neighbors=5, weights="uniform", metric="minkowski", p=2, algorithm="brute"
    )


def _cargar_de_openml(dataset_id: str) -> tuple[object, np.ndarray]:
    """Carga un conjunto del corpus desde OpenML."""
    features, labels, _, _ = descargar(int(dataset_id))
    return features, labels


# --------------------------------------------------------------------------- #
# Las variables medidas
# --------------------------------------------------------------------------- #
def regret_normalizado(
    curva: Sequence[float], optimo: float, presupuesto: int = PRESUPUESTO
) -> float:
    """Área normalizada entre la curva de mejor-hasta-ahora y el óptimo.

    Es la variable primaria del experimento 2 (v2.0 §6). Implementa la definición
    del documento: un ensayo sin ningún valor finito aún cuenta como regret pleno,
    los inevaluables consumen su turno, y el promedio se toma sobre el presupuesto
    completo. **Menor es mejor.**

    Args:
        curva: Valor de la métrica en cada ensayo. Los inevaluables vienen como
            ``NaN``.
        optimo: Óptimo de referencia de la pareja, $M(d,s)$.
        presupuesto: Ensayos sobre los que promediar.

    Returns:
        El regret normalizado, en $[0, 1]$.

    Raises:
        ValueError: Si el óptimo no es positivo, en cuyo caso la normalización no
            está definida y la pareja se excluye en lugar de producir un número sin
            sentido.
    """
    if not (math.isfinite(optimo) and optimo > 0):
        raise ValueError(
            f"El óptimo de referencia es {optimo!r} y el regret normalizado exige un "
            f"óptimo positivo. La pareja se excluye y se reporta."
        )
    acumulado = 0.0
    mejor = -math.inf
    for valor in curva[:presupuesto]:
        if math.isfinite(valor):
            mejor = max(mejor, valor)
        acumulado += 1.0 if mejor == -math.inf else (optimo - mejor) / optimo
    return acumulado / presupuesto


def ensayos_hasta_el_objetivo(
    curva: Sequence[float], objetivo: float, censura: int = CENSURA
) -> int:
    """Posición del primer ensayo que alcanza el objetivo.

    Variable **secundaria** del experimento 2, con la misma definición del
    anteproyecto que usó el experimento 1: el conteo empieza en uno, los
    inevaluables consumen su turno, y si ninguno alcanza el objetivo se devuelve la
    censura. Lo que cambia es el objetivo, que ahora es externo.

    Args:
        curva: Valor de la métrica en cada ensayo.
        objetivo: Umbral a alcanzar.
        censura: Valor a devolver si ningún ensayo lo alcanza.

    Returns:
        La posición del primer ensayo que lo alcanza, o ``censura``.
    """
    for posicion, valor in enumerate(curva, start=1):
        if math.isfinite(valor) and valor >= objetivo:
            return posicion
    return censura


def mejor_hasta(curva: Sequence[float], ensayos: int) -> float:
    """Mejor valor alcanzado en los primeros ensayos de una curva."""
    finitos = [v for v in curva[:ensayos] if math.isfinite(v)]
    return max(finitos) if finitos else float("nan")


def fraccion_inevaluable(curva: Sequence[float]) -> float:
    """Fracción de ensayos de una corrida que no se pudieron evaluar."""
    return sum(1 for v in curva if not math.isfinite(v)) / len(curva) if curva else 0.0


# --------------------------------------------------------------------------- #
# Una corrida y un pliegue
# --------------------------------------------------------------------------- #
def ejecutar_condicion(
    features: object,
    labels: np.ndarray,
    semilla: int,
    warm_start: list[dict[str, int]] | None = None,
    presupuesto: int = PRESUPUESTO,
) -> list[float]:
    """Ejecuta una condición y devuelve su curva completa de ensayos.

    Args:
        features: Características del conjunto de datos.
        labels: Etiquetas.
        semilla: Semilla del muestreador y de la partición.
        warm_start: Configuraciones a inyectar, o ``None`` para el arranque en frío.
        presupuesto: Ensayos de la corrida. Por omisión, el del pre-registro.

    Returns:
        El valor de la métrica en cada ensayo, en orden.
    """
    motor = OptunaSearchEngine(
        build_tabular_search_space(),
        n_trials=presupuesto,
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
    referencia: dict[str, dict],
    features: object,
    labels: np.ndarray,
    semillas: tuple[int, ...] = SEMILLAS,
    presupuesto: int = PRESUPUESTO,
) -> dict:
    """Ejecuta un pliegue completo: las cinco condiciones con las diez semillas.

    El recomendador se ajusta con ``base.without(dataset_id)``, de modo que ni la
    normalización ni los vecinos ven nunca el conjunto excluido.

    **No hay comprobación de censura del arranque en frío.** En el experimento 1 el
    arnés abortaba si ocurría, porque la vara era el propio máximo del frío y no
    podía fallar. Aquí la vara es externa y el frío sí puede quedarse corto: eso es
    un dato, no un error.

    Args:
        dataset_id: Identificador del conjunto excluido.
        dataset_name: Su nombre, para los mensajes.
        base: Base de meta-conocimiento completa.
        vector: Meta-características del conjunto excluido.
        referencia: Óptimo y objetivo por semilla de ese conjunto.
        features: Sus características.
        labels: Sus etiquetas.
        semillas: Semillas del protocolo. Entran por parámetro, igual que el
            cargador, para que las pruebas recorran la estructura sin ejecutar el
            protocolo entero. Por omisión, las diez del pre-registro.
        presupuesto: Ensayos por condición. La censura se deriva de él, tal como lo
            define el pre-registro.

    Returns:
        Diccionario con las curvas y las variables derivadas por semilla y
        condición.
    """
    pliegue = base.without(dataset_id)
    recomendaciones = {}
    for estrategia in CONDICIONES_CALIENTES:
        recomendador = MetaRecommender(
            n_configurations=N_CONFIGURACIONES,
            strategy=estrategia,
            random_state=SEMILLA_ALEATORIA,
        ).fit(pliegue)
        recomendaciones[estrategia] = {
            "configuraciones": recomendador.recommend(vector),
            "vecinos": [v["dataset_name"] for v in recomendador.neighbours_],
        }

    censura = presupuesto + 1
    por_semilla = []
    for semilla in semillas:
        marca = referencia[str(semilla)]
        optimo, objetivo = marca["optimo"], marca["objetivo"]
        condiciones = {}
        for condicion in (CONDICION_FRIA, *CONDICIONES_CALIENTES):
            curva = ejecutar_condicion(
                features,
                labels,
                semilla,
                None
                if condicion == CONDICION_FRIA
                else recomendaciones[condicion]["configuraciones"],
                presupuesto,
            )
            # El presupuesto y la censura se pasan explícitos y no por omisión: los
            # valores por defecto se enlazan al importar el módulo, de modo que
            # promediar el regret sobre un presupuesto distinto del que se ejecutó
            # daría un número silenciosamente equivocado.
            condiciones[condicion] = {
                "curva": curva,
                "regret": round(regret_normalizado(curva, optimo, presupuesto), 6),
                "ensayos": ensayos_hasta_el_objetivo(curva, objetivo, censura),
                "mejor": round(mejor_hasta(curva, PRESUPUESTO), 6),
                "a_presupuesto_reducido": {
                    str(p): round(mejor_hasta(curva, p), 6)
                    for p in PRESUPUESTOS_REDUCIDOS
                    if p <= presupuesto
                },
                "fraccion_inevaluable": round(fraccion_inevaluable(curva), 4),
            }
        por_semilla.append(
            {
                "semilla": semilla,
                "optimo": optimo,
                "objetivo": objetivo,
                "n_en_meseta": marca["n_en_meseta"],
                **condiciones,
            }
        )

    return {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "recomendaciones": recomendaciones,
        "meseta_mediana": statistics.median(s["n_en_meseta"] for s in por_semilla),
        "semillas": por_semilla,
    }


# --------------------------------------------------------------------------- #
# Agregación y contraste
# --------------------------------------------------------------------------- #
def observacion(pliegue: dict, condicion: str, variable: str = "regret") -> float:
    """Mediana de las diez semillas para una condición de un conjunto.

    Es la observación que entra en la prueba de Wilcoxon, una por conjunto.

    Args:
        pliegue: Resultado de :func:`ejecutar_pliegue`.
        condicion: ``frio`` o el nombre de una condición caliente.
        variable: ``regret`` o ``ensayos``.

    Returns:
        La mediana de las diez semillas.
    """
    return statistics.median(s[condicion][variable] for s in pliegue["semillas"])


def observacion_censurada(pliegue: dict, condicion: str, censura: int) -> float:
    """Mediana de los ensayos hasta el objetivo bajo otra regla de censura."""
    return statistics.median(
        ensayos_hasta_el_objetivo(s[condicion]["curva"], s["objetivo"], censura)
        for s in pliegue["semillas"]
    )


def _rango_biserial(tratamiento: list[float], control: list[float]) -> float:
    """Correlación biserial por rangos para muestras pareadas.

    Args:
        tratamiento: Observaciones de la condición contrastada.
        control: Observaciones de la condición de referencia.

    Returns:
        Valor en $[-1, 1]$; negativo cuando el tratamiento reduce la variable.
    """
    diferencias = [t - c for t, c in zip(tratamiento, control) if t != c]
    if not diferencias:
        return 0.0
    magnitudes = sorted(range(len(diferencias)), key=lambda i: abs(diferencias[i]))
    rangos = [0.0] * len(diferencias)
    for posicion, indice in enumerate(magnitudes, start=1):
        rangos[indice] = float(posicion)
    total = sum(rangos)
    positivos = sum(r for r, d in zip(rangos, diferencias) if d > 0)
    negativos = sum(r for r, d in zip(rangos, diferencias) if d < 0)
    return (positivos - negativos) / total


def contrastar(
    pliegues: list[dict],
    condicion: str,
    referencia: str,
    variable: str = "regret",
    censura: int | None = None,
) -> dict:
    """Contrasta una condición contra otra, pareando por conjunto de datos.

    La prueba es de Wilcoxon pareada de una cola en la dirección de la hipótesis
    —la condición contrastada reduce la variable— y se reporta también el valor de
    dos colas para que la lateralidad no sea un grado de libertad oculto.

    Args:
        pliegues: Resultados de todos los pliegues.
        condicion: Condición contrastada.
        referencia: Condición contra la que se compara.
        variable: ``regret`` o ``ensayos``.
        censura: Si se indica, recalcula ``ensayos`` con esa censura.

    Returns:
        Diccionario con las medianas, la reducción, los valores $p$ y el tamaño del
        efecto.
    """
    if censura is None:
        tratamiento = [observacion(p, condicion, variable) for p in pliegues]
        control = [observacion(p, referencia, variable) for p in pliegues]
    else:
        tratamiento = [observacion_censurada(p, condicion, censura) for p in pliegues]
        control = [observacion_censurada(p, referencia, censura) for p in pliegues]

    mediana_control = statistics.median(control)
    mediana_tratamiento = statistics.median(tratamiento)
    reduccion = (
        (mediana_control - mediana_tratamiento) / mediana_control
        if mediana_control
        else float("nan")
    )

    if any(t != c for t, c in zip(tratamiento, control)):
        una_cola = float(wilcoxon(tratamiento, control, alternative="less").pvalue)
        dos_colas = float(wilcoxon(tratamiento, control, alternative="two-sided").pvalue)
    else:
        una_cola = dos_colas = float("nan")

    return {
        "condicion": condicion,
        "referencia": referencia,
        "variable": variable,
        "censura": censura,
        "n_pares": len(pliegues),
        "n_empates": sum(1 for t, c in zip(tratamiento, control) if t == c),
        "mediana_referencia": round(mediana_control, 6),
        "mediana_condicion": round(mediana_tratamiento, 6),
        "reduccion": round(reduccion, 4) if math.isfinite(reduccion) else None,
        "p_una_cola": una_cola,
        "p_dos_colas": dos_colas,
        "efecto_rango_biserial": round(_rango_biserial(tratamiento, control), 4),
    }


def holm(contrastes: list[dict]) -> list[dict]:
    """Aplica la corrección de Holm--Bonferroni a la familia confirmatoria.

    Se corrige porque el pre-registro declara **dos** contrastes confirmatorios, y
    juzgar ambos a $\\alpha = 0{,}05$ sin corregir inflaría la probabilidad de
    declarar favorable alguno por azar.

    Args:
        contrastes: Los contrastes confirmatorios, con su ``p_una_cola``.

    Returns:
        Los mismos contrastes, cada uno con ``p_ajustado`` y ``cumple_criterio``.
    """
    orden = sorted(
        range(len(contrastes)),
        key=lambda i: (
            math.inf
            if not math.isfinite(contrastes[i]["p_una_cola"])
            else contrastes[i]["p_una_cola"]
        ),
    )
    ajustados: list[float] = [math.nan] * len(contrastes)
    anterior = 0.0
    for posicion, indice in enumerate(orden):
        crudo = contrastes[indice]["p_una_cola"]
        if not math.isfinite(crudo):
            ajustados[indice] = math.nan
            continue
        # Holm: p_(k) x (m - k + 1), monótono no decreciente y acotado en 1.
        candidato = min(1.0, crudo * (len(contrastes) - posicion))
        anterior = max(anterior, candidato)
        ajustados[indice] = anterior

    resultado = []
    for contraste, ajustado in zip(contrastes, ajustados):
        reduccion = contraste["reduccion"]
        magnitud = reduccion is not None and reduccion >= REDUCCION_EXIGIDA
        significancia = math.isfinite(ajustado) and ajustado < ALFA
        resultado.append(
            {
                **contraste,
                "p_ajustado": None if math.isnan(ajustado) else round(ajustado, 6),
                "cumple_magnitud": magnitud,
                "cumple_significancia": significancia,
                "cumple_criterio": magnitud and significancia,
            }
        )
    return resultado


def analizar(pliegues: list[dict], presupuesto: int = PRESUPUESTO) -> dict:
    """Produce el análisis completo: confirmatorios, secundarios y robustez.

    Args:
        pliegues: Resultados de todos los pliegues.
        presupuesto: Ensayos por condición, del que se deriva la censura.

    Returns:
        Diccionario con todos los contrastes y la tabla por conjunto de datos.
    """
    confirmatorios = holm(
        [contrastar(pliegues, c, r) for c, r in CONFIRMATORIOS]
    )

    secundarios = [
        contrastar(pliegues, c, r, variable="ensayos") for c, r in CONFIRMATORIOS
    ]
    secundarios += [
        contrastar(pliegues, "hybrid", "frequent"),
        contrastar(pliegues, "hybrid", "nearest"),
        contrastar(pliegues, "random", "frio"),
        contrastar(pliegues, "frequent", "frio"),
    ]

    robustez = [
        contrastar(pliegues, c, r, variable="ensayos", censura=censura)
        for c, r in CONFIRMATORIOS
        for censura in CENSURAS_ROBUSTEZ
    ]

    por_dataset = []
    for pliegue in pliegues:
        fila = {
            "dataset_id": pliegue["dataset_id"],
            "dataset_name": pliegue["dataset_name"],
            "meseta": pliegue["meseta_mediana"],
        }
        for condicion in (CONDICION_FRIA, *CONDICIONES_CALIENTES):
            fila[condicion] = observacion(pliegue, condicion)
            fila[f"{condicion}_ensayos"] = observacion(pliegue, condicion, "ensayos")
            fila[f"{condicion}_inevaluable"] = round(
                statistics.mean(
                    s[condicion]["fraccion_inevaluable"] for s in pliegue["semillas"]
                ),
                4,
            )
        por_dataset.append(fila)

    censurados = {
        condicion: sum(
            1
            for p in pliegues
            for s in p["semillas"]
            if s[condicion]["ensayos"] > presupuesto
        )
        for condicion in (CONDICION_FRIA, *CONDICIONES_CALIENTES)
    }

    return {
        "confirmatorios": confirmatorios,
        "secundarios": secundarios,
        "robustez": robustez,
        "por_dataset": por_dataset,
        "censurados": censurados,
        "n_corridas_por_condicion": sum(len(p["semillas"]) for p in pliegues),
    }


def ejecutar_lodo(
    base: MetaKnowledgeBase,
    cargar: CargadorDataset,
    referencias: dict[str, dict],
    vectores: dict[str, dict[str, float]] | None = None,
    semillas: tuple[int, ...] = SEMILLAS,
    presupuesto: int = PRESUPUESTO,
) -> dict:
    """Recorre el *leave-one-dataset-out* completo del experimento 2.

    Args:
        base: Base de meta-conocimiento del corpus.
        cargar: Función que entrega ``(características, etiquetas)`` por
            identificador. Entra por parámetro para que las pruebas recorran el
            protocolo entero sobre un corpus sintético.
        referencias: Óptimo y objetivo por conjunto y semilla.
        vectores: Meta-características por conjunto. Si faltan, se toman de la base.
        semillas: Semillas del protocolo. Por omisión, las diez del pre-registro.
        presupuesto: Ensayos por condición. Por omisión, los 60 del pre-registro.

    Returns:
        Diccionario con los pliegues y el análisis.

    Raises:
        KeyError: Si falta la referencia de algún conjunto del corpus. Se rechaza
            en lugar de saltarlo: un pliegue sin vara no se puede medir.
    """
    vectores = vectores or {e.dataset_id: e.meta_features for e in base}
    faltan = [e.dataset_id for e in base if e.dataset_id not in referencias]
    if faltan:
        raise KeyError(
            f"Faltan los óptimos de referencia de {faltan}. Ejecute antes "
            f"scripts/reference_optima.py: sin vara externa no hay experimento 2."
        )

    pliegues = []
    inicio = time.perf_counter()
    for numero, entrada in enumerate(base, 1):
        features, labels = cargar(entrada.dataset_id)
        pliegue = ejecutar_pliegue(
            entrada.dataset_id,
            entrada.dataset_name,
            base,
            vectores[entrada.dataset_id],
            referencias[entrada.dataset_id],
            features,
            labels,
            semillas,
            presupuesto,
        )
        pliegues.append(pliegue)
        logger.info(
            "[%2d/%d] %-38s regret frio %.4f  nearest %.4f  frequent %.4f",
            numero,
            len(base),
            entrada.dataset_name[:38],
            observacion(pliegue, "frio"),
            observacion(pliegue, "nearest"),
            observacion(pliegue, "frequent"),
        )
    return {
        "pliegues": pliegues,
        "analisis": analizar(pliegues, presupuesto),
        "minutos": round((time.perf_counter() - inicio) / 60, 2),
    }


def leer_referencias(ruta: Path) -> dict[str, dict]:
    """Lee los óptimos de referencia por conjunto y semilla.

    Args:
        ruta: Documento ``REFERENCIA-E2.json``.

    Returns:
        Diccionario de identificador a ``{semilla: {optimo, objetivo, ...}}``.

    Raises:
        SystemExit: Si el archivo no existe.
    """
    if not ruta.is_file():
        raise SystemExit(
            f"No existe {ruta}. Ejecute antes scripts/reference_optima.py."
        )
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    return {d["dataset_id"]: d["semillas"] for d in datos["datasets"]}


def main() -> int:
    """Punto de entrada.

    Returns:
        ``0`` si el experimento se ejecutó.

    Raises:
        SystemExit: Si falta la base o la referencia.
    """
    parser = argparse.ArgumentParser(description="Leave-one-dataset-out, experimento 2.")
    parser.add_argument("--limit", type=int, help="Procesar solo los primeros N.")
    args = parser.parse_args()

    openml.config.set_console_log_level(logging.WARNING)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logging.getLogger("autopipe").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    if not BASE.is_file():
        raise SystemExit(
            f"No existe {BASE}. Ejecute antes scripts/build_meta_knowledge.py con "
            f"--corpus docs/experimentos/CORPUS-E2.md."
        )
    base = MetaKnowledgeBase.load(BASE)
    referencias = leer_referencias(REFERENCIA)
    if args.limit:
        base = MetaKnowledgeBase(base.entries[: args.limit])

    logger.info("Pliegues: %d. Semillas: %d. Condiciones: %d.",
                len(base), len(SEMILLAS), 1 + len(CONDICIONES_CALIENTES))
    resultado = ejecutar_lodo(base, _cargar_de_openml, referencias)
    RESULTADOS.write_text(
        json.dumps(resultado, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("\nExperimento 2 completo en %.1f min.", resultado["minutos"])
    for contraste in resultado["analisis"]["confirmatorios"]:
        logger.info(
            "  %-8s contra %-8s  regret %.4f -> %.4f  reduccion %s  "
            "p %.4f (Holm %s)  %s",
            contraste["condicion"],
            contraste["referencia"],
            contraste["mediana_referencia"],
            contraste["mediana_condicion"],
            contraste["reduccion"],
            contraste["p_una_cola"],
            contraste["p_ajustado"],
            "CUMPLE" if contraste["cumple_criterio"] else "no cumple",
        )
    logger.info("Resultados en %s", RESULTADOS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
