# scripts/run_sensitivity_startup.py
"""Sensibilidad del experimento 2 a ``n_startup_trials = 5``.

**De dónde sale.** El spike A (2026-07-23) midió que ``n_startup_trials`` interactúa
fuertemente con el arranque en caliente. Lo fijó en **10** como configuración primaria
---el valor por defecto de TPE, que no puede atribuirse a una elección orientada al
resultado--- y declaró a la vez un **análisis de sensibilidad a 5**. El experimento 2 se
ejecutó con 10; este script ejecuta la sensibilidad.

**Qué cambia y qué no.** Cambia solo ``n_startup_trials``. Todo lo demás es idéntico al
experimento 2: los conjuntos, las diez semillas, las particiones, el presupuesto de 60, el
óptimo y el objetivo de cada pareja y **las mismas configuraciones inyectadas**, que se leen
del documento de resultados en lugar de recalcularse, para que ninguna otra cosa pueda
diferir. Las corridas con 10 no se repiten: son las del experimento 2.

**Qué condiciones.** ``frio``, ``nearest`` y ``frequent``, las tres que intervienen en los
dos contrastes confirmatorios. El spike advirtió que el efecto del parámetro había que
medirlo **también en el frío**, para distinguir un efecto propio del arranque en caliente de
uno general del muestreador con presupuestos pequeños; con el frío incluido se puede.

**El análisis se fijó antes de ejecutar**, y es este:

1. Los dos contrastes confirmatorios recalculados con 5, con la misma corrección de Holm y
   el mismo criterio. **No sustituyen** a los del experimento 2: dicen si su conclusión
   depende del valor elegido.
2. Los mismos dos pares sobre los ensayos hasta el objetivo.
3. El efecto del parámetro en cada brazo: cada condición con 5 contra sí misma con 10, con
   prueba de dos colas porque no se declaró ninguna dirección.
4. El beneficio del arranque en caliente ---regret del frío menos el de la condición--- con 5
   contra con 10, también de dos colas.

Ejecutar con:
    .\\.venv\\Scripts\\python.exe scripts\\run_sensitivity_startup.py
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
from datetime import date
from pathlib import Path

import numpy as np
import openml
import optuna
from report_lodo_e2 import redactar_veredicto
from run_lodo_e2 import (
    CONDICION_FRIA,
    CONFIRMATORIOS,
    METRICA,
    PRESUPUESTO,
    PRESUPUESTOS_REDUCIDOS,
    VAL_SIZE,
    _cargar_de_openml,
    _modelo,
    contrastar,
    ensayos_hasta_el_objetivo,
    fraccion_inevaluable,
    holm,
    mejor_hasta,
    observacion,
    regret_normalizado,
)
from scipy.stats import wilcoxon

from autopipe.search_engine import OptunaSearchEngine
from autopipe.search_space import build_tabular_search_space

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

RESULTADOS_E2 = Path("docs/experimentos/LODO-E2-resultados.json")
RESULTADOS = Path("docs/experimentos/SENSIBILIDAD-startup.json")
RESUMEN = Path("docs/experimentos/SENSIBILIDAD-startup.md")

N_STARTUP_PRIMARIO = 10
"""Valor con el que se ejecutó el experimento 2, fijado en el spike A."""

N_STARTUP_SENSIBILIDAD = 5
"""Valor de la sensibilidad declarada en el spike A."""

CONDICIONES = ("frio", "nearest", "frequent")
"""Las tres condiciones que intervienen en los dos contrastes confirmatorios."""

SUFIJO = "_n5"
"""Marca de las corridas con 5 cuando se parean con las del experimento 2."""

CLAVES_COMPARADAS = (
    "condicion", "referencia", "mediana_referencia", "mediana_condicion", "p_una_cola",
    "p_ajustado", "cumple_criterio",
)

CargadorDataset = Callable[[str], tuple[object, np.ndarray]]


# --------------------------------------------------------------------------- #
# La ejecución
# --------------------------------------------------------------------------- #
def ejecutar_condicion(
    features: object,
    labels: np.ndarray,
    semilla: int,
    warm_start: list[dict[str, int]] | None,
    n_startup_trials: int,
    presupuesto: int = PRESUPUESTO,
) -> list[float]:
    """Ejecuta una condición con un ``n_startup_trials`` dado.

    Es la función del arnés del experimento 2 con un único parámetro más. Una prueba
    comprueba que con 10 produce **exactamente** las mismas curvas que el arnés.

    Args:
        features: Características del conjunto de datos.
        labels: Etiquetas.
        semilla: Semilla del muestreador y de la partición.
        warm_start: Configuraciones a inyectar, o ``None`` para el arranque en frío.
        n_startup_trials: Ensayos al azar antes de que TPE empiece a modelar.
        presupuesto: Ensayos de la corrida.

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
        n_startup_trials=n_startup_trials,
    )
    _, _, resultados = motor.run(features, labels, _modelo(), metric=METRICA)
    return [float(r["objective"]) for r in resultados]


def medir(curva: list[float], optimo: float, objetivo: float, presupuesto: int) -> dict:
    """Deriva de una curva las mismas medidas que guarda el arnés del experimento 2.

    Args:
        curva: Valor de la métrica en cada ensayo.
        optimo: Óptimo de referencia de la pareja.
        objetivo: Objetivo de la pareja.
        presupuesto: Ensayos ejecutados, del que se deriva la censura.

    Returns:
        Diccionario con la curva, el regret, los ensayos hasta el objetivo, el mejor
        valor, el mejor valor a cada presupuesto reducido y la fracción inevaluable.
    """
    return {
        "curva": curva,
        "regret": round(regret_normalizado(curva, optimo, presupuesto), 6),
        "ensayos": ensayos_hasta_el_objetivo(curva, objetivo, presupuesto + 1),
        "mejor": round(mejor_hasta(curva, presupuesto), 6),
        "a_presupuesto_reducido": {
            str(p): round(mejor_hasta(curva, p), 6)
            for p in PRESUPUESTOS_REDUCIDOS
            if p <= presupuesto
        },
        "fraccion_inevaluable": round(fraccion_inevaluable(curva), 4),
    }


def ejecutar_pliegue(
    pliegue_e2: dict,
    features: object,
    labels: np.ndarray,
    n_startup_trials: int = N_STARTUP_SENSIBILIDAD,
    condiciones: Sequence[str] = CONDICIONES,
    presupuesto: int = PRESUPUESTO,
) -> dict:
    """Repite un pliegue del experimento 2 cambiando solo ``n_startup_trials``.

    Las semillas, el óptimo, el objetivo y las configuraciones inyectadas se toman del
    pliegue ya ejecutado, no se recalculan.

    Args:
        pliegue_e2: Pliegue tal como lo guardó el experimento 2.
        features: Características del conjunto de datos.
        labels: Etiquetas.
        n_startup_trials: Ensayos al azar antes de que TPE empiece a modelar.
        condiciones: Condiciones a ejecutar.
        presupuesto: Ensayos por corrida.

    Returns:
        Pliegue con la misma forma que los del experimento 2.
    """
    por_semilla = []
    for marca in pliegue_e2["semillas"]:
        registro = {
            "semilla": marca["semilla"],
            "optimo": marca["optimo"],
            "objetivo": marca["objetivo"],
        }
        for condicion in condiciones:
            inyectadas = (
                None
                if condicion == CONDICION_FRIA
                else pliegue_e2["recomendaciones"][condicion]["configuraciones"]
            )
            curva = ejecutar_condicion(
                features, labels, marca["semilla"], inyectadas, n_startup_trials, presupuesto
            )
            registro[condicion] = medir(curva, marca["optimo"], marca["objetivo"], presupuesto)
        por_semilla.append(registro)
    return {
        "dataset_id": pliegue_e2["dataset_id"],
        "dataset_name": pliegue_e2["dataset_name"],
        "semillas": por_semilla,
    }


def ejecutar(
    e2: dict,
    cargar: CargadorDataset,
    n_startup_trials: int = N_STARTUP_SENSIBILIDAD,
    condiciones: Sequence[str] = CONDICIONES,
    limite: int | None = None,
    presupuesto: int = PRESUPUESTO,
) -> dict:
    """Recorre los pliegues del experimento 2 con el nuevo ``n_startup_trials``.

    Args:
        e2: Documento de resultados del experimento 2.
        cargar: Función que entrega ``(características, etiquetas)`` por identificador.
        n_startup_trials: Ensayos al azar antes de que TPE empiece a modelar.
        condiciones: Condiciones a ejecutar.
        limite: Si se indica, solo los primeros pliegues.
        presupuesto: Ensayos por corrida.

    Returns:
        Diccionario con los pliegues ejecutados y la duración.
    """
    pliegues_e2 = e2["pliegues"][:limite] if limite else e2["pliegues"]
    pliegues = []
    inicio = time.perf_counter()
    for numero, pliegue_e2 in enumerate(pliegues_e2, 1):
        features, labels = cargar(pliegue_e2["dataset_id"])
        pliegue = ejecutar_pliegue(
            pliegue_e2, features, labels, n_startup_trials, condiciones, presupuesto
        )
        pliegues.append(pliegue)
        logger.info(
            "[%2d/%d] %-38s %s",
            numero,
            len(pliegues_e2),
            pliegue["dataset_name"][:38],
            "  ".join(f"{c} {observacion(pliegue, c):.4f}" for c in condiciones),
        )
    return {
        "n_startup_trials": n_startup_trials,
        "condiciones": list(condiciones),
        "pliegues": pliegues,
        "minutos": round((time.perf_counter() - inicio) / 60, 2),
    }


# --------------------------------------------------------------------------- #
# El análisis
# --------------------------------------------------------------------------- #
def combinar(
    pliegues_10: list[dict], pliegues_5: list[dict], condiciones: Sequence[str] = CONDICIONES
) -> list[dict]:
    """Parea las corridas con 5 y con 10 por conjunto y semilla.

    Args:
        pliegues_10: Pliegues del experimento 2.
        pliegues_5: Pliegues de la sensibilidad.
        condiciones: Condiciones a parear.

    Returns:
        Pliegues cuyas semillas llevan cada condición dos veces: con su nombre, la de 10,
        y con el sufijo ``_n5``, la de 5.

    Raises:
        KeyError: Si un conjunto o una semilla de la sensibilidad no existe en el
            experimento 2.
        ValueError: Si la vara de una pareja no coincide. Compararlas sería comparar
            dos objetivos distintos.
    """
    del_10 = {p["dataset_id"]: p for p in pliegues_10}
    combinados = []
    for p5 in pliegues_5:
        if p5["dataset_id"] not in del_10:
            raise KeyError(f"El conjunto {p5['dataset_name']} no está en el experimento 2.")
        semillas_10 = {s["semilla"]: s for s in del_10[p5["dataset_id"]]["semillas"]}
        semillas = []
        for s5 in p5["semillas"]:
            s10 = semillas_10.get(s5["semilla"])
            if s10 is None:
                raise KeyError(
                    f"La semilla {s5['semilla']} de {p5['dataset_name']} no está en el "
                    f"experimento 2."
                )
            if (s10["optimo"], s10["objetivo"]) != (s5["optimo"], s5["objetivo"]):
                raise ValueError(
                    f"La vara de {p5['dataset_name']} con la semilla {s5['semilla']} no "
                    f"coincide con la del experimento 2: no se comparan objetivos distintos."
                )
            semillas.append(
                {
                    "semilla": s5["semilla"],
                    "optimo": s5["optimo"],
                    "objetivo": s5["objetivo"],
                    **{c: s10[c] for c in condiciones},
                    **{c + SUFIJO: s5[c] for c in condiciones},
                }
            )
        combinados.append(
            {"dataset_id": p5["dataset_id"], "dataset_name": p5["dataset_name"],
             "semillas": semillas}
        )
    return combinados


def _recuento(tratamiento: list[float], control: list[float]) -> dict[str, int]:
    """Cuenta en cuántos conjuntos el tratamiento queda por debajo, igual o por encima."""
    return {
        "gana": sum(1 for t, c in zip(tratamiento, control) if t < c),
        "empata": sum(1 for t, c in zip(tratamiento, control) if t == c),
        "pierde": sum(1 for t, c in zip(tratamiento, control) if t > c),
    }


def efecto_por_brazo(combinados: list[dict], condicion: str) -> dict:
    """Una condición con 5 contra sí misma con 10. Menor regret es mejor.

    Args:
        combinados: Pliegues de :func:`combinar`.
        condicion: Condición a comparar.

    Returns:
        Diccionario con las dos medianas, la reducción, el recuento y el valor $p$ de dos
        colas.
    """
    contraste = contrastar(combinados, condicion + SUFIJO, condicion)
    con_5 = [observacion(p, condicion + SUFIJO) for p in combinados]
    con_10 = [observacion(p, condicion) for p in combinados]
    return {
        "condicion": condicion,
        "mediana_10": contraste["mediana_referencia"],
        "mediana_5": contraste["mediana_condicion"],
        "reduccion": contraste["reduccion"],
        "p_dos_colas": contraste["p_dos_colas"],
        "efecto_rango_biserial": contraste["efecto_rango_biserial"],
        **_recuento(con_5, con_10),
    }


def beneficio(combinados: list[dict], condicion: str) -> dict:
    """El beneficio del arranque en caliente con 5 contra con 10.

    El beneficio de una condición en un conjunto es el regret mediano del frío menos el
    de la condición: **positivo cuando la condición mejora**. Cada uno se calcula con
    su propio frío, de modo que un efecto general del muestreador no se confunde con uno
    del arranque en caliente.

    Args:
        combinados: Pliegues de :func:`combinar`.
        condicion: Condición caliente.

    Returns:
        Diccionario con los beneficios medianos, el recuento y el valor $p$ de dos colas.
    """
    con_10 = [observacion(p, CONDICION_FRIA) - observacion(p, condicion) for p in combinados]
    con_5 = [
        observacion(p, CONDICION_FRIA + SUFIJO) - observacion(p, condicion + SUFIJO)
        for p in combinados
    ]
    diferencias = [b5 - b10 for b5, b10 in zip(con_5, con_10)]
    p = (
        float(wilcoxon(con_5, con_10, alternative="two-sided").pvalue)
        if any(d != 0 for d in diferencias)
        else float("nan")
    )
    return {
        "condicion": condicion,
        "beneficio_mediano_10": round(statistics.median(con_10), 6),
        "beneficio_mediano_5": round(statistics.median(con_5), 6),
        "crece": sum(1 for d in diferencias if d > 0),
        "igual": sum(1 for d in diferencias if d == 0),
        "decrece": sum(1 for d in diferencias if d < 0),
        "p_dos_colas": p,
    }


def coincide(recalculados: list[dict], guardados: list[dict]) -> bool:
    """Comprueba que los contrastes con 10 recalculados son los del experimento 2.

    Args:
        recalculados: Contrastes recalculados sobre las corridas pareadas.
        guardados: Contrastes confirmatorios del documento del experimento 2.

    Returns:
        ``True`` si coinciden en todas las cifras que deciden.
    """
    return len(recalculados) == len(guardados) and all(
        a.get(clave) == b.get(clave)
        for a, b in zip(recalculados, guardados)
        for clave in CLAVES_COMPARADAS
    )


def analizar(
    pliegues_5: list[dict],
    e2: dict,
    condiciones: Sequence[str] = CONDICIONES,
    presupuesto: int = PRESUPUESTO,
) -> dict:
    """Produce el análisis de sensibilidad fijado antes de ejecutar.

    Args:
        pliegues_5: Pliegues ejecutados con 5.
        e2: Documento de resultados del experimento 2.
        condiciones: Condiciones ejecutadas.
        presupuesto: Ensayos por corrida.

    Returns:
        Diccionario con los contrastes con 10 y con 5, la variable secundaria, el efecto
        por brazo, el beneficio y los censurados.
    """
    combinados = combinar(e2["pliegues"], pliegues_5, condiciones)
    confirmatorios_10 = holm([contrastar(combinados, c, r) for c, r in CONFIRMATORIOS])
    completo = len(combinados) == len(e2["pliegues"])
    return {
        "n_conjuntos": len(combinados),
        "n_corridas_por_condicion": sum(len(p["semillas"]) for p in combinados),
        "confirmatorios_10": confirmatorios_10,
        "coincide_con_e2": (
            coincide(confirmatorios_10, e2["analisis"]["confirmatorios"]) if completo else None
        ),
        "confirmatorios_5": holm([contrastar(pliegues_5, c, r) for c, r in CONFIRMATORIOS]),
        "ensayos_5": [
            contrastar(pliegues_5, c, r, variable="ensayos") for c, r in CONFIRMATORIOS
        ],
        "por_brazo": [efecto_por_brazo(combinados, c) for c in condiciones],
        "beneficio": [beneficio(combinados, c) for c in condiciones if c != CONDICION_FRIA],
        "censurados": {
            c: {
                "con_10": sum(
                    1 for p in combinados for s in p["semillas"] if s[c]["ensayos"] > presupuesto
                ),
                "con_5": sum(
                    1
                    for p in combinados
                    for s in p["semillas"]
                    if s[c + SUFIJO]["ensayos"] > presupuesto
                ),
            }
            for c in condiciones
        },
    }


# --------------------------------------------------------------------------- #
# El documento
# --------------------------------------------------------------------------- #
def _n(valor: object, decimales: int = 4) -> str:
    """Presenta un número, o un guion si no está definido."""
    if isinstance(valor, bool):
        return "sí" if valor else "no"
    if not isinstance(valor, (int, float)) or not math.isfinite(valor):
        return "—"
    return f"{valor:.{decimales}f}"


def _porcentaje(valor: object) -> str:
    """Presenta una reducción relativa como porcentaje."""
    return "—" if not isinstance(valor, (int, float)) else f"{valor * 100:.1f} %"


def _tabla_confirmatorios(analisis: dict) -> str:
    """Tabula cada contraste confirmatorio con 10 y con 5, uno junto al otro."""
    filas = [
        "| Contraste | `n_startup_trials` | Regret referencia | Regret condición | Reducción "
        "| p (1 cola) | p (Holm) | Cumple |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for diez, cinco in zip(analisis["confirmatorios_10"], analisis["confirmatorios_5"]):
        for valor, c in ((N_STARTUP_PRIMARIO, diez), (N_STARTUP_SENSIBILIDAD, cinco)):
            filas.append(
                f"| `{c['condicion']}` vs `{c['referencia']}` | {valor} | "
                f"{_n(c['mediana_referencia'])} | {_n(c['mediana_condicion'])} | "
                f"{_porcentaje(c['reduccion'])} | {_n(c['p_una_cola'])} | "
                f"{_n(c['p_ajustado'])} | {_n(c['cumple_criterio'])} |"
            )
    return "\n".join(filas)


def _tabla_ensayos(analisis: dict) -> str:
    """Tabula la variable secundaria con 5."""
    filas = [
        "| Contraste | Mediana referencia | Mediana condición | Reducción | p (1 cola) |",
        "|---|---:|---:|---:|---:|",
    ]
    filas += [
        f"| `{c['condicion']}` vs `{c['referencia']}` | {_n(c['mediana_referencia'], 1)} | "
        f"{_n(c['mediana_condicion'], 1)} | {_porcentaje(c['reduccion'])} | "
        f"{_n(c['p_una_cola'])} |"
        for c in analisis["ensayos_5"]
    ]
    return "\n".join(filas)


def _tabla_brazos(analisis: dict) -> str:
    """Tabula el efecto del parámetro en cada brazo."""
    filas = [
        "| Condición | Regret con 10 | Regret con 5 | Reducción | Mejora con 5 | Igual "
        "| Empeora con 5 | p (2 colas) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    filas += [
        f"| `{b['condicion']}` | {_n(b['mediana_10'])} | {_n(b['mediana_5'])} | "
        f"{_porcentaje(b['reduccion'])} | {b['gana']} | {b['empata']} | {b['pierde']} | "
        f"{_n(b['p_dos_colas'])} |"
        for b in analisis["por_brazo"]
    ]
    return "\n".join(filas)


def _tabla_beneficio(analisis: dict) -> str:
    """Tabula el beneficio del arranque en caliente con 10 y con 5."""
    filas = [
        "| Condición | Beneficio mediano con 10 | Beneficio mediano con 5 | Crece con 5 "
        "| Igual | Decrece con 5 | p (2 colas) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    filas += [
        f"| `{b['condicion']}` | {b['beneficio_mediano_10']:+.4f} | "
        f"{b['beneficio_mediano_5']:+.4f} | {b['crece']} | {b['igual']} | {b['decrece']} | "
        f"{_n(b['p_dos_colas'])} |"
        for b in analisis["beneficio"]
    ]
    return "\n".join(filas)


def _coincidencia(valor: bool | None) -> str:
    """Enuncia si los contrastes con 10 recalculados coinciden con el experimento 2."""
    if valor is None:
        return "no aplica: la corrida no recorrió todos los conjuntos"
    if valor:
        return "**sí**, cifra por cifra"
    return "**no**. Algo difiere entre las corridas pareadas y el experimento 2"


def redactar(resultado: dict) -> str:
    """Redacta el documento de la sensibilidad.

    Args:
        resultado: Documento que escribe :func:`main`, con su análisis.

    Returns:
        El documento markdown.
    """
    analisis = resultado["analisis"]
    veredictos = "\n\n".join(
        f"{i}. {redactar_veredicto(c)}" for i, c in enumerate(analisis["confirmatorios_5"], 1)
    )
    # Se recorre en el orden declarado de las condiciones y no en el del diccionario: el
    # JSON se guarda con las claves ordenadas, y el resumen regenerado con --solo-reporte
    # tiene que ser idéntico al escrito durante la ejecución.
    censurados = "\n".join(
        f"| `{c}` | {analisis['censurados'][c]['con_10']} | "
        f"{analisis['censurados'][c]['con_5']} |"
        for c in resultado["condiciones"]
    )
    condiciones = ", ".join(f"`{c}`" for c in resultado["condiciones"])
    diez, cinco = N_STARTUP_PRIMARIO, resultado["n_startup_trials"]
    return f"""# Sensibilidad del experimento 2 a `n_startup_trials`

> **Archivo generado.** No editar a mano: se regenera con
> `.\\.venv\\Scripts\\python.exe scripts\\run_sensitivity_startup.py --solo-reporte`.

> **Análisis de sensibilidad declarado** en el spike A (2026-07-23), cuando
> `n_startup_trials` se fijó en {N_STARTUP_PRIMARIO} como configuración primaria. **No
> sustituye** a los contrastes confirmatorios del experimento 2: dice si su conclusión
> depende de ese valor.

- **Generado:** {date.today().isoformat()}
- **Datos:** `{RESULTADOS.as_posix()}` (con {resultado['n_startup_trials']}) y
  `{RESULTADOS_E2.as_posix()}` (con {N_STARTUP_PRIMARIO})
- **Duración de la ejecución:** {resultado['minutos']:.1f} min
- **Conjuntos:** {analisis['n_conjuntos']} · **Corridas por condición:**
  {analisis['n_corridas_por_condicion']} · **Condiciones:** {condiciones}
- **Los contrastes con {N_STARTUP_PRIMARIO}, recalculados sobre las corridas pareadas,
  coinciden con los del experimento 2:** {_coincidencia(analisis['coincide_con_e2'])}

## Qué cambia y qué no

Cambia solo el número de ensayos al azar antes de que TPE empiece a modelar. Las semillas,
las particiones, el presupuesto, el óptimo y el objetivo de cada pareja y **las
configuraciones inyectadas** son las del experimento 2, leídas de su documento de
resultados en lugar de recalcularse. Las corridas con {N_STARTUP_PRIMARIO} no se repiten.

## Los contrastes confirmatorios con {resultado['n_startup_trials']}

Con la misma corrección de Holm y el mismo criterio que el experimento 2: reducción de al
menos el 30 % de la mediana del regret y valor p ajustado por debajo de 0,05.

{veredictos}

{_tabla_confirmatorios(analisis)}

## La variable secundaria con {resultado['n_startup_trials']}

Los mismos dos pares sobre los ensayos hasta el objetivo, sin corrección.

{_tabla_ensayos(analisis)}

## El efecto del parámetro en cada brazo

El spike A midió el efecto solo en el brazo caliente y advirtió que había que medirlo
**también en el frío**: si el frío mejora tanto como el caliente, lo que se observa es un
efecto general del muestreador y no uno del arranque en caliente. Menor regret es mejor.
No se declaró dirección, así que la prueba es de dos colas.

{_tabla_brazos(analisis)}

## El beneficio del arranque en caliente con {diez} y con {cinco}

El beneficio es el regret mediano del frío menos el de la condición, cada uno con su propio
frío: positivo cuando la condición mejora. Si el beneficio no cambia, la conclusión del
experimento 2 no depende del parámetro aunque ambos brazos se muevan.

{_tabla_beneficio(analisis)}

## Corridas censuradas

| Condición | Con {N_STARTUP_PRIMARIO} | Con {resultado['n_startup_trials']} |
|---|---:|---:|
{censurados}
"""


def _silenciar() -> None:
    """Acalla los registros de las bibliotecas durante una ejecución larga."""
    openml.config.set_console_log_level(logging.WARNING)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logging.getLogger("autopipe").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)


def main() -> int:
    """Punto de entrada.

    Returns:
        ``0`` si la sensibilidad se ejecutó o se regeneró su resumen.

    Raises:
        SystemExit: Si falta el documento del experimento 2, o el de la sensibilidad al
            pedir solo el reporte.
    """
    parser = argparse.ArgumentParser(description="Sensibilidad a n_startup_trials.")
    parser.add_argument("--limit", type=int, help="Procesar solo los primeros N conjuntos.")
    parser.add_argument("--solo-reporte", action="store_true",
                        help="Regenerar el resumen desde el JSON, sin ejecutar.")
    parser.add_argument("--resultados", type=Path, default=RESULTADOS)
    parser.add_argument("--resumen", type=Path, default=RESUMEN)
    args = parser.parse_args()

    if not RESULTADOS_E2.is_file():
        raise SystemExit(f"No existe {RESULTADOS_E2}. Ejecute antes scripts/run_lodo_e2.py.")
    e2 = json.loads(RESULTADOS_E2.read_text(encoding="utf-8"))

    if args.solo_reporte:
        if not args.resultados.is_file():
            raise SystemExit(
                f"No existe {args.resultados}. Ejecute antes scripts/run_sensitivity_startup.py "
                f"sin --solo-reporte."
            )
        resultado = json.loads(args.resultados.read_text(encoding="utf-8"))
    else:
        _silenciar()
        logger.info("Conjuntos: %d. Condiciones: %s. n_startup_trials: %d.",
                    len(e2["pliegues"][: args.limit] if args.limit else e2["pliegues"]),
                    ", ".join(CONDICIONES), N_STARTUP_SENSIBILIDAD)
        resultado = ejecutar(e2, _cargar_de_openml, limite=args.limit)
        resultado["analisis"] = analizar(resultado["pliegues"], e2)
        args.resultados.write_text(
            json.dumps(resultado, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )

    args.resumen.write_text(redactar(resultado), encoding="utf-8")
    logger.info("Resumen escrito en %s", args.resumen)
    for contraste in resultado["analisis"]["confirmatorios_5"]:
        logger.info("%s", redactar_veredicto(contraste))
    logger.info("Coincide con el experimento 2: %s", resultado["analisis"]["coincide_con_e2"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
