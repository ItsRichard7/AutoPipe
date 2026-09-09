# scripts/analyze_e2_mechanism.py
"""Análisis exploratorio del mecanismo del experimento 2 (v2.0 §8.6).

Responde a una pregunta que los contrastes confirmatorios dejan abierta: **por qué**
la estrategia por similitud queda por detrás de las que no miran las
meta-características. No decide nada sobre la hipótesis ---eso lo hacen los
confirmatorios--- y por eso el pre-registro lo admite como exploratorio y obliga a
reportarlo marcado como tal.

**No ejecuta ningún ensayo.** Todo sale de dos fuentes ya calculadas: las
configuraciones que cada estrategia inyectó, guardadas en el documento de resultados, y
las rejillas exhaustivas de referencia, cacheadas por pareja (conjunto, semilla). Como
la rejilla contiene el valor de las 144 configuraciones sobre la partición exacta de esa
pareja, se puede saber **qué tan buena era cada configuración inyectada** sin volver a
evaluarla.

Mide dos cosas:

* **Diversidad de lo inyectado** ---distancia de Hamming media entre las cinco
  configuraciones---. Existe para poner a prueba la explicación que el spike A había
  propuesto, según la cual la similitud fallaría por reducir la dispersión. La medición
  la desmiente.
* **Calidad de lo inyectado** sobre el conjunto excluido, en regret relativo al óptimo
  de esa pareja, junto con la fracción de configuraciones inyectadas que allí resultan
  inevaluables.

Ejecutar con:
    .\\.venv\\Scripts\\python.exe scripts\\analyze_e2_mechanism.py
"""

from __future__ import annotations

import json
import logging
import math
import statistics
from collections.abc import Mapping, Sequence
from datetime import date
from itertools import combinations
from pathlib import Path

from autopipe.search_space import (
    TABULAR_SPACE_STEPS,
    TABULAR_SPACE_VERSION,
    build_tabular_search_space,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

RESULTADOS = Path("docs/experimentos/LODO-E2-resultados.json")
CACHE = Path("datasets/.reference_e2")
SALIDA = Path("docs/experimentos/MECANISMO-E2.md")

CONDICIONES = ("nearest", "frequent", "random", "hybrid")


def indice_en_la_rejilla(configuracion: Mapping[str, int]) -> int:
    """Posición de una configuración dentro de la rejilla exhaustiva.

    La rejilla la genera ``GridSearchEngine`` recorriendo los pasos en el orden
    canónico, con la opción de omitir como último índice de cada paso, de modo que su
    orden es el de un número en base mixta. Reproducirlo aquí evita tener que guardar
    la configuración de cada uno de los 144 pipelines.

    Args:
        configuracion: Índice elegido por cada paso.

    Returns:
        La posición dentro de la lista de objetivos de la rejilla.
    """
    espacio = build_tabular_search_space()
    posicion = 0
    for paso in TABULAR_SPACE_STEPS:
        posicion = posicion * (len(espacio[paso]) + 1) + configuracion[paso]
    return posicion


def hamming(una: Mapping[str, int], otra: Mapping[str, int]) -> int:
    """Número de pasos en los que dos configuraciones difieren."""
    return sum(1 for paso in TABULAR_SPACE_STEPS if una.get(paso) != otra.get(paso))


def diversidad(configuraciones: Sequence[Mapping[str, int]]) -> float:
    """Distancia de Hamming media entre todos los pares de un conjunto inyectado.

    Args:
        configuraciones: Las configuraciones que una estrategia inyectó.

    Returns:
        La media sobre todos los pares, o ``0.0`` si hay menos de dos.
    """
    pares = list(combinations(configuraciones, 2))
    return statistics.mean(hamming(a, b) for a, b in pares) if pares else 0.0


def calidad(
    configuraciones: Sequence[Mapping[str, int]], rejilla: Sequence[float], optimo: float
) -> dict:
    """Qué tan buenas son las configuraciones inyectadas sobre el conjunto excluido.

    Args:
        configuraciones: Las configuraciones inyectadas.
        rejilla: Valor de las 144 configuraciones sobre esa pareja.
        optimo: Óptimo de referencia de la pareja.

    Returns:
        Diccionario con el regret medio de las inyectadas, el de la mejor de ellas y la
        fracción que resulta inevaluable. Los dos regrets son ``None`` si ninguna
        configuración inyectada resultó evaluable.

    Raises:
        ValueError: Si el óptimo no es positivo, en cuyo caso el regret no está
            definido.
    """
    if not (math.isfinite(optimo) and optimo > 0):
        raise ValueError(f"El óptimo de referencia es {optimo!r} y debe ser positivo.")
    valores = [rejilla[indice_en_la_rejilla(c)] for c in configuraciones]
    finitos = [v for v in valores if math.isfinite(v)]
    return {
        "regret_medio": (
            statistics.mean((optimo - v) / optimo for v in finitos) if finitos else None
        ),
        "regret_de_la_mejor": (optimo - max(finitos)) / optimo if finitos else None,
        "fraccion_inevaluable": 1 - len(finitos) / len(valores) if valores else 0.0,
    }


def _rejilla(dataset_id: str, semilla: int) -> list[float]:
    """Lee la rejilla exhaustiva cacheada de una pareja.

    Args:
        dataset_id: Identificador del conjunto de datos.
        semilla: Semilla de la partición.

    Returns:
        El valor de las 144 configuraciones.

    Raises:
        SystemExit: Si la rejilla no está cacheada.
    """
    ruta = CACHE / f"{dataset_id}__{TABULAR_SPACE_VERSION}__s{semilla}.json"
    if not ruta.is_file():
        raise SystemExit(
            f"No existe {ruta}. Ejecute antes scripts/reference_optima.py: este "
            f"análisis lee las rejillas, no las recalcula."
        )
    return json.loads(ruta.read_text(encoding="utf-8"))["objetivos"]


def analizar(resultado: dict) -> dict:
    """Recorre los pliegues y agrega las dos medidas por condición.

    Args:
        resultado: Documento que escribió ``run_lodo_e2.py``.

    Returns:
        Diccionario con una entrada por condición.
    """
    filas = {f["dataset_id"]: f for f in resultado["analisis"]["por_dataset"]}
    por_condicion = {}
    for condicion in CONDICIONES:
        hammings, medios, mejores, inevaluables, vistas = [], [], [], [], set()
        for pliegue in resultado["pliegues"]:
            configuraciones = pliegue["recomendaciones"][condicion]["configuraciones"]
            hammings.append(diversidad(configuraciones))
            vistas.update(tuple(sorted(c.items())) for c in configuraciones)
            for semilla in pliegue["semillas"]:
                medida = calidad(
                    configuraciones,
                    _rejilla(pliegue["dataset_id"], semilla["semilla"]),
                    semilla["optimo"],
                )
                inevaluables.append(medida["fraccion_inevaluable"])
                if medida["regret_medio"] is not None:
                    medios.append(medida["regret_medio"])
                    mejores.append(medida["regret_de_la_mejor"])
        por_condicion[condicion] = {
            "hamming_medio": round(statistics.mean(hammings), 4),
            "configuraciones_distintas_en_el_corpus": len(vistas),
            "regret_medio_de_lo_inyectado": round(statistics.mean(medios), 6),
            "regret_de_la_mejor_inyectada": round(statistics.mean(mejores), 6),
            "fraccion_inevaluable": round(statistics.mean(inevaluables), 4),
            "regret_logrado": statistics.median(
                filas[p["dataset_id"]][condicion] for p in resultado["pliegues"]
            ),
        }
    return por_condicion


def redactar(por_condicion: dict, n_pliegues: int) -> str:
    """Redacta el documento con las dos tablas.

    Args:
        por_condicion: Agregados por condición.
        n_pliegues: Cuántos pliegues se recorrieron.

    Returns:
        El documento markdown.
    """
    orden = sorted(por_condicion, key=lambda c: por_condicion[c]["regret_logrado"])
    diversidad_filas = "\n".join(
        f"| `{c}` | {por_condicion[c]['hamming_medio']:.2f} | "
        f"{por_condicion[c]['configuraciones_distintas_en_el_corpus']} | "
        f"{por_condicion[c]['regret_logrado']:.4f} |"
        for c in sorted(por_condicion, key=lambda c: -por_condicion[c]["hamming_medio"])
    )
    calidad_filas = "\n".join(
        f"| `{c}` | {por_condicion[c]['regret_medio_de_lo_inyectado']:.4f} | "
        f"{por_condicion[c]['regret_de_la_mejor_inyectada']:.4f} | "
        f"{por_condicion[c]['fraccion_inevaluable']:.1%} | "
        f"{por_condicion[c]['regret_logrado']:.4f} |"
        for c in orden
    )
    return f"""# Mecanismo del experimento 2 — análisis exploratorio

> **Archivo generado.** No editar a mano: se regenera con
> `.\\.venv\\Scripts\\python.exe scripts\\analyze_e2_mechanism.py`.

> **Exploratorio, no confirmatorio** (pre-registro v2.0 §8.6). Describe el patrón; no
> decide nada sobre la hipótesis. No ejecuta ensayos: lee las configuraciones
> inyectadas y las rejillas exhaustivas ya cacheadas.

- **Generado:** {date.today().isoformat()}
- **Pliegues:** {n_pliegues}

## Diversidad de lo inyectado

Pone a prueba la explicación que el spike A había propuesto, según la cual la similitud
fallaría por reducir la dispersión de lo inyectado. La distancia de Hamming va de 0
—las cinco idénticas— a 4 —difieren en los cuatro pasos—.

| Condición | Hamming medio | Configuraciones distintas en todo el corpus | Regret logrado |
|---|---:|---:|---:|
{diversidad_filas}

## Calidad de lo inyectado sobre el conjunto excluido

Regret de las configuraciones inyectadas respecto del óptimo de la pareja, leído de la
rejilla exhaustiva. La columna que reproduce el orden del resultado final es la de la
**mejor de las cinco**.

| Condición | Regret medio de las 5 | Regret de la mejor de las 5 | Inevaluables | Regret logrado |
|---|---:|---:|---:|---:|
{calidad_filas}

## Lectura

La interpretación de estas dos tablas está en `LODO-E2-discusion.md`, Sección 3.
"""


def main() -> int:
    """Punto de entrada.

    Returns:
        ``0`` si el análisis se produjo.

    Raises:
        SystemExit: Si no existe el documento de resultados.
    """
    if not RESULTADOS.is_file():
        raise SystemExit(f"No existe {RESULTADOS}. Ejecute antes scripts/run_lodo_e2.py.")
    resultado = json.loads(RESULTADOS.read_text(encoding="utf-8"))
    por_condicion = analizar(resultado)
    SALIDA.write_text(redactar(por_condicion, len(resultado["pliegues"])), encoding="utf-8")

    logger.info("Análisis escrito en %s", SALIDA)
    for condicion, medidas in por_condicion.items():
        logger.info(
            "  %-9s hamming %.2f   mejor inyectada %.4f   inevaluables %.1f %%   "
            "regret %.4f",
            condicion,
            medidas["hamming_medio"],
            medidas["regret_de_la_mejor_inyectada"],
            100 * medidas["fraccion_inevaluable"],
            medidas["regret_logrado"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
