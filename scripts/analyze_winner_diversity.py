# scripts/analyze_winner_diversity.py
"""Análisis exploratorio de la diversidad de los pipelines ganadores (RE4).

Responde tres preguntas cuyas respuestas condicionan el diseño del recomendador
de arranque en caliente, y ninguna de ellas se puede contestar con el conteo de
configuraciones distintas que ya trae la base:

1. **¿Cómo es la diversidad, no solo cuánta hay?** Veinte configuraciones
   distintas entre veinticinco datasets pueden ser veinte puntos dispersos o
   veinte variantes de dos familias. Se mide la distribución marginal del índice
   ganador por paso, su entropía normalizada y la distancia de Hamming entre
   ganadores. Si un paso resulta prácticamente arbitrario, dos configuraciones
   que solo difieren en él son **la misma recomendación**, y el spike A midió que
   inyectar cinco recomendaciones parecidas y equivocadas es peor que no
   recomendar nada.

2. **¿Cuánto ancha es la meseta?** Se evalúan de forma exhaustiva los 144
   pipelines del espacio canónico sobre cada dataset, con la misma semilla y la
   misma partición que construyó la base. De ahí salen dos cosas: cuántas
   configuraciones del espacio alcanzan el 95 % del óptimo —que es literalmente
   el objetivo del *leave-one-dataset-out*— y si el ganador que la base guarda,
   hallado con cincuenta ensayos de Optuna, es de verdad el óptimo del espacio o
   una aproximación.

3. **¿Cambian las meta-características supervivientes según el pliegue?** El
   pre-registro v0.2 dice que el descarte de no finitas se calcula una sola vez
   sobre el corpus; el código de la base lo recalcula por pliegue. Las dos reglas
   no pueden gobernar a la vez el *leave-one-dataset-out*, y aquí se mide si la
   diferencia existe siquiera.

**Exploratorio, no confirmatorio.** El pre-registro declara este análisis como
exploratorio: informa el diseño del recomendador y no puede sostener la
hipótesis del proyecto.

Ejecutar con:
    .\\.venv\\Scripts\\python.exe scripts\\analyze_winner_diversity.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import time
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import openml
import optuna
from download_corpus import descargar
from sklearn.neighbors import KNeighborsClassifier

from autopipe.meta_knowledge import MetaKnowledgeBase
from autopipe.search_engine import GridSearchEngine, OptunaSearchEngine
from autopipe.search_space import (
    TABULAR_SPACE_STEPS,
    TABULAR_SPACE_VERSION,
    build_tabular_search_space,
    canonical_configuration,
    omitted_steps,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

BASE = Path("docs/experimentos/BASE-meta-conocimiento.json")
INFORME = Path("docs/experimentos/DIVERSIDAD-ganadores.md")
VALORES = Path("docs/experimentos/DIVERSIDAD-valores.json")
CACHE_REJILLA = Path("datasets/.diversity")
CACHE_ENSAYOS = Path("datasets/.diversity_trials")

SEMILLA = 42
"""Misma semilla con la que se construyó la base, para que la partición coincida."""

VAL_SIZE = 0.2
METRICA = "macro_f1"

ENSAYOS = 50
"""Presupuesto por dataset de la base, fijado en el anteproyecto."""

FRACCION_OBJETIVO = 0.95
"""Fracción del óptimo que define el objetivo del *leave-one-dataset-out*."""


def _modelo() -> KNeighborsClassifier:
    """Devuelve el modelo base congelado del pre-registro."""
    return KNeighborsClassifier(
        n_neighbors=5, weights="uniform", metric="minkowski", p=2, algorithm="brute"
    )


def _tamanos_de_paso() -> dict[str, int]:
    """Número de opciones de cada paso, sin contar la omisión.

    Returns:
        Diccionario de nombre de paso a número de opciones.
    """
    espacio = build_tabular_search_space()
    return {paso: len(espacio[paso]) for paso in TABULAR_SPACE_STEPS}


def configuracion_desde_indice(indice: int) -> dict[str, int]:
    """Traduce el índice de un pipeline de la rejilla a su configuración por paso.

    :meth:`GridSearchEngine._build_pipelines` genera los candidatos con
    ``itertools.product`` sobre las listas de opciones ampliadas con la omisión,
    de modo que el índice de un candidato es un número en base mixta cuyo dígito
    menos significativo es el último paso. Esta función lo descompone.

    Args:
        indice: Posición del pipeline en la tabla de resultados de la rejilla.

    Returns:
        Índice elegido por cada paso, en el mismo formato que guarda la base de
        meta-conocimiento.
    """
    tamanos = _tamanos_de_paso()
    configuracion: dict[str, int] = {}
    resto = indice
    for paso in reversed(TABULAR_SPACE_STEPS):
        base = tamanos[paso] + 1
        configuracion[paso] = resto % base
        resto //= base
    return configuracion


def indice_desde_configuracion(configuracion: dict[str, int]) -> int:
    """Traduce una configuración por paso a su índice en la rejilla.

    Es la inversa de :func:`configuracion_desde_indice`, y sirve para localizar
    en la rejilla exhaustiva el valor exacto de la configuración que la base
    guarda como ganadora.

    Args:
        configuracion: Índice elegido por cada paso.

    Returns:
        Posición del pipeline en la tabla de resultados de la rejilla.
    """
    tamanos = _tamanos_de_paso()
    indice = 0
    for paso in TABULAR_SPACE_STEPS:
        indice = indice * (tamanos[paso] + 1) + configuracion[paso]
    return indice


def _cache_rejilla(dataset_id: str) -> Path:
    """Ruta de caché de la rejilla exhaustiva de un dataset."""
    return CACHE_REJILLA / f"{dataset_id}__{TABULAR_SPACE_VERSION}__s{SEMILLA}.json"


def rejilla_exhaustiva(dataset_id: str, nombre: str, rehacer: bool = False) -> list[float]:
    """Evalúa los 144 pipelines del espacio canónico sobre un dataset.

    Se usa la misma semilla, la misma proporción de validación y el mismo modelo
    base que construyeron la base de meta-conocimiento, de modo que los valores
    son directamente comparables con el ganador guardado.

    Args:
        dataset_id: Identificador de OpenML del dataset.
        nombre: Nombre legible, solo para los mensajes.
        rehacer: Si se ignora la caché y se recalcula.

    Returns:
        Valor de macro-F1 de cada pipeline, en el orden de la rejilla. Los
        pipelines inevaluables aparecen como ``NaN``.
    """
    ruta = _cache_rejilla(dataset_id)
    if ruta.is_file() and not rehacer:
        return json.loads(ruta.read_text(encoding="utf-8"))["objetivos"]

    features, labels, _, _ = descargar(int(dataset_id))
    motor = GridSearchEngine(
        build_tabular_search_space(),
        val_size=VAL_SIZE,
        random_state=SEMILLA,
        task="classification",
    )
    _, _, resultados = motor.run(features, labels, _modelo(), metric=METRICA)
    objetivos = [float(r["objective"]) for r in resultados]

    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(
        json.dumps(
            {"dataset_id": dataset_id, "dataset_name": nombre, "objetivos": objetivos},
            indent=2,
        ),
        encoding="utf-8",
    )
    return objetivos


def _cache_ensayos(dataset_id: str) -> Path:
    """Ruta de caché de los 50 ensayos de un dataset."""
    return CACHE_ENSAYOS / f"{dataset_id}__{TABULAR_SPACE_VERSION}__s{SEMILLA}.json"


def empates_en_el_presupuesto(dataset_id: str, rehacer: bool = False) -> list[dict[str, int]]:
    """Configuraciones que empatan en el mejor valor **dentro de los 50 ensayos**.

    Es el empate que el arnés de construcción puede observar sin salirse del
    presupuesto que fija el anteproyecto. La rejilla exhaustiva de 144 pipelines
    que usa el resto de este análisis mide el empate *real* del espacio, pero
    usarla para elegir el ganador equivaldría a construir la base por búsqueda
    exhaustiva y no con cincuenta ensayos, que es lo pre-registrado.

    Args:
        dataset_id: Identificador de OpenML del dataset.
        rehacer: Si se ignora la caché y se recalcula.

    Returns:
        Configuraciones empatadas en el mejor valor de la corrida.
    """
    ruta = _cache_ensayos(dataset_id)
    if ruta.is_file() and not rehacer:
        return json.loads(ruta.read_text(encoding="utf-8"))["empatadas"]

    features, labels, _, _ = descargar(int(dataset_id))
    motor = OptunaSearchEngine(
        build_tabular_search_space(),
        n_trials=ENSAYOS,
        val_size=VAL_SIZE,
        random_state=SEMILLA,
        task="classification",
    )
    motor.run(features, labels, _modelo(), metric=METRICA)
    empatadas = motor.tied_configurations_ or []

    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(
        json.dumps({"dataset_id": dataset_id, "empatadas": empatadas}, indent=2),
        encoding="utf-8",
    )
    return empatadas


def impacto_de_canonizar(base: MetaKnowledgeBase, rehacer: bool = False) -> dict:
    """Mide qué cambiaría canonizar los empates dentro del presupuesto.

    Args:
        base: Base de meta-conocimiento ya cargada.
        rehacer: Si se ignoran las cachés de las corridas.

    Returns:
        Diccionario con las cifras antes y después de canonizar.
    """
    datasets = []
    for numero, entrada in enumerate(base, 1):
        empatadas = empates_en_el_presupuesto(entrada.dataset_id, rehacer)
        canonica = canonical_configuration(empatadas)
        datasets.append(
            {
                "dataset_id": entrada.dataset_id,
                "dataset_name": entrada.dataset_name,
                "n_empatadas_en_el_presupuesto": len(empatadas),
                "configuracion_canonica": canonica,
                "difiere_de_la_base": canonica != entrada.configuration,
                "omite_la_base": omitted_steps(entrada.configuration),
                "omite_la_canonica": omitted_steps(canonica),
            }
        )
        logger.info(
            "[%2d/%d] %-42s empatan en los %d ensayos: %2d%s",
            numero,
            len(base),
            entrada.dataset_name,
            ENSAYOS,
            len(empatadas),
            "  cambia" if datasets[-1]["difiere_de_la_base"] else "",
        )

    canonicas = {tuple(sorted(d["configuracion_canonica"].items())) for d in datasets}
    actuales = {tuple(sorted(e.configuration.items())) for e in base}
    return {
        "n_cambian": sum(1 for d in datasets if d["difiere_de_la_base"]),
        "n_distintas_actual": len(actuales),
        "n_distintas_canonica": len(canonicas),
        "empate_mediano_en_el_presupuesto": int(
            np.median([d["n_empatadas_en_el_presupuesto"] for d in datasets])
        ),
        "omiten_imputacion_actual": sum(1 for d in datasets if "imputation" in d["omite_la_base"]),
        "omiten_imputacion_canonica": sum(
            1 for d in datasets if "imputation" in d["omite_la_canonica"]
        ),
        "omiten_codificacion_actual": sum(1 for d in datasets if "encoding" in d["omite_la_base"]),
        "omiten_codificacion_canonica": sum(
            1 for d in datasets if "encoding" in d["omite_la_canonica"]
        ),
        "datasets": datasets,
    }


def diversidad_estructural(base: MetaKnowledgeBase) -> dict:
    """Describe la forma de la diversidad de los ganadores.

    Args:
        base: Base de meta-conocimiento ya cargada.

    Returns:
        Diccionario con el número de configuraciones distintas, la distribución
        y la entropía normalizada de cada paso, y el histograma de distancias de
        Hamming entre pares de ganadores.
    """
    tamanos = _tamanos_de_paso()
    configuraciones = [entrada.configuration for entrada in base]
    distintas = {tuple(sorted(c.items())) for c in configuraciones}

    pasos: dict[str, dict] = {}
    for paso, n_opciones in tamanos.items():
        conteo = {indice: 0 for indice in range(n_opciones + 1)}
        for configuracion in configuraciones:
            conteo[configuracion[paso]] += 1
        total = len(configuraciones)
        entropia = -sum(
            (c / total) * math.log(c / total) for c in conteo.values() if c > 0
        )
        pasos[paso] = {
            "conteo": conteo,
            "indice_omision": n_opciones,
            "entropia_normalizada": round(entropia / math.log(n_opciones + 1), 4),
        }

    distancias = [
        sum(1 for paso in TABULAR_SPACE_STEPS if a[paso] != b[paso])
        for a, b in itertools.combinations(configuraciones, 2)
    ]
    histograma = {d: distancias.count(d) for d in range(len(TABULAR_SPACE_STEPS) + 1)}

    return {
        "n_datasets": len(base),
        "n_configuraciones_distintas": len(distintas),
        "pasos": pasos,
        "hamming_media": round(float(np.mean(distancias)), 4),
        "hamming_histograma": histograma,
        "n_pares": len(distancias),
    }


def estabilidad_supervivientes(base: MetaKnowledgeBase) -> dict:
    """Mide si las meta-características supervivientes cambian según el pliegue.

    Para cada dataset se calcula el conjunto de supervivientes sobre los otros
    veinticuatro —que es lo que vería un pliegue del *leave-one-dataset-out*— y
    se compara con el del corpus completo. De las que el pliegue conserva de más,
    se cuentan las que resultan no finitas en el dataset excluido: son las que la
    intersección por consulta tendría que descartar.

    Args:
        base: Base de meta-conocimiento ya cargada.

    Returns:
        Diccionario con el número de supervivientes del corpus completo y, por
        pliegue, cuántas conserva de más y cuántas de esas caen al intersecar.
    """
    del_corpus = set(base.surviving_features())
    pliegues = []
    for entrada in base:
        del_pliegue = set(base.without(entrada.dataset_id).surviving_features())
        de_mas = sorted(del_pliegue - del_corpus)
        no_finitas = sorted(
            nombre for nombre in de_mas if not np.isfinite(entrada.meta_features[nombre])
        )
        pliegues.append(
            {
                "dataset_id": entrada.dataset_id,
                "dataset_name": entrada.dataset_name,
                "n_supervivientes": len(del_pliegue),
                "n_de_mas": len(de_mas),
                "de_mas": de_mas,
                "n_descartadas_al_intersecar": len(no_finitas),
            }
        )
    return {
        "n_supervivientes_corpus": len(del_corpus),
        "n_pliegues_que_difieren": sum(1 for p in pliegues if p["n_de_mas"] > 0),
        "pliegues": pliegues,
    }


def anchura_de_meseta(base: MetaKnowledgeBase, rehacer: bool = False) -> dict:
    """Mide, dataset por dataset, cuántas configuraciones alcanzan el objetivo.

    El objetivo es el mismo que define el protocolo: el 95 % del mejor valor de
    macro-F1. Aquí se calcula sobre el óptimo exacto del espacio, obtenido por
    rejilla exhaustiva, y no sobre el mejor de una corrida.

    Args:
        base: Base de meta-conocimiento ya cargada.
        rehacer: Si se ignora la caché de las rejillas.

    Returns:
        Diccionario con una entrada por dataset y los agregados del corpus.
    """
    n_total = math.prod(n + 1 for n in _tamanos_de_paso().values())
    datasets = []
    for numero, entrada in enumerate(base, 1):
        inicio = time.perf_counter()
        objetivos = rejilla_exhaustiva(entrada.dataset_id, entrada.dataset_name, rehacer)
        valores = np.asarray(objetivos, dtype=float)
        finitos = valores[np.isfinite(valores)]
        optimo = float(finitos.max())
        umbral = FRACCION_OBJETIVO * optimo
        en_meseta = int((finitos >= umbral).sum())
        # Cuántas configuraciones **empatan** en el óptimo. El ganador que guarda
        # la base es un miembro cualquiera de este conjunto: si tiene más de un
        # elemento, la etiqueta que el recomendador aprende es arbitraria dentro
        # del empate, no una propiedad del dataset.
        empatadas = [
            configuracion_desde_indice(i)
            for i, v in enumerate(valores)
            if np.isfinite(v) and math.isclose(float(v), optimo, abs_tol=1e-6)
        ]
        # Un empate puede ser de dos naturalezas muy distintas. Sobre un dataset
        # numérico y sin faltantes, la imputación y la codificación son la
        # identidad, de modo que 3 x 3 = 9 configuraciones son literalmente el
        # mismo pipeline: es degeneración estructural del espacio, no un empate
        # de la métrica. Si en cambio el empate abarca varios pares
        # (escalado, selección), son pipelines de verdad distintos que dan el
        # mismo valor. Contar los pares distingue ambos casos.
        pares = {(c["scaling"], c["selection"]) for c in empatadas}
        canonica = canonical_configuration(empatadas)
        valor_de_la_base = valores[indice_desde_configuracion(entrada.configuration)]
        base_es_optima = bool(
            np.isfinite(valor_de_la_base)
            and math.isclose(float(valor_de_la_base), optimo, abs_tol=1e-6)
        )

        datasets.append(
            {
                "dataset_id": entrada.dataset_id,
                "dataset_name": entrada.dataset_name,
                "n_evaluables": int(finitos.size),
                "optimo_rejilla": round(optimo, 6),
                "optimo_optuna": entrada.metric_value,
                "brecha": round(optimo - entrada.metric_value, 6),
                "n_empatadas_en_el_optimo": len(empatadas),
                "n_pares_escalado_seleccion_empatados": len(pares),
                "configuracion_canonica": canonica,
                "canonica_difiere_de_la_base": canonica != entrada.configuration,
                "omite_la_base": omitted_steps(entrada.configuration),
                "omite_la_canonica": omitted_steps(canonica),
                "ensayos_esperados_al_azar": round(n_total / en_meseta, 2),
                "configuracion_de_la_base_es_optima": base_es_optima,
                "valor_de_la_configuracion_de_la_base": round(float(valor_de_la_base), 6),
                "n_en_meseta": en_meseta,
                "fraccion_en_meseta": round(en_meseta / n_total, 4),
                "segundos": round(time.perf_counter() - inicio, 1),
            }
        )
        logger.info(
            "[%2d/%d] %-42s meseta %3d/%d  empatan %2d  optimo %.4f  base %.4f%s",
            numero,
            len(base),
            entrada.dataset_name,
            en_meseta,
            n_total,
            len(empatadas),
            optimo,
            entrada.metric_value,
            "" if base_es_optima else "  <- la base no lo halla",
        )

    en_meseta = [d["n_en_meseta"] for d in datasets]
    empates = [d["n_empatadas_en_el_optimo"] for d in datasets]
    esperados = [d["ensayos_esperados_al_azar"] for d in datasets]
    canonicas = [tuple(sorted(d["configuracion_canonica"].items())) for d in datasets]
    return {
        "esperado_mediano": round(float(np.median(esperados)), 2),
        "n_esperado_hasta_2": sum(1 for e in esperados if e <= 2),
        "n_esperado_desde_10": sum(1 for e in esperados if e >= 10),
        "n_canonica_difiere": sum(1 for d in datasets if d["canonica_difiere_de_la_base"]),
        "n_configuraciones_canonicas_distintas": len(set(canonicas)),
        "omiten_imputacion_base": sum(1 for d in datasets if "imputation" in d["omite_la_base"]),
        "omiten_imputacion_canonica": sum(
            1 for d in datasets if "imputation" in d["omite_la_canonica"]
        ),
        "omiten_codificacion_base": sum(1 for d in datasets if "encoding" in d["omite_la_base"]),
        "omiten_codificacion_canonica": sum(
            1 for d in datasets if "encoding" in d["omite_la_canonica"]
        ),
        "n_pipelines_del_espacio": n_total,
        "fraccion_objetivo": FRACCION_OBJETIVO,
        "meseta_mediana": int(np.median(en_meseta)),
        "meseta_minima": int(min(en_meseta)),
        "meseta_maxima": int(max(en_meseta)),
        "empate_mediano": int(np.median(empates)),
        "empate_maximo": int(max(empates)),
        "n_con_optimo_unico": sum(1 for e in empates if e == 1),
        "n_empate_solo_estructural": sum(
            1 for d in datasets if d["n_pares_escalado_seleccion_empatados"] == 1
        ),
        "n_base_es_optima": sum(
            1 for d in datasets if d["configuracion_de_la_base_es_optima"]
        ),
        "datasets": datasets,
    }


def _tabla_pasos(estructura: dict) -> str:
    """Redacta la tabla de distribución del índice ganador por paso.

    Args:
        estructura: Salida de :func:`diversidad_estructural`.

    Returns:
        Las filas de la tabla en markdown.
    """
    filas = []
    for paso, datos in estructura["pasos"].items():
        omision = datos["indice_omision"]
        conteo = ", ".join(
            f"`{indice}`{'(omitir)' if indice == omision else ''}: {n}"
            for indice, n in datos["conteo"].items()
        )
        filas.append(f"| `{paso}` | {conteo} | {datos['entropia_normalizada']:.3f} |")
    return "\n".join(filas)


def _tabla_meseta(meseta: dict) -> str:
    """Redacta la tabla por dataset del análisis de meseta.

    Args:
        meseta: Salida de :func:`anchura_de_meseta`.

    Returns:
        Las filas de la tabla en markdown.
    """
    return "\n".join(
        f"| {d['dataset_id']} | `{d['dataset_name']}` | {d['n_evaluables']} | "
        f"{d['optimo_rejilla']:.4f} | {d['n_empatadas_en_el_optimo']} | "
        f"{d['n_pares_escalado_seleccion_empatados']} | "
        f"{'sí' if d['configuracion_de_la_base_es_optima'] else '**no**'} | "
        f"{d['n_en_meseta']} | {100 * d['fraccion_en_meseta']:.1f} % | "
        f"{d['ensayos_esperados_al_azar']:.1f} |"
        for d in meseta["datasets"]
    )


def _informe(
    estructura: dict, meseta: dict, estabilidad: dict, canonizar: dict
) -> str:
    """Redacta el documento con los análisis.

    Args:
        estructura: Salida de :func:`diversidad_estructural`.
        meseta: Salida de :func:`anchura_de_meseta`.
        estabilidad: Salida de :func:`estabilidad_supervivientes`.
        canonizar: Salida de :func:`impacto_de_canonizar`.

    Returns:
        El documento markdown completo.
    """
    n_total = meseta["n_pipelines_del_espacio"]
    n_datasets = estructura["n_datasets"]
    n_distintas = estructura["n_configuraciones_distintas"]
    empate_mediano = meseta["empate_mediano"]
    empate_maximo = meseta["empate_maximo"]
    n_unico = meseta["n_con_optimo_unico"]
    n_canon = meseta["n_configuraciones_canonicas_distintas"]
    omite_imp = (meseta["omiten_imputacion_base"], meseta["omiten_imputacion_canonica"])
    omite_cod = (meseta["omiten_codificacion_base"], meseta["omiten_codificacion_canonica"])
    esperado = meseta["esperado_mediano"]
    can_dist = (canonizar["n_distintas_actual"], canonizar["n_distintas_canonica"])
    can_imp = (
        canonizar["omiten_imputacion_actual"],
        canonizar["omiten_imputacion_canonica"],
    )
    can_cod = (
        canonizar["omiten_codificacion_actual"],
        canonizar["omiten_codificacion_canonica"],
    )
    veredicto_canonico = (
        "**La base ya es canónica**: ninguna de sus entradas cambiaría al\n"
        "aplicar la regla. La comprobación no es una formalidad, porque el\n"
        "desempate lo aplica el arnés de construcción y esta tabla se calcula por\n"
        "separado a partir del artefacto guardado: si los dos se separaran, la\n"
        "base mezclaría dos criterios sin que nada fallara."
        if canonizar["n_cambian"] == 0
        else f"**La base no es canónica**: {canonizar['n_cambian']} de\n"
        f"{n_datasets} entradas cambiarían al aplicar la regla. Hay que\n"
        f"reconstruirla con `build_meta_knowledge.py` antes de usarla."
    )
    hamming = ", ".join(
        f"{d}: {n}" for d, n in estructura["hamming_histograma"].items() if n
    )
    difieren = [p for p in estabilidad["pliegues"] if p["n_de_mas"] > 0]
    if not difieren:
        veredicto_estabilidad = (
            f"Ninguno de los {len(estabilidad['pliegues'])} pliegues conserva "
            f"meta-características que el corpus completo descarte, de modo que "
            f"**las dos reglas coinciden sobre este corpus** y la elección entre "
            f"ellas no cambia ningún número."
        )
    else:
        detalle = "\n".join(
            f"- Pliegue que excluye `{p['dataset_name']}`: conserva "
            f"{p['n_de_mas']} de más\n"
            f"  ({', '.join(f'`{n}`' for n in p['de_mas'])}), de las cuales "
            f"{p['n_descartadas_al_intersecar']} son no finitas\n"
            f"  en el propio dataset excluido y caerían al intersecar."
            for p in difieren
        )
        verbo = "pliegue conserva" if len(difieren) == 1 else "pliegues conservan"
        veredicto_estabilidad = (
            f"**Las dos reglas no coinciden.** {len(difieren)} de "
            f"{len(estabilidad['pliegues'])} {verbo}\n"
            f"meta-características que el corpus completo descarta:\n\n{detalle}\n\n"
            f"La regla por pliegue es la que no mira el dataset excluido, pero exige\n"
            f"la intersección con las finitas de la consulta; la del corpus completo\n"
            f"usa el patrón de finitud del excluido para decidir qué columnas existen."
        )
    return f"""# Diversidad de los pipelines ganadores (RE4)

> **Archivo generado.** No editar a mano: se regenera con
> `.\\.venv\\Scripts\\python.exe scripts\\analyze_winner_diversity.py`.

- **Generado:** {date.today().isoformat()}
- **Datasets:** {estructura["n_datasets"]}
- **Versión del espacio:** `{TABULAR_SPACE_VERSION}` ({n_total} pipelines)
- **Semilla:** {SEMILLA} — la misma de la base, de modo que la partición coincide
- **Valores completos:** `DIVERSIDAD-valores.json`

**Análisis exploratorio.** El pre-registro lo declara como tal: informa el diseño
del recomendador y **no puede sostener la hipótesis** del proyecto.

## 1. Forma de la diversidad

**{n_distintas} configuraciones distintas entre {n_datasets} datasets.**
El conteo dice que hay diversidad; esta sección dice de qué tipo.

### Distribución del índice ganador por paso

La entropía normalizada vale 1 cuando todas las opciones del paso ganan por igual
y 0 cuando siempre gana la misma. Un paso de entropía alta discrimina entre
datasets; uno de entropía baja aporta poco a la identidad de una recomendación.

| Paso | Índice ganador: veces | Entropía normalizada |
|---|---|---:|
{_tabla_pasos(estructura)}

### Distancia entre ganadores

Distancia de Hamming sobre los cuatro pasos, en los {estructura["n_pares"]} pares
de datasets. Media **{estructura["hamming_media"]}** de un máximo de 4.

Pares por distancia — {hamming}

Importa porque el modo de fallo que midió el spike A es inyectar cinco
configuraciones parecidas entre sí y todas equivocadas: TPE las lee como una
región buena y deja de explorar. Cuanto más se parezcan entre sí los ganadores de
datasets vecinos, más probable es ese modo de fallo y más razón hay para exigir
diversidad entre las cinco configuraciones recomendadas.

## 2. Anchura de la meseta

Los {n_total} pipelines del espacio evaluados de forma exhaustiva sobre cada
dataset, con la misma semilla y la misma partición que construyeron la base. La
columna «meseta» cuenta cuántos alcanzan el {100 * meseta["fraccion_objetivo"]:.0f} % del
óptimo, que es **la definición del objetivo del protocolo**: cuantos más haya,
más fácil le resulta al arranque en frío tropezarse con uno al azar y menos
margen le queda al arranque en caliente.

- Mediana de configuraciones en meseta: **{meseta["meseta_mediana"]} de {n_total}**
- Rango: de {meseta["meseta_minima"]} a {meseta["meseta_maxima"]}
- La configuración que la base guarda como ganadora alcanza el óptimo exacto del
  espacio en **{meseta["n_base_es_optima"]} de {n_datasets}** datasets

### El óptimo casi nunca es único

La columna «empatan» cuenta cuántas configuraciones del espacio alcanzan
**exactamente** el óptimo. Mediana **{empate_mediano}**, máximo {empate_maximo},
y solo **{n_unico} de {n_datasets}** datasets tienen un óptimo único.

Es el hallazgo que más condiciona el recomendador: cuando varias configuraciones
empatan, la que la base guarda es **un miembro cualquiera del empate**, elegido
por el orden en que el muestreador las visitó. El recomendador estaría aprendiendo
a predecir esa elección arbitraria y no una propiedad del dataset, y dos datasets
gemelos podrían aparecer con ganadores distintos sin que nada los distinga.

La columna «pares» dice de qué naturaleza es el empate. Cuenta cuántos pares
distintos *(escalado, selección)* aparecen entre las configuraciones empatadas:

* **Un solo par** significa que el empate es **degeneración estructural del
  espacio**, no un empate de la métrica. Sobre un dataset numérico y sin
  faltantes, la imputación y la codificación son la identidad, de modo que las
  3 × 3 = 9 combinaciones de esos dos pasos son literalmente el mismo pipeline.
  El dato no dice que dé igual preprocesar: dice que en ese dataset esos dos
  pasos no tienen nada que hacer.
* **Varios pares** significa que pipelines de verdad distintos alcanzan el mismo
  valor.

Ocurre en **{meseta["n_empate_solo_estructural"]} de {n_datasets}** datasets que todo
el empate se explica por un solo par.

| ID | Nombre | Eval. | Óptimo | Empatan | Pares | Base ópt. | Meseta | % esp. | E |
|---:|---|---:|---:|---:|---:|:---:|---:|---:|---:|
{_tabla_meseta(meseta)}

### Efecto piso sobre «ensayos hasta el objetivo»

La columna **E** es el número esperado de ensayos **aleatorios** hasta caer en la
meseta, es decir {n_total} dividido entre las configuraciones que la alcanzan.

- Mediana de E: **{esperado}**
- Datasets con E ≤ 2: **{meseta["n_esperado_hasta_2"]}** — el arranque en frío
  acierta casi de inmediato
- Datasets con E ≥ 10: **{meseta["n_esperado_desde_10"]}**

Con `n_startup_trials` en 10, el arranque en frío suele cruzar el objetivo
**dentro de su propia fase aleatoria**, antes de que TPE empiece a modelar. Y una
reducción del 30 % sobre una mediana de {esperado} ensayos exige bajar a
{0.7 * esperado:.1f}, con los cinco ensayos inyectados contando dentro.

**E es una cota conservadora, y lo es en la dirección que perjudica al arranque en
caliente.** El objetivo del protocolo no es el 95 % del óptimo del espacio sino el
95 % del mejor valor que el propio arranque en frío alcanza en treinta ensayos,
que es menor o igual. El conjunto de configuraciones que califican es por tanto
mayor que la meseta medida aquí, y el frío acierta aún antes.

Es un análisis exploratorio y **no es un resultado del experimento**: describe el
espacio, no mide las dos condiciones. Su lugar es la decisión abierta sobre la
región inevaluable y la variable «ensayos hasta el objetivo», que la v1.0 del
pre-registro debe cerrar antes de ejecutar nada.

### Desempate entre configuraciones que empatan

La base guarda, entre las que empatan en el mejor valor, **la que omite menos
pasos y, a igualdad, la de índices menores**, y no la que el muestreador visitó
primero. Sin esa regla, la configuración guardada sería una propiedad de la
corrida y no del dataset.

El desempate se calcula **entre los empates observados en los {ENSAYOS} ensayos**,
no sobre la rejilla exhaustiva. Elegir el ganador mirando los {n_total} pipelines
equivaldría a construir la base por búsqueda exhaustiva, y el anteproyecto fija
{ENSAYOS} ensayos por dataset. Mediana de configuraciones empatadas dentro del
presupuesto: **{canonizar["empate_mediano_en_el_presupuesto"]}**.

{veredicto_canonico}

| | En la base | Canónica |
|---|---:|---:|
| Entradas que difieren | — | {canonizar["n_cambian"]} de {n_datasets} |
| Configuraciones distintas | {can_dist[0]} | {can_dist[1]} |
| Omiten la imputación | {can_imp[0]} | {can_imp[1]} |
| Omiten la codificación | {can_cod[0]} | {can_cod[1]} |

Las dos últimas filas son las que importan para el arranque en caliente: un
ganador que omite la imputación es **inevaluable** en un dataset receptor con
valores faltantes, y uno que omite la codificación lo es en uno con columnas
categóricas. Cada recomendación así gastaría un ensayo del presupuesto sin
producir una métrica.

Como referencia, desempatar sobre la rejilla exhaustiva —que **no** es
implementable dentro del presupuesto— dejaría {n_canon} configuraciones distintas,
con {omite_imp[1]} omisiones de imputación y {omite_cod[1]} de codificación. Es la
cota de lo que el desempate puede llegar a corregir.

La contrapartida hay que decirla: la regla reduce las recomendaciones que caen en
la región inevaluable, y eso **refuerza** la asimetría que el pre-registro dejó
abierta, porque el arranque en frío sí gasta ensayos ahí. Debe declararse en la
v1.0 y controlarse con la ablación de recomendador aleatorio.

## 3. Estabilidad de las meta-características supervivientes

El pre-registro v0.2 dice que el descarte de no finitas se calcula **una sola vez
sobre el corpus definitivo**; el código de `autopipe/meta_knowledge/` lo recalcula
**por pliegue**. Las dos reglas no pueden gobernar a la vez el
*leave-one-dataset-out*, y la diferencia entre ellas es medible.

- Supervivientes sobre el corpus completo: **{estabilidad["n_supervivientes_corpus"]}**
- Pliegues cuyo conjunto difiere: **{len(difieren)} de {len(estabilidad["pliegues"])}**

{veredicto_estabilidad}
"""


def main() -> int:
    """Punto de entrada.

    Returns:
        ``0`` si el análisis se completó.
    """
    parser = argparse.ArgumentParser(description="Diversidad de los ganadores.")
    parser.add_argument(
        "--rehacer", action="store_true", help="Ignorar la caché de las rejillas."
    )
    args = parser.parse_args()

    openml.config.set_console_log_level(logging.WARNING)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logging.getLogger("autopipe").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    if not BASE.is_file():
        raise SystemExit(
            f"No existe {BASE}. Ejecute antes scripts/build_meta_knowledge.py."
        )
    base = MetaKnowledgeBase.load(BASE)
    inicio = time.perf_counter()

    estructura = diversidad_estructural(base)
    estabilidad = estabilidad_supervivientes(base)
    meseta = anchura_de_meseta(base, args.rehacer)
    canonizar = impacto_de_canonizar(base, args.rehacer)

    VALORES.write_text(
        json.dumps(
            {
                "estructura": estructura,
                "meseta": meseta,
                "estabilidad": estabilidad,
                "canonizar": canonizar,
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    INFORME.write_text(
        _informe(estructura, meseta, estabilidad, canonizar), encoding="utf-8"
    )
    logger.info(
        "\nAnálisis completo en %.1f min. Informe en %s",
        (time.perf_counter() - inicio) / 60,
        INFORME,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
