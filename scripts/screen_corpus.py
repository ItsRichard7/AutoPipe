# scripts/screen_corpus.py
"""Cribado de sensibilidad al preprocesamiento de los candidatos del corpus (RE4).

Ejecuta el procedimiento que el pre-registro v0.2 fija en su sección de corpus, y
que quedó cerrado **antes** de esta ejecución:

1. Doce pipelines por candidato, extraídos uniformemente al azar del espacio de
   búsqueda canónico —incluida la opción de omitir cada paso— con semilla 4242.
2. Los pipelines inevaluables se descartan y se vuelve a extraer hasta reunir
   doce válidos, registrando cuántos se descartaron.
3. Evaluación con el modelo base congelado, macro-F1 y la misma partición única
   que emplea el protocolo de búsqueda.
4. Medida de sensibilidad: el IQR de los doce valores, más el rango total y el
   margen sobre el pipeline vacío con carácter descriptivo.

**Este script solo mide y ordena.** La selección del corpus, que aplica la regla
de familia y las restricciones de diversidad sobre el orden resultante, se hace
con ``--seleccionar N`` una vez elegido el tamaño.

Ejecutar con:
    .\\.venv\\Scripts\\python.exe scripts\\screen_corpus.py
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import openml
from download_corpus import descargar, leer_candidatos
from sklearn.neighbors import KNeighborsClassifier

from autopipe.evaluator.splitter import DataSplitter
from autopipe.pipeline.pipeline import Pipeline
from autopipe.search_engine.grid_search import evaluate_pipeline_safe
from autopipe.search_space import (
    TABULAR_SPACE_STEPS,
    build_tabular_search_space,
    pipeline_from_configuration,
)
from autopipe.tasks import CLASSIFICATION

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

CANDIDATOS = Path("docs/experimentos/CORPUS-candidatos.md")
VALORES = Path("docs/experimentos/CORPUS-cribado-valores.json")
SELECCION = Path("docs/experimentos/CORPUS-seleccionado.md")

FAMILIAS = {
    # Nivel 1, mismas instancias: cuatro representaciones de los mismos 2000
    # dígitos. Nivel 2, mismo dominio: el resto del reconocimiento de dígitos.
    **{
        nombre: "dígitos manuscritos"
        for nombre in (
            "mfeat-fourier",
            "mfeat-karhunen",
            "mfeat-morphological",
            "mfeat-zernike",
            "optdigits",
            "pendigits",
        )
    },
    # Nivel 2: seis proyectos del repositorio NASA MDP, con esquemas muy
    # similares aunque de proyectos distintos.
    **{
        nombre: "NASA MDP"
        for nombre in ("kc1", "kc2", "pc1", "pc3", "pc4", "jm1")
    },
}
"""Familias de origen, según ``docs/experimentos/CORPUS-criterios-inclusion.md``."""

ENCABEZADO_CORPUS = (
    "| # | OpenML ID | Nombre | IQR | Instancias | Categóricas | Clases "
    "| Faltantes | Descartadas | Licencia |\n"
    "|---:|---:|---|---:|---:|---:|---:|:---:|---:|---|"
)

SEMILLA = 4242
"""Semilla del cribado, fijada en el pre-registro."""

PIPELINES = 12
VAL_SIZE = 0.2
UMBRAL_SENSIBLE = 0.02
MAX_INTENTOS = 200


def _modelo() -> KNeighborsClassifier:
    """Devuelve el modelo base congelado del pre-registro."""
    return KNeighborsClassifier(
        n_neighbors=5, weights="uniform", metric="minkowski", p=2, algorithm="brute"
    )


def _muestrear(generador: np.random.Generator, espacio: dict) -> tuple[int, ...]:
    """Extrae una configuración uniformemente al azar del espacio.

    Cada paso aporta un índice entre ``0`` y ``len(opciones)``; el último valor
    representa omitir el paso, tal como hacen los motores de búsqueda.

    Args:
        generador: Generador aleatorio ya sembrado.
        espacio: Espacio de búsqueda canónico.

    Returns:
        Tupla de índices, uno por paso, en el orden canónico.
    """
    return tuple(
        int(generador.integers(0, len(espacio[paso]) + 1))
        for paso in TABULAR_SPACE_STEPS
    )


def _construir(configuracion: tuple[int, ...], espacio: dict) -> Pipeline:
    """Construye el pipeline que corresponde a una configuración.

    Args:
        configuracion: Índices por paso, en el orden canónico.
        espacio: Espacio de búsqueda canónico.

    Returns:
        El pipeline, sin los pasos omitidos.
    """
    return pipeline_from_configuration(
        dict(zip(TABULAR_SPACE_STEPS, configuracion)), espacio
    )


def _evaluar(pipeline: Pipeline, features, labels, train_idx, val_idx) -> float:
    """Evalúa un pipeline y devuelve su macro-F1.

    Args:
        pipeline: Pipeline a evaluar.
        features: Tabla de características.
        labels: Etiquetas.
        train_idx: Índices de entrenamiento.
        val_idx: Índices de validación.

    Returns:
        El macro-F1, o ``NaN`` si el pipeline no se pudo evaluar.
    """
    valor, _ = evaluate_pipeline_safe(
        pipeline,
        features,
        labels,
        _modelo(),
        "macro_f1",
        train_idx,
        val_idx,
        CLASSIFICATION.build_metric_calculator(0.5),
        task="classification",
    )
    return float(valor)


def cribar(features, labels) -> dict:
    """Aplica el cribado a un dataset y devuelve sus medidas.

    El muestreo es **sin reemplazo**: se extraen doce configuraciones distintas.
    Repetir una configuración daría dos valores idénticos y encogería el IQR de
    forma artificial, que es justo la medida que ordena los candidatos.

    Args:
        features: Tabla de características.
        labels: Etiquetas de clase.

    Returns:
        Diccionario con los doce valores, el IQR, el rango, el margen sobre el
        pipeline vacío y cuántas configuraciones se descartaron.
    """
    generador = np.random.default_rng(SEMILLA)
    train_idx, val_idx = DataSplitter(
        val_size=VAL_SIZE, random_state=SEMILLA
    ).split(labels)

    espacio = build_tabular_search_space()
    valores: list[float] = []
    configuraciones: list[tuple[int, ...]] = []
    vistas: set[tuple[int, ...]] = set()
    descartadas = 0

    for _ in range(MAX_INTENTOS):
        if len(valores) == PIPELINES:
            break
        configuracion = _muestrear(generador, espacio)
        if configuracion in vistas:
            continue
        vistas.add(configuracion)
        valor = _evaluar(
            _construir(configuracion, espacio), features, labels, train_idx, val_idx
        )
        if np.isnan(valor):
            descartadas += 1
            continue
        valores.append(valor)
        configuraciones.append(configuracion)

    vacio = _evaluar(Pipeline([]), features, labels, train_idx, val_idx)
    arreglo = np.array(valores)
    return {
        "valores": [round(v, 6) for v in valores],
        "configuraciones": [list(c) for c in configuraciones],
        "descartadas": descartadas,
        "iqr": round(float(np.subtract(*np.percentile(arreglo, [75, 25]))), 6),
        "rango": round(float(arreglo.max() - arreglo.min()), 6),
        "vacio": None if np.isnan(vacio) else round(float(vacio), 6),
        "margen": None if np.isnan(vacio) else round(float(arreglo.max() - vacio), 6),
    }


def _ejecutar_seleccion(objetivo: int) -> int:
    """Selecciona el corpus a partir de un cribado ya ejecutado.

    Si alguna restricción de diversidad no se cumple, **aborta** en lugar de
    intentar un intercambio automático. El pre-registro contempla desplazar
    candidatos cuando una restricción lo obliga, pero exige registrar cuál y por
    qué; con el tamaño elegido no hace falta ninguno, y prefiero no incorporar
    una lógica de intercambio que nunca llega a ejercitarse.

    Args:
        objetivo: Número de datasets a seleccionar.

    Returns:
        ``0`` si la selección cumple las restricciones, ``1`` si no.
    """
    datos = json.loads(VALORES.read_text(encoding="utf-8"))
    todos = datos["resultados"]
    seleccion, excluidos = seleccionar(todos, objetivo)
    diversidad = verificar_diversidad(seleccion)

    for nombre, (exigido, disponible, cumple) in diversidad.items():
        logger.info(
            "%-34s exigido %2d  en el corpus %2d  %s",
            nombre,
            exigido,
            disponible,
            "ok" if cumple else "NO CUMPLE",
        )
    if not all(cumple for _, _, cumple in diversidad.values()):
        logger.error(
            "\nLa selección de %d no cumple alguna restricción de diversidad. El "
            "pre-registro admite desplazar candidatos, registrando cuál y por qué; "
            "hágalo de forma explícita en lugar de automatizarlo aquí.",
            objetivo,
        )
        return 1

    SELECCION.write_text(
        _informe_seleccion(seleccion, excluidos, todos, diversidad), encoding="utf-8"
    )
    logger.info(
        "\n%d datasets seleccionados de %d candidatos. IQR de %.4f a %.4f. Informe en %s",
        len(seleccion),
        len(todos),
        seleccion[-1]["iqr"],
        seleccion[0]["iqr"],
        SELECCION,
    )
    return 0


def _regimen(instancias: int) -> str:
    """Clasifica el tamaño del dataset en uno de los tres regímenes."""
    if instancias < 1_000:
        return "centenas"
    return "millares" if instancias < 10_000 else "decenas de millares"


def seleccionar(resultados: list[dict], objetivo: int) -> tuple[list[dict], list[dict]]:
    """Aplica la regla de familia sobre el orden por IQR descendente.

    Args:
        resultados: Medidas del cribado, una por candidato.
        objetivo: Número de datasets a seleccionar.

    Returns:
        Tupla ``(seleccionados, excluidos_por_familia)``.
    """
    ordenados = sorted(resultados, key=lambda r: -r["iqr"])
    vistas: set[str] = set()
    elegibles: list[dict] = []
    excluidos: list[dict] = []
    for registro in ordenados:
        familia = FAMILIAS.get(registro["nombre"])
        if familia in vistas:
            excluidos.append({**registro, "familia": familia})
            continue
        if familia:
            vistas.add(familia)
        elegibles.append(registro)
    return elegibles[:objetivo], excluidos


def verificar_diversidad(seleccion: list[dict]) -> dict[str, tuple[int, int, bool]]:
    """Comprueba las restricciones de diversidad pre-registradas.

    Args:
        seleccion: Datasets seleccionados.

    Returns:
        Diccionario de restricción a ``(exigido, disponible, cumple)``.
    """
    regimenes = [_regimen(r["instancias"]) for r in seleccion]
    conteos = {
        "Régimen de centenas": (1, regimenes.count("centenas")),
        "Régimen de millares": (1, regimenes.count("millares")),
        "Régimen de decenas de millares": (1, regimenes.count("decenas de millares")),
        "Con variables categóricas": (5, sum(1 for r in seleccion if r["categoricas"] > 0)),
        "Con valores faltantes": (5, sum(1 for r in seleccion if r["faltantes"])),
    }
    return {
        nombre: (exigido, disponible, disponible >= exigido)
        for nombre, (exigido, disponible) in conteos.items()
    }


def _informe_seleccion(
    seleccion: list[dict], excluidos: list[dict], todos: list[dict], diversidad: dict
) -> str:
    """Redacta el documento con el corpus seleccionado."""
    filas = "\n".join(
        f"| {i} | {r['did']} | `{r['nombre']}` | {r['iqr']:.4f} | {r['instancias']} | "
        f"{r['categoricas']} | {r['clases']} | {'sí' if r['faltantes'] else 'no'} | "
        f"{r['descartadas']} | {r['licencia']} |"
        for i, r in enumerate(seleccion, 1)
    )
    familia = "\n".join(
        f"- `{r['nombre']}` (IQR {r['iqr']:.4f}): ya había un representante de la "
        f"familia «{r['familia']}»"
        for r in excluidos
    ) or "Ninguno."
    restricciones = "\n".join(
        f"| {nombre} | {exigido} | {disponible} | {'sí' if cumple else '**NO**'} |"
        for nombre, (exigido, disponible, cumple) in diversidad.items()
    )
    seleccionados = {r["did"] for r in seleccion}
    fuera = "\n".join(
        f"| {r['did']} | `{r['nombre']}` | {r['iqr']:.4f} | "
        f"{'familia' if any(e['did'] == r['did'] for e in excluidos) else 'orden'} |"
        for r in sorted(todos, key=lambda x: -x["iqr"])
        if r["did"] not in seleccionados
    )
    sensibles = sum(1 for r in seleccion if r["iqr"] >= UMBRAL_SENSIBLE)
    return f"""# Corpus seleccionado (RE4)

> **Archivo generado.** No editar a mano: se regenera con
> `.\\.venv\\Scripts\\python.exe scripts\\screen_corpus.py --seleccionar {len(seleccion)}`.

- **Generado:** {date.today().isoformat()}
- **Procedimiento:** pre-registro v0.2, sección de corpus y cribado de sensibilidad
- **Candidatos cribados:** {len(todos)}
- **Seleccionados:** {len(seleccion)}
- **Semilla del cribado:** {SEMILLA}

## Qué es este documento

La identidad del corpus es un **resultado** de aplicar el procedimiento
pre-registrado, no una regla, y por eso vive aquí y no en el pre-registro. Las
reglas que lo gobiernan quedaron fijadas en la v0.2, cerrada **antes** de esta
ejecución.

Cada candidato recibió doce pipelines extraídos al azar del espacio canónico con
la semilla {SEMILLA}, descartando y volviendo a extraer los inevaluables hasta
reunir doce válidos. La medida de sensibilidad es el IQR de esos doce valores de
macro-F1. Los valores completos, con las configuraciones que los produjeron, están
en `CORPUS-cribado-valores.json`.

## Corpus

{ENCABEZADO_CORPUS}
{filas}

**Subconjunto sensible al preprocesamiento** (IQR ≥ {UMBRAL_SENSIBLE}):
{sensibles} de {len(seleccion)}.

## Restricciones de diversidad

| Restricción | Exigido | En el corpus | ¿Cumple? |
|---|---:|---:|:---:|
{restricciones}

## Excluidos por la regla de familia

Como máximo un dataset por familia de origen, aplicada sobre el orden por IQR y
antes de las restricciones de diversidad.

{familia}

## Candidatos no seleccionados

La columna «motivo» distingue los excluidos por la regla de familia de los que
simplemente quedaron por debajo del corte en el orden por IQR.

| OpenML ID | Nombre | IQR | Motivo |
|---:|---|---:|---|
{fuera}
"""


def main() -> int:
    """Punto de entrada.

    Returns:
        ``0`` si el cribado se completó, ``1`` si la selección no cumple las
        restricciones de diversidad.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Procesar solo los primeros N.")
    parser.add_argument(
        "--seleccionar",
        type=int,
        help="Seleccionar N datasets a partir de un cribado ya ejecutado.",
    )
    args = parser.parse_args()

    if args.seleccionar:
        return _ejecutar_seleccion(args.seleccionar)

    openml.config.set_console_log_level(logging.WARNING)
    logging.getLogger("autopipe").setLevel(logging.ERROR)
    # El criterio supervisado de la selección avisa por cada columna constante.
    # Es una situación esperada y ya tratada —esas columnas quedan de últimas—,
    # y sobre 52 datasets el aviso tapa el resultado.
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    candidatos = leer_candidatos(CANDIDATOS)[: args.limit]
    resultados = []
    inicio = time.perf_counter()

    for numero, ficha in enumerate(candidatos, 1):
        features, labels, licencia, _ = descargar(ficha["did"])
        medidas = cribar(features, labels)
        resultados.append(
            {
                "did": ficha["did"],
                "nombre": ficha["nombre"],
                "instancias": int(len(features)),
                "categoricas": int(
                    features.shape[1] - len(features.select_dtypes(include="number").columns)
                ),
                "clases": int(len(np.unique(labels))),
                "faltantes": bool(features.isna().to_numpy().any()),
                "licencia": licencia,
                **medidas,
            }
        )
        logger.info(
            "[%2d/%d] %-42s IQR %.4f  descartadas %2d",
            numero,
            len(candidatos),
            ficha["nombre"],
            medidas["iqr"],
            medidas["descartadas"],
        )

    VALORES.write_text(
        json.dumps(
            {
                "generado": date.today().isoformat(),
                "semilla": SEMILLA,
                "pipelines_por_dataset": PIPELINES,
                "val_size": VAL_SIZE,
                "umbral_sensible": UMBRAL_SENSIBLE,
                "resultados": resultados,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    sensibles = sum(1 for r in resultados if r["iqr"] >= UMBRAL_SENSIBLE)
    logger.info(
        "\n%d candidatos en %.1f min. Sensibles (IQR >= %.2f): %d. Valores en %s",
        len(resultados),
        (time.perf_counter() - inicio) / 60,
        UMBRAL_SENSIBLE,
        sensibles,
        VALORES,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
