# scripts/list_corpus_candidates.py
"""Genera la lista candidata de datasets para el corpus de meta-conocimiento (RE4).

Parte de la suite curada **OpenML-CC18** y aplica los criterios de inclusión de la
Sección 4.3.1 del anteproyecto. Escribe una ficha por dataset superviviente en
``docs/experimentos/CORPUS-candidatos.md``.

Los criterios y las decisiones de procedimiento están documentados en
``docs/experimentos/CORPUS-criterios-inclusion.md``; este script solo los ejecuta.

Requiere acceso a la API de OpenML.

Ejecutar con:
    .\\.venv\\Scripts\\python.exe scripts\\list_corpus_candidates.py
"""

from __future__ import annotations

import warnings
from datetime import date
from pathlib import Path

import openml
import pandas as pd

SUITE = "OpenML-CC18"
SALIDA = Path("docs/experimentos/CORPUS-candidatos.md")

# Criterios de inclusión de la Sección 4.3.1 del anteproyecto.
MIN_INSTANCIAS = 500
MAX_INSTANCIAS = 50_000
MAX_CARACTERISTICAS = 100
MIN_CLASES = 2
MAX_CLASES = 10

COLUMNAS = [
    "did",
    "name",
    "NumberOfInstances",
    "NumberOfFeatures",
    "NumberOfClasses",
    "NumberOfInstancesWithMissingValues",
    "NumberOfSymbolicFeatures",
    "MinorityClassSize",
    "MajorityClassSize",
]


def _descargar() -> pd.DataFrame:
    """Descarga los metadatos de los datasets de la suite.

    Returns:
        Tabla con una fila por dataset de la suite.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        suite = openml.study.get_suite(SUITE)
        return openml.datasets.list_datasets(
            data_id=list(suite.data), output_format="dataframe"
        )


def _enriquecer(datos: pd.DataFrame) -> pd.DataFrame:
    """Añade las columnas derivadas que necesita la ficha.

    OpenML cuenta la columna objetivo dentro de ``NumberOfFeatures`` y dentro de
    ``NumberOfSymbolicFeatures``, de modo que ambas se corrigen para contar solo
    las variables predictoras.

    Args:
        datos: Metadatos crudos de OpenML.

    Returns:
        La misma tabla con las columnas derivadas añadidas.
    """
    datos = datos.copy()
    datos["predictoras"] = datos["NumberOfFeatures"] - 1
    datos["categoricas"] = (datos["NumberOfSymbolicFeatures"] - 1).clip(lower=0)
    datos["faltantes"] = datos["NumberOfInstancesWithMissingValues"] > 0
    datos["desbalance"] = datos["MajorityClassSize"] / datos["MinorityClassSize"]
    return datos


def _filtrar(datos: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Aplica los criterios de inclusión y contabiliza los descartes.

    Args:
        datos: Metadatos enriquecidos.

    Returns:
        Tupla ``(supervivientes, descartes)`` donde ``descartes`` cuenta cuántos
        datasets incumple cada criterio, de forma independiente.
    """
    criterios = {
        "instancias fuera de [500, 50 000]": ~datos["NumberOfInstances"].between(
            MIN_INSTANCIAS, MAX_INSTANCIAS
        ),
        "más de 100 características": datos["predictoras"] > MAX_CARACTERISTICAS,
        "número de clases fuera de [2, 10]": ~datos["NumberOfClasses"].between(
            MIN_CLASES, MAX_CLASES
        ),
    }
    descartes = {nombre: int(mascara.sum()) for nombre, mascara in criterios.items()}

    excluidos = criterios["instancias fuera de [500, 50 000]"]
    for mascara in criterios.values():
        excluidos = excluidos | mascara
    return datos[~excluidos].sort_values("NumberOfInstances"), descartes


def _regimen(instancias: float) -> str:
    """Clasifica el tamaño del dataset en uno de los tres regímenes.

    Args:
        instancias: Número de instancias.

    Returns:
        ``"centenas"``, ``"millares"`` o ``"decenas de millares"``.
    """
    if instancias < 1_000:
        return "centenas"
    if instancias < 10_000:
        return "millares"
    return "decenas de millares"


def _tabla_markdown(candidatos: pd.DataFrame) -> str:
    """Construye la tabla de fichas en Markdown.

    Args:
        candidatos: Datasets que superaron los criterios.

    Returns:
        La tabla como texto Markdown.
    """
    encabezado = (
        "| OpenML ID | Nombre | Instancias | Predictoras | Categóricas | Clases "
        "| Faltantes | Desbalance | Régimen |\n"
        "|---:|---|---:|---:|---:|---:|:---:|---:|---|"
    )
    filas = [
        f"| {int(fila.did)} | `{fila.name}` | {int(fila.NumberOfInstances)} "
        f"| {int(fila.predictoras)} | {int(fila.categoricas)} "
        f"| {int(fila.NumberOfClasses)} | {'sí' if fila.faltantes else 'no'} "
        f"| {fila.desbalance:.1f} | {_regimen(fila.NumberOfInstances)} |"
        for fila in candidatos.itertuples()
    ]
    return "\n".join([encabezado, *filas])


def _resumen(candidatos: pd.DataFrame) -> str:
    """Resume la cobertura de las restricciones de diversidad del pre-registro.

    Args:
        candidatos: Datasets que superaron los criterios.

    Returns:
        Texto con el recuento por régimen de tamaño y por característica.
    """
    regimenes = candidatos["NumberOfInstances"].map(_regimen).value_counts()
    con_categoricas = int((candidatos["categoricas"] > 0).sum())
    con_faltantes = int(candidatos["faltantes"].sum())
    multiclase = int((candidatos["NumberOfClasses"] > 2).sum())

    lineas = [
        f"- Candidatos que superan los criterios: **{len(candidatos)}**",
        "- Por régimen de tamaño:",
    ]
    for nombre in ("centenas", "millares", "decenas de millares"):
        lineas.append(f"  - {nombre}: {int(regimenes.get(nombre, 0))}")
    lineas.append(f"- Con variables categóricas: {con_categoricas}")
    lineas.append(f"- Con valores faltantes: {con_faltantes}")
    lineas.append(f"- Multiclase (más de 2 clases): {multiclase}")
    return "\n".join(lineas)


def main() -> None:
    """Descarga, filtra y escribe la lista candidata."""
    print(f"Descargando la suite {SUITE} desde OpenML...")
    datos = _enriquecer(_descargar())
    candidatos, descartes = _filtrar(datos)

    print(f"  datasets en la suite: {len(datos)}")
    for nombre, cuenta in descartes.items():
        print(f"  incumplen '{nombre}': {cuenta}")
    print(f"  supervivientes: {len(candidatos)}")

    contenido = "\n".join(
        [
            "# Lista candidata de datasets para el corpus (RE4)",
            "",
            "> **Archivo generado.** No editar a mano: se regenera con",
            "> `.\\.venv\\Scripts\\python.exe scripts\\list_corpus_candidates.py`.",
            "",
            f"- **Generado:** {date.today().isoformat()}",
            f"- **Universo:** suite curada {SUITE} ({len(datos)} datasets)",
            "- **Criterios aplicados:** Sección 4.3.1 del anteproyecto",
            "- **Decisiones de procedimiento:** `docs/experimentos/CORPUS-criterios-inclusion.md`",
            "",
            "## Resumen",
            "",
            _resumen(candidatos),
            "",
            "## Fichas",
            "",
            "«Predictoras» y «categóricas» excluyen la columna objetivo. «Desbalance» es",
            "el cociente entre la clase mayoritaria y la minoritaria.",
            "",
            _tabla_markdown(candidatos),
            "",
        ]
    )
    SALIDA.write_text(contenido, encoding="utf-8")
    print(f"Escrito: {SALIDA}")


if __name__ == "__main__":
    main()
