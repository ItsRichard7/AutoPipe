# scripts/download_corpus.py
"""Descarga los candidatos del corpus y verifica sus metadatos (RE4).

Toma la lista que produjo ``scripts/list_corpus_candidates.py``, descarga cada
dataset de OpenML y **comprueba contra los datos reales** lo que los metadatos
del repositorio afirman: número de instancias, de variables predictoras, de
variables categóricas, de clases y presencia de valores faltantes.

**Por qué hace falta.** La lista de candidatos se construyó con los metadatos que
publica OpenML, sin descargar un solo dataset; el documento de criterios lo
declara como limitación. Un recuento equivocado en el repositorio se habría
propagado a la selección del corpus sin que nada fallara: un dataset con más
clases de las declaradas, o sin los faltantes que dice tener, cambia tanto su
elegibilidad como el papel que juega en el cribado.

Extrae además el vector de meta-características de cada dataset, porque de él
sale el número que necesita el pre-registro v0.2: cuántas columnas resultan no
finitas en **algún** dataset del corpus y por tanto se descartan.

La lista de candidatos es la entrada, así que conviene regenerarla antes:

    .\\.venv\\Scripts\\python.exe scripts\\list_corpus_candidates.py
    .\\.venv\\Scripts\\python.exe scripts\\download_corpus.py
"""

from __future__ import annotations

import argparse
import logging
import re
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import openml
import pandas as pd

from autopipe.data_loader import TabularDataLoader
from autopipe.meta_features import MetaFeatureExtractor, columnas_no_finitas

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _silenciar_ruido() -> None:
    """Baja el nivel de los registros que tapan el informe.

    ``openml`` y el cargador del proyecto emiten varias líneas por descarga;
    sobre cincuenta datasets eso oculta las discrepancias, que es lo único que
    hay que leer mientras corre. ``openml`` gestiona su propio manejador de
    consola, así que ajustar el nivel del *logger* no basta: hay que usar su API.
    """
    openml.config.set_console_log_level(logging.WARNING)
    logging.getLogger("autopipe").setLevel(logging.WARNING)

CANDIDATOS = Path("docs/experimentos/CORPUS-candidatos.md")
SALIDA = Path("docs/experimentos/CORPUS-descargado.md")
CACHE_META = Path("datasets/.meta_cache")
SEMILLA = 42

FILA = re.compile(
    r"^\|\s*(\d+)\s*\|\s*`([^`]+)`\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|"
    r"\s*(\d+)\s*\|\s*(sí|no)\s*\|"
)


def leer_candidatos(ruta: Path) -> list[dict]:
    """Lee las fichas declaradas de la lista de candidatos.

    Args:
        ruta: Ruta al documento generado por ``list_corpus_candidates.py``.

    Returns:
        Lista de fichas con lo que OpenML **declara** de cada dataset.

    Raises:
        SystemExit: Si el archivo no existe o no contiene ninguna fila legible.
    """
    if not ruta.is_file():
        raise SystemExit(
            f"No existe {ruta}. Genere la lista de candidatos primero con "
            f"scripts/list_corpus_candidates.py."
        )
    fichas = []
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        encaje = FILA.match(linea)
        if encaje is None:
            continue
        did, nombre, instancias, predictoras, categoricas, clases, faltantes = (
            encaje.groups()
        )
        fichas.append(
            {
                "did": int(did),
                "nombre": nombre,
                "instancias": int(instancias),
                "predictoras": int(predictoras),
                "categoricas": int(categoricas),
                "clases": int(clases),
                "faltantes": faltantes == "sí",
            }
        )
    if not fichas:
        raise SystemExit(f"No se pudo leer ninguna ficha de {ruta}.")
    return fichas


def descargar(did: int) -> tuple[pd.DataFrame, np.ndarray, str, list[str]]:
    """Descarga un dataset de OpenML y lo carga con el cargador del proyecto.

    La tabla se pasa por :class:`~autopipe.data_loader.TabularDataLoader`, que es
    el mismo camino que recorrerán el cribado y la base de meta-conocimiento: si
    algo del dataset no es digerible, conviene descubrirlo aquí.

    Args:
        did: Identificador del dataset en OpenML.

    Returns:
        Tupla ``(características, etiquetas, licencia, atributos_ignorados)``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dataset = openml.datasets.get_dataset(
            did,
            download_data=True,
            download_qualities=False,
            download_features_meta_data=True,
        )
        X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    ignorados = list(dataset.ignore_attribute or [])
    if dataset.row_id_attribute:
        ignorados.append(str(dataset.row_id_attribute))
    tabla = X.copy()
    tabla["__objetivo__"] = np.asarray(y, dtype=object)
    features, labels = TabularDataLoader(tabla, target_column="__objetivo__").load()
    return features, labels, str(dataset.licence), ignorados


def verificar(
    ficha: dict, features: pd.DataFrame, labels: np.ndarray, ignorados: list[str]
) -> dict:
    """Compara lo que OpenML declara con lo que traen los datos.

    ``NumberOfFeatures`` cuenta también los atributos que OpenML marca como
    identificadores o ignorables (``ignore_attribute``, ``row_id_attribute``) y
    que ``get_data`` descarta al entregar la tabla. Esa diferencia es esperada y
    conveniente —una columna identificadora no aporta señal— así que se separa de
    las discrepancias que sí habría que revisar.

    Args:
        ficha: Ficha declarada, leída de la lista de candidatos.
        features: Tabla de características descargada.
        labels: Etiquetas de clase.
        ignorados: Atributos que OpenML marca como ignorables.

    Returns:
        La ficha ampliada con los valores reales, las diferencias explicadas por
        atributos ignorados y las que quedan sin explicar.
    """
    numericas = list(features.select_dtypes(include="number").columns)
    vacias = [c for c in features.columns if features[c].isna().all()]
    real = {
        "instancias": int(len(features)),
        "predictoras": int(features.shape[1]),
        "categoricas": int(features.shape[1] - len(numericas)),
        "clases": int(len(np.unique(labels))),
        "faltantes": bool(features.isna().to_numpy().any()),
    }
    diferencias = {
        clave: f"declarado {ficha[clave]}, real {real[clave]}"
        for clave in real
        if ficha[clave] != real[clave]
    }
    explicable = len(ignorados) > 0 and ficha["predictoras"] - real["predictoras"] == len(
        ignorados
    )
    if explicable:
        diferencias.pop("predictoras", None)
        diferencias.pop("categoricas", None)
    # Una columna sin ningún valor llega como ``object`` y se cuenta como
    # categórica, aunque OpenML la declare numérica. No es un error del
    # repositorio: es lo que pandas puede inferir de una columna vacía.
    if vacias and "categoricas" in diferencias:
        diferencias.pop("categoricas")
    return {
        **ficha,
        "real": real,
        "ignorados": ignorados,
        "vacias": vacias,
        "explicada": explicable,
        "discrepancias": [f"{clave}: {texto}" for clave, texto in diferencias.items()],
    }


def _tabla(fichas: list[dict]) -> str:
    """Arma la tabla markdown con una fila por dataset descargado."""
    lineas = [
        "| OpenML ID | Nombre | Instancias | Predictoras | Categóricas | Clases "
        "| Faltantes | No finitas | Licencia | Metadatos |",
        "|---:|---|---:|---:|---:|---:|:---:|---:|---|---|",
    ]
    for ficha in fichas:
        real = ficha["real"]
        if ficha["discrepancias"]:
            estado = "**revisar**"
        elif ficha["explicada"]:
            estado = f"ignora {len(ficha['ignorados'])}"
        else:
            estado = "OK"
        lineas.append(
            f"| {ficha['did']} | `{ficha['nombre']}` | {real['instancias']} | "
            f"{real['predictoras']} | {real['categoricas']} | {real['clases']} | "
            f"{'sí' if real['faltantes'] else 'no'} | {ficha['no_finitas']} | "
            f"{ficha['licencia']} | {estado} |"
        )
    return "\n".join(lineas)


def _regimen(instancias: int) -> str:
    """Clasifica el tamaño del dataset en uno de los tres regímenes.

    Args:
        instancias: Número de instancias medido sobre los datos.

    Returns:
        ``"centenas"``, ``"millares"`` o ``"decenas de millares"``.
    """
    if instancias < 1_000:
        return "centenas"
    return "millares" if instancias < 10_000 else "decenas de millares"


def _diversidad(fichas: list[dict]) -> str:
    """Arma la tabla de cobertura de las restricciones de diversidad.

    Args:
        fichas: Fichas ya verificadas contra los datos.

    Returns:
        Tabla markdown con lo exigido y lo disponible.
    """
    regimenes = [_regimen(f["real"]["instancias"]) for f in fichas]
    filas = [
        ("Régimen de centenas", 1, regimenes.count("centenas")),
        ("Régimen de millares", 1, regimenes.count("millares")),
        (
            "Régimen de decenas de millares",
            1,
            regimenes.count("decenas de millares"),
        ),
        (
            "Con variables categóricas",
            5,
            sum(1 for f in fichas if f["real"]["categoricas"] > 0),
        ),
        ("Con valores faltantes", 5, sum(1 for f in fichas if f["real"]["faltantes"])),
        ("Multiclase (más de 2 clases)", 0, sum(1 for f in fichas if f["real"]["clases"] > 2)),
    ]
    lineas = ["| Restricción | Exigido | Disponible |", "|---|---:|---:|"]
    lineas += [f"| {nombre} | {exigido} | {disponible} |" for nombre, exigido, disponible in filas]
    return "\n".join(lineas)


def _informe(fichas: list[dict], union: set[str], total_meta: int) -> str:
    """Redacta el documento de salida."""
    con_problemas = [f for f in fichas if f["discrepancias"]]
    detalle = "\n".join(
        f"- `{f['nombre']}` (ID {f['did']}): " + "; ".join(f["discrepancias"])
        for f in con_problemas
    ) or (
        f"Ninguna: los metadatos de OpenML coinciden con los datos en los "
        f"{len(fichas)} datasets."
    )
    explicadas = "\n".join(
        f"- `{f['nombre']}` (ID {f['did']}): OpenML ignora "
        f"{', '.join(f'`{a}`' for a in f['ignorados'])}"
        for f in fichas
        if f["explicada"]
    ) or "Ninguna."
    diversidad = _diversidad(fichas)
    con_vacias = "\n".join(
        f"- `{f['nombre']}` (ID {f['did']}): "
        f"{', '.join(f'`{c}`' for c in f['vacias'])}"
        for f in fichas
        if f["vacias"]
    ) or "Ninguna."
    return f"""# Corpus descargado y verificado (RE4)

> **Archivo generado.** No editar a mano: se regenera con
> `.\\.venv\\Scripts\\python.exe scripts\\download_corpus.py`.

- **Generado:** {date.today().isoformat()}
- **Entrada:** `{CANDIDATOS.as_posix()}`
- **Datasets descargados:** {len(fichas)}
- **Con discrepancias en los metadatos:** {len(con_problemas)}

## Por qué existe este documento

La lista de candidatos se construyó con los metadatos que publica OpenML, sin
descargar ningún dataset. Este paso los comprueba contra los datos reales, que es
la limitación que el documento de criterios dejó anotada para la semana 5.

## Diferencias explicadas por atributos ignorados

`NumberOfFeatures` cuenta también los atributos que OpenML marca como
identificadores o ignorables y que descarta al entregar la tabla. La diferencia es
esperada, y descartarlos es lo correcto: una columna identificadora no aporta
señal y puede introducir una correlación espuria.

{explicadas}

## Columnas enteramente vacías

Una columna sin ningún valor llega como `object` y se cuenta como categórica,
aunque OpenML la declare numérica. **No se descartan**: el pipeline las trata
correctamente —la imputación las vuelve constantes y la selección supervisada las
puntúa como las peores— y eliminarlas introduciría una transformación silenciosa
entre lo que dice el pre-registro y lo que se ejecuta.

{con_vacias}

## Discrepancias sin explicar

{detalle}

## Meta-características no finitas

Cada dataset aporta las columnas que le resultan no finitas. La regla del
pre-registro descarta las que lo sean en **algún** dataset del corpus, de modo que
el vector final es la intersección de las finitas.

- Meta-características extraídas por dataset: **{total_meta}**
- No finitas en al menos un dataset: **{len(union)}**
- **Sobrevivirían: {total_meta - len(union)}**

## Cobertura de diversidad, medida sobre los datos

El pre-registro exige, al elegir el corpus final, cubrir los tres regímenes de
tamaño y contar con al menos cinco datasets con variables categóricas y cinco con
valores faltantes. Estas cuentas ya no salen de los metadatos de OpenML sino de
los datos descargados.

{diversidad}

## Fichas verificadas

Los valores son los **reales**, medidos sobre los datos descargados. La columna
«Metadatos» dice si coinciden con lo que OpenML declaraba.

{_tabla(fichas)}
"""


def main() -> int:
    """Punto de entrada.

    Returns:
        ``0`` siempre; las discrepancias se informan, no se tratan como error.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Procesar solo los primeros N.")
    args = parser.parse_args()

    _silenciar_ruido()
    candidatos = leer_candidatos(CANDIDATOS)[: args.limit]
    extractor = MetaFeatureExtractor(random_state=SEMILLA, cache_dir=CACHE_META)
    fichas: list[dict] = []
    union: set[str] = set()
    total_meta = 0

    for numero, ficha in enumerate(candidatos, 1):
        logger.info("[%d/%d] %s", numero, len(candidatos), ficha["nombre"])
        features, labels, licencia, ignorados = descargar(ficha["did"])
        verificada = verificar(ficha, features, labels, ignorados)
        vector = extractor.extract(features, labels, dataset_id=str(ficha["did"]))
        no_finitas = columnas_no_finitas(vector)
        union |= no_finitas
        total_meta = len(vector)
        fichas.append({**verificada, "licencia": licencia, "no_finitas": len(no_finitas)})

    SALIDA.write_text(_informe(fichas, union, total_meta), encoding="utf-8")
    con_problemas = sum(1 for f in fichas if f["discrepancias"])
    logger.info(
        "\n%d datasets, %d con discrepancias. No finitas en algún dataset: %d de %d.",
        len(fichas),
        con_problemas,
        len(union),
        total_meta,
    )
    logger.info("Informe escrito en %s", SALIDA)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
