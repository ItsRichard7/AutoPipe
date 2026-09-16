# scripts/verify_engineering_criteria.py
"""Verificación con evidencia de los criterios de éxito de ingeniería (RE7).

El anteproyecto fija cuatro criterios de ingeniería, independientes del resultado de la
hipótesis (Sección 4.3.5):

a. la suite de regresión de la librería base (134 pruebas) se mantiene en verde tras la
   refactorización;
b. los módulos nuevos alcanzan al menos un 70 % de cobertura;
c. los resultados son reproducibles: la misma semilla produce el mismo resultado;
d. el flujo completo para el usuario final cabe en no más de 20 líneas de código.

Este guion los **mide**, cada uno con un procedimiento que se puede repetir:

a. Extrae ``tests/`` **tal como estaba en el commit de migración** a una carpeta temporal
   y lo ejecuta, sin tocarlo, contra el código actual. Es la lectura literal del criterio.
   Además compara cada función de prueba con la de hoy por su árbol sintáctico: un cambio
   de formato no cuenta como edición y un cambio de aserción sí.
b. Ejecuta la suite completa y lee la cobertura por archivo. «Módulo nuevo» es todo
   archivo de ``autopipe/`` que no existía en el commit de migración, y eso lo decide
   ``git``, no una lista escrita a mano.
c. Lee ``REPRODUCIBILIDAD.md``, que genera ``verify_reproducibility.py``, y exige que
   ninguna verificación difiera.
d. Cuenta las instrucciones de nivel superior de los dos ejemplos de ``examples/`` con el
   mismo criterio que ``tests/test_examples.py``.

Ejecutar con:
    .\\.venv\\Scripts\\python.exe scripts\\verify_engineering_criteria.py
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from datetime import date
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

RAIZ = Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "docs" / "CRITERIOS-ingenieria.md"
REPRODUCIBILIDAD = RAIZ / "docs" / "experimentos" / "REPRODUCIBILIDAD.md"
EJEMPLOS = (
    RAIZ / "examples" / "tabular_quickstart.py",
    RAIZ / "examples" / "warm_start_quickstart.py",
)

COMMIT_MIGRACION = "35f1db5"
"""Commit que migró la librería base: «feat: migrate AutoPipeAD base as AutoPipe foundation»."""

PRUEBAS_BASE = 134
"""Pruebas de regresión de la librería base que cita el anteproyecto."""

UMBRAL_COBERTURA = 70.0
LIMITE_INSTRUCCIONES = 20

IDENTICO, DIFIERE, OMITIDO = "idéntico", "difiere", "omitido"
"""Estados que escribe ``verify_reproducibility.py``; una prueba vigila que coincidan."""

FILA_REPRODUCIBILIDAD = re.compile(r"^\|[^|]*\|[^|]*\|\s*\*\*([^*]+)\*\*\s*\|")


# --------------------------------------------------------------------------- #
# (a) La suite de regresión de la librería base
# --------------------------------------------------------------------------- #
def _git(*argumentos: str) -> str:
    """Ejecuta ``git`` en la raíz del repositorio y devuelve su salida."""
    return subprocess.run(
        ["git", *argumentos], cwd=RAIZ, capture_output=True, text=True, encoding="utf-8",
        check=True,
    ).stdout


def _es_prueba(nodo: ast.stmt) -> bool:
    """Indica si un nodo es una función de prueba."""
    return isinstance(nodo, (ast.FunctionDef, ast.AsyncFunctionDef)) and nodo.name.startswith(
        "test"
    )


def funciones_de_prueba(fuente: str) -> dict[str, str]:
    """Funciones de prueba de un archivo, por nombre y descritas por su árbol sintáctico.

    Se comparan árboles y no texto: reformatear o comentar una prueba no la edita, pero
    cambiar una aserción, un decorador o su docstring sí.

    Args:
        fuente: Código del archivo de pruebas.

    Returns:
        Diccionario de nombre —``Clase::metodo`` para las de clase— a su árbol volcado.
    """
    funciones: dict[str, str] = {}
    for nodo in ast.parse(fuente).body:
        if _es_prueba(nodo):
            funciones[nodo.name] = ast.dump(nodo)  # type: ignore[attr-defined]
        elif isinstance(nodo, ast.ClassDef) and nodo.name.startswith("Test"):
            for hijo in nodo.body:
                if _es_prueba(hijo):
                    funciones[f"{nodo.name}::{hijo.name}"] = ast.dump(hijo)  # type: ignore
    return funciones


def comparar_funciones(anteriores: dict[str, str], actuales: dict[str, str]) -> dict:
    """Clasifica las funciones de prueba de antes según cómo están hoy.

    Args:
        anteriores: Funciones del commit de migración.
        actuales: Funciones del mismo archivo hoy.

    Returns:
        Diccionario con las listas ``identicas``, ``modificadas`` y ``ausentes``.
    """
    return {
        "identicas": [n for n, arbol in anteriores.items() if actuales.get(n) == arbol],
        "modificadas": [
            n for n, arbol in anteriores.items() if n in actuales and actuales[n] != arbol
        ],
        "ausentes": [n for n in anteriores if n not in actuales],
    }


def resumen_pytest(salida: str) -> dict[str, int]:
    """Lee los recuentos de la línea final de pytest.

    Args:
        salida: Salida estándar de pytest.

    Returns:
        Recuentos de ``passed``, ``failed``, ``errors``, ``skipped``, ``warnings`` y
        ``collected``; los que no aparecen valen cero.
    """
    recuentos = dict.fromkeys(
        ("passed", "failed", "errors", "skipped", "warnings", "collected"), 0
    )
    lineas = [linea for linea in salida.strip().splitlines() if linea.strip()]
    if not lineas:
        return recuentos
    patron = r"(\d+) (passed|failed|errors?|skipped|warnings?|tests? collected)"
    for numero, clave in re.findall(patron, lineas[-1]):
        if clave.startswith("error"):
            clave = "errors"
        elif clave.startswith("warning"):
            clave = "warnings"
        elif "collected" in clave:
            clave = "collected"
        recuentos[clave] += int(numero)
    return recuentos


def extraer_pruebas(commit: str, destino: Path) -> dict[str, str]:
    """Escribe en ``destino`` la carpeta ``tests/`` tal como estaba en ``commit``.

    Args:
        commit: Commit del que se extraen las pruebas.
        destino: Carpeta donde escribirlas.

    Returns:
        Diccionario de ruta relativa a su contenido.
    """
    fuentes = {}
    for archivo in _git("ls-tree", "-r", "--name-only", commit, "tests/").split():
        fuente = _git("show", f"{commit}:{archivo}")
        ruta = destino / archivo
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(fuente, encoding="utf-8")
        fuentes[archivo] = fuente
    return fuentes


def comparar_con_hoy(fuentes: dict[str, str]) -> dict:
    """Compara cada función de prueba migrada con la que existe hoy en el repositorio.

    Args:
        fuentes: Archivos de ``tests/`` del commit de migración.

    Returns:
        Diccionario con las listas ``identicas``, ``modificadas`` y ``ausentes``, con
        cada prueba identificada por archivo y nombre.
    """
    total: dict[str, list[str]] = {"identicas": [], "modificadas": [], "ausentes": []}
    for archivo, fuente in sorted(fuentes.items()):
        if not (archivo.endswith(".py") and Path(archivo).name.startswith("test_")):
            continue
        actual = RAIZ / archivo
        hoy = funciones_de_prueba(actual.read_text(encoding="utf-8")) if actual.is_file() else {}
        for clase, nombres in comparar_funciones(funciones_de_prueba(fuente), hoy).items():
            total[clase] += [f"{archivo}::{nombre}" for nombre in nombres]
    return total


def ejecutar_pytest(directorio: Path, *argumentos: str) -> str:
    """Ejecuta pytest sobre la carpeta ``tests`` de ``directorio`` y devuelve su salida.

    Se anula ``addopts``: la configuración de cobertura del proyecto no tiene sentido
    sobre una copia temporal de las pruebas.

    Args:
        directorio: Carpeta que contiene ``tests/``.
        argumentos: Argumentos adicionales para pytest.

    Returns:
        La salida estándar.
    """
    return subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-p", "no:cacheprovider", "-o", "addopts=",
         *argumentos],
        cwd=directorio, capture_output=True, text=True, encoding="utf-8", errors="replace",
    ).stdout


def origen_de_autopipe(directorio: Path) -> Path:
    """Ruta desde la que se importa ``autopipe`` cuando se ejecuta en ``directorio``."""
    salida = subprocess.run(
        [sys.executable, "-c", "import autopipe; print(autopipe.__file__)"],
        cwd=directorio, capture_output=True, text=True, check=True,
    ).stdout.strip()
    return Path(salida).resolve()


def evaluar_regresion(
    comparacion: dict, recoleccion: dict[str, int], ejecucion: dict[str, int], origen: Path
) -> dict:
    """Decide el criterio (a) a partir de las mediciones.

    Args:
        comparacion: Resultado de :func:`comparar_con_hoy`.
        recoleccion: Recuentos de la recolección de pytest.
        ejecucion: Recuentos de la ejecución de pytest.
        origen: Ruta desde la que se importó ``autopipe``.

    Returns:
        Diccionario con las mediciones y ``cumple``. No cumple si las pruebas se
        ejecutaron contra un ``autopipe`` que no es el del repositorio: pasar contra otra
        copia no probaría nada sobre este código.
    """
    contra_el_repositorio = origen.is_relative_to(RAIZ)
    cumple = (
        contra_el_repositorio
        and recoleccion["collected"] == PRUEBAS_BASE
        and ejecucion["passed"] == PRUEBAS_BASE
        and ejecucion["failed"] == 0
        and ejecucion["errors"] == 0
    )
    return {
        "commit": COMMIT_MIGRACION,
        "funciones": sum(len(v) for v in comparacion.values()),
        **comparacion,
        "recolectadas": recoleccion["collected"],
        "pasan": ejecucion["passed"],
        "fallan": ejecucion["failed"],
        "errores": ejecucion["errors"],
        "origen": origen,
        "contra_el_repositorio": contra_el_repositorio,
        "cumple": cumple,
    }


def criterio_regresion(commit: str = COMMIT_MIGRACION) -> dict:
    """Mide el criterio (a) sobre una copia de las pruebas del commit de migración.

    Args:
        commit: Commit del que se toman las pruebas.

    Returns:
        El resultado de :func:`evaluar_regresion`.
    """
    temporal = Path(tempfile.mkdtemp(prefix="autopipe-migracion-"))
    try:
        fuentes = extraer_pruebas(commit, temporal)
        comparacion = comparar_con_hoy(fuentes)
        recoleccion = resumen_pytest(ejecutar_pytest(temporal, "--collect-only", "-q"))
        ejecucion = resumen_pytest(ejecutar_pytest(temporal, "-q"))
        origen = origen_de_autopipe(temporal)
    finally:
        shutil.rmtree(temporal, ignore_errors=True)
    return evaluar_regresion(comparacion, recoleccion, ejecucion, origen)


# --------------------------------------------------------------------------- #
# (b) Cobertura de los módulos nuevos
# --------------------------------------------------------------------------- #
def modulos_migrados(commit: str = COMMIT_MIGRACION) -> set[str]:
    """Archivos de ``autopipe/`` que ya existían en el commit de migración."""
    return set(_git("ls-tree", "-r", "--name-only", commit, "autopipe/").split())


def medir_cobertura(ejecutar_suite: bool = True) -> tuple[dict[str, int] | None, dict]:
    """Ejecuta la suite completa, si se pide, y exporta su cobertura.

    La suite mide la cobertura por la configuración de ``pyproject.toml``, así que aquí
    no se pasa ``--cov``: duplicarlo inicializa numpy dos veces.

    Args:
        ejecutar_suite: Si es falso, se lee la cobertura de la última ejecución.

    Returns:
        Tupla con los recuentos de la suite, o ``None`` si no se ejecutó, y los datos de
        cobertura en el formato JSON de ``coverage``.

    Raises:
        SystemExit: Si no hay datos de cobertura que exportar.
    """
    resumen = None
    if ejecutar_suite:
        salida = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
            cwd=RAIZ, capture_output=True, text=True, encoding="utf-8", errors="replace",
        ).stdout
        resumen = resumen_pytest(salida)
    temporal = Path(tempfile.mkdtemp(prefix="autopipe-cobertura-"))
    try:
        destino = temporal / "cobertura.json"
        exportado = subprocess.run(
            [sys.executable, "-m", "coverage", "json", "-o", str(destino)],
            cwd=RAIZ, capture_output=True, text=True,
        )
        if exportado.returncode != 0 or not destino.is_file():
            raise SystemExit(
                "No hay datos de cobertura que exportar. Ejecute la suite o quite --sin-suite."
            )
        datos = json.loads(destino.read_text(encoding="utf-8"))
    finally:
        shutil.rmtree(temporal, ignore_errors=True)
    return resumen, datos


def criterio_cobertura(datos: dict, migrados: set[str]) -> dict:
    """Decide el criterio (b) sobre los módulos que no existían en la migración.

    La comparación con el umbral usa el porcentaje sin redondear: un módulo al 69,96 %
    no cumple aunque se muestre como 70,0 %.

    Args:
        datos: Cobertura en el formato JSON de ``coverage``.
        migrados: Archivos que ya existían en el commit de migración.

    Returns:
        Diccionario con los módulos nuevos, el de menor cobertura, los que quedan por
        debajo del umbral, la cobertura total y ``cumple``.
    """
    nuevos = []
    for archivo, valor in datos["files"].items():
        clave = archivo.replace("\\", "/")
        if clave in migrados:
            continue
        resumen = valor["summary"]
        nuevos.append(
            {
                "modulo": clave,
                "instrucciones": resumen["num_statements"],
                "cobertura": resumen["percent_covered"],
            }
        )
    nuevos.sort(key=lambda m: (m["cobertura"], m["modulo"]))
    por_debajo = [m["modulo"] for m in nuevos if m["cobertura"] < UMBRAL_COBERTURA]
    return {
        "modulos": nuevos,
        "minimo": nuevos[0] if nuevos else None,
        "por_debajo": por_debajo,
        "migrados": len(migrados),
        "total": datos["totals"]["percent_covered"],
        "instrucciones_totales": datos["totals"]["num_statements"],
        "medido": datos["meta"]["timestamp"],
        "cumple": bool(nuevos) and not por_debajo,
    }


# --------------------------------------------------------------------------- #
# (c) Reproducibilidad y (d) flujo del usuario
# --------------------------------------------------------------------------- #
def criterio_reproducibilidad(texto: str) -> dict:
    """Decide el criterio (c) a partir de ``REPRODUCIBILIDAD.md``.

    Args:
        texto: Contenido del documento.

    Returns:
        Diccionario con los recuentos por estado y ``cumple``. Una verificación omitida
        no es un fallo, pero tampoco evidencia: sin ninguna idéntica no se cumple.
    """
    estados = [m.group(1) for m in map(FILA_REPRODUCIBILIDAD.match, texto.splitlines()) if m]
    identicas, difieren = estados.count(IDENTICO), estados.count(DIFIERE)
    return {
        "verificaciones": len(estados),
        "identicas": identicas,
        "difieren": difieren,
        "omitidas": estados.count(OMITIDO),
        "cumple": identicas > 0 and difieren == 0,
    }


def contar_instrucciones(ruta: Path) -> int:
    """Cuenta las instrucciones de nivel superior de un archivo, sin su docstring.

    Es el criterio de ``tests/test_examples.py``: se cuentan instrucciones y no líneas
    físicas, porque el límite de 100 caracteres reparte una instrucción en varias líneas.
    Una prueba vigila que los dos conteos coincidan.

    Args:
        ruta: Archivo de Python.

    Returns:
        Número de instrucciones.
    """
    cuerpo = list(ast.parse(ruta.read_text(encoding="utf-8")).body)
    if cuerpo and isinstance(cuerpo[0], ast.Expr) and isinstance(cuerpo[0].value, ast.Constant):
        cuerpo = cuerpo[1:]
    return len(cuerpo)


def _relativa(ruta: Path) -> str:
    """Ruta relativa a la raíz del repositorio, si lo es."""
    return ruta.relative_to(RAIZ).as_posix() if ruta.is_relative_to(RAIZ) else ruta.as_posix()


def criterio_instrucciones(ejemplos: Sequence[Path] = EJEMPLOS) -> dict:
    """Decide el criterio (d) sobre los ejemplos del flujo de usuario.

    Args:
        ejemplos: Archivos de ejemplo.

    Returns:
        Diccionario con las instrucciones de cada ejemplo y ``cumple``.
    """
    filas = [{"ejemplo": _relativa(r), "instrucciones": contar_instrucciones(r)} for r in ejemplos]
    return {
        "ejemplos": filas,
        "cumple": bool(filas) and all(f["instrucciones"] <= LIMITE_INSTRUCCIONES for f in filas),
    }


# --------------------------------------------------------------------------- #
# El documento
# --------------------------------------------------------------------------- #
def _si(valor: bool) -> str:
    return "**sí**" if valor else "**no**"


def _lista(titulo: str, pruebas: list[str]) -> str:
    """Enumera pruebas modificadas o ausentes, si las hay."""
    if not pruebas:
        return ""
    return f"\n**{titulo}:**\n\n" + "\n".join(f"- `{p}`" for p in pruebas) + "\n"


def _tabla_resumen(
    regresion: dict, cobertura: dict, reproducibilidad: dict, instrucciones: dict
) -> str:
    """Tabula cada criterio con lo exigido, lo medido y el veredicto."""
    minimo = cobertura["minimo"]
    medido_cobertura = (
        f"Mínimo {minimo['cobertura']:.1f} % (`{minimo['modulo']}`), "
        f"{len(cobertura['modulos'])} módulos"
        if minimo
        else "Ningún módulo nuevo encontrado"
    )
    filas = [
        (
            "(a) Regresión de la librería base",
            f"Las {PRUEBAS_BASE} pruebas en verde tras la refactorización",
            f"{regresion['pasan']} de {regresion['recolectadas']} pasan, ejecutadas tal como "
            f"estaban en `{regresion['commit']}`",
            regresion["cumple"],
        ),
        (
            "(b) Cobertura de los módulos nuevos",
            f"Al menos {UMBRAL_COBERTURA:.0f} %",
            medido_cobertura,
            cobertura["cumple"],
        ),
        (
            "(c) Reproducibilidad",
            "Misma semilla, mismo resultado",
            f"{reproducibilidad['identicas']} de {reproducibilidad['verificaciones']} "
            f"verificaciones idénticas, {reproducibilidad['omitidas']} omitidas",
            reproducibilidad["cumple"],
        ),
        (
            "(d) Flujo del usuario final",
            f"No más de {LIMITE_INSTRUCCIONES} instrucciones",
            " y ".join(str(f["instrucciones"]) for f in instrucciones["ejemplos"]),
            instrucciones["cumple"],
        ),
    ]
    lineas = ["| Criterio | Exigido | Medido | Cumple |", "|---|---|---|:--:|"]
    lineas += [f"| {criterio} | {exigido} | {medido} | {_si(ok)} |"
               for criterio, exigido, medido, ok in filas]
    return "\n".join(lineas)


def redactar(
    regresion: dict,
    cobertura: dict,
    reproducibilidad: dict,
    instrucciones: dict,
    suite: dict[str, int] | None,
) -> str:
    """Redacta el documento de verificación de los criterios.

    Args:
        regresion: Resultado del criterio (a).
        cobertura: Resultado del criterio (b).
        reproducibilidad: Resultado del criterio (c).
        instrucciones: Resultado del criterio (d).
        suite: Recuentos de la suite completa, o ``None`` si no se ejecutó.

    Returns:
        El documento markdown.
    """
    criterios = (regresion, cobertura, reproducibilidad, instrucciones)
    cumplidos = sum(1 for c in criterios if c["cumple"])
    if suite is None:
        linea_suite = f"no se ejecutó en esta corrida; cobertura medida el {cobertura['medido']}"
    else:
        linea_suite = (
            f"{suite['passed']} pasan, {suite['failed']} fallan, {suite['errors']} errores"
        )
    filas_modulos = "\n".join(
        f"| `{m['modulo']}` | {m['instrucciones']} | {m['cobertura']:.1f} % |"
        for m in cobertura["modulos"]
    )
    filas_ejemplos = "\n".join(
        f"| `{f['ejemplo']}` | {f['instrucciones']} |" for f in instrucciones["ejemplos"]
    )
    return f"""# Criterios de éxito de ingeniería: verificación con evidencia

> **Archivo generado.** No editar a mano: se regenera con
> `.\\.venv\\Scripts\\python.exe scripts\\verify_engineering_criteria.py`.

> Los cuatro criterios de ingeniería del anteproyecto (Sección 4.3.5) son independientes
> del resultado de la hipótesis. Aquí cada uno se **mide** con un procedimiento repetible.

- **Generado:** {date.today().isoformat()}
- **Commit de migración de la librería base:** `{regresion['commit']}`
- **Suite completa:** {linea_suite}
- **Veredicto:** **{cumplidos} de {len(criterios)} criterios cumplidos**

{_tabla_resumen(regresion, cobertura, reproducibilidad, instrucciones)}

## (a) La suite de regresión de la librería base

Se extrajo la carpeta `tests/` tal como estaba en el commit de migración y se ejecutó,
**sin modificarla**, contra el código actual. Es la lectura literal del criterio: si las
pruebas de la librería base pasan sin tocarlas, la refactorización no rompió el caso de uso
original. `autopipe` se importó desde el repositorio: {_si(regresion['contra_el_repositorio'])}.

- **Recolectadas:** {regresion['recolectadas']} · **Pasan:** {regresion['pasan']} ·
  **Fallan:** {regresion['fallan']} · **Errores:** {regresion['errores']}
- **Funciones de prueba:** {regresion['funciones']}; la parametrización las expande en
  las pruebas recolectadas. Comparadas por su árbol sintáctico con las de hoy:
  **{len(regresion['identicas'])} idénticas**, {len(regresion['modificadas'])} modificadas y
  {len(regresion['ausentes'])} ausentes.
{_lista("Modificadas", regresion['modificadas'])}{_lista("Ausentes", regresion['ausentes'])}
## (b) Cobertura de los módulos nuevos

«Módulo nuevo» es todo archivo de `autopipe/` que no existía en el commit de migración, según
`git`; los {cobertura['migrados']} que sí existían quedan fuera del criterio. La cobertura
total del paquete es de {cobertura['total']:.1f} % sobre {cobertura['instrucciones_totales']}
instrucciones.

| Módulo nuevo | Instrucciones | Cobertura |
|---|---:|---:|
{filas_modulos}

## (c) Reproducibilidad

Se lee `docs/experimentos/REPRODUCIBILIDAD.md`, que genera
`scripts/verify_reproducibility.py`: regenera los documentos de resultados desde sus datos,
reajusta el recomendador por pliegue y reejecuta corridas ensayo por ensayo, sin tolerancia.
Una verificación omitida no cuenta como evidencia.

- **Idénticas:** {reproducibilidad['identicas']} · **Difieren:** {reproducibilidad['difieren']} ·
  **Omitidas:** {reproducibilidad['omitidas']}

## (d) Flujo del usuario final

Se cuentan instrucciones de nivel superior y no líneas físicas, con el mismo criterio que
`tests/test_examples.py`, que además ejecuta los dos ejemplos en cada corrida de la suite.

| Ejemplo | Instrucciones |
|---|---:|
{filas_ejemplos}
"""


def main() -> int:
    """Punto de entrada.

    Returns:
        ``0`` si los cuatro criterios se cumplen, ``1`` si alguno no.

    Raises:
        SystemExit: Si falta el documento de reproducibilidad.
    """
    parser = argparse.ArgumentParser(description="Criterios de éxito de ingeniería.")
    parser.add_argument(
        "--sin-suite", action="store_true",
        help="Leer la cobertura de la última ejecución de la suite en vez de ejecutarla.",
    )
    args = parser.parse_args()

    if not REPRODUCIBILIDAD.is_file():
        raise SystemExit(
            f"No existe {REPRODUCIBILIDAD}. Ejecute antes scripts/verify_reproducibility.py."
        )
    logger.info("(a) Pruebas de %s contra el código actual...", COMMIT_MIGRACION)
    regresion = criterio_regresion()
    logger.info("(b) %s", "Cobertura existente..." if args.sin_suite else "Suite completa...")
    suite, datos = medir_cobertura(ejecutar_suite=not args.sin_suite)
    cobertura = criterio_cobertura(datos, modulos_migrados())
    reproducibilidad = criterio_reproducibilidad(REPRODUCIBILIDAD.read_text(encoding="utf-8"))
    instrucciones = criterio_instrucciones()

    SALIDA.write_text(
        redactar(regresion, cobertura, reproducibilidad, instrucciones, suite), encoding="utf-8"
    )
    criterios = {
        "(a) regresión": regresion,
        "(b) cobertura": cobertura,
        "(c) reproducibilidad": reproducibilidad,
        "(d) instrucciones": instrucciones,
    }
    for nombre, criterio in criterios.items():
        logger.info("  %-22s %s", nombre, "cumple" if criterio["cumple"] else "NO CUMPLE")
    logger.info("Documento en %s", SALIDA)
    return 0 if all(c["cumple"] for c in criterios.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
