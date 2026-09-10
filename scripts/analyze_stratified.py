# scripts/analyze_stratified.py
"""Análisis estratificado por sensibilidad al preprocesamiento.

Es un **análisis secundario declarado**, no exploratorio. El pre-registro v1.0 fijó el
umbral ---un conjunto de datos es sensible al preprocesamiento si su IQR de macro-F1 es
$\\geq 0{,}02$--- *antes* de observar la distribución de los valores, y declaró que ese
mismo umbral define la estratificación del plan de análisis. El pre-registro del
experimento 2 hereda literalmente lo que no menciona, de modo que la estratificación
sigue vigente.

**En el experimento 1 quedó vacía por construcción**, tal como el propio pre-registro
anticipó: el cribado había seleccionado 25 conjuntos que superaban todos el umbral, así
que el subconjunto sensible coincidía con el corpus entero. En el experimento 2, al
ampliar el corpus a 42, **ocho conjuntos quedan por debajo** y la estratificación pasa a
tener contenido.

**El umbral no se redefine.** El pre-registro lo prohíbe explícitamente: fijarlo después
de ver la distribución para forzar una partición del corpus sería exactamente el grado
de libertad que un pre-registro existe para cerrar.

Los contrastes se calculan con la misma función del arnés que produjo los del análisis
principal, de modo que no existen dos implementaciones del mismo estadístico.

Ejecutar con:
    .\\.venv\\Scripts\\python.exe scripts\\analyze_stratified.py
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date
from pathlib import Path

from run_lodo_e2 import CONFIRMATORIOS, contrastar

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

RESULTADOS = Path("docs/experimentos/LODO-E2-resultados.json")
CORPUS = Path("docs/experimentos/CORPUS-E2.md")
SALIDA = Path("docs/experimentos/ESTRATIFICADO-E2.md")

UMBRAL = 0.02
"""IQR de macro-F1 a partir del cual un conjunto se considera sensible.

Fijado en el pre-registro v1.0 antes de observar la distribución. **No se redefine.**
"""

FILA = re.compile(r"^\|\s*\d+\s*\|\s*(\d+)\s*\|\s*`([^`]+)`\s*\|\s*([\d.]+)\s*\|")


def leer_sensibilidad(ruta: Path) -> dict[str, float]:
    """Lee el IQR de sensibilidad de cada conjunto del corpus.

    Args:
        ruta: Documento ``CORPUS-E2.md``.

    Returns:
        Diccionario de identificador a IQR.

    Raises:
        SystemExit: Si el archivo no existe o no tiene filas legibles.
    """
    if not ruta.is_file():
        raise SystemExit(f"No existe {ruta}. Ejecute antes scripts/build_corpus_e2.py.")
    valores = {
        m.group(1): float(m.group(3))
        for m in (FILA.match(linea) for linea in ruta.read_text("utf-8").splitlines())
        if m
    }
    if not valores:
        raise SystemExit(f"No se pudo leer ninguna fila de {ruta}.")
    return valores


def estratificar(pliegues: list[dict], sensibilidad: dict[str, float]) -> dict:
    """Reparte los pliegues en los dos estratos del pre-registro.

    Args:
        pliegues: Resultados de todos los pliegues.
        sensibilidad: IQR por identificador de conjunto.

    Returns:
        Diccionario con las claves ``sensibles`` e ``insensibles``.

    Raises:
        KeyError: Si algún pliegue no tiene sensibilidad conocida. Se rechaza en lugar
            de asignarlo a un estrato por omisión: una asignación silenciosa cambiaría
            el resultado del contraste.
    """
    faltan = [p["dataset_id"] for p in pliegues if p["dataset_id"] not in sensibilidad]
    if faltan:
        raise KeyError(
            f"No se conoce la sensibilidad de {faltan}, y asignarlos a un estrato por "
            f"omisión falsearía el contraste."
        )
    return {
        "sensibles": [p for p in pliegues if sensibilidad[p["dataset_id"]] >= UMBRAL],
        "insensibles": [p for p in pliegues if sensibilidad[p["dataset_id"]] < UMBRAL],
    }


def analizar(pliegues: list[dict], sensibilidad: dict[str, float]) -> dict:
    """Repite los contrastes confirmatorios dentro de cada estrato.

    Args:
        pliegues: Resultados de todos los pliegues.
        sensibilidad: IQR por identificador de conjunto.

    Returns:
        Diccionario con los contrastes por estrato y el reparto.
    """
    estratos = estratificar(pliegues, sensibilidad)
    return {
        "umbral": UMBRAL,
        "reparto": {nombre: len(grupo) for nombre, grupo in estratos.items()},
        "contrastes": {
            nombre: [
                contrastar(grupo, condicion, referencia)
                for condicion, referencia in CONFIRMATORIOS
            ]
            if grupo
            else []
            for nombre, grupo in estratos.items()
        },
        "conjuntos": {
            nombre: sorted(
                (p["dataset_name"], sensibilidad[p["dataset_id"]]) for p in grupo
            )
            for nombre, grupo in estratos.items()
        },
    }


def _porcentaje(reduccion: float | None) -> str:
    """Presenta una reducción relativa como porcentaje, o un guion si no existe."""
    return "—" if reduccion is None else f"{reduccion * 100:.1f} %"


def _fila(contraste: dict) -> str:
    """Arma la fila de un contraste."""
    return (
        f"| `{contraste['condicion']}` vs `{contraste['referencia']}` "
        f"| {contraste['mediana_referencia']:.4f} "
        f"| {contraste['mediana_condicion']:.4f} "
        f"| {_porcentaje(contraste['reduccion'])} "
        f"| {contraste['p_una_cola']:.4f} "
        f"| {contraste['efecto_rango_biserial']:.3f} |"
    )


def _tabla(contrastes: list[dict], n: int) -> str:
    """Tabula los contrastes de un estrato.

    Args:
        contrastes: Contrastes calculados dentro del estrato.
        n: Cuántos conjuntos tiene el estrato.

    Returns:
        La tabla markdown, o una nota si el estrato quedó vacío.
    """
    if not contrastes:
        return f"_Estrato vacío ({n} conjuntos): no hay contraste que calcular._"
    cabecera = (
        "| Contraste | Regret referencia | Regret condición | Reducción | p (1 cola) "
        "| Efecto |\n|---|---:|---:|---:|---:|---:|"
    )
    return cabecera + "\n" + "\n".join(_fila(c) for c in contrastes)


def redactar(analisis: dict) -> str:
    """Redacta el documento del análisis estratificado.

    Args:
        analisis: Resultado de :func:`analizar`.

    Returns:
        El documento markdown.
    """
    reparto = analisis["reparto"]
    listado = "\n".join(
        f"- `{nombre}` (IQR {iqr:.4f})"
        for nombre, iqr in analisis["conjuntos"]["insensibles"]
    ) or "_Ninguno._"
    return f"""# Análisis estratificado por sensibilidad al preprocesamiento

> **Archivo generado.** No editar a mano: se regenera con
> `.\\.venv\\Scripts\\python.exe scripts\\analyze_stratified.py`.

> **Análisis secundario declarado**, no exploratorio. El umbral y la estratificación
> los fija el pre-registro v1.0, antes de observar ninguna distribución, y el
> pre-registro del experimento 2 los hereda. No lleva corrección por multiplicidad
> porque no decide sobre la hipótesis: eso lo hacen los dos contrastes confirmatorios.

- **Generado:** {date.today().isoformat()}
- **Umbral:** IQR de macro-F1 $\\geq$ {analisis['umbral']}
- **Reparto:** {reparto['sensibles']} sensibles, {reparto['insensibles']} insensibles

## Por qué ahora tiene contenido

En el **experimento 1** este análisis quedó **vacío por construcción**: el cribado había
seleccionado 25 conjuntos que superaban todos el umbral, de modo que el subconjunto
sensible coincidía con el corpus entero. El pre-registro lo había anticipado y pedía
reportarlo como tal.

El **experimento 2** amplía el corpus a 42 incorporando conjuntos que habían quedado por
debajo del corte por sensibilidad, y **ocho de ellos caen bajo el umbral**. La
estratificación pasa a ser calculable sin haber tocado el umbral, que sigue siendo el
mismo número fijado antes de ver ningún dato.

### Conjuntos del estrato insensible

{listado}

## Estrato sensible ({reparto['sensibles']} conjuntos)

{_tabla(analisis['contrastes']['sensibles'], reparto['sensibles'])}

## Estrato insensible ({reparto['insensibles']} conjuntos)

{_tabla(analisis['contrastes']['insensibles'], reparto['insensibles'])}

## Cómo leer esto

El modelo base se eligió **sensible al preprocesamiento** precisamente para evitar un
efecto piso: si la métrica apenas variara entre pipelines, ambas condiciones alcanzarían
el objetivo de inmediato y la comparación quedaría vacía. El estrato insensible es, por
definición, el terreno donde ese riesgo es mayor, y por eso el pre-registro pidió
mirarlos por separado.

Los contrastes de este documento **no sustituyen** a los confirmatorios: el estrato
insensible tiene {reparto['insensibles']} pares y cualquier valor $p$ calculado sobre él
tiene poca potencia. Se reportan por lo que la comparación entre estratos muestra sobre
las condiciones en las que el arranque en caliente ayuda, no como una decisión sobre la
hipótesis.
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
    analisis = analizar(resultado["pliegues"], leer_sensibilidad(CORPUS))
    SALIDA.write_text(redactar(analisis), encoding="utf-8")

    logger.info("Análisis escrito en %s", SALIDA)
    logger.info(
        "Reparto: %d sensibles, %d insensibles (umbral %.2f).",
        analisis["reparto"]["sensibles"],
        analisis["reparto"]["insensibles"],
        UMBRAL,
    )
    for estrato, contrastes in analisis["contrastes"].items():
        for c in contrastes:
            logger.info(
                "  %-12s %-9s vs %-9s  regret %.4f -> %.4f  p %.4f",
                estrato,
                c["condicion"],
                c["referencia"],
                c["mediana_referencia"],
                c["mediana_condicion"],
                c["p_una_cola"],
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
