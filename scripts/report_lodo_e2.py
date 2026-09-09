# scripts/report_lodo_e2.py
"""Reporte del experimento 2 (pre-registro v2.0).

Lee el documento de resultados que escribió ``scripts/run_lodo_e2.py`` y redacta el
resumen en Markdown: veredicto, los dos contrastes confirmatorios con su corrección
de Holm, los secundarios, la robustez y la tabla por conjunto de datos.

**No calcula ningún estadístico: los transcribe.** Es la misma regla que gobierna el
reporte del experimento 1, y por la misma razón: el análisis lo fija el pre-registro
y vive junto al arnés que lo ejecuta.

La excepción, también como allá, es el veredicto: se **deriva de las cifras** en vez
de copiar la bandera ``cumple_criterio``, y denuncia la discrepancia si ambas no
coinciden. Aquí la significancia se juzga con el valor $p$ **ajustado por Holm**,
nunca con el crudo: son dos contrastes confirmatorios declarados y juzgarlos sin
corregir inflaría la probabilidad de declarar favorable alguno por azar.

Ejecutar con:
    .\\.venv\\Scripts\\python.exe scripts\\report_lodo_e2.py
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from datetime import date
from pathlib import Path

from autopipe.search_space import TABULAR_SPACE_VERSION

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

RESULTADOS = Path("docs/experimentos/LODO-E2-resultados.json")
RESUMEN = Path("docs/experimentos/LODO-E2-resultados.md")

REDUCCION_EXIGIDA = 0.30
ALFA = 0.05

PROTOCOLO = {
    "Pre-registro": "experimento 2, v2.0, etiqueta prereg-v2.0",
    "Espacio de búsqueda": f"{TABULAR_SPACE_VERSION} (144 pipelines, sin cambios)",
    "Modelo base": "KNeighborsClassifier(n_neighbors=5, metric=minkowski, p=2, brute)",
    "Métrica": "macro-F1, partición estratificada con 20 % de validación",
    "Corpus": "42 conjuntos (25 del experimento 1 + 17; los 10 de familia siguen fuera)",
    "Semillas": "101, 202, 303, 404, 505, 606, 707, 808, 909, 1010",
    "Presupuesto": "60 ensayos por condición; censura en 61",
    "Objetivo": "99 % del óptimo exhaustivo por pareja (conjunto, semilla)",
    "Variable primaria": "regret normalizado sobre la curva (menor es mejor)",
    "Variable secundaria": "ensayos hasta el objetivo, con el criterio del anteproyecto",
    "Contrastes confirmatorios": "nearest contra frio y nearest contra frequent",
    "Corrección": "Holm-Bonferroni sobre la familia de dos, α = 0,05",
}


def _numero(valor: object, decimales: int = 4) -> str:
    """Presenta un número, o un guion si no hay valor."""
    if valor is None:
        return "—"
    if isinstance(valor, bool):
        return "sí" if valor else "no"
    if isinstance(valor, (int, float)):
        return f"{valor:.{decimales}f}"
    return str(valor)


def _porcentaje(reduccion: object) -> str:
    """Presenta una reducción relativa como porcentaje."""
    if not isinstance(reduccion, (int, float)):
        return "—"
    return f"{float(reduccion) * 100:.1f} %"


def redactar_veredicto(contraste: Mapping[str, object]) -> str:
    """Enuncia el resultado de un contraste confirmatorio contra el criterio.

    El criterio del pre-registro v2.0 exige **las dos** condiciones a la vez: una
    reducción de al menos el 30 % de la mediana del regret y un valor $p$ de una
    cola **ajustado por Holm** por debajo de 0,05.

    Se deriva de las cifras y no se copia de la bandera ``cumple_criterio``. Si
    ambas discrepan, el texto lo denuncia en vez de elegir una.

    Args:
        contraste: Contraste confirmatorio tal como lo produce el arnés.

    Returns:
        El párrafo de veredicto.
    """
    reduccion = contraste.get("reduccion")
    ajustado = contraste.get("p_ajustado")
    par = f"`{contraste['condicion']}` contra `{contraste['referencia']}`"
    cabecera = (
        f"{par}: regret mediano {_numero(contraste['mediana_referencia'])} contra "
        f"{_numero(contraste['mediana_condicion'])}."
    )

    magnitud = isinstance(reduccion, (int, float)) and reduccion >= REDUCCION_EXIGIDA
    significancia = isinstance(ajustado, (int, float)) and ajustado < ALFA
    cumple = magnitud and significancia

    declarado = contraste.get("cumple_criterio")
    if declarado is not None and bool(declarado) != cumple:
        return (
            f"{cabecera} El análisis declara «cumple={bool(declarado)}» pero sus "
            f"propias cifras dicen lo contrario: reducción {_porcentaje(reduccion)} y "
            f"valor p ajustado {_numero(ajustado)}. No se enuncia ningún veredicto "
            f"hasta resolver la discrepancia."
        )

    if cumple:
        return (
            f"{cabecera} **Cumple el criterio**: reducción del "
            f"{_porcentaje(reduccion)} —se exige 30 %— con un valor p ajustado por "
            f"Holm de {_numero(ajustado)}, por debajo de 0,05."
        )

    faltas = []
    if not magnitud:
        faltas.append(
            f"la reducción es del {_porcentaje(reduccion)} y se exige un 30 %"
        )
    if not significancia:
        crudo = _numero(contraste.get("p_una_cola"))
        faltas.append(
            f"el valor p ajustado por Holm es {_numero(ajustado)} —crudo {crudo}— y "
            f"se exige menos de 0,05"
        )
    return f"{cabecera} **No cumple el criterio**: " + "; ".join(faltas) + "."


def _fila(valores: list[str]) -> str:
    """Arma una fila de tabla Markdown."""
    return "| " + " | ".join(valores) + " |"


def _tabla_confirmatorios(contrastes: list[dict]) -> str:
    """Tabula los contrastes confirmatorios con su corrección."""
    cabecera = [
        "Contraste", "Regret referencia", "Regret condición", "Reducción",
        "p (1 cola)", "p (Holm)", "Efecto", "Empates", "Cumple",
    ]
    lineas = [_fila(cabecera), "|" + "|".join(["---"] + ["---:"] * 8) + "|"]
    for c in contrastes:
        lineas.append(
            _fila(
                [
                    f"`{c['condicion']}` vs `{c['referencia']}`",
                    _numero(c["mediana_referencia"]),
                    _numero(c["mediana_condicion"]),
                    _porcentaje(c.get("reduccion")),
                    _numero(c.get("p_una_cola")),
                    _numero(c.get("p_ajustado")),
                    _numero(c.get("efecto_rango_biserial"), 3),
                    str(c.get("n_empates", 0)),
                    _numero(c.get("cumple_criterio")),
                ]
            )
        )
    return "\n".join(lineas)


def _tabla_secundarios(contrastes: list[dict]) -> str:
    """Tabula los análisis secundarios, sin corrección por multiplicidad."""
    cabecera = [
        "Contraste", "Variable", "Mediana referencia", "Mediana condición",
        "Reducción", "p (1 cola)", "Efecto",
    ]
    lineas = [_fila(cabecera), "|" + "|".join(["---", "---"] + ["---:"] * 5) + "|"]
    for c in contrastes:
        decimales = 4 if c["variable"] == "regret" else 1
        lineas.append(
            _fila(
                [
                    f"`{c['condicion']}` vs `{c['referencia']}`",
                    c["variable"],
                    _numero(c["mediana_referencia"], decimales),
                    _numero(c["mediana_condicion"], decimales),
                    _porcentaje(c.get("reduccion")),
                    _numero(c.get("p_una_cola")),
                    _numero(c.get("efecto_rango_biserial"), 3),
                ]
            )
        )
    return "\n".join(lineas)


def _tabla_robustez(contrastes: list[dict]) -> str:
    """Tabula el recálculo de la variable secundaria con censuras más severas."""
    cabecera = ["Contraste", "Censura", "Mediana referencia", "Mediana condición",
                "Reducción", "p (1 cola)"]
    lineas = [_fila(cabecera), "|" + "|".join(["---"] + ["---:"] * 5) + "|"]
    for c in contrastes:
        lineas.append(
            _fila(
                [
                    f"`{c['condicion']}` vs `{c['referencia']}`",
                    str(c["censura"]),
                    _numero(c["mediana_referencia"], 1),
                    _numero(c["mediana_condicion"], 1),
                    _porcentaje(c.get("reduccion")),
                    _numero(c.get("p_una_cola")),
                ]
            )
        )
    return "\n".join(lineas)


def _condiciones(por_dataset: list[dict]) -> list[str]:
    """Deduce las condiciones ejecutadas de las columnas de la tabla."""
    excluidas = {"dataset_id", "dataset_name", "meseta"}
    return [
        clave
        for clave in por_dataset[0]
        if clave not in excluidas and not clave.endswith(("_inevaluable", "_ensayos"))
    ]


def _tabla_datasets(por_dataset: list[dict], condiciones: list[str]) -> str:
    """Tabula el regret mediano por conjunto de datos."""
    cabecera = ["Conjunto de datos", "Meseta"] + condiciones + ["Inevaluables (frío)"]
    lineas = [_fila(cabecera), "|" + "|".join(["---"] + ["---:"] * (len(cabecera) - 1)) + "|"]
    for f in sorted(por_dataset, key=lambda f: f["nearest"] - f["frio"]):
        lineas.append(
            _fila(
                [f"`{f['dataset_name']}`", _numero(f.get("meseta"), 0)]
                + [_numero(f[c]) for c in condiciones]
                + [_numero(f.get("frio_inevaluable"), 3)]
            )
        )
    return "\n".join(lineas)


def _recuento(por_dataset: list[dict], condicion: str, referencia: str) -> tuple[int, int, int]:
    """Cuenta en cuántos conjuntos gana cada condición."""
    diferencias = [f[condicion] - f[referencia] for f in por_dataset]
    return (
        sum(1 for d in diferencias if d < 0),
        sum(1 for d in diferencias if d == 0),
        sum(1 for d in diferencias if d > 0),
    )


def redactar_resumen(resultado: dict) -> str:
    """Redacta el resumen en Markdown del experimento 2.

    Args:
        resultado: Documento completo que escribió el arnés.

    Returns:
        El documento Markdown.
    """
    analisis = resultado["analisis"]
    confirmatorios = analisis["confirmatorios"]
    por_dataset = analisis["por_dataset"]
    condiciones = _condiciones(por_dataset)
    protocolo = "\n".join(f"- **{k}:** {v}" for k, v in PROTOCOLO.items())
    veredictos = "\n\n".join(f"{i}. {redactar_veredicto(c)}"
                             for i, c in enumerate(confirmatorios, 1))

    recuentos = "\n".join(
        f"- `{c['condicion']}` contra `{c['referencia']}`: gana en **{g}**, empata en "
        f"**{e}** y pierde en **{p}** de {len(por_dataset)}."
        for c in confirmatorios
        for g, e, p in [_recuento(por_dataset, c["condicion"], c["referencia"])]
    )

    censurados = "\n".join(
        f"- `{condicion}`: {n} de {analisis['n_corridas_por_condicion']}"
        for condicion, n in analisis["censurados"].items()
    )

    return f"""# Resultados del experimento 2 (RE6)

> **Archivo generado.** No editar a mano: se regenera con
> `.\\.venv\\Scripts\\python.exe scripts\\report_lodo_e2.py`.

- **Generado:** {date.today().isoformat()}
- **Datos:** `{RESULTADOS.as_posix()}`
- **Duración de la ejecución:** {resultado['minutos']:.1f} min
- **Conjuntos de datos:** {len(por_dataset)} · **Corridas por condición:**
  {analisis['n_corridas_por_condicion']}

## Veredicto

La hipótesis del trabajo se juega en los **dos contrastes confirmatorios**, y solo en
ellos. El primero pregunta si el arranque en caliente meta-aprendido reduce el costo
de la búsqueda; el segundo, si la **similitud** aporta algo por encima de un buen
valor por defecto.

{veredictos}

{recuentos}

## Protocolo ejecutado

{protocolo}

## Contrastes confirmatorios

Menor regret es mejor. El valor $p$ que decide es el **ajustado por Holm**: son dos
contrastes declarados y juzgarlos a 0,05 sin corregir inflaría la probabilidad de
declarar favorable alguno por azar. El efecto es la correlación biserial por rangos
para muestras pareadas, negativa cuando la condición reduce el regret.

{_tabla_confirmatorios(confirmatorios)}

## Análisis secundarios

Declarados como tales en el pre-registro. **No sustituyen a los confirmatorios** y no
llevan corrección por multiplicidad, porque no se usan para decidir sobre la
hipótesis. Los dos primeros repiten los contrastes confirmatorios sobre la variable
del experimento 1, que es lo que permite comparar los dos experimentos en los mismos
términos.

{_tabla_secundarios(analisis['secundarios'])}

## Robustez al valor de censura

El regret **no se censura**, de modo que el análisis primario es inmune a esta regla.
Estas filas recalculan solo la variable secundaria con los castigos más severos que
el pre-registro fijó antes de ejecutar.

{_tabla_robustez(analisis['robustez'])}

## Corridas censuradas

Con vara externa el arranque en frío **sí puede quedarse corto**, a diferencia del
experimento 1, donde era imposible por construcción.

{censurados}

## Resultado por conjunto de datos

Regret mediano de las diez semillas, ordenado por la diferencia entre `nearest` y
`frio`. «Meseta» es cuántas de las 144 configuraciones alcanzan el objetivo del 99 %,
recalculada para este experimento: la del experimento 1 estaba medida al 95 % y no
corresponde al objetivo vigente.

{_tabla_datasets(por_dataset, condiciones)}
"""


def main() -> int:
    """Punto de entrada.

    Returns:
        ``0`` si el reporte se generó.

    Raises:
        SystemExit: Si no existe el documento de resultados.
    """
    if not RESULTADOS.is_file():
        raise SystemExit(f"No existe {RESULTADOS}. Ejecute antes scripts/run_lodo_e2.py.")
    resultado = json.loads(RESULTADOS.read_text(encoding="utf-8"))
    RESUMEN.write_text(redactar_resumen(resultado), encoding="utf-8")

    logger.info("Resumen escrito en %s", RESUMEN)
    for contraste in resultado["analisis"]["confirmatorios"]:
        logger.info("%s", redactar_veredicto(contraste))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
