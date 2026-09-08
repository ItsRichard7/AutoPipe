# scripts/report_lodo.py
"""Produce los reportes comparativos del experimento LODO (RE6).

Lee el documento de resultados que escribió ``scripts/run_lodo.py`` y genera dos
salidas de la misma fuente: el reporte HTML —con las gráficas, para leerlo— y un
resumen en Markdown —para versionarlo y citarlo desde el informe—.

**No recalcula nada.** Todos los estadísticos vienen del arnés, que es donde el
pre-registro los fija. Si este script los recomputara existirían dos
implementaciones del mismo contraste y nada garantizaría que coincidan.

Ejecutar con:
    .\\.venv\\Scripts\\python.exe scripts\\report_lodo.py
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path

from autopipe.reporter import ComparativeReportGenerator, redactar_veredicto
from autopipe.search_space import TABULAR_SPACE_VERSION

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

RESULTADOS = Path("docs/experimentos/LODO-resultados.json")
RESUMEN = Path("docs/experimentos/LODO-resultados.md")
REPORTES = Path("docs/experimentos/reportes")

PROTOCOLO = {
    "Pre-registro": "v1.0, etiqueta prereg-v1.0",
    "Espacio de búsqueda": f"{TABULAR_SPACE_VERSION} (144 pipelines con omisión)",
    "Modelo base": "KNeighborsClassifier(n_neighbors=5, metric=minkowski, p=2, brute)",
    "Métrica": "macro-F1, partición estratificada con 20 % de validación",
    "Semillas": "101, 202, 303, 404, 505 (idénticas en ambas condiciones)",
    "Presupuesto": "30 ensayos por condición; censura en 31",
    "Arranque en caliente": "5 configuraciones inyectadas con enqueue_trial, que cuentan",
    "Objetivo": "95 % del mejor valor del arranque en frío, por pareja (dataset, semilla)",
    "Prueba": "Wilcoxon pareada de una cola, α = 0,05",
}


def _fila_markdown(valores: list[str]) -> str:
    """Arma una fila de tabla Markdown."""
    return "| " + " | ".join(valores) + " |"


def _numero(valor: object, decimales: int = 2) -> str:
    """Presenta un número, o un guion si no hay valor."""
    if valor is None:
        return "—"
    if isinstance(valor, bool):
        return "sí" if valor else "no"
    if isinstance(valor, (int, float)):
        return f"{valor:.{decimales}f}"
    return str(valor)


def _tabla_contrastes(contrastes: list[dict], con_censura: bool = False) -> str:
    """Tabula una lista de contrastes en Markdown.

    Args:
        contrastes: Contrastes a tabular.
        con_censura: Si se incluye la columna del valor de censura.

    Returns:
        La tabla Markdown.
    """
    con_papel = any("papel" in c for c in contrastes)
    cabecera = (
        ["Estrategia"]
        + (["Papel"] if con_papel else [])
        + (["Censura"] if con_censura else [])
        + [
            "Mediana frío", "Mediana caliente", "Reducción", "p (1 cola)",
            "p (2 colas)", "Censurados", "Cumple el criterio",
        ]
    )
    lineas = [
        _fila_markdown(cabecera),
        "|" + "|".join(["---"] + ["---:"] * (len(cabecera) - 1)) + "|",
    ]
    for c in contrastes:
        reduccion = c.get("reduccion")
        fila = [f"`{c['estrategia']}`"]
        if con_papel:
            fila.append(f"**{c['papel']}**" if c["papel"] == "primario" else c["papel"])
        if con_censura:
            fila.append(str(c["censura"]))
        fila += [
            _numero(c["mediana_frio"], 1),
            _numero(c["mediana_caliente"], 1),
            "—" if reduccion is None else f"{reduccion * 100:.1f} %",
            _numero(c.get("p_una_cola"), 4),
            _numero(c.get("p_dos_colas"), 4),
            str(c.get("censurados_caliente", 0)),
            _numero(c.get("confirma")),
        ]
        lineas.append(_fila_markdown(fila))
    return "\n".join(lineas)


def _tabla_datasets(por_dataset: list[dict], condiciones: list[str]) -> str:
    """Tabula el resultado por conjunto de datos en Markdown.

    Args:
        por_dataset: Filas por conjunto de datos.
        condiciones: Condiciones ejecutadas, empezando por ``frio``.

    Returns:
        La tabla Markdown.
    """
    cabecera = ["Conjunto de datos", "Meseta"] + condiciones + ["Inevaluables (frío)"]
    lineas = [
        _fila_markdown(cabecera),
        "|" + "|".join(["---"] + ["---:"] * (len(cabecera) - 1)) + "|",
    ]
    for fila in sorted(por_dataset, key=lambda f: f["nearest"] - f["frio"]):
        lineas.append(
            _fila_markdown(
                [f"`{fila['dataset_name']}`", _numero(fila.get("meseta"), 0)]
                + [_numero(fila[c], 1) for c in condiciones]
                + [_numero(fila.get("frio_inevaluable"), 3)]
            )
        )
    return "\n".join(lineas)


def _recuento(por_dataset: list[dict]) -> tuple[int, int, int]:
    """Cuenta en cuántos conjuntos ganó cada condición.

    Args:
        por_dataset: Filas por conjunto de datos.

    Returns:
        Tupla ``(gana_caliente, empatan, gana_frio)``.
    """
    diferencias = [f["nearest"] - f["frio"] for f in por_dataset]
    return (
        sum(1 for d in diferencias if d < 0),
        sum(1 for d in diferencias if d == 0),
        sum(1 for d in diferencias if d > 0),
    )


def redactar_resumen(resultado: dict, ruta_html: Path) -> str:
    """Redacta el resumen en Markdown del experimento.

    Args:
        resultado: Documento completo que escribió el arnés.
        ruta_html: Ruta del reporte HTML generado, para citarla.

    Returns:
        El documento Markdown.
    """
    analisis = resultado["analisis"]
    primario = analisis["primario"]
    por_dataset = analisis["por_dataset"]
    primaria = primario["estrategia"]
    otras = [
        c
        for c in por_dataset[0]
        if c not in ("dataset_id", "dataset_name", "meseta", "frio", primaria)
        and not c.endswith(("_inevaluable", "_a_15"))
    ]
    condiciones = ["frio", primaria, *otras]
    gana_caliente, empatan, gana_frio = _recuento(por_dataset)
    protocolo = "\n".join(f"- **{k}:** {v}" for k, v in PROTOCOLO.items())
    tabla_primaria = _tabla_contrastes(
        [
            {**primario, "papel": "primario"},
            *({**c, "papel": "control"} for c in analisis["secundarios"]),
        ]
    )

    return f"""# Resultados del experimento LODO (RE6)

> **Archivo generado.** No editar a mano: se regenera con
> `.\\.venv\\Scripts\\python.exe scripts\\report_lodo.py`.

- **Generado:** {date.today().isoformat()}
- **Datos:** `{RESULTADOS.as_posix()}`
- **Reporte con gráficas:** `{ruta_html.as_posix()}`
- **Duración de la ejecución:** {resultado['minutos']:.1f} min

## Veredicto

{redactar_veredicto(primario)}

De los {primario['n_pares']} conjuntos de datos, el arranque en caliente alcanzó el
objetivo antes en **{gana_caliente}**, empató en **{empatan}** y llegó después en
**{gana_frio}**.

## Protocolo ejecutado

{protocolo}

## Contraste primario y controles secundarios

Las estrategias de control comparten todo con la primaria salvo cómo eligen las cinco
configuraciones: `random` inyecta las ganadoras de conjuntos al azar y `frequent` las
más frecuentes del pliegue. Si `nearest` no se separa de ellas, la similitud entre
conjuntos de datos no está aportando nada.

**La hipótesis del trabajo se juega únicamente en la fila primaria.** Que un control
cumpla el criterio no la confirma: es un hallazgo sobre ese control, y sobre lo que su
ventaja implica para el meta-aprendizaje.

{tabla_primaria}

## Robustez al valor de censura

El análisis primario castiga con {primario['censura']} a las corridas que no alcanzan
el objetivo. Estos recálculos usan los castigos más severos que el pre-registro fijó
antes de ejecutar.

{_tabla_contrastes(analisis['robustez'], con_censura=True)}

## Resultado por conjunto de datos

Cada celda es la mediana de las cinco semillas, ordenada por la diferencia entre
`nearest` y `frio`. «Meseta» es el número de configuraciones del espacio que alcanzan
el 95 % del óptimo, y «inevaluables» la fracción de ensayos que no se pudieron
evaluar: las dos covariables declaradas en el pre-registro, medidas antes de ejecutar.

{_tabla_datasets(por_dataset, condiciones)}
"""


def main() -> int:
    """Punto de entrada.

    Returns:
        ``0`` si los reportes se generaron.

    Raises:
        SystemExit: Si no existe el documento de resultados.
    """
    parser = argparse.ArgumentParser(description="Reportes comparativos del LODO.")
    parser.add_argument("--sin-pdf", action="store_true", help="No intentar el PDF.")
    args = parser.parse_args()

    if not RESULTADOS.is_file():
        raise SystemExit(f"No existe {RESULTADOS}. Ejecute antes scripts/run_lodo.py.")
    resultado = json.loads(RESULTADOS.read_text(encoding="utf-8"))

    rutas = ComparativeReportGenerator(REPORTES).generate_comparison(
        resultado["analisis"], protocolo=PROTOCOLO, pdf=not args.sin_pdf
    )
    RESUMEN.write_text(redactar_resumen(resultado, rutas["html"]), encoding="utf-8")

    logger.info("Resumen escrito en %s", RESUMEN)
    logger.info("%s", redactar_veredicto(resultado["analisis"]["primario"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
