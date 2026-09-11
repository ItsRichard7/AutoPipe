# scripts/tabulate_by_dataset.py
"""Tabla por conjunto de datos de los dos experimentos, completa.

El pre-registro v1.0 declaró entre sus análisis secundarios una **tabla por conjunto**
con sensibilidad, margen sobre el pipeline vacío, ensayos hasta el objetivo por semilla y
condición, casos censurados, desempeño a quince ensayos y las dos covariables. Pidió
además que los conjuntos planos aparecieran **en su propia fila con su sensibilidad
visible**, no promediados dentro de la mediana global.

Los resúmenes de los dos experimentos traen una tabla por conjunto, pero de medianas: les
faltan la sensibilidad, el margen, el detalle por semilla y los censurados. Todo eso ya
estaba medido ---en el cribado del corpus y en las corridas de los arneses--- y aquí solo
se cruza. **No ejecuta ningún ensayo.**

El pre-registro v2.0 hereda literalmente lo que no menciona, y no menciona esta tabla, de
modo que rige también para el experimento 2, con su propia censura y sus presupuestos
reducidos.

El documento del experimento 2 lleva además el análisis **exploratorio** que su
pre-registro declaró y que faltaba: la relación entre la anchura de la meseta y el
beneficio observado.

Ejecutar con:
    .\\.venv\\Scripts\\python.exe scripts\\tabulate_by_dataset.py
"""

from __future__ import annotations

import json
import logging
import statistics
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import run_lodo
import run_lodo_e2
from analyze_reduced_budget import mejor_valor
from analyze_stratified import UMBRAL
from scipy.stats import pearsonr, spearmanr

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

CRIBADO = Path("docs/experimentos/CORPUS-cribado-valores.json")


@dataclass(frozen=True)
class Experimento:
    """Lo que distingue a un experimento a la hora de tabularlo.

    Attributes:
        nombre: Nombre para el documento.
        preregistro: De dónde sale la obligación de la tabla.
        resultados: Documento de resultados del arnés.
        salida: Documento a escribir.
        arnes: Script que produce los resultados, para los mensajes.
        condiciones: Condiciones en el orden en que se tabulan.
        censura: Valor que el arnés asigna a una corrida que no alcanza el objetivo.
        presupuestos: Presupuestos reducidos que el arnés guardó.
        meseta: Fracción del óptimo a la que se midió la meseta.
        explorar_meseta: Si su pre-registro declaró la relación entre meseta y
            beneficio.
    """

    nombre: str
    preregistro: str
    resultados: Path
    salida: Path
    arnes: str
    condiciones: tuple[str, ...]
    censura: int
    presupuestos: tuple[int, ...]
    meseta: str
    explorar_meseta: bool


EXPERIMENTO_1 = Experimento(
    nombre="experimento 1",
    preregistro="declarado en el pre-registro v1.0",
    resultados=Path("docs/experimentos/LODO-resultados.json"),
    salida=Path("docs/experimentos/LODO-por-dataset.md"),
    arnes="run_lodo.py",
    condiciones=("frio", *run_lodo.ESTRATEGIAS_CALIENTE),
    censura=run_lodo.CENSURA,
    presupuestos=(run_lodo.PRESUPUESTO_REDUCIDO,),
    meseta="95 %",
    explorar_meseta=False,
)

EXPERIMENTO_2 = Experimento(
    nombre="experimento 2",
    preregistro=(
        "declarado en el pre-registro v1.0 y heredado por el v2.0, que hereda "
        "literalmente lo que no menciona"
    ),
    resultados=Path("docs/experimentos/LODO-E2-resultados.json"),
    salida=Path("docs/experimentos/LODO-E2-por-dataset.md"),
    arnes="run_lodo_e2.py",
    condiciones=("frio", *run_lodo_e2.CONDICIONES_CALIENTES),
    censura=run_lodo_e2.CENSURA,
    presupuestos=run_lodo_e2.PRESUPUESTOS_REDUCIDOS,
    meseta="99 %",
    explorar_meseta=True,
)

TABULADOS = (EXPERIMENTO_1, EXPERIMENTO_2)


# --------------------------------------------------------------------------- #
# Las fuentes
# --------------------------------------------------------------------------- #
def leer_cribado(ruta: Path) -> dict[str, dict]:
    """Lee la sensibilidad y el margen sobre el pipeline vacío de cada candidato.

    Args:
        ruta: Documento ``CORPUS-cribado-valores.json``.

    Returns:
        Diccionario de identificador de OpenML, como texto, a sus medidas del cribado.
        El margen es ``None`` cuando el pipeline vacío no se pudo evaluar.

    Raises:
        SystemExit: Si el archivo no existe.
    """
    if not ruta.is_file():
        raise SystemExit(f"No existe {ruta}. Ejecute antes scripts/screen_corpus.py.")
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    return {
        str(r["did"]): {
            "iqr": r["iqr"],
            "margen": r["margen"],
            "categoricas": r["categoricas"],
            "faltantes": r["faltantes"],
        }
        for r in datos["resultados"]
    }


def resumir_condicion(semillas: list[dict], condicion: str, experimento: Experimento) -> dict:
    """Resume una condición de un conjunto a partir de sus corridas.

    Args:
        semillas: Resultados del conjunto por semilla.
        condicion: Condición a resumir.
        experimento: Experimento al que pertenecen, del que se toma la censura.

    Returns:
        Diccionario con los ensayos por semilla, su mediana, los censurados, el regret
        si el experimento lo midió, el mejor valor a cada presupuesto reducido y la
        fracción media de ensayos inevaluables.
    """
    ensayos = [int(s[condicion]["ensayos"]) for s in semillas]
    return {
        "condicion": condicion,
        "ensayos": ensayos,
        "mediana": statistics.median(ensayos),
        "censurados": sum(1 for e in ensayos if e >= experimento.censura),
        "regret": (
            statistics.median(s[condicion]["regret"] for s in semillas)
            if "regret" in semillas[0][condicion]
            else None
        ),
        "mejor": {
            k: statistics.median(mejor_valor(s[condicion], k) for s in semillas)
            for k in experimento.presupuestos
        },
        "inevaluable": statistics.mean(s[condicion]["fraccion_inevaluable"] for s in semillas),
    }


def tabular(resultado: dict, experimento: Experimento, cribado: dict[str, dict]) -> list[dict]:
    """Cruza las corridas de un experimento con el cribado del corpus.

    Args:
        resultado: Documento que escribió el arnés.
        experimento: Experimento que se tabula.
        cribado: Medidas del cribado por identificador.

    Returns:
        Una fila por conjunto, ordenadas de más a menos sensible, para que los conjuntos
        planos queden juntos y visibles al final.

    Raises:
        KeyError: Si algún conjunto no aparece en el cribado. Se rechaza en lugar de
            dejar la celda vacía: la sensibilidad es justo lo que el pre-registro pide
            ver fila por fila.
    """
    faltan = [p["dataset_name"] for p in resultado["pliegues"] if p["dataset_id"] not in cribado]
    if faltan:
        raise KeyError(f"No se conoce el cribado de {faltan}.")
    mesetas = {
        f["dataset_id"]: f.get("meseta")
        for f in resultado.get("analisis", {}).get("por_dataset", [])
    }
    filas = [
        {
            "dataset_id": p["dataset_id"],
            "dataset_name": p["dataset_name"],
            **cribado[p["dataset_id"]],
            "meseta": mesetas.get(p["dataset_id"]),
            "semillas": [s["semilla"] for s in p["semillas"]],
            "condiciones": [
                resumir_condicion(p["semillas"], c, experimento) for c in experimento.condiciones
            ],
        }
        for p in resultado["pliegues"]
    ]
    return sorted(filas, key=lambda f: (-f["iqr"], f["dataset_name"]))


# --------------------------------------------------------------------------- #
# El análisis exploratorio del experimento 2
# --------------------------------------------------------------------------- #
def _correlaciones(x: list[float], y: list[float]) -> dict:
    """Correlaciones de Spearman y de Pearson, con sus valores p de dos colas."""
    rho, p_rho = spearmanr(x, y)
    r, p_r = pearsonr(x, y)
    return {
        "n": len(x),
        "spearman": round(float(rho), 4),
        "p_spearman": float(p_rho),
        "pearson": round(float(r), 4),
        "p_pearson": float(p_r),
    }


def meseta_y_beneficio(filas: list[dict]) -> dict:
    """Relación entre la anchura de la meseta y el beneficio frente al arranque en frío.

    **Exploratorio**: el pre-registro v2.0 lo declaró así y de él no se deriva ninguna
    afirmación sobre la hipótesis. El beneficio de una condición en un conjunto es el
    regret mediano del frío menos el de la condición: **positivo cuando la condición
    mejora**. Una correlación negativa dice que el beneficio crece donde la meseta es
    estrecha.

    Se acompaña de la relación entre la meseta y el regret del propio frío, que es la
    covariable que ya reportan las discusiones, para poder situar una junto a la otra.

    Args:
        filas: Filas de :func:`tabular` de un experimento que midió el regret.

    Returns:
        Diccionario con la relación de referencia y una entrada por condición caliente.

    Raises:
        ValueError: Si el experimento no midió el regret.
    """
    if any(c["regret"] is None for f in filas for c in f["condiciones"]):
        raise ValueError(
            "La relación con el beneficio se mide sobre el regret, y este experimento "
            "no lo midió."
        )
    mesetas = [float(f["meseta"]) for f in filas]
    regrets = [{c["condicion"]: c["regret"] for c in f["condiciones"]} for f in filas]
    condiciones = []
    for condicion in regrets[0]:
        if condicion == "frio":
            continue
        beneficio = [r["frio"] - r[condicion] for r in regrets]
        condiciones.append(
            {
                "condicion": condicion,
                **_correlaciones(mesetas, beneficio),
                "con_beneficio": sum(1 for b in beneficio if b > 0),
                "beneficio_mediano": round(statistics.median(beneficio), 6),
            }
        )
    return {
        "referencia": _correlaciones(mesetas, [r["frio"] for r in regrets]),
        "condiciones": condiciones,
    }


# --------------------------------------------------------------------------- #
# El documento
# --------------------------------------------------------------------------- #
def _numero(valor: float | None) -> str:
    """Presenta un número sin ceros de relleno, o un guion si no hay valor."""
    return "—" if valor is None else f"{valor:g}"


def _celdas_de_conjunto(fila: dict) -> list[str]:
    """Las columnas propias del conjunto, que solo se escriben en su primera fila."""
    marca = " †" if fila["iqr"] < UMBRAL else ""
    margen = "n/e" if fila["margen"] is None else f"{fila['margen']:.4f}"
    return [
        f"`{fila['dataset_name']}`",
        f"{fila['iqr']:.4f}{marca}",
        margen,
        _numero(fila["meseta"]),
    ]


def _tabla(filas: list[dict], experimento: Experimento) -> str:
    """Tabula el experimento con una fila por conjunto y condición."""
    con_regret = experimento.explorar_meseta
    cabecera = (
        ["Conjunto", "IQR", "Margen sobre vacío", f"Meseta {experimento.meseta}", "Condición",
         "Ensayos por semilla", "Mediana", "Censurados"]
        + (["Regret"] if con_regret else [])
        + [f"Mejor a {k}" for k in experimento.presupuestos]
        + ["Inevaluables"]
    )
    alineacion = ["---", "---:", "---:", "---:", "---", "---"] + ["---:"] * (len(cabecera) - 6)
    lineas = ["| " + " | ".join(cabecera) + " |", "|" + "|".join(alineacion) + "|"]
    for fila in filas:
        for posicion, c in enumerate(fila["condiciones"]):
            celdas = _celdas_de_conjunto(fila) if posicion == 0 else [""] * 4
            celdas += [
                f"`{c['condicion']}`",
                ", ".join(str(e) for e in c["ensayos"]),
                _numero(c["mediana"]),
                str(c["censurados"]),
            ]
            if con_regret:
                celdas.append(f"{c['regret']:.4f}")
            celdas += [f"{c['mejor'][k]:.4f}" for k in experimento.presupuestos]
            celdas.append(f"{c['inevaluable']:.3f}")
            lineas.append("| " + " | ".join(celdas) + " |")
    return "\n".join(lineas)


def _seccion_meseta(relacion: dict) -> str:
    """Redacta la sección exploratoria sobre la meseta y el beneficio."""
    ref = relacion["referencia"]
    filas = [
        "| Relación | n | Spearman | p | Pearson | p | Conjuntos con beneficio "
        "| Beneficio mediano |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| Meseta y regret del frío (referencia) | {ref['n']} | {ref['spearman']:+.3f} | "
        f"{ref['p_spearman']:.4f} | {ref['pearson']:+.3f} | {ref['p_pearson']:.4f} | — | — |",
    ]
    filas += [
        f"| Meseta y beneficio de `{c['condicion']}` | {c['n']} | {c['spearman']:+.3f} | "
        f"{c['p_spearman']:.4f} | {c['pearson']:+.3f} | {c['p_pearson']:.4f} | "
        f"{c['con_beneficio']} | {c['beneficio_mediano']:+.4f} |"
        for c in relacion["condiciones"]
    ]
    tabla = "\n".join(filas)
    return f"""
## Anchura de la meseta y beneficio

> **Exploratorio, no confirmatorio**, declarado así en el pre-registro v2.0. No se
> deriva de él ninguna afirmación sobre la hipótesis.

El **beneficio** de una condición en un conjunto es el regret mediano del frío menos el de
la condición: positivo cuando la condición mejora. Una correlación **negativa** dice que el
beneficio crece donde la meseta es estrecha. Se dan las dos: la de Spearman, por rangos,
porque la meseta es un conteo muy asimétrico, y la de Pearson, que es la que usan las
discusiones para la covariable. Los valores p son de dos colas y no llevan corrección.

{tabla}
"""


def redactar(experimento: Experimento, filas: list[dict], relacion: dict | None = None) -> str:
    """Redacta el documento de la tabla por conjunto de un experimento.

    Args:
        experimento: Experimento que se tabula.
        filas: Filas de :func:`tabular`.
        relacion: Resultado de :func:`meseta_y_beneficio`, si corresponde.

    Returns:
        El documento markdown.
    """
    planos = [f for f in filas if f["iqr"] < UMBRAL]
    sin_vacio = [f for f in filas if f["margen"] is None]
    con_columnas = sum(1 for f in sin_vacio if f["categoricas"] or f["faltantes"])
    semillas = ", ".join(str(s) for s in filas[0]["semillas"])
    regret = (
        "- **Regret**: mediana del regret normalizado, la variable primaria de este "
        "experimento.\n"
        if experimento.explorar_meseta
        else ""
    )
    presupuestos = ", ".join(str(k) for k in experimento.presupuestos)
    return f"""# Tabla por conjunto de datos — {experimento.nombre}

> **Archivo generado.** No editar a mano: se regenera con
> `.\\.venv\\Scripts\\python.exe scripts\\tabulate_by_dataset.py`.

> **Análisis secundario**, {experimento.preregistro}. Cruza lo que ya estaba medido
> —el cribado del corpus y las corridas del arnés— y no ejecuta ningún ensayo.

- **Generado:** {date.today().isoformat()}
- **Datos:** `{experimento.resultados.as_posix()}` y `{CRIBADO.as_posix()}`
- **Conjuntos:** {len(filas)}; {len(planos)} por debajo del umbral de sensibilidad y
  {len(sin_vacio)} con el pipeline vacío inevaluable
- **Censura:** {experimento.censura}

## Cómo leer la tabla

- **IQR**: sensibilidad al preprocesamiento, medida en el cribado sobre doce pipelines al
  azar. La tabla va de más a menos sensible y **†** marca los conjuntos por debajo del
  umbral de {UMBRAL}, que el pre-registro pide ver en su propia fila y no promediados.
- **Margen sobre vacío**: mejor de esos doce pipelines menos el desempeño sin
  preprocesamiento. **n/e** cuando el pipeline vacío no se puede evaluar; de los
  {len(sin_vacio)} conjuntos en que ocurre, {con_columnas} tienen columnas categóricas o
  valores faltantes.
- **Meseta**: cuántas de las 144 configuraciones alcanzan el {experimento.meseta} del
  óptimo; es una de las dos covariables declaradas.
- **Ensayos por semilla**: ensayos hasta el objetivo, en el orden de las semillas
  {semillas}. Un {experimento.censura} es una corrida censurada.
- **Mediana** y **Censurados**: sobre esas mismas semillas.
{regret}- **Mejor a k**: mediana del mejor macro-F1 tras k ensayos ({presupuestos}).
- **Inevaluables**: fracción media de ensayos que no se pudieron evaluar; es la otra
  covariable declarada.

## Tabla

{_tabla(filas, experimento)}
{_seccion_meseta(relacion) if relacion is not None else ""}"""


def main() -> int:
    """Punto de entrada.

    Returns:
        ``0`` si los documentos se escribieron.

    Raises:
        SystemExit: Si falta el cribado o algún documento de resultados.
    """
    cribado = leer_cribado(CRIBADO)
    for experimento in TABULADOS:
        if not experimento.resultados.is_file():
            raise SystemExit(
                f"No existe {experimento.resultados}. Ejecute antes scripts/{experimento.arnes}."
            )
    for experimento in TABULADOS:
        resultado = json.loads(experimento.resultados.read_text(encoding="utf-8"))
        filas = tabular(resultado, experimento, cribado)
        relacion = meseta_y_beneficio(filas) if experimento.explorar_meseta else None
        experimento.salida.write_text(redactar(experimento, filas, relacion), encoding="utf-8")
        logger.info("Tabla del %s escrita en %s", experimento.nombre, experimento.salida)
        if relacion is not None:
            ref = relacion["referencia"]
            logger.info("  meseta y regret del frío: Spearman %+.3f, Pearson %+.3f",
                        ref["spearman"], ref["pearson"])
            for c in relacion["condiciones"]:
                logger.info("  meseta y beneficio de %-9s Spearman %+.3f (p %.4f), "
                            "Pearson %+.3f (p %.4f)", c["condicion"], c["spearman"],
                            c["p_spearman"], c["pearson"], c["p_pearson"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
