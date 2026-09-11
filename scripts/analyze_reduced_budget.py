# scripts/analyze_reduced_budget.py
"""Desempeño a presupuesto reducido en los dos experimentos.

Es la segunda variable que el anteproyecto define en su Sección 4.3.3 ---el mejor
valor de la métrica alcanzado con un presupuesto menor que el completo--- y que los dos
pre-registros declararon antes de ejecutar:

* **Experimento 1** (pre-registro v1.0, plan de análisis): mejor macro-F1 tras
  **quince** ensayos, «como medida descriptiva». No forma parte del criterio de
  confirmación y no se somete a contraste.
* **Experimento 2** (pre-registro v2.0, análisis secundarios): mejor valor a los **5, 15
  y 30** ensayos. El de 30 permite comparar con el experimento 1 a igual presupuesto.

Los dos arneses guardaron ese valor en cada corrida. **Este script no ejecuta ningún
ensayo**: lo lee tal como quedó. La variable estaba medida y no se había reportado.

**Mayor es mejor.** A diferencia del regret y de los ensayos hasta el objetivo, aquí la
condición favorable es la que alcanza un valor *más alto*, de modo que la prueba de una
cola va en la dirección contraria a la de los contrastes principales.

Los contrastes del experimento 2 son **los mismos dos pares que los confirmatorios**, a
cada presupuesto, fijados así antes de mirar los valores: elegir los pares después abriría
un grado de libertad. No llevan corrección por multiplicidad porque, como todo análisis
secundario del pre-registro, no deciden sobre la hipótesis.

Ejecutar con:
    .\\.venv\\Scripts\\python.exe scripts\\analyze_reduced_budget.py
"""

from __future__ import annotations

import json
import logging
import math
import statistics
from datetime import date
from pathlib import Path

from run_lodo import ESTRATEGIAS_CALIENTE as CALIENTES_E1
from run_lodo import PRESUPUESTO as PRESUPUESTO_COMPLETO_E1
from run_lodo import PRESUPUESTO_REDUCIDO as PRESUPUESTO_E1
from run_lodo_e2 import CONDICIONES_CALIENTES as CALIENTES_E2
from run_lodo_e2 import CONFIRMATORIOS, _rango_biserial
from run_lodo_e2 import PRESUPUESTOS_REDUCIDOS as PRESUPUESTOS_E2
from scipy.stats import wilcoxon

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

RESULTADOS_E1 = Path("docs/experimentos/LODO-resultados.json")
RESULTADOS_E2 = Path("docs/experimentos/LODO-E2-resultados.json")
SALIDA = Path("docs/experimentos/PRESUPUESTO-REDUCIDO.md")

PRESUPUESTO_COMUN = PRESUPUESTO_COMPLETO_E1
"""Presupuesto al que se comparan los dos experimentos: el completo del primero."""


# --------------------------------------------------------------------------- #
# La observación
# --------------------------------------------------------------------------- #
def mejor_valor(corrida: dict, presupuesto: int) -> float:
    """Mejor valor que el arnés guardó para una corrida a un presupuesto dado.

    El arnés del experimento 1 guardó un único número, el de quince ensayos; el del
    experimento 2, uno por presupuesto. Se leen los dos sin recalcularlos desde la
    curva, para que no existan dos implementaciones de la misma medida.

    Args:
        corrida: Resultado de una condición en una semilla.
        presupuesto: Ensayos a los que se lee el mejor valor.

    Returns:
        El mejor valor guardado.

    Raises:
        KeyError: Si la corrida no guardó ese presupuesto.
        ValueError: Si el valor guardado no es finito. Ningún pre-registro fija qué
            valor darle a una corrida sin ensayos evaluables, y asignarle uno aquí
            sería decidirlo después de ver los datos.
    """
    guardado = corrida["a_presupuesto_reducido"]
    if isinstance(guardado, dict):
        valor = guardado[str(presupuesto)]
    elif presupuesto == PRESUPUESTO_E1:
        valor = guardado
    else:
        raise KeyError(
            f"La corrida solo guardó el mejor valor a {PRESUPUESTO_E1} ensayos, no a "
            f"{presupuesto}."
        )
    if not math.isfinite(valor):
        raise ValueError(
            f"El mejor valor a {presupuesto} ensayos es {valor!r}. Ningún pre-registro "
            f"fija cómo tratar una corrida sin ensayos evaluables y no se decide aquí."
        )
    return float(valor)


def observacion(pliegue: dict, condicion: str, presupuesto: int) -> float:
    """Mediana de las semillas de un conjunto: la observación que entra en el contraste.

    Args:
        pliegue: Resultado de un pliegue.
        condicion: Condición a leer.
        presupuesto: Ensayos a los que se lee el mejor valor.

    Returns:
        La mediana sobre las semillas.
    """
    return statistics.median(mejor_valor(s[condicion], presupuesto) for s in pliegue["semillas"])


def _recuento(diferencias: list[float]) -> dict[str, int]:
    """Cuenta en cuántos conjuntos la condición queda por encima, igual o por debajo."""
    return {
        "gana": sum(1 for d in diferencias if d > 0),
        "empata": sum(1 for d in diferencias if d == 0),
        "pierde": sum(1 for d in diferencias if d < 0),
    }


def comparar(pliegues: list[dict], condicion: str, referencia: str, presupuesto: int) -> dict:
    """Compara una condición contra otra a un presupuesto, pareando por conjunto.

    La prueba es de Wilcoxon pareada de una cola en la dirección favorable a la
    condición, que aquí es un valor **mayor**; se reporta también la de dos colas. El
    efecto es la correlación biserial por rangos del arnés, **positiva** cuando la
    condición alcanza valores más altos.

    Args:
        pliegues: Resultados de todos los pliegues.
        condicion: Condición contrastada.
        referencia: Condición contra la que se compara.
        presupuesto: Ensayos a los que se lee el mejor valor.

    Returns:
        Diccionario con las medianas, el recuento, los valores $p$ y el efecto.
    """
    tratamiento = [observacion(p, condicion, presupuesto) for p in pliegues]
    control = [observacion(p, referencia, presupuesto) for p in pliegues]
    diferencias = [t - c for t, c in zip(tratamiento, control)]

    if any(d != 0 for d in diferencias):
        una_cola = float(wilcoxon(tratamiento, control, alternative="greater").pvalue)
        dos_colas = float(wilcoxon(tratamiento, control, alternative="two-sided").pvalue)
    else:
        una_cola = dos_colas = float("nan")

    return {
        "condicion": condicion,
        "referencia": referencia,
        "presupuesto": presupuesto,
        "n_pares": len(pliegues),
        "mediana_referencia": round(statistics.median(control), 6),
        "mediana_condicion": round(statistics.median(tratamiento), 6),
        "diferencia_mediana": round(statistics.median(diferencias), 6),
        **_recuento(diferencias),
        "p_una_cola": una_cola,
        "p_dos_colas": dos_colas,
        "efecto_rango_biserial": round(_rango_biserial(tratamiento, control), 4),
    }


def medianas(pliegues: list[dict], condiciones: tuple[str, ...], presupuesto: int) -> dict:
    """Mediana entre conjuntos de la observación de cada condición.

    Args:
        pliegues: Resultados de todos los pliegues.
        condiciones: Condiciones a resumir.
        presupuesto: Ensayos a los que se lee el mejor valor.

    Returns:
        Diccionario de condición a mediana.
    """
    return {
        c: round(statistics.median(observacion(p, c, presupuesto) for p in pliegues), 6)
        for c in condiciones
    }


# --------------------------------------------------------------------------- #
# Los análisis
# --------------------------------------------------------------------------- #
def analizar_e1(resultado: dict) -> dict:
    """Mejor valor a quince ensayos en el experimento 1, descriptivo.

    Args:
        resultado: Documento que escribió ``run_lodo.py``.

    Returns:
        Diccionario con las medianas y la comparación de cada condición caliente con
        el frío.
    """
    pliegues = resultado["pliegues"]
    return {
        "presupuesto": PRESUPUESTO_E1,
        "n_conjuntos": len(pliegues),
        "medianas": medianas(pliegues, ("frio", *CALIENTES_E1), PRESUPUESTO_E1),
        "contra_frio": [comparar(pliegues, c, "frio", PRESUPUESTO_E1) for c in CALIENTES_E1],
    }


def analizar_e2(resultado: dict) -> dict:
    """Mejor valor a 5, 15 y 30 ensayos en el experimento 2, secundario.

    Args:
        resultado: Documento que escribió ``run_lodo_e2.py``.

    Returns:
        Diccionario con las medianas por presupuesto y los contrastes de los dos pares
        confirmatorios a cada presupuesto.
    """
    pliegues = resultado["pliegues"]
    return {
        "presupuestos": list(PRESUPUESTOS_E2),
        "n_conjuntos": len(pliegues),
        "medianas": {
            str(k): medianas(pliegues, ("frio", *CALIENTES_E2), k) for k in PRESUPUESTOS_E2
        },
        "contrastes": [
            comparar(pliegues, c, r, k) for k in PRESUPUESTOS_E2 for c, r in CONFIRMATORIOS
        ],
    }


def comparar_experimentos(e1: dict, e2: dict) -> dict:
    """Mejor valor a treinta ensayos en los dos experimentos, sobre lo que comparten.

    Los dos comparten conjuntos y semillas. **El arranque en frío tiene que dar
    exactamente lo mismo**: misma semilla, misma partición y mismo muestreador, cuyo
    comportamiento en los primeros treinta ensayos no depende del presupuesto total. Si
    no coincidiera, algo habría cambiado entre los dos arneses y la comparación de las
    condiciones calientes no significaría nada; por eso se cuenta y se reporta en vez de
    suponerse.

    Args:
        e1: Documento del experimento 1.
        e2: Documento del experimento 2.

    Returns:
        Diccionario con la comprobación del frío y, por condición, las medianas de los
        dos experimentos y el recuento de conjuntos.
    """
    del_uno = {(p["dataset_id"], s["semilla"]): s for p in e1["pliegues"] for s in p["semillas"]}
    comunes = [
        (p["dataset_id"], [(del_uno[(p["dataset_id"], s["semilla"])], s) for s in p["semillas"]
                           if (p["dataset_id"], s["semilla"]) in del_uno])
        for p in e2["pliegues"]
    ]
    comunes = [(dataset, parejas) for dataset, parejas in comunes if parejas]

    parejas = [pareja for _, lista in comunes for pareja in lista]
    frio_identico = sum(
        1 for uno, dos in parejas
        if uno["frio"]["mejor"] == mejor_valor(dos["frio"], PRESUPUESTO_COMUN)
    )

    por_condicion = {}
    for condicion in ("frio", *CALIENTES_E1):
        uno = [statistics.median(float(a[condicion]["mejor"]) for a, _ in lista)
               for _, lista in comunes]
        dos = [statistics.median(mejor_valor(b[condicion], PRESUPUESTO_COMUN) for _, b in lista)
               for _, lista in comunes]
        por_condicion[condicion] = {
            "mediana_e1": round(statistics.median(uno), 6),
            "mediana_e2": round(statistics.median(dos), 6),
            **_recuento([d - u for u, d in zip(uno, dos)]),
        }

    return {
        "presupuesto": PRESUPUESTO_COMUN,
        "n_conjuntos": len(comunes),
        "n_parejas": len(parejas),
        "frio_identico": frio_identico,
        "base_e1": len(e1["pliegues"]) - 1,
        "base_e2": len(e2["pliegues"]) - 1,
        "condiciones": por_condicion,
    }


# --------------------------------------------------------------------------- #
# El documento
# --------------------------------------------------------------------------- #
def _p(valor: float) -> str:
    """Presenta un valor p, o un guion si no está definido."""
    return "—" if not math.isfinite(valor) else f"{valor:.4f}"


def _tabla_e1(analisis: dict) -> str:
    """Tabula el experimento 1, sin valores p: la medida es descriptiva."""
    filas = [
        "| Condición | Mejor a 15 (mediana) | Diferencia mediana con `frio` | Gana | Empata "
        "| Pierde |",
        "|---|---:|---:|---:|---:|---:|",
        f"| `frio` | {analisis['medianas']['frio']:.4f} | — | — | — | — |",
    ]
    filas += [
        f"| `{c['condicion']}` | {c['mediana_condicion']:.4f} | "
        f"{c['diferencia_mediana']:+.4f} | {c['gana']} | {c['empata']} | {c['pierde']} |"
        for c in analisis["contra_frio"]
    ]
    return "\n".join(filas)


def _tabla_medianas_e2(analisis: dict) -> str:
    """Tabula las medianas del experimento 2 por presupuesto y condición."""
    condiciones = list(next(iter(analisis["medianas"].values())))
    filas = [
        "| Presupuesto | " + " | ".join(f"`{c}`" for c in condiciones) + " |",
        "|---:|" + "---:|" * len(condiciones),
    ]
    filas += [
        f"| {k} | " + " | ".join(f"{v:.4f}" for v in valores.values()) + " |"
        for k, valores in analisis["medianas"].items()
    ]
    return "\n".join(filas)


def _tabla_contrastes_e2(analisis: dict) -> str:
    """Tabula los contrastes secundarios del experimento 2."""
    filas = [
        "| Presupuesto | Contraste | Mediana referencia | Mediana condición | Diferencia "
        "mediana | Gana | Empata | Pierde | p (1 cola) | Efecto |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    filas += [
        f"| {c['presupuesto']} | `{c['condicion']}` vs `{c['referencia']}` | "
        f"{c['mediana_referencia']:.4f} | {c['mediana_condicion']:.4f} | "
        f"{c['diferencia_mediana']:+.4f} | {c['gana']} | {c['empata']} | {c['pierde']} | "
        f"{_p(c['p_una_cola'])} | {c['efecto_rango_biserial']:+.3f} |"
        for c in analisis["contrastes"]
    ]
    return "\n".join(filas)


def _tabla_experimentos(comparacion: dict) -> str:
    """Tabula los dos experimentos a igual presupuesto."""
    filas = [
        "| Condición | Experimento 1 | Experimento 2 | Sube en el 2 | Igual | Baja en el 2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    filas += [
        f"| `{condicion}` | {v['mediana_e1']:.4f} | {v['mediana_e2']:.4f} | {v['gana']} | "
        f"{v['empata']} | {v['pierde']} |"
        for condicion, v in comparacion["condiciones"].items()
    ]
    return "\n".join(filas)


def _comprobacion_del_frio(comparacion: dict) -> str:
    """Enuncia la comprobación del arranque en frío, y la alarma si falla."""
    identico, total = comparacion["frio_identico"], comparacion["n_parejas"]
    texto = (
        f"El arranque en frío coincide en **{identico} de {total}** parejas (conjunto, "
        f"semilla)."
    )
    if identico != total:
        texto += (
            " **No coincide en todas**: algo cambió entre los dos arneses y la "
            "comparación de las condiciones calientes no se puede interpretar hasta "
            "resolverlo."
        )
    return texto


def redactar(e1: dict, e2: dict, comparacion: dict) -> str:
    """Redacta el documento del desempeño a presupuesto reducido.

    Args:
        e1: Resultado de :func:`analizar_e1`.
        e2: Resultado de :func:`analizar_e2`.
        comparacion: Resultado de :func:`comparar_experimentos`.

    Returns:
        El documento markdown.
    """
    return f"""# Desempeño a presupuesto reducido

> **Archivo generado.** No editar a mano: se regenera con
> `.\\.venv\\Scripts\\python.exe scripts\\analyze_reduced_budget.py`.

> **Segunda variable del anteproyecto** (Sección 4.3.3): el mejor macro-F1 alcanzado
> con un presupuesto menor que el completo. Los dos pre-registros la declararon antes de
> ejecutar —descriptiva en el experimento 1, secundaria en el 2— y los dos arneses la
> midieron en cada corrida. **No decide sobre la hipótesis**: eso lo hacen los ensayos
> hasta el objetivo en el experimento 1 y el regret en el 2.

- **Generado:** {date.today().isoformat()}
- **Datos:** `{RESULTADOS_E1.as_posix()}` y `{RESULTADOS_E2.as_posix()}`
- **Mayor es mejor.** Cada observación es la mediana de las semillas de un conjunto;
  «gana» cuenta los conjuntos en que la condición queda por encima de la referencia.

## Experimento 1: {e1['presupuesto']} ensayos

Medida **descriptiva**, tal como la declaró el pre-registro v1.0: no se somete a
contraste. {e1['n_conjuntos']} conjuntos de datos.

{_tabla_e1(e1)}

## Experimento 2: {", ".join(str(k) for k in e2['presupuestos'])} ensayos

Análisis **secundario** del pre-registro v2.0. {e2['n_conjuntos']} conjuntos de datos.

### Medianas por condición

{_tabla_medianas_e2(e2)}

### Contrastes

Los mismos dos pares que los contrastes confirmatorios, a cada presupuesto. Wilcoxon
pareada de una cola en la dirección favorable a la condición —ahora, un valor
**mayor**—, sin corrección por multiplicidad. El efecto es la correlación biserial por
rangos, **positiva** cuando la condición mejora.

{_tabla_contrastes_e2(e2)}

## Los dos experimentos a igual presupuesto

Mejor valor a {comparacion['presupuesto']} ensayos sobre los {comparacion['n_conjuntos']}
conjuntos y las semillas que los dos experimentos comparten.

{_comprobacion_del_frio(comparacion)}

Es la comprobación de que la comparación es legítima: con la misma semilla, la misma
partición y el mismo muestreador, el frío no tiene por qué cambiar.

Lo que sí cambia en las condiciones calientes es la base con la que se ajusta el
recomendador: {comparacion['base_e1']} conjuntos en cada pliegue del experimento 1 y
{comparacion['base_e2']} en el 2.

{_tabla_experimentos(comparacion)}
"""


def main() -> int:
    """Punto de entrada.

    Returns:
        ``0`` si el análisis se produjo.

    Raises:
        SystemExit: Si falta alguno de los dos documentos de resultados.
    """
    for ruta, arnes in ((RESULTADOS_E1, "run_lodo.py"), (RESULTADOS_E2, "run_lodo_e2.py")):
        if not ruta.is_file():
            raise SystemExit(f"No existe {ruta}. Ejecute antes scripts/{arnes}.")
    uno = json.loads(RESULTADOS_E1.read_text(encoding="utf-8"))
    dos = json.loads(RESULTADOS_E2.read_text(encoding="utf-8"))

    e1, e2 = analizar_e1(uno), analizar_e2(dos)
    comparacion = comparar_experimentos(uno, dos)
    SALIDA.write_text(redactar(e1, e2, comparacion), encoding="utf-8")

    logger.info("Análisis escrito en %s", SALIDA)
    for c in e1["contra_frio"]:
        logger.info("  E1 a %d  %-9s %.4f contra frio %.4f", c["presupuesto"],
                    c["condicion"], c["mediana_condicion"], c["mediana_referencia"])
    for c in e2["contrastes"]:
        logger.info("  E2 a %2d  %-9s vs %-9s %.4f -> %.4f  p %s", c["presupuesto"],
                    c["condicion"], c["referencia"], c["mediana_referencia"],
                    c["mediana_condicion"], _p(c["p_una_cola"]))
    logger.info("  Frío idéntico en %d de %d parejas.", comparacion["frio_identico"],
                comparacion["n_parejas"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
