# tests/test_examples.py
"""Pruebas de los ejemplos ejecutables de ``examples/``.

Verifican el criterio de éxito de ingeniería de la Sección 4.3.5 del
anteproyecto: el flujo completo para la persona que usa la librería debe caber en
no más de 20 líneas de código.
"""

from __future__ import annotations

import ast
import runpy
from pathlib import Path

import pytest

EJEMPLOS = Path(__file__).resolve().parents[1] / "examples"
QUICKSTART = EJEMPLOS / "tabular_quickstart.py"
WARM_START = EJEMPLOS / "warm_start_quickstart.py"
LIMITE_INSTRUCCIONES = 20


def _contar_instrucciones(ruta: Path) -> int:
    """Cuenta las instrucciones de nivel superior de un archivo de Python.

    Se cuentan instrucciones y no líneas físicas porque el límite de 100
    caracteres del proyecto obliga a repartir una sola instrucción en varias
    líneas: contar líneas físicas mediría el estilo de formato y no la
    complejidad del flujo. La cadena de documentación del módulo no cuenta.

    Args:
        ruta: Ruta al archivo de Python.

    Returns:
        Número de instrucciones de nivel superior.
    """
    arbol = ast.parse(ruta.read_text(encoding="utf-8"))
    cuerpo = list(arbol.body)
    if cuerpo and isinstance(cuerpo[0], ast.Expr) and isinstance(cuerpo[0].value, ast.Constant):
        cuerpo = cuerpo[1:]
    return len(cuerpo)


def test_quickstart_exists() -> None:
    assert QUICKSTART.is_file()


def test_the_user_flow_fits_in_twenty_statements() -> None:
    """Criterio de éxito de ingeniería del anteproyecto, Sección 4.3.5."""
    instrucciones = _contar_instrucciones(QUICKSTART)

    assert instrucciones <= LIMITE_INSTRUCCIONES, (
        f"El flujo del ejemplo usa {instrucciones} instrucciones y el criterio "
        f"admite {LIMITE_INSTRUCCIONES}."
    )


def test_quickstart_runs_end_to_end(capsys: pytest.CaptureFixture[str]) -> None:
    """El ejemplo documentado debe ejecutarse de verdad, no solo compilar."""
    runpy.run_path(str(QUICKSTART), run_name="__main__")

    salida = capsys.readouterr().out
    assert "Pipeline ganador" in salida
    assert "macro-F1" in salida


# --------------------------------------------------------------------------- #
# El flujo con arranque en caliente
#
# Es el que da nombre al proyecto, y el que un lector del informe va a querer
# copiar. Se le exige lo mismo que al flujo en frío: caber en el presupuesto de
# instrucciones del criterio de éxito y ejecutarse de verdad.
# --------------------------------------------------------------------------- #
def test_warm_start_example_exists() -> None:
    assert WARM_START.is_file()


def test_the_warm_start_flow_also_fits_in_twenty_statements() -> None:
    """Criterio de éxito de ingeniería del anteproyecto, Sección 4.3.5.

    El flujo en caliente necesita cuatro pasos más que el frío —extraer las
    meta-características, cargar la base, ajustar el recomendador y pedirle las
    configuraciones—, de modo que es el que de verdad pone a prueba el límite.
    """
    instrucciones = _contar_instrucciones(WARM_START)

    assert instrucciones <= LIMITE_INSTRUCCIONES, (
        f"El flujo con arranque en caliente usa {instrucciones} instrucciones y "
        f"el criterio admite {LIMITE_INSTRUCCIONES}."
    )


def test_warm_start_example_runs_end_to_end(capsys: pytest.CaptureFixture[str]) -> None:
    """Recorre de verdad la cadena entera: base, recomendador y motor.

    Es la única prueba que ejercita las tres piezas juntas contra la base de
    meta-conocimiento real. Si alguna cambiara de formato, aquí se vería.
    """
    runpy.run_path(str(WARM_START), run_name="__main__")

    salida = capsys.readouterr().out
    assert "Inyectadas 5 configuraciones" in salida
    assert "Pipeline ganador" in salida
    assert "macro-F1" in salida


def test_the_warm_start_example_does_not_run_the_experiment() -> None:
    """El ejemplo no puede consultar la base excluyendo un dataset del corpus.

    Suena a detalle y no lo es: si el ejemplo hiciera un pliegue de verdad,
    estaría ejecutando el experimento fuera del protocolo, y quien lo copiara
    creería que así se mide la hipótesis. El dataset del ejemplo es sintético y
    la base se usa entera, que es lo correcto **solo** porque el consultado no
    pertenece al corpus.
    """
    arbol = ast.parse(WARM_START.read_text(encoding="utf-8"))
    llamadas = [
        nodo.func.attr
        for nodo in ast.walk(arbol)
        if isinstance(nodo, ast.Call) and isinstance(nodo.func, ast.Attribute)
    ]

    assert "without" not in llamadas
    # El docstring sí tiene que explicar que en el LODO hay que usarlo.
    assert "base.without(d)" in ast.get_docstring(arbol)
