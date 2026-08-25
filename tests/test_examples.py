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
