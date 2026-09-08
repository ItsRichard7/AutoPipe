# tests/test_report_lodo.py
"""Pruebas del script que redacta los reportes del LODO.

Lo que estas pruebas protegen es una sola propiedad, y es la que hace citable al
documento: **el resumen no calcula ningún estadístico**. Todo —medianas, valores p,
reducción y veredicto— viene del arnés, que es donde el pre-registro los fija. Si el
script los recomputara habría dos implementaciones del mismo contraste y nada
garantizaría que coincidan, de modo que hay pruebas escritas para fallar si alguien
introduce un cálculo propio.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import report_lodo as reporte  # noqa: E402

from autopipe.reporter import redactar_veredicto  # noqa: E402


def _contraste(estrategia: str = "nearest", **cambios: object) -> dict:
    """Arma un contraste con la forma que produce el arnés."""
    base = {
        "estrategia": estrategia,
        "censura": 31,
        "n_pares": 3,
        "mediana_frio": 8.0,
        "mediana_caliente": 4.0,
        "reduccion": 0.5,
        "p_una_cola": 0.0123,
        "p_dos_colas": 0.0246,
        "confirma": True,
        "censurados_frio": 0,
        "censurados_caliente": 1,
    }
    return {**base, **cambios}


def _fila(nombre: str, frio: float, nearest: float) -> dict:
    """Arma una fila de la tabla por conjunto de datos."""
    fila = {"dataset_id": nombre, "dataset_name": nombre, "meseta": 9}
    for condicion, valor in (("frio", frio), ("nearest", nearest), ("random", 6.0),
                             ("frequent", 5.0)):
        fila[condicion] = valor
        fila[f"{condicion}_inevaluable"] = 0.0
        fila[f"{condicion}_a_15"] = 0.75
    return fila


@pytest.fixture
def resultado() -> dict:
    """Documento completo, con la forma que escribe ``run_lodo.py``."""
    return {
        "minutos": 21.5,
        "pliegues": [],
        "analisis": {
            "primario": _contraste(),
            "secundarios": [
                _contraste("random", reduccion=0.05, p_una_cola=0.7, confirma=False),
                _contraste("frequent", reduccion=0.20, p_una_cola=0.3, confirma=False),
            ],
            "robustez": [
                _contraste(censura=60, reduccion=0.55),
                _contraste(censura=300, reduccion=0.61),
            ],
            "por_dataset": [
                _fila("gana-el-caliente", 8.0, 2.0),
                _fila("gana-el-frio", 3.0, 11.0),
                _fila("empatan", 5.0, 5.0),
            ],
        },
    }


# --------------------------------------------------------------------------- #
# El resumen no calcula: lee
# --------------------------------------------------------------------------- #
def test_the_summary_carries_the_same_verdict_as_the_report(resultado: dict) -> None:
    texto = reporte.redactar_resumen(resultado, Path("reportes/lodo.html"))

    assert redactar_veredicto(resultado["analisis"]["primario"]) in texto


def test_the_p_value_is_transcribed_and_not_recomputed(resultado: dict) -> None:
    """Prueba por sabotaje: un valor p imposible de derivar de estas cifras.

    Los tres conjuntos de la fixture no podrían producir jamás un valor p de
    0,4242. Si aparece en el documento es porque se leyó del análisis, que es la
    única fuente admisible. Si alguien hiciera que el script lo calculara, esta
    prueba fallaría.
    """
    resultado["analisis"]["primario"]["p_una_cola"] = 0.4242

    texto = reporte.redactar_resumen(resultado, Path("reportes/lodo.html"))

    assert "0.4242" in texto


def test_an_adverse_result_is_stated_as_such(resultado: dict) -> None:
    """Un resultado desfavorable tiene que salir del documento sin suavizarse."""
    resultado["analisis"]["primario"].update(
        {"reduccion": 0.02, "p_una_cola": 0.8, "confirma": False}
    )

    texto = reporte.redactar_resumen(resultado, Path("reportes/lodo.html"))

    assert "no se confirma" in texto
    assert "2.0 %" in texto


# --------------------------------------------------------------------------- #
# Lo que el documento tiene que contener
# --------------------------------------------------------------------------- #
def test_the_summary_counts_who_won_on_each_dataset(resultado: dict) -> None:
    gana_caliente, empatan, gana_frio = reporte._recuento(
        resultado["analisis"]["por_dataset"]
    )

    assert (gana_caliente, empatan, gana_frio) == (1, 1, 1)


def test_the_datasets_are_ordered_by_the_difference(resultado: dict) -> None:
    """El orden es el del gráfico pareado: primero donde más ayudó el caliente."""
    texto = reporte.redactar_resumen(resultado, Path("reportes/lodo.html"))

    posiciones = [
        texto.index(f"`{nombre}`")
        for nombre in ("gana-el-caliente", "empatan", "gana-el-frio")
    ]

    assert posiciones == sorted(posiciones)


def test_the_summary_identifies_the_frozen_protocol(resultado: dict) -> None:
    """Sin la identificación del protocolo, la tabla no es citable."""
    texto = reporte.redactar_resumen(resultado, Path("reportes/lodo.html"))

    assert "prereg-v1.0" in texto
    assert "tabular-v1" in texto
    assert "101, 202, 303, 404, 505" in texto
    assert "Wilcoxon" in texto


def test_the_robustness_table_shows_both_censoring_values(resultado: dict) -> None:
    texto = reporte.redactar_resumen(resultado, Path("reportes/lodo.html"))

    assert "| 60 |" in texto
    assert "| 300 |" in texto


def test_a_control_that_meets_the_criterion_is_not_read_as_a_confirmation(
    resultado: dict,
) -> None:
    """El caso real del experimento: `frequent` cumple y `nearest` no.

    Sin distinguir el papel de cada fila, un lector concluiría que la hipótesis se
    confirmó. La tabla tiene que decir cuál es la primaria, y el texto que la
    hipótesis se juega solo ahí.
    """
    resultado["analisis"]["primario"].update({"p_una_cola": 0.31, "confirma": False})
    resultado["analisis"]["secundarios"][1].update(
        {"reduccion": 0.33, "p_una_cola": 0.02, "confirma": True}
    )

    texto = reporte.redactar_resumen(resultado, Path("reportes/lodo.html"))

    assert "**primario**" in texto
    assert "| control |" in texto
    assert "se juega únicamente en la fila primaria" in texto
    assert "no se confirma" in texto


def test_the_primary_condition_is_columned_next_to_the_cold_start(
    resultado: dict,
) -> None:
    """Leer el contraste no debería exigir saltar sobre las columnas de control."""
    texto = reporte.redactar_resumen(resultado, Path("reportes/lodo.html"))

    cabecera = next(
        linea for linea in texto.splitlines() if linea.startswith("| Conjunto de datos")
    )

    assert cabecera.index("frio") < cabecera.index("nearest") < cabecera.index("random")


def test_the_secondary_controls_are_reported(resultado: dict) -> None:
    texto = reporte.redactar_resumen(resultado, Path("reportes/lodo.html"))

    assert "`random`" in texto
    assert "`frequent`" in texto


def test_a_new_strategy_reaches_the_table_without_touching_the_script(
    resultado: dict,
) -> None:
    for fila in resultado["analisis"]["por_dataset"]:
        fila["inventada"] = 1.0
        fila["inventada_inevaluable"] = 0.0
        fila["inventada_a_15"] = 0.5

    texto = reporte.redactar_resumen(resultado, Path("reportes/lodo.html"))

    assert "| inventada |" in texto


def test_the_covariates_are_not_listed_as_conditions(resultado: dict) -> None:
    texto = reporte.redactar_resumen(resultado, Path("reportes/lodo.html"))

    assert "| frio_a_15 |" not in texto
    assert "| frio_inevaluable |" not in texto


# --------------------------------------------------------------------------- #
# El punto de entrada
# --------------------------------------------------------------------------- #
def test_it_refuses_to_run_without_results(monkeypatch: pytest.MonkeyPatch,
                                           tmp_path: Path) -> None:
    monkeypatch.setattr(reporte, "RESULTADOS", tmp_path / "no-existe.json")
    monkeypatch.setattr(sys, "argv", ["report_lodo.py"])

    with pytest.raises(SystemExit, match="run_lodo"):
        reporte.main()


def test_it_writes_both_documents(
    resultado: dict, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    datos = tmp_path / "LODO-resultados.json"
    datos.write_text(json.dumps(resultado), encoding="utf-8")
    monkeypatch.setattr(reporte, "RESULTADOS", datos)
    monkeypatch.setattr(reporte, "RESUMEN", tmp_path / "LODO-resultados.md")
    monkeypatch.setattr(reporte, "REPORTES", tmp_path / "reportes")
    monkeypatch.setattr(sys, "argv", ["report_lodo.py", "--sin-pdf"])

    assert reporte.main() == 0
    assert (tmp_path / "LODO-resultados.md").is_file()
    assert (tmp_path / "reportes" / "lodo.html").is_file()
