# tests/test_report_lodo_e2.py
"""Pruebas del reporte del experimento 2.

Protegen la misma propiedad que las del reporte del experimento 1 —el documento
transcribe los estadísticos y no los recalcula— más una que es propia de este:
**la significancia se juzga con el valor p ajustado por Holm y nunca con el crudo**.
Hay una prueba por sabotaje para cada una.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import report_lodo_e2 as reporte  # noqa: E402


def _confirmatorio(
    condicion: str = "nearest",
    referencia: str = "frio",
    reduccion: float | None = 0.45,
    p_una_cola: float = 0.004,
    p_ajustado: float | None = 0.008,
    cumple: bool | None = None,
) -> dict:
    """Contraste confirmatorio con la forma que produce el arnés."""
    if cumple is None:
        cumple = (
            reduccion is not None
            and reduccion >= 0.30
            and p_ajustado is not None
            and p_ajustado < 0.05
        )
    return {
        "condicion": condicion,
        "referencia": referencia,
        "variable": "regret",
        "censura": None,
        "n_pares": 42,
        "n_empates": 3,
        "mediana_referencia": 0.2000,
        "mediana_condicion": 0.1100,
        "reduccion": reduccion,
        "p_una_cola": p_una_cola,
        "p_dos_colas": p_una_cola * 2,
        "efecto_rango_biserial": -0.62,
        "p_ajustado": p_ajustado,
        "cumple_magnitud": reduccion is not None and reduccion >= 0.30,
        "cumple_significancia": p_ajustado is not None and p_ajustado < 0.05,
        "cumple_criterio": cumple,
    }


def _fila(nombre: str, frio: float, nearest: float) -> dict:
    """Fila de la tabla por conjunto de datos."""
    fila = {"dataset_id": nombre, "dataset_name": nombre, "meseta": 12}
    for condicion, valor in (("frio", frio), ("nearest", nearest), ("frequent", 0.15),
                             ("random", 0.18), ("hybrid", 0.12)):
        fila[condicion] = valor
        fila[f"{condicion}_ensayos"] = 7.0
        fila[f"{condicion}_inevaluable"] = 0.05
    return fila


@pytest.fixture
def resultado() -> dict:
    """Documento completo, con la forma que escribe ``run_lodo_e2.py``."""
    return {
        "minutos": 154.2,
        "pliegues": [],
        "analisis": {
            "confirmatorios": [
                _confirmatorio(),
                _confirmatorio(referencia="frequent", reduccion=0.10, p_una_cola=0.4,
                               p_ajustado=0.4),
            ],
            "secundarios": [
                {**_confirmatorio(), "variable": "ensayos",
                 "mediana_referencia": 18.0, "mediana_condicion": 9.0},
                {**_confirmatorio(condicion="hybrid", referencia="frequent")},
            ],
            "robustez": [
                {**_confirmatorio(), "variable": "ensayos", "censura": 120,
                 "mediana_referencia": 18.0, "mediana_condicion": 9.0},
                {**_confirmatorio(), "variable": "ensayos", "censura": 600,
                 "mediana_referencia": 18.0, "mediana_condicion": 9.0},
            ],
            "por_dataset": [
                _fila("gana-el-caliente", 0.30, 0.05),
                _fila("gana-el-frio", 0.05, 0.30),
                _fila("empatan", 0.10, 0.10),
            ],
            "censurados": {"frio": 12, "nearest": 4, "frequent": 3, "random": 9,
                           "hybrid": 2},
            "n_corridas_por_condicion": 420,
        },
    }


# --------------------------------------------------------------------------- #
# El veredicto usa el valor ajustado, no el crudo
# --------------------------------------------------------------------------- #
def test_the_verdict_passes_when_both_conditions_hold() -> None:
    texto = reporte.redactar_veredicto(_confirmatorio())

    assert "Cumple el criterio" in texto
    assert "No cumple" not in texto


def test_the_verdict_judges_with_the_holm_adjusted_value() -> None:
    """Prueba por sabotaje de la corrección por multiplicidad.

    Un crudo de 0,03 pasaría solo si nadie corrigiera. Con dos confirmatorios
    declarados, Holm lo lleva a 0,06 y el criterio no se cumple. La prueba falla si
    alguien hace que el veredicto mire el crudo.
    """
    texto = reporte.redactar_veredicto(
        _confirmatorio(p_una_cola=0.03, p_ajustado=0.06)
    )

    assert "No cumple el criterio" in texto
    assert "0.0600" in texto
    assert "crudo 0.0300" in texto


def test_the_verdict_names_only_the_condition_that_failed() -> None:
    solo_magnitud = reporte.redactar_veredicto(
        _confirmatorio(reduccion=0.12, p_ajustado=0.001)
    )

    assert "se exige un 30 %" in solo_magnitud
    assert "Holm" not in solo_magnitud


def test_the_verdict_denounces_a_flag_that_contradicts_the_numbers() -> None:
    """El reporte no puede afirmar lo que sus propias cifras desmienten."""
    texto = reporte.redactar_veredicto(
        _confirmatorio(reduccion=0.01, p_ajustado=0.9, cumple=True)
    )

    assert "discrepancia" in texto
    assert "Cumple el criterio" not in texto


def test_an_undefined_p_value_does_not_pass() -> None:
    texto = reporte.redactar_veredicto(_confirmatorio(p_ajustado=None))

    assert "No cumple el criterio" in texto


# --------------------------------------------------------------------------- #
# El documento
# --------------------------------------------------------------------------- #
def test_the_summary_carries_both_confirmatory_verdicts(resultado: dict) -> None:
    texto = reporte.redactar_resumen(resultado)

    for contraste in resultado["analisis"]["confirmatorios"]:
        assert reporte.redactar_veredicto(contraste) in texto


def test_the_p_values_are_transcribed_and_not_recomputed(resultado: dict) -> None:
    """Un valor imposible de derivar de tres conjuntos: si aparece, se leyó."""
    resultado["analisis"]["confirmatorios"][0]["p_ajustado"] = 0.4242

    texto = reporte.redactar_resumen(resultado)

    assert "0.4242" in texto


def test_the_summary_identifies_the_frozen_protocol(resultado: dict) -> None:
    texto = reporte.redactar_resumen(resultado)

    assert "prereg-v2.0" in texto
    assert "Holm" in texto
    assert "99 % del óptimo exhaustivo" in texto
    assert "1010" in texto


def test_the_secondary_analyses_are_marked_as_such(resultado: dict) -> None:
    texto = reporte.redactar_resumen(resultado)

    assert "No sustituyen a los confirmatorios" in texto
    assert "llevan corrección por multiplicidad" in texto


def test_the_censored_runs_are_reported(resultado: dict) -> None:
    """Con vara externa el frío puede censurarse, y eso hay que decirlo."""
    texto = reporte.redactar_resumen(resultado)

    assert "sí puede quedarse corto" in texto
    assert "`frio`: 12 de 420" in texto


def test_the_datasets_are_ordered_by_the_difference(resultado: dict) -> None:
    texto = reporte.redactar_resumen(resultado)

    posiciones = [
        texto.index(f"`{n}`") for n in ("gana-el-caliente", "empatan", "gana-el-frio")
    ]

    assert posiciones == sorted(posiciones)


def test_the_trial_count_columns_are_not_mistaken_for_conditions(resultado: dict) -> None:
    """`frio_ensayos` acompaña a cada condición; no es una condición."""
    condiciones = reporte._condiciones(resultado["analisis"]["por_dataset"])

    assert condiciones == ["frio", "nearest", "frequent", "random", "hybrid"]


def test_a_new_condition_reaches_the_table_without_touching_the_script(
    resultado: dict,
) -> None:
    for fila in resultado["analisis"]["por_dataset"]:
        fila["inventada"] = 0.2
        fila["inventada_ensayos"] = 5.0
        fila["inventada_inevaluable"] = 0.0

    assert "| inventada |" in reporte.redactar_resumen(resultado)


def test_it_refuses_to_run_without_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(reporte, "RESULTADOS", tmp_path / "no-existe.json")

    with pytest.raises(SystemExit, match="run_lodo_e2"):
        reporte.main()


def test_it_writes_the_summary(
    resultado: dict, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    datos = tmp_path / "LODO-E2-resultados.json"
    datos.write_text(json.dumps(resultado), encoding="utf-8")
    monkeypatch.setattr(reporte, "RESULTADOS", datos)
    monkeypatch.setattr(reporte, "RESUMEN", tmp_path / "LODO-E2-resultados.md")

    assert reporte.main() == 0
    assert (tmp_path / "LODO-E2-resultados.md").is_file()
