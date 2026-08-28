# tests/test_corpus_selection.py
"""Pruebas de la selección del corpus a partir del cribado de sensibilidad.

La regla de familia y las restricciones de diversidad son decisiones
**pre-registradas**, no detalles de implementación: la regla de familia protege la
independencia entre pliegues del *leave-one-dataset-out*, y las restricciones de
diversidad garantizan que la codificación y la imputación del espacio de búsqueda
queden ejercitadas. Se prueban con datos sintéticos, sin red.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

openml = pytest.importorskip("openml", reason="el script del cribado importa openml")

from screen_corpus import seleccionar, verificar_diversidad  # noqa: E402


def registro(
    nombre: str,
    iqr: float,
    instancias: int = 800,
    categoricas: int = 0,
    faltantes: bool = False,
) -> dict:
    """Arma un registro de cribado sintético."""
    return {
        "did": abs(hash(nombre)) % 10_000,
        "nombre": nombre,
        "iqr": iqr,
        "instancias": instancias,
        "categoricas": categoricas,
        "faltantes": faltantes,
        "clases": 2,
        "descartadas": 0,
        "licencia": "Public",
    }


# --------------------------------------------------------------------------- #
# Orden y regla de familia
# --------------------------------------------------------------------------- #
def test_the_candidates_are_ordered_by_descending_iqr() -> None:
    candidatos = [registro("a", 0.1), registro("b", 0.3), registro("c", 0.2)]

    seleccion, _ = seleccionar(candidatos, objetivo=3)

    assert [r["nombre"] for r in seleccion] == ["b", "c", "a"]


def test_only_one_dataset_per_family_survives() -> None:
    """Es un problema de validez, no de diversidad.

    Los cuatro ``mfeat-*`` son representaciones de las mismas 2000 instancias: si
    entraran dos, el recomendador tendría el dataset excluido en su base bajo otra
    representación y la independencia entre pliegues se rompería.
    """
    candidatos = [
        registro("mfeat-morphological", 0.30),
        registro("mfeat-zernike", 0.25),
        registro("mfeat-fourier", 0.20),
        registro("otro", 0.10),
    ]

    seleccion, excluidos = seleccionar(candidatos, objetivo=4)

    assert [r["nombre"] for r in seleccion] == ["mfeat-morphological", "otro"]
    assert {r["nombre"] for r in excluidos} == {"mfeat-zernike", "mfeat-fourier"}


def test_the_family_representative_is_the_one_with_the_highest_iqr() -> None:
    candidatos = [registro("pc1", 0.10), registro("kc2", 0.40), registro("jm1", 0.20)]

    seleccion, _ = seleccionar(candidatos, objetivo=3)

    assert [r["nombre"] for r in seleccion] == ["kc2"]


def test_digit_recognition_collections_share_one_family() -> None:
    """``optdigits`` y ``pendigits`` compiten con el representante de ``mfeat-*``."""
    candidatos = [
        registro("mfeat-karhunen", 0.30),
        registro("optdigits", 0.28),
        registro("pendigits", 0.26),
    ]

    seleccion, excluidos = seleccionar(candidatos, objetivo=3)

    assert [r["nombre"] for r in seleccion] == ["mfeat-karhunen"]
    assert len(excluidos) == 2


def test_the_excluded_records_say_which_family_displaced_them() -> None:
    candidatos = [registro("kc1", 0.30), registro("pc4", 0.20)]

    _, excluidos = seleccionar(candidatos, objetivo=2)

    assert excluidos[0]["familia"] == "NASA MDP"


def test_a_dataset_outside_every_family_is_never_excluded() -> None:
    candidatos = [registro(f"d{i}", 0.5 - i / 100) for i in range(5)]

    seleccion, excluidos = seleccionar(candidatos, objetivo=5)

    assert len(seleccion) == 5
    assert excluidos == []


def test_the_target_size_caps_the_selection() -> None:
    candidatos = [registro(f"d{i}", 0.5 - i / 100) for i in range(10)]

    seleccion, _ = seleccionar(candidatos, objetivo=4)

    assert len(seleccion) == 4


# --------------------------------------------------------------------------- #
# Restricciones de diversidad
# --------------------------------------------------------------------------- #
def test_the_three_size_regimes_are_counted_apart() -> None:
    seleccion = [
        registro("chico", 0.1, instancias=800),
        registro("mediano", 0.1, instancias=5_000),
        registro("grande", 0.1, instancias=40_000),
    ]

    diversidad = verificar_diversidad(seleccion)

    assert diversidad["Régimen de centenas"][1] == 1
    assert diversidad["Régimen de millares"][1] == 1
    assert diversidad["Régimen de decenas de millares"][1] == 1


def test_a_missing_size_regime_fails_its_constraint() -> None:
    seleccion = [registro(f"d{i}", 0.1, instancias=800) for i in range(5)]

    diversidad = verificar_diversidad(seleccion)

    assert diversidad["Régimen de centenas"][2] is True
    assert diversidad["Régimen de millares"][2] is False


def test_five_datasets_with_categoricals_are_required() -> None:
    """Menos de cinco dejaría la codificación del espacio sin ejercitar."""
    cuatro = [registro(f"d{i}", 0.1, categoricas=3) for i in range(4)]

    assert verificar_diversidad(cuatro)["Con variables categóricas"] == (5, 4, False)
    assert verificar_diversidad(
        cuatro + [registro("quinto", 0.1, categoricas=1)]
    )["Con variables categóricas"] == (5, 5, True)


def test_five_datasets_with_missing_values_are_required() -> None:
    cuatro = [registro(f"d{i}", 0.1, faltantes=True) for i in range(4)]

    assert verificar_diversidad(cuatro)["Con valores faltantes"] == (5, 4, False)
    assert verificar_diversidad(
        cuatro + [registro("quinto", 0.1, faltantes=True)]
    )["Con valores faltantes"] == (5, 5, True)


def test_the_regime_boundaries_are_the_pre_registered_ones() -> None:
    """Centenas por debajo de mil, decenas de millares desde diez mil."""
    limites = [
        registro("justo_debajo_de_mil", 0.1, instancias=999),
        registro("mil", 0.1, instancias=1_000),
        registro("justo_debajo_de_diez_mil", 0.1, instancias=9_999),
        registro("diez_mil", 0.1, instancias=10_000),
    ]

    diversidad = verificar_diversidad(limites)

    assert diversidad["Régimen de centenas"][1] == 1
    assert diversidad["Régimen de millares"][1] == 2
    assert diversidad["Régimen de decenas de millares"][1] == 1
