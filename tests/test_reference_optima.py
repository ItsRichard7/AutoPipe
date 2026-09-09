# tests/test_reference_optima.py
"""Pruebas de los óptimos de referencia del experimento 2.

La que sostiene todo el experimento es
``test_no_search_can_beat_its_own_reference``: si la rejilla y la búsqueda no
usaran la misma partición, el óptimo de referencia no acotaría las curvas, el
*regret* podría salir negativo y el objetivo del 99\\ % sería inalcanzable o trivial
según el caso. Es un invariante, no una probabilidad, y por eso se comprueba.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

pytest.importorskip("openml", reason="el guion importa el descargador del corpus")

import reference_optima as referencia  # noqa: E402
import run_lodo_e2 as arnes  # noqa: E402

FILAS = 60


@pytest.fixture(autouse=True)
def cache_aislada(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Aparta la caché de rejillas a un directorio temporal.

    Sin esto, las rejillas sintéticas de estas pruebas se escribirían junto a las
    del experimento real bajo la misma clave y se leerían unas por otras: dos
    pruebas con tablas distintas compartirían resultado y alguna pasaría sin
    comprobar nada.
    """
    monkeypatch.setattr(referencia, "CACHE", tmp_path / "rejillas")


def _tabla(semilla: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Tabla sintética con señal, ruido, faltantes y una columna categórica."""
    rng = np.random.default_rng(semilla)
    clase = np.tile([0, 1], FILAS // 2)
    senal = clase * 2.0 + rng.normal(0, 0.6, size=FILAS)
    senal[rng.random(FILAS) < 0.1] = np.nan
    tabla = pd.DataFrame(
        {
            "senal": senal,
            "ruido": rng.normal(0, 1.0, size=FILAS),
            "cat": np.where(rng.random(FILAS) < 0.8, clase, 1 - clase).astype(str),
        }
    )
    return tabla, clase


# --------------------------------------------------------------------------- #
# El invariante que sostiene la vara externa
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("semilla", [101, 202])
def test_no_search_can_beat_its_own_reference(semilla: int) -> None:
    """Ningún ensayo puede superar el óptimo exhaustivo de su misma partición.

    La rejilla evalúa las 144 configuraciones del espacio con el mismo
    ``val_size`` y el mismo ``random_state`` que la búsqueda, así que el máximo de
    la rejilla acota por arriba cualquier valor que la búsqueda alcance. Si esta
    prueba fallara, indicaría que las particiones no coinciden, y todo el
    experimento 2 estaría midiendo contra una vara que no le corresponde.
    """
    features, labels = _tabla(semilla)

    rejilla = referencia.rejilla("no-cacheado", semilla, features, labels)
    optimo = referencia.resumir(rejilla)["optimo"]
    curva = arnes.ejecutar_condicion(features, labels, semilla, presupuesto=20)

    finitos = [v for v in curva if math.isfinite(v)]
    assert finitos, "La búsqueda no evaluó ni un solo ensayo."
    assert max(finitos) <= optimo + 1e-9


def test_the_regret_of_a_real_search_is_never_negative() -> None:
    """Corolario del invariante, y la razón por la que importa."""
    features, labels = _tabla(303)

    optimo = referencia.resumir(
        referencia.rejilla("no-cacheado", 303, features, labels)
    )["optimo"]
    curva = arnes.ejecutar_condicion(features, labels, 303, presupuesto=20)

    assert arnes.regret_normalizado(curva, optimo, 20) >= 0.0


def test_a_different_seed_gives_a_different_partition() -> None:
    """Si no lo hiciera, cachear la rejilla por semilla no tendría sentido."""
    features, labels = _tabla(404)

    una = referencia.rejilla("no-cacheado", 101, features, labels)
    otra = referencia.rejilla("no-cacheado", 202, features, labels)

    assert una != otra


# --------------------------------------------------------------------------- #
# El resumen de una rejilla
# --------------------------------------------------------------------------- #
def test_the_summary_reports_the_optimum_and_its_target() -> None:
    resumen = referencia.resumir([0.5, 1.0, 0.9])

    assert resumen["optimo"] == 1.0
    assert resumen["objetivo"] == pytest.approx(0.99)


def test_the_optimum_is_not_rounded_down() -> None:
    """Prueba por sabotaje: redondear la vara la deja por debajo de lo alcanzable.

    Con ``round(optimo, 6)``, un óptimo de 0,8333333333 se guardaría como 0,833333
    y una búsqueda que lo encuentra lo superaría. El regret saldría negativo y el
    invariante del pre-registro quedaría roto por una decisión cosmética.
    """
    optimo = 5 / 6

    assert referencia.resumir([0.1, optimo])["optimo"] == optimo


def test_the_plateau_counts_the_configurations_that_reach_the_target() -> None:
    """La meseta al 99 %, que es la covariable declarada del experimento 2."""
    resumen = referencia.resumir([1.0, 0.995, 0.98, 0.5])

    assert resumen["n_en_meseta"] == 2


def test_the_plateau_is_measured_at_ninety_nine_percent_not_ninety_five() -> None:
    """Prueba por sabotaje de la covariable.

    La meseta del experimento 1 se midió al 95 %, y reutilizarla aquí daría una
    covariable que no corresponde al objetivo vigente. Con esta rejilla, al 95 %
    entrarían tres configuraciones y al 99 % solo una.
    """
    rejilla = [1.0, 0.97, 0.96, 0.2]

    assert referencia.FRACCION_OBJETIVO == 0.99
    assert referencia.resumir(rejilla)["n_en_meseta"] == 1


def test_unevaluable_configurations_are_counted_but_not_ranked() -> None:
    nan = float("nan")

    resumen = referencia.resumir([1.0, nan, nan, 0.5])

    assert resumen["optimo"] == 1.0
    assert resumen["n_evaluables"] == 2
    assert resumen["fraccion_inevaluable"] == pytest.approx(0.5)


def test_a_grid_with_nothing_evaluable_is_refused() -> None:
    """Sin óptimo no hay vara, y una pareja sin vara no se puede medir."""
    with pytest.raises(ValueError, match="evaluable"):
        referencia.resumir([float("nan"), float("nan")])


# --------------------------------------------------------------------------- #
# Las constantes del protocolo
# --------------------------------------------------------------------------- #
def test_the_seeds_are_the_ten_of_the_pre_registration() -> None:
    assert referencia.SEMILLAS == (101, 202, 303, 404, 505, 606, 707, 808, 909, 1010)
    assert referencia.SEMILLAS == arnes.SEMILLAS, (
        "La referencia y el arnés tienen que recorrer las mismas parejas: si "
        "difirieran, faltarían varas o sobrarían rejillas."
    )


def test_the_reference_uses_the_same_validation_split_as_the_harness() -> None:
    assert referencia.VAL_SIZE == arnes.VAL_SIZE
    assert referencia.METRICA == arnes.METRICA
