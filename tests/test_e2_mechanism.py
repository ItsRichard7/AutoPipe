# tests/test_e2_mechanism.py
"""Pruebas del análisis exploratorio del mecanismo del experimento 2.

La que sostiene todo lo demás es
``test_the_grid_index_matches_the_engine_ordering``: el análisis lee el valor de cada
configuración inyectada por su **posición** en la rejilla exhaustiva, y si esa posición
no coincidiera con el orden en que el motor genera los pipelines, el documento
reportaría la calidad de configuraciones distintas de las que se inyectaron **sin
producir ningún error**. Se comprueba contra el motor, una por una, las 144.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import analyze_e2_mechanism as mecanismo  # noqa: E402

from autopipe.search_engine import GridSearchEngine  # noqa: E402
from autopipe.search_space import (  # noqa: E402
    TABULAR_SPACE_STEPS,
    build_tabular_search_space,
    pipeline_from_configuration,
)

CONFIG_CERO = dict.fromkeys(TABULAR_SPACE_STEPS, 0)


def _todas_las_configuraciones() -> list[dict[str, int]]:
    """Las 144 configuraciones del espacio, incluida la omisión de cada paso."""
    espacio = build_tabular_search_space()
    configuraciones = [{}]
    for paso in TABULAR_SPACE_STEPS:
        configuraciones = [
            {**parcial, paso: indice}
            for parcial in configuraciones
            for indice in range(len(espacio[paso]) + 1)
        ]
    return configuraciones


# --------------------------------------------------------------------------- #
# La posición en la rejilla
# --------------------------------------------------------------------------- #
def test_the_grid_index_matches_the_engine_ordering() -> None:
    """Prueba por sabotaje del índice.

    Si el orden mixto no reprodujera el del motor, el análisis leería el valor de otra
    configuración y publicaría números plausibles y equivocados. Se comprueba contra
    los pipelines que el propio motor genera.
    """
    motor = GridSearchEngine(build_tabular_search_space(), task="classification")
    pipelines = motor._build_pipelines()

    assert len(pipelines) == 144
    for configuracion in _todas_las_configuraciones():
        posicion = mecanismo.indice_en_la_rejilla(configuracion)
        esperado = pipeline_from_configuration(configuracion)
        assert pipelines[posicion].get_params() == esperado.get_params()


def test_every_configuration_gets_its_own_position() -> None:
    posiciones = [mecanismo.indice_en_la_rejilla(c) for c in _todas_las_configuraciones()]

    assert sorted(posiciones) == list(range(144))


def test_the_all_zero_configuration_is_the_first() -> None:
    assert mecanismo.indice_en_la_rejilla(CONFIG_CERO) == 0


# --------------------------------------------------------------------------- #
# Diversidad
# --------------------------------------------------------------------------- #
def test_identical_configurations_have_no_diversity() -> None:
    assert mecanismo.diversidad([CONFIG_CERO, dict(CONFIG_CERO)]) == 0.0


def test_configurations_that_differ_everywhere_are_maximally_diverse() -> None:
    otra = {"imputation": 1, "scaling": 1, "encoding": 1, "selection": 1}

    assert mecanismo.diversidad([CONFIG_CERO, otra]) == 4.0


def test_the_diversity_is_the_mean_over_all_pairs() -> None:
    """Tres configuraciones, tres pares: distancias 1, 2 y 1, media 4/3."""
    a = dict(CONFIG_CERO)
    b = {**CONFIG_CERO, "scaling": 1}
    c = {**CONFIG_CERO, "scaling": 1, "encoding": 1}

    assert mecanismo.diversidad([a, b, c]) == pytest.approx(4 / 3)


def test_a_single_configuration_has_no_pairs() -> None:
    assert mecanismo.diversidad([CONFIG_CERO]) == 0.0


# --------------------------------------------------------------------------- #
# Calidad de lo inyectado
# --------------------------------------------------------------------------- #
def _rejilla_falsa(valores: dict[int, float]) -> list[float]:
    """Rejilla de 144 posiciones con los valores indicados y el resto en cero."""
    rejilla = [0.0] * 144
    for posicion, valor in valores.items():
        rejilla[posicion] = valor
    return rejilla


def test_the_quality_is_the_regret_of_what_was_injected() -> None:
    """Dos inyectadas que valen 0,8 y 0,6 con óptimo 1,0: regrets 0,2 y 0,4."""
    otra = {**CONFIG_CERO, "selection": 1}
    rejilla = _rejilla_falsa({0: 0.8, mecanismo.indice_en_la_rejilla(otra): 0.6})

    medida = mecanismo.calidad([CONFIG_CERO, otra], rejilla, 1.0)

    assert medida["regret_medio"] == pytest.approx(0.3)
    assert medida["regret_de_la_mejor"] == pytest.approx(0.2)
    assert medida["fraccion_inevaluable"] == 0.0


def test_unevaluable_injected_configurations_are_counted_apart() -> None:
    """No entran en el regret, pero su fracción se reporta: es un ensayo perdido."""
    otra = {**CONFIG_CERO, "selection": 1}
    rejilla = _rejilla_falsa(
        {0: 0.9, mecanismo.indice_en_la_rejilla(otra): float("nan")}
    )

    medida = mecanismo.calidad([CONFIG_CERO, otra], rejilla, 1.0)

    assert medida["regret_medio"] == pytest.approx(0.1)
    assert medida["fraccion_inevaluable"] == pytest.approx(0.5)


def test_everything_unevaluable_gives_no_regret_instead_of_a_wrong_one() -> None:
    rejilla = _rejilla_falsa({0: float("nan")})

    medida = mecanismo.calidad([CONFIG_CERO], rejilla, 1.0)

    assert medida["regret_medio"] is None
    assert medida["regret_de_la_mejor"] is None
    assert medida["fraccion_inevaluable"] == 1.0


@pytest.mark.parametrize("optimo", [0.0, -1.0, float("nan")])
def test_a_non_positive_reference_is_refused(optimo: float) -> None:
    with pytest.raises(ValueError, match="positivo"):
        mecanismo.calidad([CONFIG_CERO], _rejilla_falsa({0: 0.5}), optimo)


# --------------------------------------------------------------------------- #
# El documento
# --------------------------------------------------------------------------- #
def _por_condicion() -> dict:
    """Agregados con la forma que produce :func:`analizar`."""
    return {
        c: {
            "hamming_medio": h,
            "configuraciones_distintas_en_el_corpus": n,
            "regret_medio_de_lo_inyectado": 0.09,
            "regret_de_la_mejor_inyectada": mejor,
            "fraccion_inevaluable": inev,
            "regret_logrado": logrado,
        }
        for c, h, n, mejor, inev, logrado in [
            ("nearest", 2.58, 28, 0.0352, 0.029, 0.0058),
            ("frequent", 2.10, 6, 0.0150, 0.0, 0.0039),
            ("random", 2.34, 10, 0.0209, 0.0, 0.0028),
            ("hybrid", 2.54, 26, 0.0231, 0.024, 0.0042),
        ]
    }


def test_the_document_is_marked_as_exploratory() -> None:
    """Sin la marca, un lector podría tomarlo por un contraste de la hipótesis."""
    texto = mecanismo.redactar(_por_condicion(), 42)

    assert "Exploratorio, no confirmatorio" in texto
    assert "§8.6" in texto
    assert "No ejecuta ensayos" in texto


def test_the_quality_table_is_ordered_by_the_result() -> None:
    texto = mecanismo.redactar(_por_condicion(), 42)
    calidad = texto.split("## Calidad de lo inyectado")[1]

    posiciones = [calidad.index(f"`{c}`") for c in ("random", "frequent", "hybrid", "nearest")]

    assert posiciones == sorted(posiciones)


def test_the_diversity_table_shows_the_ordering_is_inverted() -> None:
    """Es el punto del análisis: el más diverso es el que peor rinde."""
    texto = mecanismo.redactar(_por_condicion(), 42)
    diversidad = texto.split("## Calidad de lo inyectado")[0]

    assert diversidad.index("`nearest`") < diversidad.index("`frequent`")


def test_it_refuses_to_run_without_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(mecanismo, "RESULTADOS", tmp_path / "no-existe.json")

    with pytest.raises(SystemExit, match="run_lodo_e2"):
        mecanismo.main()
