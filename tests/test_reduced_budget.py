# tests/test_reduced_budget.py
"""Pruebas del análisis del desempeño a presupuesto reducido.

Lo que más importa proteger es la **dirección**: aquí mayor es mejor, al revés que en el
regret y en los ensayos hasta el objetivo. Una prueba de una cola copiada del arnés sin
invertirla daría valores p cercanos a uno justo cuando la condición mejora, y nada más
fallaría. Hay una prueba por sabotaje para eso, otra para los presupuestos y otra para la
comprobación del arranque en frío entre los dos experimentos.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

pytest.importorskip("openml", reason="el análisis importa los arneses de los experimentos")

import analyze_reduced_budget as reducido  # noqa: E402

CONDICIONES_E1 = ("frio", "nearest", "random", "frequent")
CONDICIONES_E2 = ("frio", "nearest", "frequent", "random", "hybrid")


def _corrida(valor: float, e2: bool = True) -> dict:
    """Corrida con la forma que guardan los arneses."""
    guardado = {"5": valor - 0.1, "15": valor, "30": valor + 0.05} if e2 else valor
    return {
        "curva": [valor],
        "ensayos": 3,
        "mejor": round(valor + 0.05, 6),
        "a_presupuesto_reducido": guardado,
        "fraccion_inevaluable": 0.0,
    }


def _pliegue(
    nombre: str,
    valores: dict[str, float],
    semillas: tuple[int, ...] = (101, 202, 303),
    e2: bool = True,
) -> dict:
    """Pliegue con el mismo valor en todas sus semillas."""
    return {
        "dataset_id": nombre,
        "dataset_name": f"d{nombre}",
        "recomendaciones": {},
        "semillas": [
            {"semilla": s, **{c: _corrida(v, e2) for c, v in valores.items()}}
            for s in semillas
        ],
    }


def _todos(condiciones: tuple[str, ...], valor: float, **cambios: float) -> dict[str, float]:
    """Mismo valor para todas las condiciones, salvo las que se indiquen."""
    return {c: cambios.get(c, valor) for c in condiciones}


# --------------------------------------------------------------------------- #
# Los presupuestos y la lectura
# --------------------------------------------------------------------------- #
def test_the_budgets_are_the_ones_the_pre_registrations_fixed() -> None:
    """Prueba por sabotaje de los presupuestos.

    El anteproyecto fija quince (la mitad de treinta), el pre-registro v2.0 añade cinco
    y treinta, y la comparación entre experimentos se hace al presupuesto completo del
    primero. Los tres vienen de los arneses; si alguien los cambiara en uno de ellos, el
    documento seguiría generándose y diría otra cosa.
    """
    assert reducido.PRESUPUESTO_E1 == 15
    assert tuple(reducido.PRESUPUESTOS_E2) == (5, 15, 30)
    assert reducido.PRESUPUESTO_COMUN == 30


def test_experiment_two_stores_one_value_per_budget() -> None:
    corrida = {"a_presupuesto_reducido": {"5": 0.5, "15": 0.7, "30": 0.8}}

    assert reducido.mejor_valor(corrida, 5) == 0.5
    assert reducido.mejor_valor(corrida, 15) == 0.7


def test_experiment_one_only_stored_fifteen() -> None:
    corrida = {"a_presupuesto_reducido": 0.7}

    assert reducido.mejor_valor(corrida, 15) == 0.7
    with pytest.raises(KeyError, match="15"):
        reducido.mejor_valor(corrida, 30)


def test_a_non_finite_value_is_refused_instead_of_imputed() -> None:
    """Ningún pre-registro fija qué valor darle; decidirlo aquí sería hacerlo a posteriori."""
    with pytest.raises(ValueError, match="no se decide aquí"):
        reducido.mejor_valor({"a_presupuesto_reducido": {"15": math.nan}}, 15)


def test_the_observation_is_the_median_of_the_seeds() -> None:
    pliegue = _pliegue("a", {"frio": 0.5})
    for semilla, valor in zip(pliegue["semillas"], (0.2, 0.9, 0.6)):
        semilla["frio"]["a_presupuesto_reducido"]["15"] = valor

    assert reducido.observacion(pliegue, "frio", 15) == 0.6


# --------------------------------------------------------------------------- #
# La dirección del contraste
# --------------------------------------------------------------------------- #
def test_higher_is_better_in_the_one_tailed_test() -> None:
    """Prueba por sabotaje de la dirección.

    La condición queda por encima del frío en los ocho conjuntos, con diferencias
    distintas para que la prueba tenga rangos. Si alguien copiara la cola del arnés
    —«menor es mejor»— el valor p saldría cercano a uno.
    """
    pliegues = [
        _pliegue(str(i), {"frio": 0.5, "nearest": 0.5 + 0.01 * (i + 1)}) for i in range(8)
    ]

    contraste = reducido.comparar(pliegues, "nearest", "frio", 15)

    assert contraste["p_una_cola"] < 0.01
    assert contraste["efecto_rango_biserial"] > 0
    assert (contraste["gana"], contraste["empata"], contraste["pierde"]) == (8, 0, 0)
    assert contraste["diferencia_mediana"] > 0


def test_a_condition_that_gets_worse_is_not_favoured() -> None:
    pliegues = [
        _pliegue(str(i), {"frio": 0.5, "nearest": 0.5 - 0.01 * (i + 1)}) for i in range(8)
    ]

    contraste = reducido.comparar(pliegues, "nearest", "frio", 15)

    assert contraste["p_una_cola"] > 0.9
    assert contraste["efecto_rango_biserial"] < 0
    assert contraste["pierde"] == 8


def test_identical_conditions_produce_no_p_value_instead_of_a_spurious_one() -> None:
    pliegues = [_pliegue(str(i), {"frio": 0.5, "nearest": 0.5}) for i in range(4)]

    contraste = reducido.comparar(pliegues, "nearest", "frio", 15)

    assert math.isnan(contraste["p_una_cola"])
    assert contraste["empata"] == 4


# --------------------------------------------------------------------------- #
# Los análisis
# --------------------------------------------------------------------------- #
def _e1() -> dict:
    return {
        "pliegues": [
            _pliegue(str(i), _todos(CONDICIONES_E1, 0.5, nearest=0.5 + 0.01 * i), e2=False)
            for i in range(6)
        ]
    }


def _e2() -> dict:
    return {
        "pliegues": [
            _pliegue(str(i), _todos(CONDICIONES_E2, 0.5, nearest=0.5 + 0.01 * i),
                     semillas=(101, 202, 303, 404))
            for i in range(6)
        ]
    }


def test_experiment_one_compares_every_warm_condition_with_cold() -> None:
    analisis = reducido.analizar_e1(_e1())

    assert [c["condicion"] for c in analisis["contra_frio"]] == ["nearest", "random", "frequent"]
    assert all(c["referencia"] == "frio" for c in analisis["contra_frio"])
    assert analisis["presupuesto"] == 15


def test_experiment_two_contrasts_are_the_confirmatory_pairs_at_each_budget() -> None:
    """Fijados antes de ver los valores; elegir otros después abriría un grado de libertad."""
    analisis = reducido.analizar_e2(_e2())

    pares = [(c["presupuesto"], c["condicion"], c["referencia"]) for c in analisis["contrastes"]]
    assert pares == [
        (k, c, r) for k in (5, 15, 30) for c, r in (("nearest", "frio"), ("nearest", "frequent"))
    ]
    assert list(analisis["medianas"]) == ["5", "15", "30"]


def test_the_experiments_are_compared_only_on_what_they_share() -> None:
    """El experimento 2 tiene una semilla y un conjunto que el 1 no tiene."""
    e2 = _e2()
    e2["pliegues"].append(_pliegue("solo-e2", _todos(CONDICIONES_E2, 0.9)))

    comparacion = reducido.comparar_experimentos(_e1(), e2)

    assert comparacion["n_conjuntos"] == 6
    assert comparacion["n_parejas"] == 6 * 3
    assert comparacion["base_e1"] == 5
    assert comparacion["base_e2"] == 6


def test_the_cold_start_check_counts_identical_pairs() -> None:
    comparacion = reducido.comparar_experimentos(_e1(), _e2())

    assert comparacion["frio_identico"] == comparacion["n_parejas"]


def test_a_cold_start_mismatch_is_reported_and_not_hidden() -> None:
    """Si el frío no coincide, la comparación de los calientes no significa nada."""
    e2 = _e2()
    e2["pliegues"][0]["semillas"][0]["frio"]["a_presupuesto_reducido"]["30"] = 0.1

    comparacion = reducido.comparar_experimentos(_e1(), e2)
    texto = reducido.redactar(reducido.analizar_e1(_e1()), reducido.analizar_e2(e2), comparacion)

    assert comparacion["frio_identico"] == comparacion["n_parejas"] - 1
    assert "No coincide en todas" in texto


# --------------------------------------------------------------------------- #
# El documento
# --------------------------------------------------------------------------- #
def _texto() -> str:
    e1, e2 = _e1(), _e2()
    return reducido.redactar(
        reducido.analizar_e1(e1), reducido.analizar_e2(e2), reducido.comparar_experimentos(e1, e2)
    )


def test_experiment_one_is_reported_as_descriptive_without_p_values() -> None:
    """El pre-registro v1.0 la declaró descriptiva: ponerle un valor p la ascendería."""
    texto = _texto()
    seccion = texto[texto.index("## Experimento 1") : texto.index("## Experimento 2")]

    assert "descriptiva" in seccion
    assert "p (1 cola)" not in seccion


def test_the_document_says_higher_is_better_and_that_it_does_not_decide() -> None:
    texto = _texto()

    assert "Mayor es mejor" in texto
    assert "No decide sobre la hipótesis" in texto
    assert "sin corrección por multiplicidad" in texto


def test_the_document_carries_the_cold_start_check() -> None:
    assert "coincide en **18 de 18**" in _texto()


def test_it_refuses_to_run_without_experiment_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(reducido, "RESULTADOS_E1", tmp_path / "no-existe.json")

    with pytest.raises(SystemExit, match="run_lodo.py"):
        reducido.main()


def test_it_refuses_to_run_without_experiment_two(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    uno = tmp_path / "e1.json"
    uno.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(reducido, "RESULTADOS_E1", uno)
    monkeypatch.setattr(reducido, "RESULTADOS_E2", tmp_path / "no-existe.json")

    with pytest.raises(SystemExit, match="run_lodo_e2.py"):
        reducido.main()


def test_it_writes_the_document(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    uno, dos = tmp_path / "e1.json", tmp_path / "e2.json"
    uno.write_text(json.dumps(_e1()), encoding="utf-8")
    dos.write_text(json.dumps(_e2()), encoding="utf-8")
    monkeypatch.setattr(reducido, "RESULTADOS_E1", uno)
    monkeypatch.setattr(reducido, "RESULTADOS_E2", dos)
    monkeypatch.setattr(reducido, "SALIDA", tmp_path / "PRESUPUESTO-REDUCIDO.md")

    assert reducido.main() == 0
    assert (tmp_path / "PRESUPUESTO-REDUCIDO.md").is_file()
