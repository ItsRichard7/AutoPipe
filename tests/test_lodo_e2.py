# tests/test_lodo_e2.py
"""Pruebas del arnés del experimento 2.

Se dividen en dos clases, como las del experimento 1: las que fijan la
**definición** de las variables contra curvas construidas a mano, donde la
respuesta se puede calcular mentalmente, y las que recorren el protocolo entero
sobre un corpus sintético comprobando las propiedades que el pre-registro v2.0
declara.

Tres de ellas están escritas para fallar si alguien revierte un cambio deliberado:
que el arranque en frío **ya puede censurarse**, que el regret se promedia sobre el
presupuesto ejecutado y no sobre uno enlazado al importar, y que la corrección de
Holm se aplica de verdad sobre la familia confirmatoria.
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

pytest.importorskip("openml", reason="el arnés importa el descargador del corpus")

import run_lodo_e2 as arnes  # noqa: E402

from autopipe.meta_knowledge import MetaKnowledgeBase, MetaKnowledgeEntry  # noqa: E402
from autopipe.search_space import TABULAR_SPACE_VERSION  # noqa: E402

CONFIG_A = {"imputation": 0, "scaling": 0, "encoding": 0, "selection": 0}
CONFIG_B = {"imputation": 1, "scaling": 1, "encoding": 1, "selection": 1}
CONFIG_C = {"imputation": 0, "scaling": 2, "encoding": 1, "selection": 2}

FILAS_JUGUETE = 40


# --------------------------------------------------------------------------- #
# La variable primaria: regret normalizado
# --------------------------------------------------------------------------- #
def test_a_run_that_starts_at_the_optimum_has_no_regret() -> None:
    assert arnes.regret_normalizado([1.0, 1.0, 1.0], 1.0, 3) == 0.0


def test_a_run_that_never_evaluates_anything_has_full_regret() -> None:
    """Un ensayo sin ningún valor finito aún cuenta como regret pleno (v2.0 §6)."""
    nan = float("nan")

    assert arnes.regret_normalizado([nan, nan, nan], 0.9, 3) == 1.0


def test_the_regret_is_computed_by_hand() -> None:
    """Curva 0,5 -> 0,8 -> 0,8 con óptimo 1,0: deficits 0,5 + 0,2 + 0,2 entre 3."""
    valor = arnes.regret_normalizado([0.5, 0.8, 0.4], 1.0, 3)

    assert valor == pytest.approx((0.5 + 0.2 + 0.2) / 3)


def test_unevaluable_trials_consume_their_turn_without_reducing_the_regret() -> None:
    """El NaN del medio no mejora el mejor-hasta-ahora ni se descuenta."""
    nan = float("nan")

    valor = arnes.regret_normalizado([0.5, nan, 1.0], 1.0, 3)

    assert valor == pytest.approx((0.5 + 0.5 + 0.0) / 3)


def test_the_regret_averages_over_the_budget_not_over_the_curve() -> None:
    """Prueba por sabotaje del promedio.

    Si se promediara sobre la longitud de la curva en vez de sobre el presupuesto,
    una corrida que alcanza el óptimo pronto y otra que lo alcanza tarde saldrían
    iguales cuando la curva se acorta. El presupuesto es parte de la definición.
    """
    corta = arnes.regret_normalizado([0.0, 1.0], 1.0, 2)
    larga = arnes.regret_normalizado([0.0, 1.0], 1.0, 4)

    assert corta == pytest.approx(0.5)
    assert larga == pytest.approx(0.25)


def test_trials_beyond_the_budget_are_ignored() -> None:
    assert arnes.regret_normalizado([0.0, 1.0, 1.0, 1.0], 1.0, 2) == pytest.approx(0.5)


@pytest.mark.parametrize("optimo", [0.0, -1.0, float("nan"), float("inf")])
def test_a_non_positive_reference_is_refused(optimo: float) -> None:
    """Sin óptimo positivo la normalización no existe: se excluye, no se inventa."""
    with pytest.raises(ValueError, match="óptimo"):
        arnes.regret_normalizado([0.5], optimo, 1)


def test_the_regret_never_leaves_the_unit_interval() -> None:
    for curva in ([0.1, 0.2], [float("nan"), 0.9], [0.9, 0.9]):
        valor = arnes.regret_normalizado(curva, 0.9, 2)
        assert 0.0 <= valor <= 1.0


# --------------------------------------------------------------------------- #
# La variable secundaria conserva la definición del anteproyecto
# --------------------------------------------------------------------------- #
def test_the_count_still_starts_at_one() -> None:
    assert arnes.ensayos_hasta_el_objetivo([0.9, 0.5], 0.8) == 1


def test_reaching_exactly_the_target_still_counts() -> None:
    assert arnes.ensayos_hasta_el_objetivo([0.5, 0.8], 0.8) == 2


def test_never_reaching_an_external_target_is_censored() -> None:
    """Con vara externa esto ya puede pasarle a cualquier condición."""
    assert arnes.ensayos_hasta_el_objetivo([0.1, 0.2], 0.99) == arnes.CENSURA


# --------------------------------------------------------------------------- #
# La corrección de Holm
# --------------------------------------------------------------------------- #
def _contraste(condicion: str, p: float, reduccion: float | None = 0.5) -> dict:
    """Contraste mínimo con lo que Holm necesita."""
    return {"condicion": condicion, "referencia": "frio", "p_una_cola": p,
            "reduccion": reduccion}


def test_holm_multiplies_the_smallest_p_by_the_family_size() -> None:
    ajustados = arnes.holm([_contraste("a", 0.01), _contraste("b", 0.04)])

    assert ajustados[0]["p_ajustado"] == pytest.approx(0.02)
    assert ajustados[1]["p_ajustado"] == pytest.approx(0.04)


def test_holm_is_monotone() -> None:
    """El ajustado nunca puede bajar al avanzar en el orden."""
    ajustados = arnes.holm([_contraste("a", 0.03), _contraste("b", 0.031)])

    assert ajustados[0]["p_ajustado"] == pytest.approx(0.06)
    assert ajustados[1]["p_ajustado"] == pytest.approx(0.06)


def test_holm_caps_the_adjusted_value_at_one() -> None:
    ajustados = arnes.holm([_contraste("a", 0.8), _contraste("b", 0.9)])

    assert all(c["p_ajustado"] <= 1.0 for c in ajustados)


def test_a_p_value_that_only_passes_uncorrected_does_not_meet_the_criterion() -> None:
    """Prueba por sabotaje de la corrección.

    0,03 pasaría solo si nadie corrigiera por multiplicidad. Con dos contrastes
    confirmatorios declarados, Holm lo lleva a 0,06 y el criterio no se cumple. La
    prueba falla si alguien quita la corrección.
    """
    ajustados = arnes.holm([_contraste("a", 0.03), _contraste("b", 0.5)])

    assert ajustados[0]["p_una_cola"] == 0.03
    assert ajustados[0]["p_ajustado"] == pytest.approx(0.06)
    assert ajustados[0]["cumple_significancia"] is False
    assert ajustados[0]["cumple_criterio"] is False


def test_the_criterion_needs_both_conditions() -> None:
    magnitud_corta = arnes.holm([_contraste("a", 0.001, reduccion=0.10)])[0]
    ambas = arnes.holm([_contraste("a", 0.001, reduccion=0.40)])[0]

    assert magnitud_corta["cumple_significancia"] is True
    assert magnitud_corta["cumple_magnitud"] is False
    assert magnitud_corta["cumple_criterio"] is False
    assert ambas["cumple_criterio"] is True


def test_an_undefined_p_value_does_not_meet_the_criterion() -> None:
    ajustado = arnes.holm([_contraste("a", float("nan"))])[0]

    assert ajustado["p_ajustado"] is None
    assert ajustado["cumple_criterio"] is False


# --------------------------------------------------------------------------- #
# El contraste
# --------------------------------------------------------------------------- #
def _pliegue(nombre: str, regrets: dict[str, list[float]]) -> dict:
    """Pliegue mínimo con las medianas que el contraste necesita."""
    n = len(next(iter(regrets.values())))
    semillas = []
    for i in range(n):
        semilla = {"semilla": 100 + i, "optimo": 1.0, "objetivo": 0.99, "n_en_meseta": 5}
        for condicion, valores in regrets.items():
            semilla[condicion] = {
                "curva": [valores[i]],
                "regret": valores[i],
                "ensayos": 1,
                "fraccion_inevaluable": 0.0,
            }
        semillas.append(semilla)
    return {
        "dataset_id": nombre,
        "dataset_name": nombre,
        "meseta_mediana": 5,
        "recomendaciones": {},
        "semillas": semillas,
    }


def test_the_observation_is_the_median_of_the_seeds() -> None:
    pliegue = _pliegue("d", {"frio": [0.1, 0.5, 0.3]})

    assert arnes.observacion(pliegue, "frio") == 0.3


def test_a_clear_reduction_is_detected() -> None:
    pliegues = [
        _pliegue(f"d{i}", {"frio": [0.8], "nearest": [0.2]}) for i in range(12)
    ]

    contraste = arnes.contrastar(pliegues, "nearest", "frio")

    assert contraste["mediana_referencia"] == 0.8
    assert contraste["mediana_condicion"] == 0.2
    assert contraste["reduccion"] == pytest.approx(0.75)
    assert contraste["p_una_cola"] < 0.05
    assert contraste["efecto_rango_biserial"] == -1.0


def test_no_difference_gives_no_p_value_instead_of_a_spurious_one() -> None:
    pliegues = [_pliegue(f"d{i}", {"frio": [0.4], "nearest": [0.4]}) for i in range(8)]

    contraste = arnes.contrastar(pliegues, "nearest", "frio")

    assert contraste["n_empates"] == 8
    assert math.isnan(contraste["p_una_cola"])
    assert contraste["reduccion"] == 0.0


def test_the_number_of_ties_is_reported() -> None:
    """Wilcoxon los descarta, así que cuántos hay es información del contraste."""
    pliegues = [_pliegue("a", {"frio": [0.5], "nearest": [0.5]}),
                _pliegue("b", {"frio": [0.5], "nearest": [0.2]})]

    assert arnes.contrastar(pliegues, "nearest", "frio")["n_empates"] == 1


def test_the_effect_size_changes_sign_with_the_direction() -> None:
    mejora = [_pliegue(f"d{i}", {"frio": [0.8], "nearest": [0.2]}) for i in range(6)]
    empeora = [_pliegue(f"d{i}", {"frio": [0.2], "nearest": [0.8]}) for i in range(6)]

    assert arnes.contrastar(mejora, "nearest", "frio")["efecto_rango_biserial"] < 0
    assert arnes.contrastar(empeora, "nearest", "frio")["efecto_rango_biserial"] > 0


def test_a_contrast_can_use_any_pair_of_conditions() -> None:
    """El listón exigente es `nearest` contra `frequent`, no contra el frío."""
    pliegues = [
        _pliegue(f"d{i}", {"frio": [0.9], "nearest": [0.3], "frequent": [0.5]})
        for i in range(10)
    ]

    contraste = arnes.contrastar(pliegues, "nearest", "frequent")

    assert contraste["referencia"] == "frequent"
    assert contraste["mediana_referencia"] == 0.5


# --------------------------------------------------------------------------- #
# El protocolo entero, sobre un corpus sintético
# --------------------------------------------------------------------------- #
def _tabla(semilla: int, separacion: float) -> tuple[pd.DataFrame, np.ndarray]:
    """Tabla sintética pequeña, con señal, ruido y una columna categórica."""
    rng = np.random.default_rng(semilla)
    clase = np.tile([0, 1], FILAS_JUGUETE // 2)
    tabla = pd.DataFrame(
        {
            "senal": clase * separacion + rng.normal(0, 0.5, size=FILAS_JUGUETE),
            "ruido": rng.normal(0, 1.0, size=FILAS_JUGUETE),
            "cat": np.where(rng.random(FILAS_JUGUETE) < 0.8, clase, 1 - clase).astype(str),
        }
    )
    return tabla, clase


SEMILLAS_JUGUETE = (101, 202)
PRESUPUESTO_JUGUETE = 12
"""Protocolo recortado de la corrida de juguete.

El real son 10 semillas x 5 condiciones x 60 ensayos, que sobre tres conjuntos
serían 9000 ensayos por prueba. Estas pruebas verifican la **estructura**; los
valores del protocolo los fija ``test_the_constants_match_the_pre_registration``.

Van por parámetro y **no** parcheando las constantes del módulo: un parcheo de
alcance de módulo seguiría activo cuando corriera la prueba que compara las
constantes con el pre-registro, y esa prueba leería los valores recortados.
"""


@pytest.fixture(scope="module")
def corpus_de_juguete() -> tuple[MetaKnowledgeBase, dict, dict]:
    """Tres conjuntos sintéticos con sus entradas en la base y sus referencias.

    Tres es el mínimo: con dos, cada pliegue dejaría un solo vecino y el
    recomendador rechaza —con razón— una base cuyas meta-características no varían.
    """
    filas = [("0", 3.0, CONFIG_A), ("1", 2.0, CONFIG_B), ("2", 1.0, CONFIG_C)]
    entradas = [
        MetaKnowledgeEntry(
            dataset_id=i,
            dataset_name=f"juguete{i}",
            meta_features={"a": sep, "b": sep * 2},
            configuration=cfg,
            pipeline_params={},
            metric="macro_f1",
            metric_value=0.8,
            seed=42,
            n_trials=50,
            n_failed_trials=0,
            space_version=TABULAR_SPACE_VERSION,
        )
        for i, sep, cfg in filas
    ]
    datos = {i: _tabla(int(i), sep) for i, sep, _ in filas}
    referencias = {
        i: {
            str(s): {"optimo": 1.0, "objetivo": 0.99, "n_en_meseta": 4}
            for s in SEMILLAS_JUGUETE
        }
        for i, _, _ in filas
    }
    return MetaKnowledgeBase(entradas), datos, referencias


@pytest.fixture(scope="module")
def resultado_juguete(corpus_de_juguete) -> dict:
    """Ejecuta el LODO de juguete **una sola vez** para todas las pruebas."""
    base, datos, referencias = corpus_de_juguete
    return arnes.ejecutar_lodo(
        base, lambda i: datos[i], referencias,
        semillas=SEMILLAS_JUGUETE, presupuesto=PRESUPUESTO_JUGUETE,
    )


def test_the_whole_protocol_runs_on_a_toy_corpus(
    corpus_de_juguete, resultado_juguete
) -> None:
    base, _, _ = corpus_de_juguete

    assert len(resultado_juguete["pliegues"]) == len(base)
    for pliegue in resultado_juguete["pliegues"]:
        assert len(pliegue["semillas"]) == len(SEMILLAS_JUGUETE)
        for semilla in pliegue["semillas"]:
            for condicion in ("frio", *arnes.CONDICIONES_CALIENTES):
                assert len(semilla[condicion]["curva"]) == PRESUPUESTO_JUGUETE
                assert 0.0 <= semilla[condicion]["regret"] <= 1.0


def test_the_five_conditions_all_run(resultado_juguete) -> None:
    condiciones = set(resultado_juguete["pliegues"][0]["semillas"][0])

    assert {"frio", "nearest", "frequent", "random", "hybrid"} <= condiciones


def test_no_fold_sees_its_own_dataset_among_the_neighbours(resultado_juguete) -> None:
    """La garantía de ausencia de fuga en el nivel meta."""
    for pliegue in resultado_juguete["pliegues"]:
        for recomendacion in pliegue["recomendaciones"].values():
            assert pliegue["dataset_name"] not in recomendacion["vecinos"]


def test_the_cold_start_is_no_longer_guarded_against_censoring(corpus_de_juguete) -> None:
    """Prueba por sabotaje del cambio más delicado del experimento 2.

    En el experimento 1 el arnés abortaba si el arranque en frío quedaba censurado,
    porque con la vara autorreferencial era imposible. Aquí la vara es externa y
    quedarse corto es un dato, no un error. Con una referencia inalcanzable **todas**
    las corridas se censuran, y el arnés tiene que terminar igual. La prueba falla si
    alguien reintroduce aquella guarda.
    """
    base, datos, _ = corpus_de_juguete
    # El objetivo se pone por encima del óptimo a propósito: así es inalcanzable por
    # construcción y no por suerte del muestreador.
    imposible = {
        i: {
            str(s): {"optimo": 1.0, "objetivo": 1.5, "n_en_meseta": 0}
            for s in SEMILLAS_JUGUETE
        }
        for i in ("0", "1", "2")
    }

    resultado = arnes.ejecutar_lodo(
        base, lambda i: datos[i], imposible,
        semillas=SEMILLAS_JUGUETE, presupuesto=PRESUPUESTO_JUGUETE,
    )

    censurados = resultado["analisis"]["censurados"]
    assert censurados["frio"] == len(base) * len(SEMILLAS_JUGUETE)
    assert resultado["analisis"]["confirmatorios"]


def test_a_missing_reference_is_refused(corpus_de_juguete) -> None:
    """Un pliegue sin vara no se puede medir: se rechaza en vez de saltarlo."""
    base, datos, referencias = corpus_de_juguete
    incompleta = {k: v for k, v in referencias.items() if k != "1"}

    with pytest.raises(KeyError, match="reference_optima"):
        arnes.ejecutar_lodo(
            base, lambda i: datos[i], incompleta,
            semillas=SEMILLAS_JUGUETE, presupuesto=PRESUPUESTO_JUGUETE,
        )


def test_the_analysis_reports_both_confirmatory_contrasts(resultado_juguete) -> None:
    confirmatorios = resultado_juguete["analisis"]["confirmatorios"]

    assert len(confirmatorios) == 2
    assert (confirmatorios[0]["condicion"], confirmatorios[0]["referencia"]) == (
        "nearest",
        "frio",
    )
    assert (confirmatorios[1]["condicion"], confirmatorios[1]["referencia"]) == (
        "nearest",
        "frequent",
    )
    assert all("p_ajustado" in c for c in confirmatorios)


def test_the_secondary_analyses_repeat_the_contrasts_on_the_trial_count(
    resultado_juguete,
) -> None:
    """Es lo que permite comparar los dos experimentos en los mismos términos."""
    secundarios = resultado_juguete["analisis"]["secundarios"]
    sobre_ensayos = [c for c in secundarios if c["variable"] == "ensayos"]

    assert len(sobre_ensayos) == 2
    assert {(c["condicion"], c["referencia"]) for c in sobre_ensayos} == set(
        arnes.CONFIRMATORIOS
    )


def test_the_hybrid_is_contrasted_against_both_of_its_parents(resultado_juguete) -> None:
    pares = {
        (c["condicion"], c["referencia"])
        for c in resultado_juguete["analisis"]["secundarios"]
    }

    assert ("hybrid", "frequent") in pares
    assert ("hybrid", "nearest") in pares


def test_the_robustness_recomputes_only_the_censored_variable(resultado_juguete) -> None:
    """El regret no se censura, así que el análisis primario es inmune a esa regla."""
    robustez = resultado_juguete["analisis"]["robustez"]

    assert {c["censura"] for c in robustez} == set(arnes.CENSURAS_ROBUSTEZ)
    assert all(c["variable"] == "ensayos" for c in robustez)


def test_the_per_dataset_table_carries_the_plateau_covariate(resultado_juguete) -> None:
    for fila in resultado_juguete["analisis"]["por_dataset"]:
        assert fila["meseta"] == 4
        assert "frio_inevaluable" in fila
        assert "nearest_ensayos" in fila


# --------------------------------------------------------------------------- #
# Las constantes son las del pre-registro
# --------------------------------------------------------------------------- #
def test_the_constants_match_the_pre_registration() -> None:
    """Si alguien las cambiara, el arnés dejaría de medir lo pre-registrado.

    Ninguna prueba de comportamiento se enteraría, porque el protocolo seguiría
    ejecutándose: solo mediría otra cosa.
    """
    assert arnes.SEMILLAS == (101, 202, 303, 404, 505, 606, 707, 808, 909, 1010)
    assert arnes.PRESUPUESTO == 60
    assert arnes.CENSURA == 61
    assert arnes.CENSURAS_ROBUSTEZ == (120, 600)
    assert arnes.PRESUPUESTOS_REDUCIDOS == (5, 15, 30)
    assert arnes.N_CONFIGURACIONES == 5
    assert arnes.CONDICIONES_CALIENTES == ("nearest", "frequent", "random", "hybrid")
    assert arnes.CONFIRMATORIOS == (("nearest", "frio"), ("nearest", "frequent"))
    assert arnes.REDUCCION_EXIGIDA == 0.30
    assert arnes.ALFA == 0.05


def test_the_primary_condition_is_unchanged_from_experiment_one() -> None:
    """`nearest` es lo que permite atribuir al instrumento lo que mejore."""
    assert arnes.CONDICION_PRIMARIA == "nearest"
    assert arnes.CONFIRMATORIOS[0] == ("nearest", "frio")
    assert arnes.SEMILLA_ALEATORIA == 202
