# tests/test_validate_meta_knowledge.py
"""Pruebas de la validación de la base de meta-conocimiento.

Cada prueba **rompe una entrada a propósito** y comprueba que la validación lo
detecte. Una validación que nunca se ha visto fallar no prueba nada: pasaría igual
sobre una base defectuosa, que es justo el caso que existe para descubrir.

Un defecto de la base no produce un error visible, produce recomendaciones
equivocadas. Por eso las comprobaciones son del artefacto guardado y no del código
que lo generó.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

pytest.importorskip("openml", reason="la validación importa el arnés, que usa openml")

import validate_meta_knowledge as validacion  # noqa: E402

from autopipe.meta_knowledge import MetaKnowledgeBase, MetaKnowledgeEntry  # noqa: E402
from autopipe.search_space import (  # noqa: E402
    TABULAR_SPACE_VERSION,
    build_tabular_search_space,
    pipeline_from_configuration,
)

CONFIGURACION = {"imputation": 0, "scaling": 1, "encoding": 0, "selection": 3}


def entrada(**cambios) -> MetaKnowledgeEntry:
    """Arma una entrada sana, con los cambios indicados."""
    base = MetaKnowledgeEntry(
        dataset_id="1",
        dataset_name="sano",
        meta_features={"a": 1.0, "b": 2.0},
        configuration=dict(CONFIGURACION),
        pipeline_params=pipeline_from_configuration(CONFIGURACION).get_params(),
        metric="macro_f1",
        metric_value=0.8,
        seed=validacion.SEMILLA,
        n_trials=validacion.ENSAYOS,
        n_failed_trials=3,
        space_version=TABULAR_SPACE_VERSION,
        seconds=2.0,
    )
    return replace(base, **cambios)


# --------------------------------------------------------------------------- #
# La entrada sana no genera falsos positivos
# --------------------------------------------------------------------------- #
def test_a_healthy_entry_reports_nothing() -> None:
    assert validacion.verificar_campos(entrada()) == []
    assert validacion.verificar_configuracion(entrada()) == []


# --------------------------------------------------------------------------- #
# Cobertura
# --------------------------------------------------------------------------- #
def test_a_missing_dataset_is_detected() -> None:
    base = MetaKnowledgeBase([entrada()])

    problemas = validacion.verificar_cobertura(base, [(1, "sano"), (2, "ausente")])

    assert any("faltan" in p for p in problemas)


def test_an_extra_dataset_is_detected() -> None:
    """Una entrada de un dataset que ya no está en el corpus contamina la base."""
    base = MetaKnowledgeBase([entrada(), entrada(dataset_id="9", dataset_name="viejo")])

    problemas = validacion.verificar_cobertura(base, [(1, "sano")])

    assert any("no están en el corpus" in p for p in problemas)


def test_exact_coverage_reports_nothing() -> None:
    base = MetaKnowledgeBase([entrada()])

    assert validacion.verificar_cobertura(base, [(1, "sano")]) == []


# --------------------------------------------------------------------------- #
# Campos
# --------------------------------------------------------------------------- #
def test_an_empty_meta_feature_vector_is_detected() -> None:
    assert any(
        "meta-características" in p for p in validacion.verificar_campos(entrada(meta_features={}))
    )


@pytest.mark.parametrize("valor", [-0.1, 1.5, float("nan")])
def test_a_metric_outside_zero_one_is_detected(valor: float) -> None:
    problemas = validacion.verificar_campos(entrada(metric_value=valor))

    assert any("fuera de [0, 1]" in p for p in problemas)


def test_a_different_seed_is_detected() -> None:
    """La semilla la fija el pre-registro; otra invalida la comparación."""
    assert any("semilla" in p for p in validacion.verificar_campos(entrada(seed=7)))


def test_a_different_trial_budget_is_detected() -> None:
    assert any("ensayos" in p for p in validacion.verificar_campos(entrada(n_trials=30)))


def test_more_failed_trials_than_trials_is_detected() -> None:
    problemas = validacion.verificar_campos(entrada(n_failed_trials=99))

    assert any("no son creíbles" in p for p in problemas)


def test_an_entry_where_every_trial_failed_is_detected() -> None:
    """Con todos los ensayos fallidos, el ganador se eligió sin comparación."""
    rota = entrada(n_failed_trials=validacion.ENSAYOS)

    assert any("no significa nada" in p for p in validacion.verificar_campos(rota))


def test_an_entry_from_another_space_version_is_detected() -> None:
    assert any(
        "versión del espacio" in p
        for p in validacion.verificar_campos(entrada(space_version="tabular-v0"))
    )


# --------------------------------------------------------------------------- #
# Interpretabilidad de la configuración
# --------------------------------------------------------------------------- #
def test_a_configuration_with_missing_steps_is_detected() -> None:
    rota = entrada(configuration={"imputation": 0})

    assert any("no son los" in p for p in validacion.verificar_configuracion(rota))


def test_an_out_of_range_index_is_detected() -> None:
    rota = entrada(configuration={**CONFIGURACION, "scaling": 99})

    assert any("no se pudo resolver" in p for p in validacion.verificar_configuracion(rota))


def test_parameters_that_do_not_match_the_indices_are_detected() -> None:
    """Es la comprobación que detectaría que la base y el espacio se separaron.

    Los índices siguen siendo enteros válidos, así que nada falla al leerlos: lo
    que cambia es a qué transformaciones apuntan.
    """
    rota = entrada(pipeline_params={"0_ImputeMissing": {"numeric_strategy": "median"}})

    problemas = validacion.verificar_configuracion(rota)

    assert any("no coincide con los parámetros guardados" in p for p in problemas)


def test_a_configuration_that_skips_every_step_is_interpretable() -> None:
    """Omitir todos los pasos es una configuración legítima, no un defecto."""
    espacio = build_tabular_search_space()
    todos_omitidos = {paso: len(opciones) for paso, opciones in espacio.items()}
    vacia = entrada(configuration=todos_omitidos, pipeline_params={})

    assert validacion.verificar_configuracion(vacia) == []


# --------------------------------------------------------------------------- #
# Alineación
# --------------------------------------------------------------------------- #
def test_a_base_where_nothing_survives_is_detected() -> None:
    """Sin supervivientes el recomendador no tiene con qué comparar datasets."""
    base = MetaKnowledgeBase(
        [
            entrada(dataset_id="1", meta_features={"a": float("nan"), "b": 1.0}),
            entrada(dataset_id="2", meta_features={"a": 1.0, "b": float("nan")}),
        ]
    )

    problemas = validacion.verificar_alineacion(base)

    assert any("ninguna meta-característica sobrevive" in p for p in problemas)


def test_an_aligned_base_reports_nothing() -> None:
    base = MetaKnowledgeBase([entrada(dataset_id="1"), entrada(dataset_id="2")])

    assert validacion.verificar_alineacion(base) == []
