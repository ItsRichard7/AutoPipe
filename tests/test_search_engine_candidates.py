# tests/test_search_engine_candidates.py
"""Pruebas de dos comportamientos de los motores que antes eran implícitos.

1. **Los candidatos se pueden inspeccionar sin evaluarlos.** ``build_pipelines`` es la
   versión pública de lo que ``run`` recorre; antes solo existía el método privado, y un
   notebook lo usaba.
2. **Una búsqueda sin ningún candidato evaluable ya no termina en silencio.** El motor
   sigue devolviendo lo mismo que antes —cambiarlo alteraría resultados ya reportados—,
   pero registra un aviso, porque ese «ganador» no ganó nada.

Van en un archivo propio para no tocar los de la librería base, cuyas funciones de prueba
se verifican idénticas a las del commit de migración.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest
from sklearn.ensemble import IsolationForest

from autopipe import (
    GaussianBlur,
    GridSearchEngine,
    Normalize,
    OptunaSearchEngine,
    build_tabular_search_space,
)

AVISO_REJILLA = "Ningún pipeline pudo evaluarse"
AVISO_OPTUNA = "Ningún ensayo pudo evaluarse"


class ModeloQueFalla:
    """Modelo base que no puede entrenar: vuelve inevaluables a todos los candidatos."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        raise ValueError("este modelo no entrena")

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise ValueError("este modelo no predice")


@pytest.fixture
def datos() -> tuple[list[np.ndarray], np.ndarray]:
    """Veinte imágenes pequeñas, seis de ellas anómalas."""
    rng = np.random.default_rng(0)
    imagenes = [rng.integers(0, 255, size=(8, 8), dtype=np.uint8) for _ in range(20)]
    return imagenes, np.array([0] * 14 + [1] * 6)


def _espacio() -> dict:
    return {
        "blur": [GaussianBlur(kernel_size=3), GaussianBlur(kernel_size=5)],
        "normalize": [Normalize(mean=0.5, std=0.5)],
    }


def _parametros(pipelines: list) -> list:
    return [p.get_params() for p in pipelines]


# --------------------------------------------------------------------------- #
# Los candidatos, sin evaluarlos
# --------------------------------------------------------------------------- #
def test_the_candidates_include_the_option_of_skipping_each_step() -> None:
    """(2 opciones + omitir) × (1 opción + omitir) = 6 candidatos."""
    assert len(GridSearchEngine(_espacio()).build_pipelines()) == 6


def test_the_canonical_tabular_space_has_144_candidates() -> None:
    motor = GridSearchEngine(build_tabular_search_space(), task="classification")

    assert len(motor.build_pipelines()) == 144


def test_the_public_method_is_the_one_the_private_name_still_points_to() -> None:
    """El nombre anterior se conserva para los guiones y pruebas que ya lo usan."""
    motor = GridSearchEngine(_espacio())

    assert _parametros(motor.build_pipelines()) == _parametros(motor._build_pipelines())


def test_run_evaluates_exactly_the_candidates_it_exposes_in_the_same_order(datos) -> None:
    imagenes, etiquetas = datos
    motor = GridSearchEngine(_espacio(), val_size=0.3, random_state=0)

    _, _, resultados = motor.run(imagenes, etiquetas, IsolationForest(random_state=0))

    assert [r["params"] for r in resultados] == _parametros(motor.build_pipelines())


def test_each_call_builds_fresh_pipelines() -> None:
    motor = GridSearchEngine(_espacio())

    assert motor.build_pipelines() is not motor.build_pipelines()


# --------------------------------------------------------------------------- #
# Ningún candidato evaluable
# --------------------------------------------------------------------------- #
def test_grid_search_warns_when_no_pipeline_could_be_evaluated(datos, caplog) -> None:
    caplog.set_level(logging.WARNING, logger="autopipe.search_engine")
    imagenes, etiquetas = datos
    motor = GridSearchEngine(_espacio(), val_size=0.3, random_state=0)

    mejor, metricas, resultados = motor.run(imagenes, etiquetas, ModeloQueFalla())

    assert AVISO_REJILLA in caplog.text
    assert all(math.isnan(r["objective"]) for r in resultados)
    # Lo que devuelve no cambia: el primer candidato, con sus métricas NaN.
    assert mejor.get_params() == motor.build_pipelines()[0].get_params()
    assert all(math.isnan(valor) for valor in metricas.values())


def test_grid_search_does_not_warn_when_some_pipeline_was_evaluated(datos, caplog) -> None:
    caplog.set_level(logging.WARNING, logger="autopipe.search_engine")
    imagenes, etiquetas = datos
    motor = GridSearchEngine(_espacio(), val_size=0.3, random_state=0)

    motor.run(imagenes, etiquetas, IsolationForest(random_state=0))

    assert AVISO_REJILLA not in caplog.text


def test_optuna_warns_when_no_trial_could_be_evaluated(datos, caplog) -> None:
    caplog.set_level(logging.WARNING, logger="autopipe.search_engine")
    imagenes, etiquetas = datos
    motor = OptunaSearchEngine(_espacio(), n_trials=3, val_size=0.3, random_state=0)

    _, metricas, resultados = motor.run(imagenes, etiquetas, ModeloQueFalla())

    assert AVISO_OPTUNA in caplog.text
    assert len(resultados) == 3
    assert all(math.isnan(valor) for valor in metricas.values())


def test_optuna_does_not_warn_when_some_trial_was_evaluated(datos, caplog) -> None:
    caplog.set_level(logging.WARNING, logger="autopipe.search_engine")
    imagenes, etiquetas = datos
    motor = OptunaSearchEngine(_espacio(), n_trials=3, val_size=0.3, random_state=0)

    motor.run(imagenes, etiquetas, IsolationForest(random_state=0))

    assert AVISO_OPTUNA not in caplog.text
