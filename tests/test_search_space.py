# tests/test_search_space.py
"""Pruebas del espacio de búsqueda tabular canónico y versionado.

La prueba central de este archivo es :func:`test_the_index_of_every_option_is_frozen`.
Los motores de búsqueda registran por ensayo **índices enteros** sobre las listas
de opciones, de modo que reordenar una lista cambiaría el significado de todas
las configuraciones ya guardadas sin que nada fallara. Esa prueba convierte el
reordenamiento en un fallo visible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import KNeighborsClassifier

from autopipe.pipeline.base import BaseTransform
from autopipe.search_engine.grid_search import GridSearchEngine
from autopipe.search_engine.optuna_search import OptunaSearchEngine
from autopipe.search_space import (
    TABULAR_SPACE_STEPS,
    TABULAR_SPACE_VERSION,
    build_tabular_search_space,
    canonical_configuration,
    describe_tabular_search_space,
    omitted_steps,
    pipeline_from_configuration,
    validate_space_version,
)

# Composición congelada de tabular-v1: paso, índice, clase y parámetros.
# Cualquier cambio aquí obliga a cambiar TABULAR_SPACE_VERSION.
COMPOSICION_CONGELADA = [
    ("imputation", 0, "ImputeMissing", {"numeric_strategy": "mean"}),
    ("imputation", 1, "ImputeMissing", {"numeric_strategy": "median"}),
    ("scaling", 0, "ScaleNumeric", {"method": "standard"}),
    ("scaling", 1, "ScaleNumeric", {"method": "minmax"}),
    ("scaling", 2, "ScaleNumeric", {"method": "robust"}),
    ("encoding", 0, "EncodeCategorical", {"method": "onehot"}),
    ("encoding", 1, "EncodeCategorical", {"method": "ordinal"}),
    ("selection", 0, "SelectFeatures", {"fraction": 0.25, "criterion": "f_classif"}),
    ("selection", 1, "SelectFeatures", {"fraction": 0.5, "criterion": "f_classif"}),
    ("selection", 2, "SelectFeatures", {"fraction": 0.75, "criterion": "f_classif"}),
]

# 3 x 4 x 3 x 4: cada paso aporta sus opciones más la de omitirse.
COMBINACIONES = 144


@pytest.fixture
def datos_tabulares() -> tuple[pd.DataFrame, np.ndarray]:
    """Tabla mixta con faltantes, tres clases y señal separable.

    La columna categórica se desalinea de la clase en una de cada cinco filas. Si
    coincidiera con ella exactamente, sus indicadoras separarían las clases a la
    perfección y ``f_classif`` puntuaría ``+inf``: un caso legítimo, pero poco
    representativo de un dataset real.
    """
    rng = np.random.default_rng(0)
    n = 90
    clase = np.tile([0, 1, 2], n // 3)
    categoria = np.where(clase == 0, "norte", np.where(clase == 1, "sur", "este"))
    desalineadas = rng.choice(n, size=n // 5, replace=False)
    categoria[desalineadas] = rng.choice(["norte", "sur", "este"], size=n // 5)
    tabla = pd.DataFrame(
        {
            "señal": clase * 3.0 + rng.normal(0, 0.4, size=n),
            "ruido": rng.normal(0, 1.0, size=n),
            "escala_grande": rng.normal(0, 1.0, size=n) * 500,
            "cat": categoria,
        }
    )
    tabla.loc[:4, "ruido"] = np.nan
    return tabla, clase


# --------------------------------------------------------------------------- #
# Identidad congelada del espacio
# --------------------------------------------------------------------------- #
def test_the_index_of_every_option_is_frozen() -> None:
    """Ningún paso puede reordenarse sin cambiar la versión del espacio.

    Un índice de Optuna solo significa algo respecto de un orden concreto. Si
    esta prueba falla, o bien el cambio se revierte, o bien se incrementa
    ``TABULAR_SPACE_VERSION`` y se invalidan las configuraciones ya guardadas.
    """
    descripcion = [
        (fila["step"], fila["index"], fila["transform"], fila["params"])
        for fila in describe_tabular_search_space()
    ]

    assert descripcion == COMPOSICION_CONGELADA


def test_step_order_is_frozen() -> None:
    """El orden de los pasos es parte de la identidad de la versión."""
    espacio = build_tabular_search_space()

    assert list(espacio) == list(TABULAR_SPACE_STEPS)
    assert TABULAR_SPACE_STEPS == ("imputation", "scaling", "encoding", "selection")


def test_scaling_comes_before_encoding() -> None:
    """El escalado no puede tocar las indicadoras one-hot.

    Estandarizarlas llevaría las categorías raras a valores extremos, que con un
    modelo base de vecinos más cercanos dominaría la distancia.
    """
    pasos = list(TABULAR_SPACE_STEPS)

    assert pasos.index("scaling") < pasos.index("encoding")


def test_selection_is_the_last_step() -> None:
    """Selecciona cuando la tabla ya es enteramente numérica."""
    assert TABULAR_SPACE_STEPS[-1] == "selection"


def test_every_option_is_a_transform() -> None:
    espacio = build_tabular_search_space()

    for opciones in espacio.values():
        assert all(isinstance(opcion, BaseTransform) for opcion in opciones)


def test_no_option_names_a_column() -> None:
    """El espacio debe ser transferible: ningún parámetro depende del dataset."""
    for fila in describe_tabular_search_space():
        for valor in fila["params"].values():
            assert isinstance(valor, (str, float, int))
            if isinstance(valor, float):
                assert 0 < valor <= 1


# --------------------------------------------------------------------------- #
# Instancias frescas
# --------------------------------------------------------------------------- #
def test_each_call_returns_fresh_instances() -> None:
    """Compartir instancias arrastraría el estado de una búsqueda a la siguiente."""
    primero = build_tabular_search_space()
    segundo = build_tabular_search_space()

    for paso in TABULAR_SPACE_STEPS:
        for uno, otro in zip(primero[paso], segundo[paso]):
            assert uno is not otro
            assert uno.get_params() == otro.get_params()


def test_a_fitted_space_does_not_contaminate_the_next(datos_tabulares) -> None:
    tabla, _ = datos_tabulares
    primero = build_tabular_search_space()
    primero["imputation"][0].fit(tabla)

    segundo = build_tabular_search_space()

    assert segundo["imputation"][0].fitted is False


# --------------------------------------------------------------------------- #
# Versionado
# --------------------------------------------------------------------------- #
def test_the_current_version_is_accepted() -> None:
    validate_space_version(TABULAR_SPACE_VERSION)


def test_a_configuration_from_another_version_is_rejected() -> None:
    """Se rechaza en lugar de ejecutarse: es el fallo mudo que evita el versionado."""
    with pytest.raises(ValueError, match="tabular-v1"):
        validate_space_version("tabular-v0")


def test_the_version_identifier_is_not_empty() -> None:
    assert isinstance(TABULAR_SPACE_VERSION, str)
    assert TABULAR_SPACE_VERSION


# --------------------------------------------------------------------------- #
# De índices a pipeline
# --------------------------------------------------------------------------- #
def test_a_configuration_resolves_to_its_pipeline() -> None:
    """Es la operación inversa de lo que registran los motores."""
    pipeline = pipeline_from_configuration(
        {"imputation": 0, "scaling": 1, "encoding": 0, "selection": 3}
    )

    assert [type(paso).__name__ for paso in pipeline.steps] == [
        "ImputeMissing",
        "ScaleNumeric",
        "EncodeCategorical",
    ]


def test_the_last_index_of_a_step_means_skipping_it() -> None:
    espacio = build_tabular_search_space()
    todos_omitidos = {paso: len(opciones) for paso, opciones in espacio.items()}

    assert pipeline_from_configuration(todos_omitidos).steps == []


def test_a_configuration_missing_a_step_is_rejected() -> None:
    with pytest.raises(ValueError, match="faltan los pasos"):
        pipeline_from_configuration({"imputation": 0})


def test_an_index_outside_its_step_range_is_rejected() -> None:
    """Un índice fuera de rango delata que la configuración es de otro espacio."""
    with pytest.raises(ValueError, match="fuera de rango"):
        pipeline_from_configuration(
            {"imputation": 0, "scaling": 99, "encoding": 0, "selection": 0}
        )


def test_a_negative_index_is_rejected() -> None:
    with pytest.raises(ValueError, match="fuera de rango"):
        pipeline_from_configuration(
            {"imputation": -1, "scaling": 0, "encoding": 0, "selection": 0}
        )


def test_the_resolved_steps_follow_the_canonical_order() -> None:
    pipeline = pipeline_from_configuration(
        {"imputation": 0, "scaling": 0, "encoding": 0, "selection": 0}
    )

    assert list(pipeline.get_params()) == [
        "0_ImputeMissing",
        "1_ScaleNumeric",
        "2_EncodeCategorical",
        "3_SelectFeatures",
    ]


# --------------------------------------------------------------------------- #
# El espacio atraviesa los dos motores
# --------------------------------------------------------------------------- #
def test_grid_search_covers_the_whole_space(datos_tabulares) -> None:
    """Las 144 combinaciones se evalúan y alguna gana con métrica real."""
    tabla, etiquetas = datos_tabulares
    motor = GridSearchEngine(
        build_tabular_search_space(),
        val_size=0.3,
        random_state=42,
        task="classification",
    )

    mejor, metricas, resultados = motor.run(
        tabla, etiquetas, KNeighborsClassifier(n_neighbors=5, algorithm="brute")
    )

    assert len(resultados) == COMBINACIONES
    assert not np.isnan(metricas["macro_f1"])
    assert 0.0 <= metricas["macro_f1"] <= 1.0
    assert len(mejor.steps) <= len(TABULAR_SPACE_STEPS)


def test_optuna_search_runs_on_the_canonical_space(datos_tabulares) -> None:
    """El motor que llevará el arranque en caliente opera sobre el mismo espacio."""
    tabla, etiquetas = datos_tabulares
    motor = OptunaSearchEngine(
        build_tabular_search_space(),
        n_trials=12,
        val_size=0.3,
        random_state=101,
        task="classification",
    )

    _, metricas, resultados = motor.run(
        tabla, etiquetas, KNeighborsClassifier(n_neighbors=5, algorithm="brute")
    )

    assert len(resultados) == 12
    assert not np.isnan(metricas["macro_f1"])


def test_the_same_seed_selects_the_same_winner(datos_tabulares) -> None:
    """Criterio de reproducibilidad del anteproyecto, Sección 4.3.5."""
    tabla, etiquetas = datos_tabulares
    modelo = KNeighborsClassifier(n_neighbors=5, algorithm="brute")

    ganadores = [
        OptunaSearchEngine(
            build_tabular_search_space(),
            n_trials=12,
            val_size=0.3,
            random_state=101,
            task="classification",
        )
        .run(tabla, etiquetas, modelo)[0]
        .get_params()
        for _ in range(2)
    ]

    assert ganadores[0] == ganadores[1]


# ---------------------------------------------------------------------------
# Desempate entre configuraciones que empatan
#
# Sobre un dataset numérico y sin faltantes, la imputación y la codificación son
# la identidad: varias combinaciones de esos dos pasos son el mismo pipeline y
# empatan exactamente. Cuál de ellas guarda una búsqueda depende del orden de
# visita del muestreador, que es una propiedad de la corrida y no del dataset.
# ---------------------------------------------------------------------------


def test_omitted_steps_detects_the_omission_index() -> None:
    """Un paso está omitido cuando su índice es el número de opciones."""
    configuracion = {"imputation": 2, "scaling": 0, "encoding": 2, "selection": 3}
    assert omitted_steps(configuracion) == ["imputation", "encoding", "selection"]


def test_omitted_steps_returns_nothing_when_all_steps_are_present() -> None:
    """Una configuración sin omisiones no reporta ningún paso."""
    assert omitted_steps({"imputation": 0, "scaling": 1, "encoding": 1, "selection": 2}) == []


def test_the_canonical_configuration_prefers_omitting_fewer_steps() -> None:
    """Entre empatadas gana la que omite menos pasos, sea cual sea el orden.

    Es la razón de ser de la regla, y no es estética: omitir la codificación no
    cuesta nada en un dataset sin columnas categóricas, pero vuelve el pipeline
    **inevaluable** en el dataset receptor que sí las tenga, de modo que el
    ensayo inyectado se pierde sin producir una métrica.
    """
    omite_dos = {"imputation": 2, "scaling": 0, "encoding": 2, "selection": 0}
    omite_cero = {"imputation": 0, "scaling": 0, "encoding": 0, "selection": 0}
    assert canonical_configuration([omite_dos, omite_cero]) == omite_cero
    assert canonical_configuration([omite_cero, omite_dos]) == omite_cero


def test_the_canonical_configuration_breaks_remaining_ties_by_index() -> None:
    """Con las mismas omisiones, gana la de índices menores en el orden canónico."""
    baja = {"imputation": 0, "scaling": 0, "encoding": 0, "selection": 0}
    alta = {"imputation": 0, "scaling": 1, "encoding": 0, "selection": 0}
    assert canonical_configuration([alta, baja]) == baja


def test_the_canonical_configuration_is_deterministic() -> None:
    """El resultado no depende del orden en que se reciban las empatadas."""
    empatadas = [
        {"imputation": i, "scaling": 2, "encoding": e, "selection": 1}
        for i in range(3)
        for e in range(3)
    ]
    esperada = {"imputation": 0, "scaling": 2, "encoding": 0, "selection": 1}
    assert canonical_configuration(empatadas) == esperada
    assert canonical_configuration(list(reversed(empatadas))) == esperada


def test_a_single_configuration_is_its_own_canonical() -> None:
    """Sin empate no hay nada que desempatar."""
    unica = {"imputation": 1, "scaling": 2, "encoding": 0, "selection": 3}
    assert canonical_configuration([unica]) == unica


def test_the_canonical_configuration_rejects_an_empty_set() -> None:
    """Pedir la canónica de nada es un error, no un valor por omisión."""
    with pytest.raises(ValueError, match="No hay configuraciones"):
        canonical_configuration([])


def test_the_canonical_configuration_is_executable(datos_tabulares) -> None:
    """La canónica de un empate se resuelve a un pipeline que corre.

    Cierra el circuito: la regla existe para que la configuración transferida sea
    ejecutable en el dataset receptor, así que se comprueba que lo sea.
    """
    tabla, etiquetas = datos_tabulares
    empatadas = [
        {"imputation": 2, "scaling": 0, "encoding": 2, "selection": 3},
        {"imputation": 0, "scaling": 0, "encoding": 0, "selection": 3},
    ]
    pipeline = pipeline_from_configuration(canonical_configuration(empatadas))
    assert omitted_steps(canonical_configuration(empatadas)) == ["selection"]
    transformada = pipeline.fit_transform(tabla, etiquetas)
    assert transformada.to_numpy(dtype=float).shape[0] == len(tabla)
