# tests/test_tabular_protocol.py
"""Garantías del protocolo de evaluación sobre datos tabulares.

Las pruebas de ``test_tabular_transforms.py`` verifican cada transformación por
separado. Estas verifican la garantía **a nivel de protocolo**: que
:func:`~autopipe.search_engine.grid_search.evaluate_pipeline` ajuste el pipeline
completo únicamente con ``train_idx``, también cuando el pipeline lleva un paso
**supervisado**.

La selección supervisada es el caso de fuga más fuerte del proyecto. Un escalador
que viera la validación filtraría una media; una selección que la viera elegiría
**otras columnas**, y la matriz de características con la que se entrena el modelo
sería distinta. Por eso estas pruebas están escritas para **fallar** si alguien
revierte el ajuste solo con entrenamiento (véase
``docs/decisiones/ADR-001-contrato-fit-transform.md``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import KNeighborsClassifier

from autopipe.parallel_runner.runner import ParallelRunner
from autopipe.pipeline.base import Dataset
from autopipe.pipeline.pipeline import Pipeline
from autopipe.pipeline.tabular_transforms import (
    EncodeCategorical,
    ImputeMissing,
    ScaleNumeric,
    SelectFeatures,
)
from autopipe.search_engine.grid_search import (
    GridSearchEngine,
    evaluate_pipeline,
    evaluate_pipeline_safe,
)
from autopipe.tasks import CLASSIFICATION

AJUSTES: list[dict] = []
"""Registro de los ajustes observados por :class:`SelectorEspia`.

Es una lista de módulo y no un atributo de instancia a propósito:
``evaluate_pipeline`` clona el pipeline antes de ajustarlo, así que lo que la
copia escriba en sí misma se pierde. La lista global sí sobrevive, porque la
copia la referencia por nombre.
"""


class SelectorEspia(SelectFeatures):
    """Selección que anota con cuántas filas se la ajustó y qué eligió.

    Debe ser de nivel superior del módulo para que ``pickle`` pueda serializarla.
    """

    def _fit(self, X: Dataset, y: np.ndarray | None = None) -> None:
        """Ajusta y registra la observación."""
        super()._fit(X, y)
        AJUSTES.append(
            {"filas": len(X), "columnas": list(self.selected_columns_ or [])}
        )


@pytest.fixture(autouse=True)
def _limpiar_registro():
    """Cada prueba empieza con el registro vacío."""
    AJUSTES.clear()
    yield
    AJUSTES.clear()


@pytest.fixture
def datos_con_trampa() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Tabla donde la columna informativa cambia entre entrenamiento y validación.

    En las cuatro filas de entrenamiento separa las clases la columna ``a``; en
    las cuatro de validación, la ``b``. Ajustar con el dataset completo elige
    ``b``, así que la elección delata de inmediato con qué se ajustó.
    """
    datos = pd.DataFrame(
        {
            "a": [0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0],
            "b": [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 5.0, 5.0],
        }
    )
    etiquetas = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    return datos, etiquetas, np.arange(0, 4), np.arange(4, 8)


def _evaluar(pipeline: Pipeline, datos, etiquetas, train_idx, val_idx) -> tuple:
    """Atraviesa el protocolo con kNN y macro-F1."""
    return evaluate_pipeline(
        pipeline,
        datos,
        etiquetas,
        KNeighborsClassifier(n_neighbors=1, algorithm="brute"),
        "macro_f1",
        train_idx,
        val_idx,
        CLASSIFICATION.build_metric_calculator(0.5),
        task="classification",
    )


# --------------------------------------------------------------------------- #
# La trampa discrimina de verdad
# --------------------------------------------------------------------------- #
def test_a_leaked_fit_would_choose_a_different_column(datos_con_trampa) -> None:
    """Sin esto, las pruebas de abajo pasarían aunque hubiera fuga.

    Verifica que el dataset distingue las dos conductas: ajustar solo con
    entrenamiento elige ``a``; ajustar con el dataset completo elige ``b``.
    """
    datos, etiquetas, train_idx, _ = datos_con_trampa

    honesto = SelectFeatures(0.5, "f_classif").fit(
        datos.iloc[train_idx], etiquetas[train_idx]
    )
    con_fuga = SelectFeatures(0.5, "f_classif").fit(datos, etiquetas)

    assert honesto.selected_columns_ == ["a"]
    assert con_fuga.selected_columns_ == ["b"]


# --------------------------------------------------------------------------- #
# La garantía a nivel de protocolo
# --------------------------------------------------------------------------- #
def test_the_protocol_fits_the_selection_only_on_training(datos_con_trampa) -> None:
    """Prueba anti-fuga tabular con un paso supervisado.

    Falla si ``evaluate_pipeline`` vuelve a ajustar con el dataset completo: se
    registrarían ocho filas en lugar de cuatro, y la columna elegida sería otra.
    """
    datos, etiquetas, train_idx, val_idx = datos_con_trampa

    _evaluar(Pipeline([SelectorEspia(0.5, "f_classif")]), datos, etiquetas, train_idx, val_idx)

    assert len(AJUSTES) == 1
    assert AJUSTES[0]["filas"] == len(train_idx)
    assert AJUSTES[0]["columnas"] == ["a"]


def test_the_canonical_order_fits_the_selection_only_on_training(
    datos_con_trampa,
) -> None:
    """La garantía se mantiene con los cuatro pasos del espacio canónico.

    El ajuste del pipeline es secuencial, así que la selección se ajusta sobre la
    salida de los pasos anteriores. Esa salida también debe venir solo del
    entrenamiento.
    """
    datos, etiquetas, train_idx, val_idx = datos_con_trampa
    pipeline = Pipeline(
        [
            ImputeMissing("mean"),
            ScaleNumeric("standard"),
            EncodeCategorical("onehot"),
            SelectorEspia(0.5, "f_classif"),
        ]
    )

    _evaluar(pipeline, datos, etiquetas, train_idx, val_idx)

    assert AJUSTES[0]["filas"] == len(train_idx)
    assert AJUSTES[0]["columnas"] == ["a"]


def test_validation_is_transformed_with_the_training_choice(datos_con_trampa) -> None:
    """La validación se recorta con las columnas que eligió el entrenamiento."""
    datos, etiquetas, train_idx, val_idx = datos_con_trampa
    selector = SelectFeatures(0.5, "f_classif")

    Pipeline([selector]).fit(datos.iloc[train_idx], etiquetas[train_idx])
    validacion = selector.transform(datos.iloc[val_idx])

    assert list(validacion.columns) == ["a"]


def test_each_candidate_gets_its_own_fit(datos_con_trampa) -> None:
    """Evaluar dos candidatos no puede reutilizar el ajuste del primero.

    Los motores comparten instancias del espacio de búsqueda entre candidatos, de
    modo que sin el clonado de ``evaluate_pipeline`` el segundo heredaría el
    estado del primero.
    """
    datos, etiquetas, train_idx, val_idx = datos_con_trampa
    compartido = SelectorEspia(0.5, "f_classif")

    _evaluar(Pipeline([compartido]), datos, etiquetas, train_idx, val_idx)
    _evaluar(Pipeline([compartido]), datos, etiquetas, train_idx, val_idx)

    assert len(AJUSTES) == 2
    assert compartido.fitted is False


# --------------------------------------------------------------------------- #
# Las etiquetas llegan a cualquier posición del pipeline
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("posicion", [0, 1, 2])
def test_the_supervised_step_receives_labels_at_any_position(
    datos_con_trampa, posicion: int
) -> None:
    """El paso supervisado necesita ``y`` esté donde esté.

    ``Pipeline.fit`` ajusta el último paso sin transformarlo, así que la última
    posición recorre un camino distinto de las demás y hay que probarla también.
    """
    datos, etiquetas, train_idx, val_idx = datos_con_trampa
    pasos: list = [ImputeMissing("mean"), ScaleNumeric("standard")]
    pasos.insert(posicion, SelectorEspia(0.5, "f_classif"))

    _evaluar(Pipeline(pasos), datos, etiquetas, train_idx, val_idx)

    assert AJUSTES[0]["filas"] == len(train_idx)


# --------------------------------------------------------------------------- #
# Candidatos inevaluables
# --------------------------------------------------------------------------- #
def test_an_unevaluable_candidate_yields_nan_instead_of_raising() -> None:
    """Omitir la imputación deja nulos que el criterio supervisado no puede puntuar.

    El candidato debe quedar en ``NaN`` y no interrumpir la búsqueda: es lo que
    permite que el espacio canónico ofrezca la omisión de cada paso.
    """
    datos = pd.DataFrame(
        {"x": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0], "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
    )
    etiquetas = np.array([0, 0, 1, 1, 0, 1])

    valor, metricas = evaluate_pipeline_safe(
        Pipeline([SelectFeatures(0.5, "f_classif")]),
        datos,
        etiquetas,
        KNeighborsClassifier(n_neighbors=1, algorithm="brute"),
        "macro_f1",
        np.arange(0, 4),
        np.arange(4, 6),
        CLASSIFICATION.build_metric_calculator(0.5),
        task="classification",
    )

    assert np.isnan(valor)
    assert set(metricas) == set(CLASSIFICATION.valid_metrics)
    assert all(np.isnan(v) for v in metricas.values())


def test_when_every_candidate_is_unevaluable_the_winner_is_the_first() -> None:
    """Caracteriza una deuda conocida: el ganador se elige en silencio.

    Si ningún candidato es evaluable, ``GridSearchEngine`` devuelve el primero
    con métricas ``NaN`` y sin avisar de que no hubo comparación posible. Esta
    prueba **no aprueba** esa conducta: la fija para que un cambio futuro sea
    deliberado y visible. La deuda está registrada en ``CLAUDE.md``.

    El caso dejó de ser hipotético con el espacio canónico: un dataset con
    columnas categóricas y un espacio sin codificación lo produce.
    """
    datos = pd.DataFrame(
        {"num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "cat": [*"aabbab"]}
    )
    etiquetas = np.array([0, 0, 1, 1, 0, 1])
    motor = GridSearchEngine(
        {"scaling": [ScaleNumeric("standard")]},
        val_size=0.34,
        random_state=42,
        task="classification",
    )

    mejor, metricas, resultados = motor.run(
        datos, etiquetas, KNeighborsClassifier(n_neighbors=1, algorithm="brute")
    )

    assert all(np.isnan(registro["objective"]) for registro in resultados)
    assert np.isnan(metricas["macro_f1"])
    assert mejor.get_params() == resultados[0]["params"]


# --------------------------------------------------------------------------- #
# El tercer consumidor: ejecución paralela
# --------------------------------------------------------------------------- #
def test_the_tabular_pipeline_survives_the_worker_processes() -> None:
    """Un pipeline tabular con selección debe viajar por ``loky`` sin cambiar.

    Se ejecuta con ``n_jobs=2`` a propósito: con un solo proceso nada se
    serializaría y la prueba no verificaría nada.
    """
    rng = np.random.default_rng(0)
    n = 60
    clase = np.tile([0, 1, 2], n // 3)
    datos = pd.DataFrame(
        {
            "señal": clase * 3.0 + rng.normal(0, 0.4, size=n),
            "ruido": rng.normal(0, 1.0, size=n),
            "grande": rng.normal(0, 1.0, size=n) * 500,
        }
    )
    pipelines = [
        Pipeline([ImputeMissing("mean"), ScaleNumeric("standard")]),
        Pipeline(
            [
                ImputeMissing("mean"),
                ScaleNumeric("standard"),
                SelectFeatures(0.5, "f_classif"),
            ]
        ),
    ]
    runner = ParallelRunner(
        n_jobs=2, val_size=0.4, random_state=42, task="classification"
    )

    reporte = runner.benchmark(
        pipelines, datos, clase, KNeighborsClassifier(n_neighbors=3, algorithm="brute")
    )

    assert reporte["consistent"] is True
    assert len(reporte["results"]) == 2
    for registro in reporte["results"]:
        assert not np.isnan(registro["objective"])
