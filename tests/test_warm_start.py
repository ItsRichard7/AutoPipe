# tests/test_warm_start.py
"""Pruebas del arranque en caliente del motor de Optuna.

Cada comprobación de validación existe por una medición del spike A
(``docs/experimentos/SPIKE-A-enqueue-trial.md``), no por precaución genérica.
``enqueue_trial`` falla de dos formas peligrosas y en direcciones opuestas: un
índice fuera de rango **aborta el estudio completo**, y una clave desconocida se
**acepta en silencio**, aplicando una recomendación parcial que nadie pidió. Las
pruebas de este archivo fijan que ninguna de las dos llegue a ocurrir.

Se fija además la propiedad de la que va a depender el experimento: los ensayos
inyectados se consumen **primero**, en orden, y **descuentan del presupuesto**.
El pre-registro cuenta los cinco del arranque en caliente dentro de «ensayos
hasta el objetivo», así que si esa propiedad cambiara, la variable medida
significaría otra cosa sin que nada fallara.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import KNeighborsClassifier

from autopipe.meta_knowledge import MetaKnowledgeBase, MetaKnowledgeEntry
from autopipe.meta_recommender import MetaRecommender
from autopipe.pipeline.pipeline import Pipeline
from autopipe.search_engine.optuna_search import OptunaSearchEngine, validate_warm_start
from autopipe.search_space import (
    TABULAR_SPACE_VERSION,
    build_tabular_search_space,
)
from tests.test_serch_engine import Identity, MeanThresholdModel, Zeroer

ESPACIO = {"step": [Identity(), Zeroer()]}
"""Un paso con dos opciones más la de omitirlo: índices válidos 0, 1 y 2."""


@pytest.fixture
def dataset() -> tuple[np.ndarray, np.ndarray]:
    """Dataset donde las anomalías son más brillantes que las normales."""
    rng = np.random.default_rng(0)
    normal = rng.integers(0, 20, size=(10, 8, 8), dtype=np.int64)
    anomaly = rng.integers(200, 256, size=(10, 8, 8), dtype=np.int64)
    return np.concatenate([normal, anomaly], axis=0), np.array([0] * 10 + [1] * 10)


# --------------------------------------------------------------------------- #
# Validación de lo que se inyecta
# --------------------------------------------------------------------------- #
def test_a_valid_configuration_is_normalised_to_python_ints() -> None:
    """Los índices salen como ``int``, vengan como vengan."""
    validadas = validate_warm_start([{"step": np.int64(1)}], ESPACIO, n_trials=5)

    assert validadas == [{"step": 1}]
    assert isinstance(validadas[0]["step"], int)


def test_the_skip_index_is_a_valid_warm_start_value() -> None:
    """Omitir el paso es una opción legítima del espacio, no un desbordamiento."""
    assert validate_warm_start([{"step": 2}], ESPACIO, n_trials=5) == [{"step": 2}]


def test_more_configurations_than_trials_is_rejected() -> None:
    """Las que sobran no se ejecutarían, y descartarlas en silencio sería peor."""
    configuraciones = [{"step": 0}, {"step": 1}, {"step": 2}]

    with pytest.raises(ValueError, match="no llegarían a ejecutarse"):
        validate_warm_start(configuraciones, ESPACIO, n_trials=2)


def test_a_missing_step_is_rejected() -> None:
    """Optuna muestrearía por su cuenta el paso que falta, sin avisar."""
    with pytest.raises(ValueError, match="faltan \\['step'\\]"):
        validate_warm_start([{}], ESPACIO, n_trials=5)


def test_an_unknown_key_is_rejected() -> None:
    """El spike A midió que Optuna la ignora en silencio."""
    with pytest.raises(ValueError, match="sobran \\['paso_z'\\]"):
        validate_warm_start([{"step": 0, "paso_z": 1}], ESPACIO, n_trials=5)


def test_an_index_above_the_range_is_rejected() -> None:
    """El spike A midió que aborta el estudio completo."""
    with pytest.raises(ValueError, match="abortaría el estudio completo"):
        validate_warm_start([{"step": 99}], ESPACIO, n_trials=5)


def test_a_negative_index_is_rejected() -> None:
    with pytest.raises(ValueError, match="fuera de rango"):
        validate_warm_start([{"step": -1}], ESPACIO, n_trials=5)


def test_a_non_integer_index_is_rejected() -> None:
    """Los parámetros que el muestreador registra por ensayo son enteros."""
    with pytest.raises(ValueError, match="no es un entero"):
        validate_warm_start([{"step": 1.0}], ESPACIO, n_trials=5)


def test_a_boolean_index_is_rejected() -> None:
    """``True`` valdría 1 en silencio, que es justo el fallo mudo a evitar."""
    with pytest.raises(ValueError, match="no es un entero"):
        validate_warm_start([{"step": True}], ESPACIO, n_trials=5)


def test_a_repeated_configuration_is_warned_about(caplog) -> None:
    """Inyectar dos veces la misma gasta dos ensayos para el mismo valor.

    Se avisa pero no se rechaza: el motor ejecuta lo que se le pide, y quién
    recomienda es quien decide si repetir tiene sentido.
    """
    with caplog.at_level("WARNING"):
        validadas = validate_warm_start([{"step": 0}, {"step": 0}], ESPACIO, n_trials=5)

    assert len(validadas) == 2
    assert "repite una ya inyectada" in caplog.text


def test_the_engine_validates_when_it_is_built(dataset) -> None:
    """Una configuración imposible falla antes de gastar cómputo."""
    with pytest.raises(ValueError, match="abortaría el estudio completo"):
        OptunaSearchEngine(ESPACIO, n_trials=4, warm_start=[{"step": 7}])


# --------------------------------------------------------------------------- #
# Comportamiento sin arranque en caliente: nada cambia
# --------------------------------------------------------------------------- #
def test_without_warm_start_the_engine_is_unchanged(dataset) -> None:
    """El parámetro es aditivo: omitirlo deja el motor exactamente como estaba."""
    images, labels = dataset
    sin_parametro = OptunaSearchEngine(ESPACIO, n_trials=6, val_size=0.5, random_state=7)
    con_none = OptunaSearchEngine(
        ESPACIO, n_trials=6, val_size=0.5, random_state=7, warm_start=None
    )

    _, _, primero = sin_parametro.run(images, labels, MeanThresholdModel())
    _, _, segundo = con_none.run(images, labels, MeanThresholdModel())

    assert sin_parametro.warm_start == []
    assert [r["objective"] for r in primero] == [r["objective"] for r in segundo]
    assert [r["params"] for r in primero] == [r["params"] for r in segundo]


def test_an_empty_warm_start_is_accepted(dataset) -> None:
    images, labels = dataset
    motor = OptunaSearchEngine(ESPACIO, n_trials=4, val_size=0.5, random_state=7, warm_start=[])

    _, _, resultados = motor.run(images, labels, MeanThresholdModel())

    assert motor.warm_start == []
    assert len(resultados) == 4


# --------------------------------------------------------------------------- #
# Las propiedades de las que depende el experimento
# --------------------------------------------------------------------------- #
def test_the_injected_trials_run_first_and_in_order(dataset) -> None:
    """Los ensayos inyectados se consumen primero, en el orden de inyección.

    El pre-registro cuenta los cinco del arranque en caliente dentro de «ensayos
    hasta el objetivo», y el arnés del experimento va a identificarlos por su
    posición. Si Optuna cambiara ese orden, la variable medida significaría otra
    cosa sin que nada fallara.
    """
    images, labels = dataset
    inyectadas = [{"step": 2}, {"step": 1}, {"step": 0}]
    motor = OptunaSearchEngine(
        ESPACIO, n_trials=8, val_size=0.5, random_state=3, warm_start=inyectadas
    )

    _, _, resultados = motor.run(images, labels, MeanThresholdModel())

    opciones = ESPACIO["step"]
    for posicion, configuracion in enumerate(inyectadas):
        indice = configuracion["step"]
        pasos = [] if indice == len(opciones) else [opciones[indice]]
        assert resultados[posicion]["params"] == Pipeline(pasos).get_params()


def test_the_injected_trials_count_against_the_budget(dataset) -> None:
    """Con tres inyectadas y ocho de presupuesto se ejecutan ocho, no once.

    Es la razón por la que el pre-registro las cuenta: no contarlas daría al
    arranque en caliente una ventaja puramente contable.
    """
    images, labels = dataset
    motor = OptunaSearchEngine(
        ESPACIO,
        n_trials=8,
        val_size=0.5,
        random_state=3,
        warm_start=[{"step": 0}, {"step": 1}, {"step": 2}],
    )

    _, _, resultados = motor.run(images, labels, MeanThresholdModel())

    assert len(resultados) == 8


def test_warm_starting_changes_the_search(dataset) -> None:
    """Inyectar altera la secuencia de ensayos: el efecto es real, no cosmético."""
    images, labels = dataset
    frio = OptunaSearchEngine(ESPACIO, n_trials=6, val_size=0.5, random_state=3)
    caliente = OptunaSearchEngine(
        ESPACIO, n_trials=6, val_size=0.5, random_state=3, warm_start=[{"step": 2}, {"step": 1}]
    )

    _, _, sin_inyectar = frio.run(images, labels, MeanThresholdModel())
    _, _, con_inyectar = caliente.run(images, labels, MeanThresholdModel())

    assert [r["params"] for r in sin_inyectar] != [r["params"] for r in con_inyectar]


def test_warm_starting_is_reproducible(dataset) -> None:
    """Criterio de reproducibilidad del anteproyecto, Sección 4.3.5."""
    images, labels = dataset
    inyectadas = [{"step": 1}, {"step": 2}]
    corridas = [
        OptunaSearchEngine(
            ESPACIO, n_trials=6, val_size=0.5, random_state=5, warm_start=inyectadas
        ).run(images, labels, MeanThresholdModel())[2]
        for _ in range(2)
    ]

    assert [r["params"] for r in corridas[0]] == [r["params"] for r in corridas[1]]
    assert [r["objective"] for r in corridas[0]] == [r["objective"] for r in corridas[1]]


def test_an_unevaluable_injected_configuration_does_not_abort_the_study() -> None:
    """Una inyectada que no se puede evaluar queda en ``NaN`` y la búsqueda sigue.

    Es distinto del índice fuera de rango, que sí aborta: aquí la configuración
    es legítima y simplemente no corre sobre este dataset. Pasa de verdad, porque
    el recomendador transfiere configuraciones halladas en otros datasets.
    """
    rng = np.random.default_rng(0)
    imagenes = np.empty(20, dtype=object)
    for i in range(20):
        lado = 8 if i < 10 else 16
        imagenes[i] = rng.integers(0, 255, size=(lado, lado), dtype=np.uint8)
    etiquetas = np.array([0] * 10 + [1] * 10)

    motor = OptunaSearchEngine(
        ESPACIO, n_trials=5, val_size=0.5, random_state=1, warm_start=[{"step": 2}]
    )

    _, _, resultados = motor.run(imagenes, etiquetas, MeanThresholdModel())

    assert len(resultados) == 5
    assert math.isnan(resultados[0]["objective"])


# --------------------------------------------------------------------------- #
# Integración: del recomendador al motor
# --------------------------------------------------------------------------- #
@pytest.fixture
def tabla() -> tuple[pd.DataFrame, np.ndarray]:
    """Tabla mixta con faltantes y tres clases separables."""
    rng = np.random.default_rng(0)
    n = 90
    clase = np.tile([0, 1, 2], n // 3)
    tabla = pd.DataFrame(
        {
            "señal": clase * 3.0 + rng.normal(0, 0.6, size=n),
            "ruido": rng.normal(0, 1.0, size=n),
            "grande": rng.normal(0, 1.0, size=n) * 400,
            "cat": np.where(clase == 0, "norte", np.where(clase == 1, "sur", "este")),
        }
    )
    tabla.loc[:4, "ruido"] = np.nan
    return tabla, clase


def _entrada(dataset_id: str, valor: float, configuracion: dict[str, int]) -> MetaKnowledgeEntry:
    """Arma una entrada de la base para la prueba de integración."""
    return MetaKnowledgeEntry(
        dataset_id=dataset_id,
        dataset_name=f"d{dataset_id}",
        meta_features={"nr_inst": valor, "nr_attr": valor * 2},
        configuration=configuracion,
        pipeline_params={},
        metric="macro_f1",
        metric_value=0.7,
        seed=42,
        n_trials=50,
        n_failed_trials=0,
        space_version=TABULAR_SPACE_VERSION,
    )


def test_the_recommender_output_feeds_the_engine_unchanged(tabla) -> None:
    """Lo que el recomendador devuelve entra en el motor sin adaptarlo.

    Es la costura entre RE5 y el motor de búsqueda, y la prueba que verifica que
    las dos piezas hablan el mismo idioma: índices por paso sobre el espacio
    canónico vigente. Si una de las dos cambiara de formato, esto caería.
    """
    datos, etiquetas = tabla
    base = MetaKnowledgeBase(
        [
            _entrada("0", 1.0, {"imputation": 0, "scaling": 0, "encoding": 0, "selection": 3}),
            _entrada("1", 2.0, {"imputation": 1, "scaling": 2, "encoding": 0, "selection": 1}),
            _entrada("2", 3.0, {"imputation": 0, "scaling": 1, "encoding": 1, "selection": 2}),
        ]
    )
    recomendador = MetaRecommender(n_configurations=3).fit(base)

    configuraciones = recomendador.recommend({"nr_inst": 1.2, "nr_attr": 2.4})

    motor = OptunaSearchEngine(
        build_tabular_search_space(),
        n_trials=8,
        val_size=0.3,
        random_state=101,
        task="classification",
        warm_start=configuraciones,
    )
    ganador, metricas, resultados = motor.run(
        datos, etiquetas, KNeighborsClassifier(n_neighbors=5, algorithm="brute")
    )

    assert motor.warm_start == configuraciones
    assert len(resultados) == 8
    assert not math.isnan(metricas["macro_f1"])
    assert ganador is not None


def test_a_recommendation_from_another_space_version_never_reaches_the_engine() -> None:
    """El rechazo ocurre en el recomendador, antes de llegar a ``enqueue_trial``.

    Las dos capas validan, y a propósito: el recomendador comprueba la versión
    del espacio y el motor comprueba los índices contra su propio espacio, que no
    tiene por qué ser el canónico tabular.
    """
    base = MetaKnowledgeBase(
        [
            MetaKnowledgeEntry(
                dataset_id="0",
                dataset_name="viejo",
                meta_features={"a": 1.0},
                configuration={"imputation": 0, "scaling": 0, "encoding": 0, "selection": 0},
                pipeline_params={},
                metric="macro_f1",
                metric_value=0.5,
                seed=42,
                n_trials=50,
                n_failed_trials=0,
                space_version="tabular-v0",
            )
        ]
    )

    with pytest.raises(ValueError, match="tabular-v1"):
        MetaRecommender().fit(base)
