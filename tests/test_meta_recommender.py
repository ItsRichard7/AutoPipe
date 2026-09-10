# tests/test_meta_recommender.py
"""Pruebas del recomendador de arranque en caliente por vecinos más cercanos.

Las pruebas que más pesan son las anti-fuga del nivel meta. El pre-registro exige
que la normalización y la métrica de distancia se ajusten **únicamente con los
datasets de entrenamiento de cada pliegue**, y una violación de esa regla no
produce ningún error: produce una ventaja inflada para el arranque en caliente
que resulta indetectable a posteriori. Por eso están escritas para **fallar si
alguien ajusta con el corpus completo**, comparando los parámetros aprendidos
contra los calculados a mano sobre los datasets del pliegue.

Es el mismo patrón que las anti-fuga del núcleo, que verifican que el pipeline se
ajuste solo con ``train_idx``.
"""

from __future__ import annotations

import numpy as np
import pytest

from autopipe.meta_knowledge import MetaKnowledgeBase, MetaKnowledgeEntry
from autopipe.meta_recommender import ESTRATEGIAS, MetaRecommender
from autopipe.pipeline.base import NotFittedError
from autopipe.search_space import TABULAR_SPACE_VERSION

# Cuatro configuraciones distintas y válidas en tabular-v1.
CONFIG_A = {"imputation": 0, "scaling": 0, "encoding": 0, "selection": 0}
CONFIG_B = {"imputation": 1, "scaling": 1, "encoding": 1, "selection": 1}
CONFIG_C = {"imputation": 0, "scaling": 2, "encoding": 0, "selection": 2}
CONFIG_D = {"imputation": 1, "scaling": 0, "encoding": 1, "selection": 3}


def entrada(
    dataset_id: str,
    meta_features: dict[str, float],
    configuration: dict[str, int],
    space_version: str = TABULAR_SPACE_VERSION,
) -> MetaKnowledgeEntry:
    """Arma una entrada mínima de la base.

    Args:
        dataset_id: Identificador del dataset.
        meta_features: Vector de meta-características.
        configuration: Configuración ganadora.
        space_version: Versión del espacio de búsqueda.

    Returns:
        La entrada construida.
    """
    return MetaKnowledgeEntry(
        dataset_id=dataset_id,
        dataset_name=f"d{dataset_id}",
        meta_features=meta_features,
        configuration=configuration,
        pipeline_params={},
        metric="macro_f1",
        metric_value=0.5,
        seed=42,
        n_trials=50,
        n_failed_trials=0,
        space_version=space_version,
    )


@pytest.fixture
def corpus() -> MetaKnowledgeBase:
    """Corpus sintético de cinco datasets alineados sobre un eje.

    El dataset ``4`` es un valor extremo en ``b``: es lo que hace visible la
    diferencia entre ajustar por pliegue y ajustar con el corpus completo.
    """
    filas = [
        ("0", {"a": 0.0, "b": 0.0}, CONFIG_A),
        ("1", {"a": 1.0, "b": 1.0}, CONFIG_B),
        ("2", {"a": 2.0, "b": 2.0}, CONFIG_C),
        ("3", {"a": 3.0, "b": 3.0}, CONFIG_D),
        ("4", {"a": 4.0, "b": 1000.0}, CONFIG_A),
    ]
    return MetaKnowledgeBase([entrada(i, mf, c) for i, mf, c in filas])


# --------------------------------------------------------------------------- #
# Construcción
# --------------------------------------------------------------------------- #
def test_the_number_of_configurations_must_be_positive() -> None:
    with pytest.raises(ValueError, match="al menos una configuración"):
        MetaRecommender(n_configurations=0)


def test_an_unsupported_scaling_is_rejected() -> None:
    with pytest.raises(ValueError, match="no soportado"):
        MetaRecommender(scaling="minmax")


def test_an_unsupported_metric_is_rejected() -> None:
    with pytest.raises(ValueError, match="no soportada"):
        MetaRecommender(metric="cosine")


def test_the_defaults_are_the_pre_registered_ones() -> None:
    """Cinco configuraciones, escalado robusto y L1: lo primario del protocolo."""
    recomendador = MetaRecommender()

    assert recomendador.n_configurations == 5
    assert recomendador.scaling == "robust"
    assert recomendador.metric == "manhattan"


# --------------------------------------------------------------------------- #
# Ajuste
# --------------------------------------------------------------------------- #
def test_fitting_an_empty_base_is_rejected() -> None:
    with pytest.raises(ValueError, match="base vacía"):
        MetaRecommender().fit(MetaKnowledgeBase([]))


def test_fit_returns_the_recommender(corpus) -> None:
    recomendador = MetaRecommender()

    assert recomendador.fit(corpus) is recomendador


def test_a_base_from_another_space_version_is_rejected() -> None:
    """Los índices de otra versión designarían otras transformaciones."""
    base = MetaKnowledgeBase(
        [entrada("0", {"a": 1.0}, CONFIG_A, space_version="tabular-v0")]
    )

    with pytest.raises(ValueError, match="tabular-v1"):
        MetaRecommender().fit(base)


def test_a_base_without_surviving_features_is_rejected() -> None:
    """Sin ninguna meta-característica finita no hay espacio en el que medir."""
    base = MetaKnowledgeBase(
        [
            entrada("0", {"a": float("nan")}, CONFIG_A),
            entrada("1", {"a": 1.0}, CONFIG_B),
        ]
    )

    with pytest.raises(ValueError, match="Ninguna meta-característica es finita"):
        MetaRecommender().fit(base)


def test_a_base_whose_features_do_not_vary_is_rejected() -> None:
    """Con todas las distancias en cero el orden de vecinos sería arbitrario."""
    base = MetaKnowledgeBase(
        [entrada(str(i), {"a": 7.0, "b": 3.0}, CONFIG_A) for i in range(3)]
    )

    with pytest.raises(ValueError, match="Ninguna meta-característica varía"):
        MetaRecommender().fit(base)


def test_a_feature_without_variation_is_dropped(corpus) -> None:
    """Una columna constante en el pliegue no entra en la distancia.

    Dejarla con escala uno la metería en bruto: si su magnitud es grande, esa
    sola columna dominaría la distancia sin aportar nada para ordenar vecinos.
    """
    base = MetaKnowledgeBase(
        [
            entrada("0", {"varia": 0.0, "constante": 5000.0}, CONFIG_A),
            entrada("1", {"varia": 1.0, "constante": 5000.0}, CONFIG_B),
            entrada("2", {"varia": 2.0, "constante": 5000.0}, CONFIG_C),
        ]
    )

    recomendador = MetaRecommender().fit(base)

    assert recomendador.feature_names_ == ["varia"]


# --------------------------------------------------------------------------- #
# Ausencia de fuga en el nivel meta
#
# Escritas para fallar si alguien ajusta la normalización con el corpus completo.
# --------------------------------------------------------------------------- #
def test_the_scaling_is_fitted_only_with_the_fold(corpus) -> None:
    """El centro y la escala salen de los cuatro del pliegue, no de los cinco.

    Se comparan contra los estadísticos calculados a mano sobre las entradas que
    quedan. Si alguien ajustara con el corpus completo, el dataset ``4`` movería
    ambos y esta prueba caería.
    """
    pliegue = corpus.without("4")
    nombres = pliegue.surviving_features()
    matriz, _ = pliegue.feature_matrix(nombres)
    centro_esperado = np.median(matriz, axis=0)
    cuartiles = np.percentile(matriz, [25, 75], axis=0)
    escala_esperada = cuartiles[1] - cuartiles[0]

    recomendador = MetaRecommender().fit(pliegue)

    assert recomendador.feature_names_ == nombres
    assert np.allclose(recomendador.center_, centro_esperado)
    assert np.allclose(recomendador.scale_, escala_esperada)


def test_the_held_out_dataset_changes_the_fitted_scaling(corpus) -> None:
    """Incluir o excluir el dataset retenido da parámetros distintos.

    Sin esta comprobación, la anterior pasaría igual si el pliegue y el corpus
    completo dieran por casualidad los mismos números, y no probaría nada.
    """
    del_pliegue = MetaRecommender(scaling="standard").fit(corpus.without("4"))
    del_corpus = MetaRecommender(scaling="standard").fit(corpus)

    assert del_pliegue.feature_names_ == del_corpus.feature_names_
    assert not np.allclose(del_pliegue.scale_, del_corpus.scale_)


def test_fitting_with_the_full_corpus_changes_the_recommendation(corpus) -> None:
    """La diferencia no es cosmética: cambia qué se inyecta.

    Con el valor extremo dentro, la escala de ``b`` se dispara y esa dimensión
    deja de pesar en la distancia, de modo que el vecino más cercano cambia. Es
    la razón de que la fuga en el nivel meta importe: no rompe nada, mueve el
    resultado.
    """
    consulta = {"a": 3.6, "b": 900.0}

    del_pliegue = MetaRecommender(scaling="standard", n_configurations=1)
    del_pliegue.fit(corpus.without("4")).recommend(consulta)
    del_corpus = MetaRecommender(scaling="standard", n_configurations=1)
    del_corpus.fit(corpus).recommend(consulta)

    assert del_pliegue.neighbours_[0]["dataset_id"] != del_corpus.neighbours_[0]["dataset_id"]


def test_no_fold_ever_sees_its_own_dataset(corpus) -> None:
    """Recorre el leave-one-dataset-out entero comprobando la exclusión.

    Es la garantía del protocolo expresada de la forma más directa posible: en
    ningún pliegue el dataset excluido puede aparecer entre los vecinos que
    aportan una recomendación.
    """
    for excluido in corpus.dataset_ids:
        vector = corpus.entry_for(excluido).meta_features
        recomendador = MetaRecommender(n_configurations=3).fit(corpus.without(excluido))

        recomendador.recommend(vector)

        vistos = [vecino["dataset_id"] for vecino in recomendador.neighbours_]
        assert excluido not in vistos


def test_recommending_does_not_alter_the_fitted_state(corpus) -> None:
    """Consultar no reajusta nada: el vector consultado no entra en la escala."""
    recomendador = MetaRecommender().fit(corpus.without("4"))
    centro, escala = recomendador.center_.copy(), recomendador.scale_.copy()

    recomendador.recommend({"a": 500.0, "b": 500.0})

    assert np.allclose(recomendador.center_, centro)
    assert np.allclose(recomendador.scale_, escala)


def test_the_reported_distances_come_only_from_the_fitted_scaling(corpus) -> None:
    """Las distancias se reproducen a mano con el centro y la escala del ajuste.

    Es la prueba que de verdad clava la garantía. Comprobar que los atributos no
    cambian tras consultar **no basta**: una implementación que recalculara la
    escala en una variable local, incluyendo el vector consultado, dejaría los
    atributos intactos y contaminaría igualmente cada distancia. Verificado
    saboteando el módulo de esa forma exacta: esta prueba cae y la de los
    atributos no.

    La consulta va dentro del rango del pliegue pero descentrada, para que el
    recorte no la absorba y la comprobación mida de verdad el escalado.
    """
    pliegue = corpus.without("4")
    recomendador = MetaRecommender(n_configurations=4).fit(pliegue)
    consulta = {"a": 2.4, "b": 0.3}

    recomendador.recommend(consulta)

    matriz, nombres = pliegue.feature_matrix(recomendador.used_features_)
    centro, escala = recomendador.center_, recomendador.scale_
    bruto = np.array([consulta[n] for n in nombres])
    vector = (np.clip(bruto, recomendador.lower_, recomendador.upper_) - centro) / escala
    esperadas = np.abs((matriz - centro) / escala - vector).sum(axis=1)

    identificadores = pliegue.dataset_ids
    for vecino in recomendador.neighbours_:
        posicion = identificadores.index(vecino["dataset_id"])
        assert vecino["distance"] == pytest.approx(esperadas[posicion])


# --------------------------------------------------------------------------- #
# Recomendación
# --------------------------------------------------------------------------- #
def test_recommending_before_fitting_is_rejected() -> None:
    with pytest.raises(NotFittedError, match="no está ajustado"):
        MetaRecommender().recommend({"a": 1.0})


def test_the_closest_dataset_comes_first(corpus) -> None:
    """El primer inyectado es el ganador del vecino más cercano."""
    recomendador = MetaRecommender(n_configurations=2).fit(corpus.without("4"))

    configuraciones = recomendador.recommend({"a": 0.9, "b": 0.9})

    assert configuraciones[0] == CONFIG_B
    assert recomendador.neighbours_[0]["dataset_id"] == "1"


def test_the_neighbours_come_out_in_increasing_distance(corpus) -> None:
    recomendador = MetaRecommender(n_configurations=4).fit(corpus.without("4"))

    recomendador.recommend({"a": 0.0, "b": 0.0})

    distancias = [vecino["distance"] for vecino in recomendador.neighbours_]
    assert distancias == sorted(distancias)


def test_repeated_configurations_are_skipped(corpus) -> None:
    """Dos vecinos con el mismo ganador aportan una sola recomendación.

    Inyectar la misma configuración dos veces gastaría dos de los treinta
    ensayos del presupuesto para obtener el mismo valor.
    """
    base = MetaKnowledgeBase(
        [
            entrada("0", {"a": 0.0, "b": 0.0}, CONFIG_A),
            entrada("1", {"a": 1.0, "b": 1.0}, CONFIG_A),
            entrada("2", {"a": 2.0, "b": 2.0}, CONFIG_B),
        ]
    )
    recomendador = MetaRecommender(n_configurations=2).fit(base)

    configuraciones = recomendador.recommend({"a": 0.0, "b": 0.0})

    assert configuraciones == [CONFIG_A, CONFIG_B]
    assert [v["dataset_id"] for v in recomendador.neighbours_] == ["0", "2"]


def test_it_walks_outwards_until_it_gathers_enough(corpus) -> None:
    """Pide cinco distintas y las reúne recorriendo más vecinos que cinco."""
    base = MetaKnowledgeBase(
        [
            entrada("0", {"a": 0.0, "b": 0.0}, CONFIG_A),
            entrada("1", {"a": 1.0, "b": 1.0}, CONFIG_A),
            entrada("2", {"a": 2.0, "b": 2.0}, CONFIG_A),
            entrada("3", {"a": 3.0, "b": 3.0}, CONFIG_B),
            entrada("4", {"a": 4.0, "b": 4.0}, CONFIG_C),
        ]
    )
    recomendador = MetaRecommender(n_configurations=3).fit(base)

    configuraciones = recomendador.recommend({"a": 0.0, "b": 0.0})

    assert configuraciones == [CONFIG_A, CONFIG_B, CONFIG_C]
    assert len(recomendador.neighbours_) == 3


def test_fewer_distinct_configurations_than_requested_are_returned(corpus, caplog) -> None:
    """Si la base no tiene cinco distintas, se devuelven las que hay y se avisa."""
    base = MetaKnowledgeBase(
        [entrada(str(i), {"a": float(i), "b": float(i)}, CONFIG_A) for i in range(3)]
    )
    recomendador = MetaRecommender(n_configurations=5).fit(base)

    with caplog.at_level("WARNING"):
        configuraciones = recomendador.recommend({"a": 0.0, "b": 0.0})

    assert configuraciones == [CONFIG_A]
    assert "se devuelven las que hay" in caplog.text


def test_recommendations_are_reproducible(corpus) -> None:
    """Criterio de reproducibilidad del anteproyecto, Sección 4.3.5."""
    consulta = {"a": 1.7, "b": 1.7}
    primero = MetaRecommender().fit(corpus.without("4")).recommend(consulta)
    segundo = MetaRecommender().fit(corpus.without("4")).recommend(consulta)

    assert primero == segundo


def test_ties_between_neighbours_are_broken_deterministically() -> None:
    """Dos vecinos equidistantes se ordenan por su posición en la base.

    Sin un criterio fijo, dos ejecuciones con la misma semilla podrían inyectar
    configuraciones distintas.
    """
    base = MetaKnowledgeBase(
        [
            entrada("0", {"a": -1.0, "b": 0.0}, CONFIG_A),
            entrada("1", {"a": 1.0, "b": 0.0}, CONFIG_B),
            entrada("2", {"a": 5.0, "b": 1.0}, CONFIG_C),
        ]
    )
    recomendador = MetaRecommender(n_configurations=1).fit(base)

    configuraciones = recomendador.recommend({"a": 0.0, "b": 0.0})

    assert configuraciones == [CONFIG_A]


# --------------------------------------------------------------------------- #
# Recorte de la consulta al rango del pliegue
#
# El escalado robusto controla los valores extremos dentro del conjunto de
# ajuste, pero no los de la consulta, y en el leave-one-dataset-out el dataset
# excluido puede caer muy fuera del corpus. Sin recorte, medido sobre el corpus
# real, una sola meta-característica llega a aportar el 71% de la distancia.
# --------------------------------------------------------------------------- #
def test_a_query_outside_the_fold_range_is_clipped(corpus) -> None:
    """Un valor por encima del máximo del pliegue se trata como ese máximo."""
    recomendador = MetaRecommender(n_configurations=1).fit(corpus.without("4"))

    recomendador.recommend({"a": 10_000.0, "b": 3.0})

    assert recomendador.clipped_features_ == ["a"]


def test_a_query_inside_the_fold_range_is_untouched(corpus) -> None:
    recomendador = MetaRecommender(n_configurations=1).fit(corpus.without("4"))

    recomendador.recommend({"a": 1.5, "b": 2.5})

    assert recomendador.clipped_features_ == []


def test_clipping_bounds_what_one_feature_can_contribute(corpus) -> None:
    """Alejar aún más un valor ya fuera de rango no cambia nada.

    Es la propiedad que se buscaba: el recomendador no puede medir *cuánto* fuera
    del corpus cae un dataset, solo que cae fuera, así que una característica
    desbocada deja de poder dominar la distancia por su magnitud.
    """
    lejos = MetaRecommender(n_configurations=3).fit(corpus.without("4"))
    lejos.recommend({"a": 1_000.0, "b": 1.0})
    lejisimos = MetaRecommender(n_configurations=3).fit(corpus.without("4"))
    lejisimos.recommend({"a": 1_000_000.0, "b": 1.0})

    assert [v["distance"] for v in lejos.neighbours_] == [
        v["distance"] for v in lejisimos.neighbours_
    ]
    assert [v["dataset_id"] for v in lejos.neighbours_] == [
        v["dataset_id"] for v in lejisimos.neighbours_
    ]


def test_the_clipping_range_comes_only_from_the_fold(corpus) -> None:
    """El rango de recorte lo fija el pliegue, no el corpus completo.

    Si se tomara del corpus, el dataset excluido ampliaría el rango con su propio
    valor extremo y dejaría de recortarse a sí mismo: la fuga entraría por aquí.
    """
    pliegue = corpus.without("4")
    matriz, nombres = pliegue.feature_matrix(pliegue.surviving_features())

    recomendador = MetaRecommender().fit(pliegue)

    assert recomendador.feature_names_ == nombres
    assert np.allclose(recomendador.lower_, matriz.min(axis=0))
    assert np.allclose(recomendador.upper_, matriz.max(axis=0))
    assert recomendador.upper_[nombres.index("b")] == pytest.approx(3.0)


# --------------------------------------------------------------------------- #
# Intersección de meta-características con la consulta
# --------------------------------------------------------------------------- #
def test_a_non_finite_feature_in_the_query_is_dropped(corpus) -> None:
    """Una columna que el pliegue conserva y la consulta no tiene finita se cae.

    El pre-registro descarta las no finitas en lugar de imputarlas; la
    intersección extiende esa regla al vector consultado. Meterla propagaría un
    ``NaN`` a todas las distancias por igual.
    """
    recomendador = MetaRecommender(n_configurations=1).fit(corpus.without("4"))

    recomendador.recommend({"a": 0.9, "b": float("nan")})

    assert recomendador.used_features_ == ["a"]


def test_a_feature_missing_from_the_query_is_dropped(corpus) -> None:
    recomendador = MetaRecommender(n_configurations=1).fit(corpus.without("4"))

    recomendador.recommend({"a": 0.9})

    assert recomendador.used_features_ == ["a"]


def test_a_query_sharing_no_finite_feature_is_rejected(corpus) -> None:
    """Sin columnas comunes no hay espacio en el que medir la distancia."""
    recomendador = MetaRecommender().fit(corpus.without("4"))

    with pytest.raises(ValueError, match="no comparte ninguna meta-característica"):
        recomendador.recommend({"a": float("nan"), "b": float("inf")})


def test_extra_features_in_the_query_are_ignored(corpus) -> None:
    """La consulta puede traer meta-características que la base no tiene."""
    recomendador = MetaRecommender(n_configurations=1).fit(corpus.without("4"))

    configuraciones = recomendador.recommend({"a": 0.9, "b": 0.9, "sobra": 3.0})

    assert configuraciones == [CONFIG_B]
    assert "sobra" not in recomendador.used_features_


# --------------------------------------------------------------------------- #
# Validación de lo que se inyecta
# --------------------------------------------------------------------------- #
def test_a_configuration_out_of_range_is_rejected() -> None:
    """No se devuelve una configuración que ``enqueue_trial`` no podría ejecutar.

    El spike A midió que un índice fuera de rango **aborta el estudio completo**,
    así que validar antes de devolver no es opcional.
    """
    rota = {"imputation": 99, "scaling": 0, "encoding": 0, "selection": 0}
    base = MetaKnowledgeBase(
        [
            entrada("0", {"a": 0.0, "b": 0.0}, rota),
            entrada("1", {"a": 1.0, "b": 1.0}, CONFIG_B),
        ]
    )
    recomendador = MetaRecommender(n_configurations=1).fit(base)

    with pytest.raises(ValueError, match="abortaría la búsqueda entera"):
        recomendador.recommend({"a": 0.0, "b": 0.0})


def test_an_incomplete_configuration_is_rejected() -> None:
    """Una configuración a la que le falta un paso tampoco se inyecta.

    El spike A midió que una clave desconocida se ignora **en silencio**, de modo
    que Optuna no avisaría: se aplicaría una recomendación parcial que nadie
    pidió.
    """
    incompleta = {"imputation": 0, "scaling": 0}
    base = MetaKnowledgeBase(
        [
            entrada("0", {"a": 0.0, "b": 0.0}, incompleta),
            entrada("1", {"a": 1.0, "b": 1.0}, CONFIG_B),
        ]
    )
    recomendador = MetaRecommender(n_configurations=1).fit(base)

    with pytest.raises(ValueError, match="no se puede resolver"):
        recomendador.recommend({"a": 0.0, "b": 0.0})


# --------------------------------------------------------------------------- #
# Alternativas declaradas para el análisis de sensibilidad
# --------------------------------------------------------------------------- #
def test_standard_scaling_uses_the_mean_and_the_deviation(corpus) -> None:
    pliegue = corpus.without("4")
    matriz, _ = pliegue.feature_matrix(pliegue.surviving_features())

    recomendador = MetaRecommender(scaling="standard").fit(pliegue)

    assert np.allclose(recomendador.center_, matriz.mean(axis=0))
    assert np.allclose(recomendador.scale_, matriz.std(axis=0))


def test_the_euclidean_metric_is_available(corpus) -> None:
    """Alternativa declarada de antemano, para no elegirla al ver resultados."""
    recomendador = MetaRecommender(metric="euclidean", n_configurations=1)

    configuraciones = recomendador.fit(corpus.without("4")).recommend({"a": 0.9, "b": 0.9})

    assert configuraciones == [CONFIG_B]


def test_the_two_metrics_can_disagree() -> None:
    """L1 y L2 no son intercambiables, que es por qué la elección se pre-registra.

    Un vecino se separa un poco en tres ejes y otro se separa bastante en uno
    solo. La L1 suma las diferencias y prefiere al segundo; la L2 las eleva al
    cuadrado, castiga la diferencia grande y prefiere al primero. Las dos
    ordenan los vecinos al revés, de modo que **la elección de distancia cambia
    qué se inyecta** y no puede tomarse después de ver los resultados.
    """
    base = MetaKnowledgeBase(
        [
            entrada("repartido", {"a": 1.0, "b": 1.0, "c": 1.0, "d": 0.0}, CONFIG_A),
            entrada("concentrado", {"a": 0.0, "b": 0.0, "c": 0.0, "d": 3.0}, CONFIG_B),
            entrada("lejos", dict.fromkeys("abcd", 8.0), CONFIG_C),
            entrada("lejos_opuesto", dict.fromkeys("abcd", -8.0), CONFIG_C),
        ]
    )
    consulta = dict.fromkeys("abcd", 0.0)

    manhattan = MetaRecommender(metric="manhattan", n_configurations=1)
    euclidean = MetaRecommender(metric="euclidean", n_configurations=1)

    assert manhattan.fit(base).recommend(consulta) == [CONFIG_B]
    assert euclidean.fit(base).recommend(consulta) == [CONFIG_A]


# --------------------------------------------------------------------------- #
# Estrategias de recomendación
#
# Las cuatro devuelven ganadoras de datasets reales de la base y saltan las
# repetidas, de modo que lo único que las distingue es de qué datasets toman las
# configuraciones. Es lo que permite compararlas sin que la comparación se
# contamine con la calidad intrínseca de lo inyectado.
# --------------------------------------------------------------------------- #
def test_an_unsupported_strategy_is_rejected() -> None:
    with pytest.raises(ValueError, match="no soportada"):
        MetaRecommender(strategy="oracle")


def test_the_default_strategy_is_the_primary_one() -> None:
    assert MetaRecommender().strategy == "nearest"


@pytest.fixture
def base_con_frecuencias() -> MetaKnowledgeBase:
    """Base donde CONFIG_A gana tres veces, CONFIG_B dos y CONFIG_C una.

    El orden por cercanía al origen es el inverso al orden por frecuencia, de
    modo que las dos estrategias no pueden coincidir por casualidad.
    """
    filas = [
        ("0", 0.0, CONFIG_C),
        ("1", 1.0, CONFIG_B),
        ("2", 2.0, CONFIG_B),
        ("3", 3.0, CONFIG_A),
        ("4", 4.0, CONFIG_A),
        ("5", 5.0, CONFIG_A),
    ]
    return MetaKnowledgeBase([entrada(i, {"a": v, "b": v * 2}, c) for i, v, c in filas])


def test_every_strategy_returns_winners_from_the_base(base_con_frecuencias) -> None:
    """Ninguna inventa configuraciones: todas salen de un dataset real."""
    ganadoras = [e.configuration for e in base_con_frecuencias]
    for estrategia in ESTRATEGIAS:
        recomendador = MetaRecommender(
            n_configurations=2, strategy=estrategia, random_state=7
        ).fit(base_con_frecuencias)

        for configuracion in recomendador.recommend({"a": 0.0, "b": 0.0}):
            assert configuracion in ganadoras


def test_every_strategy_returns_distinct_configurations(base_con_frecuencias) -> None:
    """Inyectar repetidas gastaría presupuesto para obtener el mismo valor."""
    for estrategia in ESTRATEGIAS:
        recomendador = MetaRecommender(
            n_configurations=3, strategy=estrategia, random_state=7
        ).fit(base_con_frecuencias)

        configuraciones = recomendador.recommend({"a": 0.0, "b": 0.0})

        assert len(configuraciones) == 3
        for posicion, una in enumerate(configuraciones):
            assert una not in configuraciones[:posicion]


def test_the_frequent_strategy_ignores_the_meta_features(base_con_frecuencias) -> None:
    """Devuelve las que más ganan, esté donde esté el dataset consultado.

    Es el control que aísla la contribución del meta-aprendizaje: si la
    estrategia primaria no le gana a esto, las meta-características no aportan
    nada y bastaría recomendar siempre lo mismo.
    """
    recomendador = MetaRecommender(n_configurations=2, strategy="frequent")
    recomendador.fit(base_con_frecuencias)

    cerca = recomendador.recommend({"a": 0.0, "b": 0.0})
    lejos = recomendador.recommend({"a": 99.0, "b": 99.0})

    assert cerca == [CONFIG_A, CONFIG_B]
    assert cerca == lejos


def test_the_nearest_strategy_does_depend_on_the_meta_features(
    base_con_frecuencias,
) -> None:
    """La primaria sí cambia con la consulta: es lo que la frecuente no hace."""
    recomendador = MetaRecommender(n_configurations=1, strategy="nearest")
    recomendador.fit(base_con_frecuencias)

    assert recomendador.recommend({"a": 0.0, "b": 0.0}) == [CONFIG_C]
    assert recomendador.recommend({"a": 5.0, "b": 10.0}) == [CONFIG_A]


def test_the_random_strategy_is_reproducible(base_con_frecuencias) -> None:
    """Criterio de reproducibilidad del anteproyecto, Sección 4.3.5."""
    consulta = {"a": 0.0, "b": 0.0}
    primero = MetaRecommender(n_configurations=2, strategy="random", random_state=3)
    segundo = MetaRecommender(n_configurations=2, strategy="random", random_state=3)

    esperado = primero.fit(base_con_frecuencias).recommend(consulta)
    assert segundo.fit(base_con_frecuencias).recommend(consulta) == esperado


def test_the_random_strategy_ignores_distance(base_con_frecuencias) -> None:
    """Con distintas semillas no siempre elige al más cercano.

    Es toda su razón de ser como ablación: si acabara eligiendo siempre a los
    vecinos, no aislaría la contribución de la similitud.
    """
    consulta = {"a": 0.0, "b": 0.0}
    elegidos = {
        tuple(
            sorted(
                MetaRecommender(
                    n_configurations=1, strategy="random", random_state=semilla
                )
                .fit(base_con_frecuencias)
                .recommend(consulta)[0]
                .items()
            )
        )
        for semilla in range(12)
    }

    assert len(elegidos) > 1


def test_the_diverse_strategy_spreads_the_recommendations() -> None:
    """Prefiere separarse aunque haya candidatas más cercanas y parecidas.

    Fija el comportamiento de la estrategia, no su justificación: la explicación
    del spike A por la que se diseñó quedó refutada en el experimento 2. Se
    construyen dos vecinos próximos con configuraciones casi
    idénticas y uno más lejano que difiere en todo; la primaria toma los dos más
    cercanos y la diversa cambia el segundo por el lejano.
    """
    casi_igual = {**CONFIG_A, "selection": 1}
    filas = [("cerca", 0.0, CONFIG_A), ("casi", 1.0, casi_igual), ("lejos", 4.0, CONFIG_B)]
    base = MetaKnowledgeBase([entrada(i, {"a": v, "b": v}, c) for i, v, c in filas])
    consulta = {"a": 0.0, "b": 0.0}

    primaria = MetaRecommender(n_configurations=2, strategy="nearest").fit(base)
    diversa = MetaRecommender(n_configurations=2, strategy="diverse").fit(base)

    assert primaria.recommend(consulta) == [CONFIG_A, casi_igual]
    assert diversa.recommend(consulta) == [CONFIG_A, CONFIG_B]


def test_the_diverse_strategy_still_starts_from_the_closest() -> None:
    """La primera inyectada sigue siendo la del vecino más cercano."""
    filas = [("cerca", 0.0, CONFIG_A), ("media", 2.0, CONFIG_B), ("lejos", 9.0, CONFIG_C)]
    base = MetaKnowledgeBase([entrada(i, {"a": v, "b": v}, c) for i, v, c in filas])

    diversa = MetaRecommender(n_configurations=2, strategy="diverse").fit(base)

    assert diversa.recommend({"a": 0.0, "b": 0.0})[0] == CONFIG_A
    assert diversa.neighbours_[0]["dataset_name"] == "dcerca"


def test_the_diverse_strategy_limits_how_far_it_looks() -> None:
    """Solo considera el doble de candidatas de las que va a elegir.

    Sin ese tope, la diversidad se compraría yendo a buscar datasets
    arbitrariamente lejanos, que es justo lo que un recomendador por similitud no
    debe hacer.
    """
    lejanisimo = {"imputation": 2, "scaling": 3, "encoding": 2, "selection": 3}
    filas = [
        ("c0", 0.0, CONFIG_A),
        ("c1", 1.0, {**CONFIG_A, "selection": 1}),
        ("c2", 2.0, {**CONFIG_A, "selection": 2}),
        ("c3", 3.0, {**CONFIG_A, "scaling": 1}),
        ("c4", 50.0, lejanisimo),
    ]
    base = MetaKnowledgeBase([entrada(i, {"a": v, "b": v}, c) for i, v, c in filas])

    diversa = MetaRecommender(n_configurations=2, strategy="diverse").fit(base)
    configuraciones = diversa.recommend({"a": 0.0, "b": 0.0})

    # Con n_configurations=2 la reserva son las cuatro candidatas más cercanas, de
    # modo que la quinta queda fuera pese a ser, con diferencia, la más distinta
    # de todas. Sin el tope la diversidad la habría elegido.
    assert len(configuraciones) == 2
    assert configuraciones[0] == CONFIG_A
    assert lejanisimo not in configuraciones


def test_every_strategy_validates_before_returning() -> None:
    """Ninguna deja pasar una configuración que abortaría el estudio."""
    rota = {"imputation": 99, "scaling": 0, "encoding": 0, "selection": 0}
    base = MetaKnowledgeBase(
        [entrada(str(i), {"a": float(i), "b": float(i)}, rota) for i in range(3)]
    )

    for estrategia in ESTRATEGIAS:
        recomendador = MetaRecommender(
            n_configurations=1, strategy=estrategia, random_state=1
        ).fit(base)

        with pytest.raises(ValueError, match="abortaría la búsqueda entera"):
            recomendador.recommend({"a": 0.0, "b": 0.0})


def test_every_strategy_records_its_neighbours(base_con_frecuencias) -> None:
    """La recomendación es auditable venga de donde venga."""
    for estrategia in ESTRATEGIAS:
        recomendador = MetaRecommender(
            n_configurations=2, strategy=estrategia, random_state=5
        ).fit(base_con_frecuencias)

        configuraciones = recomendador.recommend({"a": 0.0, "b": 0.0})

        assert len(recomendador.neighbours_) == len(configuraciones)
        for vecino, configuracion in zip(recomendador.neighbours_, configuraciones):
            assert vecino["configuration"] == configuracion
            assert vecino["dataset_id"] in base_con_frecuencias.dataset_ids


# --------------------------------------------------------------------------- #
# La estrategia híbrida (pre-registro del experimento 2)
# --------------------------------------------------------------------------- #
def _corpus_hibrido() -> MetaKnowledgeBase:
    """Corpus donde la ganadora frecuente y la cercana no coinciden.

    ``CONFIG_A`` gana en tres datasets lejanos, de modo que es la más frecuente.
    Los dos datasets más cercanos al origen traen ``CONFIG_B`` y ``CONFIG_C``. Sin
    esa separación, la prueba no distinguiría una estrategia de la otra.
    """
    filas = [
        ("0", {"a": 0.0, "b": 0.0}, CONFIG_B),
        ("1", {"a": 1.0, "b": 1.0}, CONFIG_C),
        ("2", {"a": 8.0, "b": 8.0}, CONFIG_A),
        ("3", {"a": 9.0, "b": 9.0}, CONFIG_A),
        ("4", {"a": 10.0, "b": 10.0}, CONFIG_A),
        ("5", {"a": 11.0, "b": 11.0}, CONFIG_D),
    ]
    return MetaKnowledgeBase([entrada(i, mf, c) for i, mf, c in filas])


def test_the_hybrid_strategy_is_supported() -> None:
    assert "hybrid" in ESTRATEGIAS
    assert MetaRecommender(strategy="hybrid").strategy == "hybrid"


def test_the_hybrid_starts_with_the_most_frequent_winner() -> None:
    """Las dos primeras salen de `frequent`, que es lo que el pre-registro fija."""
    base = _corpus_hibrido()
    consulta = {"a": 0.0, "b": 0.0}

    frecuentes = MetaRecommender(strategy="frequent").fit(base).recommend(consulta)
    hibridas = MetaRecommender(strategy="hybrid").fit(base).recommend(consulta)

    assert hibridas[:2] == frecuentes[:2]


def test_the_hybrid_continues_with_the_nearest_neighbours() -> None:
    base = _corpus_hibrido()
    consulta = {"a": 0.0, "b": 0.0}

    cercanas = MetaRecommender(strategy="nearest").fit(base).recommend(consulta)
    hibridas = MetaRecommender(strategy="hybrid").fit(base).recommend(consulta)

    resto = [c for c in hibridas[2:]]
    assert resto == [c for c in cercanas if c not in hibridas[:2]][: len(resto)]


def test_the_hybrid_differs_from_both_of_its_parents() -> None:
    """Si coincidiera con una de las dos, no aportaría una condición nueva."""
    base = _corpus_hibrido()
    consulta = {"a": 0.0, "b": 0.0}

    hibridas = MetaRecommender(strategy="hybrid").fit(base).recommend(consulta)
    cercanas = MetaRecommender(strategy="nearest").fit(base).recommend(consulta)
    frecuentes = MetaRecommender(strategy="frequent").fit(base).recommend(consulta)

    assert hibridas != cercanas
    assert hibridas != frecuentes


def test_the_hybrid_never_repeats_a_configuration() -> None:
    """Inyectar duplicados sería pérdida pura de presupuesto."""
    base = _corpus_hibrido()

    hibridas = MetaRecommender(strategy="hybrid").fit(base).recommend({"a": 0.0, "b": 0.0})

    assert len(hibridas) == len({tuple(sorted(c.items())) for c in hibridas})


def test_the_hybrid_is_deterministic() -> None:
    base = _corpus_hibrido()
    consulta = {"a": 0.5, "b": 0.5}

    una = MetaRecommender(strategy="hybrid").fit(base).recommend(consulta)
    otra = MetaRecommender(strategy="hybrid").fit(base).recommend(consulta)

    assert una == otra


def test_the_hybrid_split_is_the_one_the_pre_registration_fixes() -> None:
    """El reparto 2 y 3 se declara y no se explora."""
    from autopipe.meta_recommender.recommender import N_HIBRIDO_FRECUENTES

    assert N_HIBRIDO_FRECUENTES == 2
