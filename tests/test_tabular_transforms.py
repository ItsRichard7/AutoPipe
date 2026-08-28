# tests/test_tabular_transforms.py
"""Pruebas unitarias para autopipe.pipeline.tabular_transforms."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import KNeighborsClassifier

from autopipe.pipeline.base import NotFittedError, StatefulTransform
from autopipe.pipeline.pipeline import Pipeline
from autopipe.pipeline.tabular_transforms import (
    CATEGORIA_DESCONOCIDA,
    EncodeCategorical,
    ImputeMissing,
    ScaleNumeric,
    SelectFeatures,
)
from autopipe.search_engine.grid_search import evaluate_pipeline
from autopipe.tasks import CLASSIFICATION

# SelectFeatures entra con el criterio no supervisado: las pruebas de contrato
# común ajustan sin etiquetas, y el criterio supervisado las exige.
TRANSFORMACIONES = (
    ImputeMissing(),
    ScaleNumeric(),
    EncodeCategorical(),
    SelectFeatures(criterion="variance"),
)


@pytest.fixture
def mixta() -> pd.DataFrame:
    """Tabla con dos numéricas y una categórica."""
    return pd.DataFrame(
        {
            "num_a": [1.0, 2.0, 3.0, 4.0],
            "num_b": [10.0, 20.0, 30.0, 40.0],
            "cat": ["x", "y", "x", "y"],
        }
    )


# --------------------------------------------------------------------------- #
# Contrato común
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("transform", TRANSFORMACIONES, ids=lambda t: type(t).__name__)
def test_are_stateful_transforms(transform: StatefulTransform) -> None:
    assert isinstance(transform, StatefulTransform)
    assert transform.fitted is False


@pytest.mark.parametrize("transform", TRANSFORMACIONES, ids=lambda t: type(t).__name__)
def test_transform_before_fit_raises(transform: StatefulTransform, mixta) -> None:
    with pytest.raises(NotFittedError):
        transform.transform(mixta)


@pytest.mark.parametrize("transform", TRANSFORMACIONES, ids=lambda t: type(t).__name__)
def test_reject_input_that_is_not_a_table(transform: StatefulTransform) -> None:
    """El mensaje debe orientar hacia el cargador tabular."""
    with pytest.raises(TypeError, match="DataFrame"):
        transform.fit(np.array([[1.0, 2.0], [3.0, 4.0]]))


@pytest.mark.parametrize("transform", TRANSFORMACIONES, ids=lambda t: type(t).__name__)
def test_parameters_do_not_mention_columns(transform: StatefulTransform, mixta) -> None:
    """Ningún parámetro puede depender del dataset.

    Si una transformación recibiera nombres o posiciones de columnas, el espacio
    de búsqueda dejaría de ser transferible entre datasets y el arranque en
    caliente no podría inyectar configuraciones aprendidas en otro dataset.

    Un parámetro numérico solo es admisible si es **relativo**: una fracción en
    ``(0, 1]``. Un número absoluto de columnas no sería transferible, porque
    conservar diez columnas significa algo distinto en un dataset de doce que en
    uno de cien (pre-registro, Sección 3.1).
    """
    for nombre, valor in transform.get_params().items():
        assert not isinstance(valor, (list, tuple, dict, set)), (
            f"El parámetro {nombre!r} es una colección; podría enumerar columnas."
        )
        assert valor not in mixta.columns, (
            f"El parámetro {nombre!r} nombra una columna del dataset."
        )
        if isinstance(valor, (int, float)) and not isinstance(valor, bool):
            assert 0 < float(valor) <= 1, (
                f"El parámetro numérico {nombre!r} vale {valor!r}, fuera de (0, 1]: "
                f"sería un valor absoluto y no una fracción."
            )


@pytest.mark.parametrize("transform", TRANSFORMACIONES, ids=lambda t: type(t).__name__)
def test_missing_column_at_transform_time_raises(
    transform: StatefulTransform, mixta
) -> None:
    """Transformar una tabla incompleta debe fallar, no rellenar en silencio."""
    transform.fit(mixta)

    with pytest.raises(ValueError, match="faltan columnas"):
        transform.transform(mixta.drop(columns=["num_a"]))


@pytest.mark.parametrize("transform", TRANSFORMACIONES, ids=lambda t: type(t).__name__)
def test_fit_does_not_mutate_the_input(transform: StatefulTransform, mixta) -> None:
    original = mixta.copy()
    transform.fit(mixta).transform(mixta)

    pd.testing.assert_frame_equal(mixta, original)


# --------------------------------------------------------------------------- #
# ImputeMissing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("estrategia", "esperado"), [("mean", 2.0), ("median", 2.0)]
)
def test_numeric_imputation_uses_the_configured_statistic(
    estrategia: str, esperado: float
) -> None:
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    validacion = pd.DataFrame({"x": [np.nan]})

    resultado = ImputeMissing(estrategia).fit(train).transform(validacion)

    assert resultado["x"].iloc[0] == pytest.approx(esperado)


def test_categorical_imputation_uses_the_mode() -> None:
    train = pd.DataFrame({"c": ["a", "a", "b"]})
    validacion = pd.DataFrame({"c": [None]})

    resultado = ImputeMissing().fit(train).transform(validacion)

    assert resultado["c"].iloc[0] == "a"


def test_fill_value_comes_from_training_not_from_validation() -> None:
    """Es la garantía anti-fuga de esta transformación."""
    train = pd.DataFrame({"x": [0.0, 10.0]})          # media de entrenamiento = 5.0
    validacion = pd.DataFrame({"x": [1000.0, np.nan]})  # media propia = 1000.0

    resultado = ImputeMissing("mean").fit(train).transform(validacion)

    assert resultado["x"].iloc[1] == pytest.approx(5.0)


def test_fully_empty_column_falls_back_without_propagating_nulls() -> None:
    train = pd.DataFrame({"x": [np.nan, np.nan], "c": [None, None]})

    resultado = ImputeMissing().fit(train).transform(train)

    assert resultado["x"].tolist() == [0.0, 0.0]
    assert resultado["c"].tolist() == ["faltante", "faltante"]


def test_invalid_imputation_strategy_raises() -> None:
    with pytest.raises(ValueError, match="numeric_strategy"):
        ImputeMissing("moda")


# --------------------------------------------------------------------------- #
# ScaleNumeric
# --------------------------------------------------------------------------- #
def test_standard_scaling_centres_the_training_set() -> None:
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})

    resultado = ScaleNumeric("standard").fit_transform(train)

    assert resultado["x"].mean() == pytest.approx(0.0)
    assert resultado["x"].std(ddof=0) == pytest.approx(1.0)


def test_minmax_scaling_maps_training_to_the_unit_interval() -> None:
    train = pd.DataFrame({"x": [10.0, 20.0, 30.0]})

    resultado = ScaleNumeric("minmax").fit_transform(train)

    assert resultado["x"].tolist() == pytest.approx([0.0, 0.5, 1.0])


def test_robust_scaling_uses_median_and_interquartile_range() -> None:
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})

    resultado = ScaleNumeric("robust").fit_transform(train)

    assert resultado["x"].iloc[2] == pytest.approx(0.0)


def test_scaling_uses_training_statistics_on_validation() -> None:
    """Es la garantía anti-fuga de esta transformación."""
    train = pd.DataFrame({"x": [0.0, 10.0]})
    validacion = pd.DataFrame({"x": [100.0, 200.0]})

    escalador = ScaleNumeric("minmax").fit(train)
    resultado = escalador.transform(validacion)

    assert resultado["x"].tolist() == pytest.approx([10.0, 20.0])
    assert resultado["x"].min() != pytest.approx(0.0)


def test_constant_column_is_centred_and_not_divided_by_zero() -> None:
    train = pd.DataFrame({"x": [5.0, 5.0, 5.0]})

    resultado = ScaleNumeric("standard").fit_transform(train)

    assert resultado["x"].tolist() == [0.0, 0.0, 0.0]


def test_scaling_leaves_categorical_columns_untouched(mixta) -> None:
    resultado = ScaleNumeric().fit_transform(mixta)

    assert resultado["cat"].tolist() == mixta["cat"].tolist()


def test_invalid_scaling_method_raises() -> None:
    with pytest.raises(ValueError, match="method"):
        ScaleNumeric("zscore")


# --------------------------------------------------------------------------- #
# EncodeCategorical
# --------------------------------------------------------------------------- #
def test_onehot_creates_one_column_per_training_category(mixta) -> None:
    resultado = EncodeCategorical("onehot").fit_transform(mixta)

    assert list(resultado.columns) == ["num_a", "num_b", "cat_x", "cat_y"]
    assert resultado["cat_x"].tolist() == [1, 0, 1, 0]


def test_ordinal_replaces_the_column_with_integer_codes(mixta) -> None:
    resultado = EncodeCategorical("ordinal").fit_transform(mixta)

    assert list(resultado.columns) == ["num_a", "num_b", "cat"]
    assert resultado["cat"].tolist() == [0, 1, 0, 1]


def test_unseen_category_does_not_add_columns_in_onehot() -> None:
    """El número de columnas lo fija el ajuste; la validación no puede alterarlo.

    Si una categoría vista solo en validación creara una columna nueva, la matriz
    de características de validación no encajaría con la de entrenamiento.
    """
    train = pd.DataFrame({"c": ["a", "b"]})
    validacion = pd.DataFrame({"c": ["a", "z"]})

    codificador = EncodeCategorical("onehot").fit(train)
    resultado = codificador.transform(validacion)

    assert list(resultado.columns) == ["c_a", "c_b"]
    assert resultado.iloc[1].tolist() == [0, 0]


def test_unseen_category_gets_a_reserved_code_in_ordinal() -> None:
    train = pd.DataFrame({"c": ["a", "b"]})
    validacion = pd.DataFrame({"c": ["z"]})

    codificador = EncodeCategorical("ordinal").fit(train)

    assert codificador.transform(validacion)["c"].iloc[0] == CATEGORIA_DESCONOCIDA


def test_encoded_output_is_fully_numeric(mixta) -> None:
    resultado = EncodeCategorical("onehot").fit_transform(mixta)

    assert all(pd.api.types.is_numeric_dtype(resultado[col]) for col in resultado.columns)


def test_column_order_is_deterministic() -> None:
    """Dos ajustes sobre los mismos datos producen el mismo orden de columnas."""
    train = pd.DataFrame({"z": [1.0, 2.0], "c": ["b", "a"], "a": [3.0, 4.0]})

    primero = EncodeCategorical().fit_transform(train)
    segundo = EncodeCategorical().fit_transform(train)

    assert list(primero.columns) == list(segundo.columns) == ["z", "a", "c_a", "c_b"]


def test_invalid_encoding_method_raises() -> None:
    with pytest.raises(ValueError, match="method"):
        EncodeCategorical("target")


@pytest.mark.parametrize("metodo", ["onehot", "ordinal"])
def test_columns_of_dtype_category_are_encoded(metodo: str) -> None:
    """Es el dtype con el que OpenML entrega las columnas categóricas.

    En modo ordinal, mapear una columna categórica devuelve otra columna
    categórica, y rellenarla con el código de categoría desconocida fallaba con
    ``TypeError`` por no estar ese código entre sus categorías.
    """
    train = pd.DataFrame({"c": pd.Series(["a", "b", "a", "b"], dtype="category")})
    validacion = pd.DataFrame({"c": pd.Series(["a", "z", None, "b"], dtype="category")})

    codificador = EncodeCategorical(metodo).fit(train)
    resultado = codificador.transform(validacion)

    assert all(pd.api.types.is_numeric_dtype(resultado[col]) for col in resultado.columns)
    assert not resultado.isna().to_numpy().any()


def test_an_unseen_category_of_dtype_category_gets_the_unknown_code() -> None:
    train = pd.DataFrame({"c": pd.Series(["a", "b"], dtype="category")})
    validacion = pd.DataFrame({"c": pd.Series(["z"], dtype="category")})

    resultado = EncodeCategorical("ordinal").fit(train).transform(validacion)

    assert resultado["c"].iloc[0] == CATEGORIA_DESCONOCIDA


# --------------------------------------------------------------------------- #
# SelectFeatures
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fraccion", [0.0, -0.5, 1.5, 2])
def test_invalid_fraction_raises(fraccion: float) -> None:
    """Solo se admiten fracciones en (0, 1]: un absoluto no sería transferible."""
    with pytest.raises(ValueError, match="fraction"):
        SelectFeatures(fraccion)


def test_boolean_fraction_raises() -> None:
    """``True`` es un ``int`` en Python y colaría como fracción válida."""
    with pytest.raises(ValueError, match="fraction"):
        SelectFeatures(True)


def test_invalid_criterion_raises() -> None:
    with pytest.raises(ValueError, match="criterion"):
        SelectFeatures(0.5, "chi2")


@pytest.mark.parametrize(
    ("fraccion", "esperadas"), [(0.25, 1), (0.5, 2), (0.75, 3), (1.0, 4)]
)
def test_fraction_determines_how_many_columns_survive(
    fraccion: float, esperadas: int
) -> None:
    """El número conservado es ceil(fracción x columnas), con un mínimo de una."""
    train = pd.DataFrame({nombre: [1.0, 2.0, 3.0] for nombre in "abcd"})

    resultado = SelectFeatures(fraccion, "variance").fit_transform(train)

    assert resultado.shape[1] == esperadas


def test_a_tiny_fraction_still_keeps_one_column() -> None:
    train = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})

    resultado = SelectFeatures(0.01, "variance").fit_transform(train)

    assert resultado.shape[1] == 1


def test_variance_criterion_keeps_the_most_variable_columns() -> None:
    train = pd.DataFrame(
        {"casi_constante": [1.0, 1.0, 1.1], "dispersa": [0.0, 50.0, 100.0]}
    )

    resultado = SelectFeatures(0.5, "variance").fit_transform(train)

    assert list(resultado.columns) == ["dispersa"]


def test_f_classif_keeps_the_column_that_separates_the_classes() -> None:
    """El criterio supervisado debe preferir la columna informativa."""
    train = pd.DataFrame(
        {"ruido": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0], "señal": [0.0, 0.0, 0.0, 9.0, 9.0, 9.0]}
    )
    labels = np.array([0, 0, 0, 1, 1, 1])

    resultado = SelectFeatures(0.5, "f_classif").fit_transform(train, labels)

    assert list(resultado.columns) == ["señal"]


def test_f_classif_without_labels_raises() -> None:
    """El criterio supervisado no puede ajustarse a ciegas."""
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0]})

    with pytest.raises(ValueError, match="supervisado"):
        SelectFeatures(0.5, "f_classif").fit(train)


def test_f_classif_with_missing_values_points_at_imputation() -> None:
    """El mensaje debe decir qué paso falta, no filtrar el error de sklearn."""
    train = pd.DataFrame({"x": [1.0, np.nan, 3.0, 4.0], "y": [1.0, 2.0, 3.0, 4.0]})
    labels = np.array([0, 0, 1, 1])

    with pytest.raises(ValueError, match="imputación"):
        SelectFeatures(0.5, "f_classif").fit(train, labels)


def test_non_numeric_columns_pass_through_untouched(mixta) -> None:
    """La fracción se aplica solo a las numéricas; las demás nunca se descartan."""
    resultado = SelectFeatures(0.5, "variance").fit_transform(mixta)

    assert "cat" in resultado.columns
    assert list(resultado.columns) == ["num_b", "cat"]


def test_output_keeps_the_original_column_order() -> None:
    """Se conserva el orden de la tabla, no el del puntaje."""
    train = pd.DataFrame(
        {"primera": [0.0, 100.0], "segunda": [0.0, 1.0], "tercera": [0.0, 50.0]}
    )

    resultado = SelectFeatures(0.6, "variance").fit_transform(train)

    assert list(resultado.columns) == ["primera", "tercera"]


def test_ties_are_broken_by_column_position() -> None:
    """Con puntajes idénticos gana la columna más a la izquierda, siempre."""
    train = pd.DataFrame({"a": [0.0, 1.0], "b": [0.0, 1.0], "c": [0.0, 1.0]})

    resultado = SelectFeatures(0.33, "variance").fit_transform(train)

    assert list(resultado.columns) == ["a"]


def test_a_column_without_finite_values_is_the_worst_ranked() -> None:
    """Una columna sin varianza definida no puede ganarle a una informativa."""
    train = pd.DataFrame({"vacia": [np.nan, np.nan], "util": [0.0, 10.0]})

    resultado = SelectFeatures(0.5, "variance").fit_transform(train)

    assert list(resultado.columns) == ["util"]


def test_constant_column_is_the_worst_ranked_under_f_classif() -> None:
    """Con f_classif una columna constante puntúa NaN; debe quedar de última."""
    train = pd.DataFrame(
        {"constante": [7.0, 7.0, 7.0, 7.0], "señal": [0.0, 0.0, 9.0, 9.0]}
    )
    labels = np.array([0, 0, 1, 1])

    resultado = SelectFeatures(0.5, "f_classif").fit_transform(train, labels)

    assert list(resultado.columns) == ["señal"]


def test_selection_is_reproducible() -> None:
    """Criterio de reproducibilidad: dos ajustes iguales eligen lo mismo."""
    rng = np.random.default_rng(7)
    train = pd.DataFrame(rng.normal(size=(30, 8)), columns=[f"c{i}" for i in range(8)])
    labels = np.tile([0, 1], 15)

    primero = SelectFeatures(0.5, "f_classif").fit(train, labels)
    segundo = SelectFeatures(0.5, "f_classif").fit(train, labels)

    assert primero.selected_columns_ == segundo.selected_columns_


def test_table_with_no_numeric_columns_keeps_everything() -> None:
    train = pd.DataFrame({"c1": ["a", "b"], "c2": ["x", "y"]})

    resultado = SelectFeatures(0.5, "variance").fit_transform(train)

    assert list(resultado.columns) == ["c1", "c2"]


def test_selection_comes_from_training_not_from_validation() -> None:
    """Es la garantía anti-fuga de esta transformación.

    En el entrenamiento la columna informativa es ``a``; en la validación es
    ``b``. Si la selección se ajustara con el conjunto completo, o con la
    validación, conservaría ``b``.
    """
    train = pd.DataFrame({"a": [0.0, 0.0, 9.0, 9.0], "b": [1.0, 1.0, 1.0, 1.1]})
    validacion = pd.DataFrame({"a": [1.0, 1.0, 1.0, 1.1], "b": [0.0, 0.0, 9.0, 9.0]})
    etiquetas = np.array([0, 0, 1, 1])

    selector = SelectFeatures(0.5, "f_classif").fit(train, etiquetas)

    assert selector.selected_columns_ == ["a"]
    assert list(selector.transform(validacion).columns) == ["a"]


# --------------------------------------------------------------------------- #
# Contrato por muestra
# --------------------------------------------------------------------------- #
def test_apply_on_a_single_sample_matches_the_batch(mixta) -> None:
    codificador = EncodeCategorical("onehot").fit(mixta)

    por_lote = codificador.transform(mixta).to_numpy()
    por_muestra = codificador.apply(mixta.iloc[0].to_numpy())

    np.testing.assert_allclose(por_muestra.astype(float), por_lote[0].astype(float))


# --------------------------------------------------------------------------- #
# Encadenamiento y protocolo de evaluación
# --------------------------------------------------------------------------- #
def test_pipeline_chains_imputation_encoding_and_scaling() -> None:
    """El ajuste secuencial hace que cada paso vea la salida del anterior."""
    train = pd.DataFrame(
        {"num": [1.0, np.nan, 3.0, 5.0], "cat": ["a", "b", None, "a"]}
    )
    pipeline = Pipeline([ImputeMissing("mean"), EncodeCategorical("onehot"), ScaleNumeric()])

    resultado = pipeline.fit_transform(train)

    assert list(resultado.columns) == ["num", "cat_a", "cat_b"]
    assert not resultado.isna().to_numpy().any()
    assert resultado["num"].mean() == pytest.approx(0.0)


def test_tabular_pipeline_runs_through_the_evaluation_protocol() -> None:
    """Un pipeline tabular completo atraviesa el protocolo con kNN y macro-F1."""
    rng = np.random.default_rng(0)
    n = 60
    datos = pd.DataFrame(
        {
            "num": rng.normal(size=n),
            "escala_grande": rng.normal(size=n) * 1000,
            "cat": rng.choice(["a", "b", "c"], size=n),
        }
    )
    datos.loc[:5, "num"] = np.nan
    labels = np.tile([0, 1, 2], n // 3)

    pipeline = Pipeline(
        [ImputeMissing("median"), EncodeCategorical("onehot"), ScaleNumeric("standard")]
    )
    valor, metrics = evaluate_pipeline(
        pipeline,
        datos,
        labels,
        KNeighborsClassifier(n_neighbors=5, algorithm="brute"),
        "macro_f1",
        np.arange(0, 45),
        np.arange(45, n),
        CLASSIFICATION.build_metric_calculator(0.5),
        task="classification",
    )

    assert set(metrics) == set(CLASSIFICATION.valid_metrics)
    assert 0.0 <= valor <= 1.0


def test_evaluation_does_not_fit_the_transforms_on_validation() -> None:
    """Prueba anti-fuga tabular a nivel de protocolo.

    Entrenamiento y validación tienen escalas muy distintas. El escalador se
    ajusta con el entrenamiento, así que la validación transformada **no** puede
    quedar en el intervalo unitario.
    """
    datos = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 1000.0, 1001.0, 1002.0, 1003.0]})
    train_idx, val_idx = np.arange(0, 4), np.arange(4, 8)

    escalador = ScaleNumeric("minmax")
    Pipeline([escalador]).fit(datos.iloc[train_idx])
    validacion = escalador.transform(datos.iloc[val_idx])

    assert validacion["x"].min() > 1.0
