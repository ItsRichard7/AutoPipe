# tests/test_tabular_loader.py
"""Pruebas unitarias para autopipe.data_loader.tabular.TabularDataLoader."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from autopipe.data_loader.tabular import TabularDataLoader
from autopipe.pipeline.pipeline import Pipeline
from autopipe.search_engine.grid_search import evaluate_pipeline
from autopipe.tasks import CLASSIFICATION


@pytest.fixture
def frame() -> pd.DataFrame:
    """Tabla mixta: dos numéricas, una categórica y la etiqueta al final."""
    return pd.DataFrame(
        {
            "edad": [25, 40, 35, 50, 28, 46],
            "ingreso": [1.5, 3.2, 2.8, 4.1, 1.9, 3.7],
            "region": ["norte", "sur", "norte", "sur", "norte", "sur"],
            "clase": ["no", "si", "no", "si", "no", "si"],
        }
    )


@pytest.fixture
def csv_path(frame: pd.DataFrame, tmp_path: Path) -> Path:
    """Escribe la tabla en un CSV temporal."""
    destino = tmp_path / "dataset.csv"
    frame.to_csv(destino, index=False)
    return destino


# --------------------------------------------------------------------------- #
# Carga básica
# --------------------------------------------------------------------------- #
def test_loads_from_a_dataframe(frame: pd.DataFrame) -> None:
    features, labels = TabularDataLoader(frame).load()

    assert list(features.columns) == ["edad", "ingreso", "region"]
    assert features.shape == (6, 3)
    assert labels.tolist() == [0, 1, 0, 1, 0, 1]


def test_loads_from_a_csv(csv_path: Path, frame: pd.DataFrame) -> None:
    desde_csv, labels_csv = TabularDataLoader(csv_path).load()
    desde_frame, labels_frame = TabularDataLoader(frame).load()

    pd.testing.assert_frame_equal(desde_csv, desde_frame)
    np.testing.assert_array_equal(labels_csv, labels_frame)


def test_does_not_mutate_the_source_dataframe(frame: pd.DataFrame) -> None:
    original = frame.copy()
    TabularDataLoader(frame).load()

    pd.testing.assert_frame_equal(frame, original)


def test_column_types_are_reported(frame: pd.DataFrame) -> None:
    """La distinción numérica/categórica es lo que permite a las
    transformaciones detectar los tipos sin recibirlos como parámetro."""
    loader = TabularDataLoader(frame)
    loader.load()

    assert loader.numeric_columns_ == ["edad", "ingreso"]
    assert loader.categorical_columns_ == ["region"]
    assert loader.feature_names_ == ["edad", "ingreso", "region"]


def test_preserves_column_dtypes(frame: pd.DataFrame) -> None:
    features, _ = TabularDataLoader(frame).load()

    assert pd.api.types.is_numeric_dtype(features["edad"])
    assert not pd.api.types.is_numeric_dtype(features["region"])


# --------------------------------------------------------------------------- #
# Columna objetivo
# --------------------------------------------------------------------------- #
def test_target_can_be_named(frame: pd.DataFrame) -> None:
    features, labels = TabularDataLoader(frame, target_column="region").load()

    assert list(features.columns) == ["edad", "ingreso", "clase"]
    assert labels.tolist() == [0, 1, 0, 1, 0, 1]


def test_target_can_be_positional(frame: pd.DataFrame) -> None:
    features, _ = TabularDataLoader(frame, target_column=0).load()

    assert list(features.columns) == ["ingreso", "region", "clase"]


def test_unknown_target_name_raises(frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="no existe"):
        TabularDataLoader(frame, target_column="inexistente").load()


def test_out_of_range_target_position_raises(frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="fuera de rango"):
        TabularDataLoader(frame, target_column=99).load()


def test_missing_values_in_the_target_raise(frame: pd.DataFrame) -> None:
    frame.loc[0, "clase"] = None
    with pytest.raises(ValueError, match="valores faltantes"):
        TabularDataLoader(frame).load()


def test_single_class_target_raises(frame: pd.DataFrame) -> None:
    frame["clase"] = "si"
    with pytest.raises(ValueError, match="una sola clase"):
        TabularDataLoader(frame).load()


def test_dataset_without_features_raises() -> None:
    with pytest.raises(ValueError, match="ninguna columna de características"):
        TabularDataLoader(pd.DataFrame({"clase": [0, 1, 0]})).load()


# --------------------------------------------------------------------------- #
# Codificación de etiquetas
# --------------------------------------------------------------------------- #
def test_classes_are_exposed_in_encoding_order(frame: pd.DataFrame) -> None:
    loader = TabularDataLoader(frame)
    _, labels = loader.load()

    assert loader.classes_.tolist() == ["no", "si"]
    assert loader.classes_[labels].tolist() == frame["clase"].tolist()


def test_binary_integer_labels_are_left_untouched() -> None:
    """Se preserva la convención de detección de anomalías (0 normal, 1 anomalía)."""
    datos = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [0, 1, 0, 1]})
    loader = TabularDataLoader(datos)
    _, labels = loader.load()

    assert labels.tolist() == [0, 1, 0, 1]
    assert loader.classes_.tolist() == [0, 1]


def test_multiclass_labels_are_encoded_from_zero() -> None:
    datos = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": ["c", "a", "b"]})
    loader = TabularDataLoader(datos)
    _, labels = loader.load()

    assert loader.classes_.tolist() == ["a", "b", "c"]
    assert labels.tolist() == [2, 0, 1]


# --------------------------------------------------------------------------- #
# Lectura de archivos
# --------------------------------------------------------------------------- #
def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        TabularDataLoader(tmp_path / "no-existe.csv").load()


def test_unsupported_extension_raises(tmp_path: Path) -> None:
    ruta = tmp_path / "dataset.parquet"
    ruta.write_text("contenido", encoding="utf-8")
    with pytest.raises(ValueError, match="Extensión no soportada"):
        TabularDataLoader(ruta).load()


def test_empty_dataset_raises() -> None:
    with pytest.raises(ValueError, match="vacío"):
        TabularDataLoader(pd.DataFrame()).load()


def test_custom_na_values_are_honoured(tmp_path: Path) -> None:
    """El signo de interrogación es el marcador de faltante habitual en UCI."""
    ruta = tmp_path / "dataset.csv"
    ruta.write_text("x,y\n1,a\n?,b\n3,a\n", encoding="utf-8")

    features, _ = TabularDataLoader(ruta, na_values=["?"]).load()

    assert features["x"].isna().sum() == 1
    assert pd.api.types.is_numeric_dtype(features["x"])


def test_missing_values_in_features_are_warned(
    frame: pd.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    frame.loc[0, "ingreso"] = None
    with caplog.at_level(logging.WARNING, logger="autopipe.data_loader.tabular"):
        TabularDataLoader(frame).load()

    assert any("valores faltantes" in record.message for record in caplog.records)


# --------------------------------------------------------------------------- #
# Integración con el protocolo de evaluación
# --------------------------------------------------------------------------- #
def test_loaded_dataset_runs_through_the_evaluation_protocol() -> None:
    """Un dataset numérico cargado atraviesa el protocolo de punta a punta."""
    rng = np.random.default_rng(0)
    datos = pd.DataFrame(
        {
            "a": rng.normal(size=40),
            "b": rng.normal(size=40),
            "clase": np.tile([0, 1, 2, 0], 10),
        }
    )
    features, labels = TabularDataLoader(datos).load()

    valor, metrics = evaluate_pipeline(
        Pipeline([]),
        features,
        labels,
        LogisticRegression(max_iter=200),
        "macro_f1",
        np.arange(0, 30),
        np.arange(30, 40),
        CLASSIFICATION.build_metric_calculator(0.5),
        task="classification",
    )

    assert set(metrics) == set(CLASSIFICATION.valid_metrics)
    assert valor == metrics["macro_f1"]


def test_categorical_columns_without_encoding_fail_with_a_clear_message(
    frame: pd.DataFrame,
) -> None:
    """Sin un paso de codificación, el error debe decir qué falta.

    Es el mensaje que verá quien construya un pipeline tabular incompleto, así
    que tiene que señalar la causa y no un fallo genérico de conversión.
    """
    features, labels = TabularDataLoader(frame).load()

    with pytest.raises(ValueError, match="paso de codificación"):
        evaluate_pipeline(
            Pipeline([]),
            features,
            labels,
            LogisticRegression(max_iter=200),
            "macro_f1",
            np.array([0, 1, 2, 3]),
            np.array([4, 5]),
            CLASSIFICATION.build_metric_calculator(0.5),
            task="classification",
        )
