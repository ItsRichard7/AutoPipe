# tests/test_evaluator.py
"""Pruebas unitarias para el módulo autopipe.evaluator."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pytest

from autopipe.evaluator.metrics import MetricCalculator
from autopipe.evaluator.splitter import DataSplitter


@pytest.fixture
def labels() -> np.ndarray:
    """20 etiquetas balanceadas (10 normales, 10 anómalas)."""
    return np.array([0] * 10 + [1] * 10)


# --------------------------------------------------------------------------- #
# DataSplitter
# --------------------------------------------------------------------------- #
def test_split_partitions_are_disjoint_and_complete(labels: np.ndarray) -> None:
    splitter = DataSplitter(val_size=0.25, random_state=42)
    train_idx, val_idx = splitter.split(labels)

    assert len(val_idx) == 5
    assert len(train_idx) == 15
    assert set(train_idx).isdisjoint(set(val_idx))
    assert sorted(np.concatenate([train_idx, val_idx])) == list(range(len(labels)))


def test_split_is_stratified(labels: np.ndarray) -> None:
    splitter = DataSplitter(val_size=0.5, random_state=0)
    _, val_idx = splitter.split(labels)
    val_labels = labels[val_idx]

    assert int((val_labels == 0).sum()) == 5
    assert int((val_labels == 1).sum()) == 5


def test_split_is_reproducible(labels: np.ndarray) -> None:
    a = DataSplitter(val_size=0.3, random_state=7).split(labels)
    b = DataSplitter(val_size=0.3, random_state=7).split(labels)

    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])


@pytest.mark.parametrize("invalid", [0.0, 1.0, -0.1, 1.5])
def test_invalid_val_size_raises(invalid: float) -> None:
    with pytest.raises(ValueError):
        DataSplitter(val_size=invalid)


def test_k_fold_split_covers_all_indices(labels: np.ndarray) -> None:
    splitter = DataSplitter(random_state=1)
    folds = splitter.k_fold_split(labels, n_splits=5)

    assert len(folds) == 5
    all_val = np.concatenate([val for _, val in folds])
    assert sorted(all_val) == list(range(len(labels)))
    for train_idx, val_idx in folds:
        assert set(train_idx).isdisjoint(set(val_idx))


def test_k_fold_is_stratified(labels: np.ndarray) -> None:
    splitter = DataSplitter(random_state=1)
    folds = splitter.k_fold_split(labels, n_splits=5)
    for _, val_idx in folds:
        assert int((labels[val_idx] == 0).sum()) == 2
        assert int((labels[val_idx] == 1).sum()) == 2


def test_k_fold_non_stratified_covers_all_indices(labels: np.ndarray) -> None:
    splitter = DataSplitter(random_state=3, stratify=False)
    folds = splitter.k_fold_split(labels, n_splits=4)

    assert len(folds) == 4
    all_val = np.concatenate([val for _, val in folds])
    assert sorted(all_val) == list(range(len(labels)))


def test_k_fold_invalid_n_splits_raises(labels: np.ndarray) -> None:
    with pytest.raises(ValueError):
        DataSplitter().k_fold_split(labels, n_splits=1)


def test_get_config_returns_params() -> None:
    config = DataSplitter(val_size=0.3, random_state=5, stratify=False).get_config()
    assert config == {"val_size": 0.3, "random_state": 5, "stratify": False}


@pytest.mark.parametrize("filename", ["config.json", "config.yaml", "config.yml"])
def test_save_and_load_config_roundtrip(
    labels: np.ndarray, tmp_path: Path, filename: str
) -> None:
    original = DataSplitter(val_size=0.25, random_state=42, stratify=True)
    config_path = tmp_path / filename
    original.save_config(config_path)

    loaded = DataSplitter.load_config(config_path)
    assert loaded.get_config() == original.get_config()
    assert np.array_equal(loaded.split(labels)[1], original.split(labels)[1])


def test_save_config_unsupported_extension_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        DataSplitter().save_config(tmp_path / "config.txt")


def test_load_config_unsupported_extension_raises(tmp_path: Path) -> None:
    bogus = tmp_path / "config.txt"
    bogus.write_text("nope", encoding="utf-8")
    with pytest.raises(ValueError):
        DataSplitter.load_config(bogus)


# --------------------------------------------------------------------------- #
# MetricCalculator
# --------------------------------------------------------------------------- #
def test_perfect_predictions() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    metrics = MetricCalculator().calculate(y_true, y_pred)

    assert metrics["auc_roc"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["false_positive_rate"] == 0.0


def test_known_confusion_matrix_values() -> None:
    # tn=2, fp=2, fn=0, tp=2  ->  fpr = 2/4 = 0.5, recall = 1.0
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 1, 1])
    metrics = MetricCalculator().calculate(y_true, y_pred)

    assert metrics["recall"] == 1.0
    assert metrics["false_positive_rate"] == 0.5
    assert metrics["precision"] == pytest.approx(0.5)


def test_returns_all_expected_keys() -> None:
    metrics = MetricCalculator().calculate(np.array([0, 1]), np.array([0, 1]))
    assert set(metrics) == {
        "auc_roc",
        "f1",
        "precision",
        "recall",
        "false_positive_rate",
    }


def test_continuous_scores_are_thresholded() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = MetricCalculator(threshold=0.5).calculate(y_true, y_pred)

    assert metrics["auc_roc"] == 1.0
    assert metrics["f1"] == 1.0


def test_auc_with_separate_scores() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = MetricCalculator().calculate(y_true, y_pred, y_score=y_score)

    assert metrics["auc_roc"] == 1.0


def test_single_class_auc_is_nan(caplog: pytest.LogCaptureFixture) -> None:
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([1, 1, 0, 1])
    with caplog.at_level(logging.WARNING, logger="autopipe.evaluator.metrics"):
        metrics = MetricCalculator().calculate(y_true, y_pred)

    assert math.isnan(metrics["auc_roc"])
    assert metrics["recall"] == pytest.approx(0.75)
    assert any("AUC-ROC" in record.message for record in caplog.records)
