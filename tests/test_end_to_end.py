# tests/test_end_to_end.py
"""Prueba de integración del flujo de alto nivel de AutoPipe."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from sklearn.ensemble import IsolationForest

from autopipe import GaussianBlur, GridSearchEngine, ReportGenerator, Resize


@pytest.fixture
def dataset() -> tuple[np.ndarray, np.ndarray]:
    """Dataset donde las anomalías son notablemente más brillantes."""
    rng = np.random.default_rng(0)
    normal = rng.integers(0, 20, size=(15, 16, 16), dtype=np.uint8)
    anomaly = rng.integers(200, 256, size=(15, 16, 16), dtype=np.uint8)
    images = np.concatenate([normal, anomaly], axis=0)
    labels = np.array([0] * 15 + [1] * 15)
    return images, labels


def test_high_level_api_in_under_20_lines(dataset, tmp_path: Path) -> None:
    """El usuario obtiene pipeline ganador, métricas y reporte sin tocar internos."""
    images, labels = dataset

    # --- API de alto nivel (flujo completo del usuario) ------------------- #
    search_space = {
        "resize": [Resize(width=8, height=8)],
        "blur": [GaussianBlur(kernel_size=3), GaussianBlur(kernel_size=5)],
    }
    engine = GridSearchEngine(search_space, val_size=0.4, random_state=42)
    best_pipeline, metrics, results = engine.run(
        images, labels, IsolationForest(random_state=0), metric="auc_roc"
    )
    report = ReportGenerator(output_dir=tmp_path).generate(
        results, best_pipeline, metrics, metric="auc_roc", pdf=False
    )
    # --------------------------------------------------------------------- #

    # (1 resize + omitir) x (2 blur + omitir) = 6 combinaciones evaluadas.
    assert len(results) == 6
    assert set(metrics) == {"auc_roc", "f1", "precision", "recall", "false_positive_rate"}
    assert report["html"].exists()
    assert report["config"].exists()


def test_high_level_api_is_reproducible(dataset) -> None:
    # OpenCV en un solo hilo para eliminar su no-determinismo de reducción.
    previous_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        images, labels = dataset
        search_space = {"blur": [GaussianBlur(kernel_size=3), GaussianBlur(kernel_size=5)]}

        run_a = GridSearchEngine(search_space, val_size=0.4, random_state=42).run(
            images, labels, IsolationForest(random_state=0)
        )
        run_b = GridSearchEngine(search_space, val_size=0.4, random_state=42).run(
            images, labels, IsolationForest(random_state=0)
        )
    finally:
        cv2.setNumThreads(previous_threads)

    # Requisito de la especificación: misma semilla => mismo pipeline ganador.
    assert run_a[0].get_params() == run_b[0].get_params()
    assert [r["objective"] for r in run_a[2]] == [r["objective"] for r in run_b[2]]
