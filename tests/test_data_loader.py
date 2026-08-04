# tests/test_data_loader.py
"""Pruebas unitarias para autopipe.data_loader.loader.ImageDataLoader."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from autopipe.data_loader.loader import ImageDataLoader


def _count_images(directory: Path) -> int:
    """Cuenta los archivos con extensión soportada en un directorio."""
    return sum(
        1
        for path in directory.iterdir()
        if path.suffix.lower() in ImageDataLoader.SUPPORTED_EXTENSIONS
    )


def _make_dataset(root: Path, n_normal: int, n_anomaly: int) -> None:
    """Crea un dataset sintético de imágenes 8x8 RGB."""
    for subdir, count in (("normal", n_normal), ("anomaly", n_anomaly)):
        target = root / subdir
        target.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            array = np.zeros((8, 8, 3), dtype=np.uint8)
            Image.fromarray(array).save(target / f"img_{index:02d}.png")


def test_load_returns_arrays_and_labels(fixture_dataset: Path) -> None:
    loader = ImageDataLoader(fixture_dataset)
    images, labels = loader.load()

    expected = _count_images(fixture_dataset / "normal") + _count_images(
        fixture_dataset / "anomaly"
    )
    assert isinstance(images, np.ndarray)
    assert isinstance(labels, np.ndarray)
    assert len(images) == expected
    assert len(labels) == expected


def test_labels_are_binary_and_ordered(fixture_dataset: Path) -> None:
    loader = ImageDataLoader(fixture_dataset)
    _, labels = loader.load()

    n_normal = _count_images(fixture_dataset / "normal")
    n_anomaly = _count_images(fixture_dataset / "anomaly")

    assert set(np.unique(labels)).issubset({0, 1})
    assert np.all(labels[:n_normal] == 0)
    assert np.all(labels[n_normal:] == 1)
    assert int(labels.sum()) == n_anomaly


def test_accepts_str_path(tmp_path: Path) -> None:
    _make_dataset(tmp_path, n_normal=2, n_anomaly=1)
    loader = ImageDataLoader(str(tmp_path))
    images, labels = loader.load()

    assert len(images) == 3
    assert len(labels) == 3


def test_homogeneous_images_are_stacked(tmp_path: Path) -> None:
    _make_dataset(tmp_path, n_normal=3, n_anomaly=2)
    images, _ = ImageDataLoader(tmp_path).load()

    assert images.dtype != object
    assert images.shape == (5, 8, 8, 3)


def test_heterogeneous_images_return_object_array(tmp_path: Path) -> None:
    (tmp_path / "normal").mkdir(parents=True)
    (tmp_path / "anomaly").mkdir(parents=True)
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(
        tmp_path / "normal" / "a.png"
    )
    Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(
        tmp_path / "anomaly" / "b.png"
    )

    images, labels = ImageDataLoader(tmp_path).load()

    assert images.dtype == object
    assert len(images) == 2
    assert images[0].shape == (8, 8, 3)
    assert images[1].shape == (16, 16, 3)


def test_unsupported_files_are_ignored(tmp_path: Path) -> None:
    _make_dataset(tmp_path, n_normal=2, n_anomaly=1)
    (tmp_path / "normal" / "notes.txt").write_text("ignore me")

    images, _ = ImageDataLoader(tmp_path).load()

    assert len(images) == 3


def test_unsupported_files_emit_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    _make_dataset(tmp_path, n_normal=2, n_anomaly=1)
    (tmp_path / "normal" / "notes.txt").write_text("ignore me")

    with caplog.at_level(logging.WARNING, logger="autopipe.data_loader.loader"):
        ImageDataLoader(tmp_path).load()

    assert any("no soportada" in record.message for record in caplog.records)


def test_missing_root_raises(tmp_path: Path) -> None:
    loader = ImageDataLoader(tmp_path / "does_not_exist")
    with pytest.raises(FileNotFoundError):
        loader.load()


def test_missing_subdir_raises(tmp_path: Path) -> None:
    (tmp_path / "normal").mkdir()
    loader = ImageDataLoader(tmp_path)
    with pytest.raises(FileNotFoundError):
        loader.load()


def test_empty_dataset_raises(tmp_path: Path) -> None:
    (tmp_path / "normal").mkdir()
    (tmp_path / "anomaly").mkdir()
    loader = ImageDataLoader(tmp_path)
    with pytest.raises(ValueError):
        loader.load()


def test_corrupt_image_raises(tmp_path: Path) -> None:
    (tmp_path / "normal").mkdir()
    (tmp_path / "anomaly").mkdir()
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(
        tmp_path / "anomaly" / "ok.png"
    )
    (tmp_path / "normal" / "broken.png").write_bytes(b"not a real png")

    loader = ImageDataLoader(tmp_path)
    with pytest.raises(ValueError):
        loader.load()
