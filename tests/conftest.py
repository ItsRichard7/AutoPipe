# tests/conftest.py
"""Fixtures compartidos de pytest: dataset sintético generado al vuelo.

Sustituye a los antiguos binarios versionados en tests/fixtures/: las imágenes
se generan una vez por sesión de pytest en un directorio temporal, con semilla
fija para garantizar reproducibilidad.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture(scope="session")
def fixture_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Genera un dataset sintético con subdirectorios ``normal/`` y ``anomaly/``.

    Args:
        tmp_path_factory: Fábrica de directorios temporales de pytest.

    Returns:
        Ruta raíz del dataset generado (10 imágenes normales, 5 anómalas,
        64x64 RGB ``uint8``).
    """
    root = tmp_path_factory.mktemp("dataset")
    rng = np.random.default_rng(0)
    for folder, count in (("normal", 10), ("anomaly", 5)):
        target = root / folder
        target.mkdir()
        for index in range(count):
            array = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
            Image.fromarray(array).save(target / f"img_{index:02d}.png")
    return root
