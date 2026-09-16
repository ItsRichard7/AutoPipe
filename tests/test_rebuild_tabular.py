# tests/test_rebuild_tabular.py
"""Pruebas de la reconstrucción de ganadores tabulares desde su configuración guardada.

``ReportGenerator.rebuild_pipeline`` solo conocía las transformaciones de imagen: un
ganador tabular se guardaba en el reporte y después no se podía reconstruir. Estas pruebas
recorren los 144 candidatos del espacio canónico ida y vuelta por JSON y YAML, comprueban
que el pipeline reconstruido se ajusta igual que el original, y vigilan que ninguna
transformación predefinida quede fuera del registro.

Van en un archivo propio para no tocar ``test_reporter.py``, que es de la librería base.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import autopipe.pipeline as modulo_pipeline
from autopipe import (
    BaseTransform,
    GridSearchEngine,
    NotFittedError,
    ReportGenerator,
    StatefulTransform,
    TabularDataLoader,
    TabularTransform,
    build_tabular_search_space,
)
from autopipe.reporter.report import _TRANSFORM_REGISTRY


def _tabla() -> tuple[pd.DataFrame, np.ndarray]:
    """Tabla con una columna numérica informativa, ruido, faltantes y una categórica."""
    rng = np.random.default_rng(0)
    clase = np.tile([0, 1, 2], 30)
    medida = clase * 2.0 + rng.normal(0, 0.5, size=90)
    medida[::11] = np.nan
    tabla = pd.DataFrame(
        {
            "medida": medida,
            "ruido": rng.normal(0, 1.0, size=90),
            "categoria": pd.Series(np.where(rng.random(90) < 0.8, clase, 2)).map(
                {0: "norte", 1: "sur", 2: "este"}
            ),
            "clase": clase,
        }
    )
    return TabularDataLoader(tabla).load()


def _firma(pipeline) -> list[tuple[str, dict]]:
    """Tipo y parámetros de cada paso, que es lo que define a un pipeline sin ajustar."""
    return [(type(paso).__name__, paso.get_params()) for paso in pipeline.steps]


@pytest.mark.parametrize("formato", ["json", "yaml"])
def test_every_canonical_tabular_pipeline_survives_the_round_trip(
    tmp_path: Path, formato: str
) -> None:
    """Los 144 candidatos del espacio canónico, guardados y reconstruidos."""
    generador = ReportGenerator(output_dir=tmp_path)
    candidatos = GridSearchEngine(
        build_tabular_search_space(), task="classification"
    ).build_pipelines()

    for original in candidatos:
        ruta = generador.save_pipeline_config(original, config_format=formato)
        reconstruido = ReportGenerator.rebuild_pipeline(
            ReportGenerator.load_pipeline_config(ruta) or {}
        )
        assert _firma(reconstruido) == _firma(original)


def test_a_rebuilt_tabular_pipeline_is_unfitted_and_fits_like_the_original(
    tmp_path: Path,
) -> None:
    features, labels = _tabla()
    original = build_tabular_search_space()
    pipeline = GridSearchEngine(original, task="classification").build_pipelines()[0]
    ruta = ReportGenerator(output_dir=tmp_path).save_pipeline_config(pipeline)
    reconstruido = ReportGenerator.rebuild_pipeline(ReportGenerator.load_pipeline_config(ruta))

    with pytest.raises(NotFittedError):
        reconstruido.transform(features)

    esperado = pipeline.fit(features, labels).transform(features)
    obtenido = reconstruido.fit(features, labels).transform(features)
    pd.testing.assert_frame_equal(obtenido, esperado)


def test_every_built_in_transform_is_registered() -> None:
    """Prueba por sabotaje del registro.

    Si la librería exporta una transformación concreta que no está registrada, un
    ganador que la use no se puede reconstruir, y nada más fallaría hasta que alguien lo
    intentara con un reporte real.
    """
    abstractas = {BaseTransform, StatefulTransform, TabularTransform}
    exportadas = {
        nombre
        for nombre, objeto in vars(modulo_pipeline).items()
        if inspect.isclass(objeto)
        and issubclass(objeto, BaseTransform)
        and objeto not in abstractas
        and not inspect.isabstract(objeto)
    }

    assert exportadas
    assert exportadas <= set(_TRANSFORM_REGISTRY)


def test_an_unknown_transform_is_still_refused() -> None:
    with pytest.raises(ValueError, match="no registrada"):
        ReportGenerator.rebuild_pipeline({"0_Inventada": {"x": 1}})
