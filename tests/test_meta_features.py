# tests/test_meta_features.py
"""Pruebas del extractor de meta-características.

Tres de estas pruebas verifican condiciones de las que depende el recomendador y
no simples detalles de implementación:

* **Alineación**: el conjunto de nombres debe ser el mismo para cualquier
  dataset. Si no lo fuera, el recomendador compararía vectores de longitudes o
  significados distintos sin que nada fallara.
* **Determinismo**: dos extracciones del mismo dataset deben dar el mismo vector.
  El spike B midió que ``hubs.mean`` y ``hubs.sd`` difieren en torno a 3e-15
  entre corridas por orden de sumación; el redondeo lo absorbe.
* **Submuestra de tamaño fijo**: las medidas de complejidad cambian con el tamaño
  de la muestra, así que dos datasets solo son comparables si se extrajeron con
  el mismo tamaño.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from autopipe.meta_features import (
    MetaFeatureError,
    MetaFeatureExtractor,
    columnas_no_finitas,
)
from autopipe.meta_features import extractor as modulo


def tabla(
    n: int = 300,
    columnas: int = 8,
    clases: int = 2,
    faltantes: bool = False,
    categorica: bool = False,
    semilla: int = 0,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Genera un dataset tabular sintético reproducible."""
    rng = np.random.default_rng(semilla)
    datos = pd.DataFrame(
        rng.normal(size=(n, columnas)), columns=[f"c{i}" for i in range(columnas)]
    )
    if categorica:
        datos["cat"] = rng.choice(["a", "b", "c"], size=n)
    if faltantes:
        datos.iloc[rng.choice(n, n // 10, replace=False), 0] = np.nan
    etiquetas = np.tile(np.arange(clases), n // clases + 1)[:n]
    return datos, etiquetas


def iguales(uno: dict[str, float], otro: dict[str, float]) -> bool:
    """Compara dos vectores tratando ``NaN`` como igual a ``NaN``."""
    if set(uno) != set(otro):
        return False
    return all(
        (math.isnan(uno[k]) and math.isnan(otro[k])) or uno[k] == otro[k] for k in uno
    )


@pytest.fixture(scope="module")
def extractor() -> MetaFeatureExtractor:
    return MetaFeatureExtractor(random_state=42)


@pytest.fixture(scope="module")
def vector_base(extractor) -> dict[str, float]:
    datos, etiquetas = tabla()
    return extractor.extract(datos, etiquetas)


# --------------------------------------------------------------------------- #
# Alineación: la condición que el recomendador necesita
# --------------------------------------------------------------------------- #
def test_the_feature_names_are_the_same_across_very_different_datasets(
    extractor, vector_base
) -> None:
    """Sin esto, el recomendador compararía vectores desalineados en silencio."""
    otros = [
        tabla(n=600, columnas=15, clases=5, semilla=1),
        tabla(n=400, columnas=4, clases=3, categorica=True, semilla=2),
        tabla(n=250, columnas=10, clases=2, faltantes=True, semilla=3),
    ]

    for datos, etiquetas in otros:
        assert set(extractor.extract(datos, etiquetas)) == set(vector_base)


def test_the_vector_is_not_empty(vector_base) -> None:
    assert len(vector_base) > 50


def test_every_value_is_a_float(vector_base) -> None:
    assert all(isinstance(valor, float) for valor in vector_base.values())


# --------------------------------------------------------------------------- #
# Determinismo
# --------------------------------------------------------------------------- #
def test_two_extractions_give_the_same_vector(extractor) -> None:
    """Criterio de reproducibilidad: el vector debe ser estable bit a bit."""
    datos, etiquetas = tabla(n=400, columnas=10, clases=3, semilla=7)

    assert iguales(extractor.extract(datos, etiquetas), extractor.extract(datos, etiquetas))


def test_values_are_rounded_to_the_configured_precision() -> None:
    """El redondeo es lo que absorbe el ruido de coma flotante de pymfe."""
    datos, etiquetas = tabla(n=200, columnas=6, semilla=4)

    vector = MetaFeatureExtractor(random_state=42, decimals=3).extract(datos, etiquetas)

    finitos = [v for v in vector.values() if math.isfinite(v)]
    assert finitos
    assert all(v == round(v, 3) for v in finitos)


# --------------------------------------------------------------------------- #
# Submuestra de tamaño fijo y estratificada
# --------------------------------------------------------------------------- #
def test_a_dataset_smaller_than_the_sample_is_used_whole(extractor) -> None:
    datos, etiquetas = tabla(n=120, clases=2, semilla=5)

    muestra, etiquetas_muestra = extractor._submuestra(datos, etiquetas)

    assert len(muestra) == 120
    assert len(etiquetas_muestra) == 120


def test_a_larger_dataset_is_cut_to_the_fixed_size() -> None:
    """El tamaño es fijo, no una fracción: es lo que hace comparables dos datasets."""
    corto = MetaFeatureExtractor(random_state=42, sample_size=100)
    datos, etiquetas = tabla(n=500, clases=2, semilla=6)

    muestra, etiquetas_muestra = corto._submuestra(datos, etiquetas)

    assert len(muestra) == 100
    assert len(etiquetas_muestra) == 100


def test_the_subsample_preserves_the_class_proportions() -> None:
    corto = MetaFeatureExtractor(random_state=42, sample_size=100)
    datos = pd.DataFrame(np.random.default_rng(8).normal(size=(500, 5)))
    etiquetas = np.repeat([0, 1], [400, 100])            # 80 % / 20 %

    _, etiquetas_muestra = corto._submuestra(datos, etiquetas)

    assert (etiquetas_muestra == 0).sum() == 80
    assert (etiquetas_muestra == 1).sum() == 20


def test_the_subsample_is_reproducible() -> None:
    corto = MetaFeatureExtractor(random_state=42, sample_size=100)
    datos, etiquetas = tabla(n=500, clases=3, semilla=9)

    primera, _ = corto._submuestra(datos, etiquetas)
    segunda, _ = corto._submuestra(datos, etiquetas)

    assert list(primera.index) == list(segunda.index)


def test_a_different_seed_gives_a_different_subsample() -> None:
    datos, etiquetas = tabla(n=500, clases=3, semilla=10)

    una, _ = MetaFeatureExtractor(random_state=1, sample_size=100)._submuestra(datos, etiquetas)
    otra, _ = MetaFeatureExtractor(random_state=2, sample_size=100)._submuestra(datos, etiquetas)

    assert list(una.index) != list(otra.index)


def test_it_falls_back_to_simple_sampling_when_stratification_is_impossible(
    caplog,
) -> None:
    """Una clase con un solo miembro impide estratificar; describir vale más."""
    corto = MetaFeatureExtractor(random_state=42, sample_size=100)
    datos = pd.DataFrame(np.random.default_rng(11).normal(size=(300, 4)))
    etiquetas = np.array([0] * 299 + [1])

    with caplog.at_level("WARNING"):
        muestra, etiquetas_muestra = corto._submuestra(datos, etiquetas)

    assert len(muestra) == 100
    assert len(etiquetas_muestra) == 100
    assert "estratificar" in caplog.text


# --------------------------------------------------------------------------- #
# Valores faltantes: el fallo que documentó el spike B
# --------------------------------------------------------------------------- #
def test_a_dataset_with_missing_values_is_extracted(extractor, vector_base) -> None:
    """Sin la imputación previa, el grupo de complejidad falla con mensaje vacío."""
    datos, etiquetas = tabla(n=300, columnas=8, faltantes=True, semilla=12)

    vector = extractor.extract(datos, etiquetas)

    assert set(vector) == set(vector_base)


def test_a_dataset_with_categorical_columns_is_extracted(extractor, vector_base) -> None:
    datos, etiquetas = tabla(n=300, columnas=6, categorica=True, semilla=13)

    assert set(extractor.extract(datos, etiquetas)) == set(vector_base)


def test_a_plain_array_is_accepted(extractor, vector_base) -> None:
    datos, etiquetas = tabla(n=250, columnas=6, semilla=14)

    vector = extractor.extract(datos.to_numpy(), etiquetas)

    assert set(vector) == set(vector_base)


# --------------------------------------------------------------------------- #
# Caché
# --------------------------------------------------------------------------- #
def test_the_cache_writes_a_file_and_reads_it_back(tmp_path) -> None:
    con_cache = MetaFeatureExtractor(random_state=42, cache_dir=tmp_path)
    datos, etiquetas = tabla(n=250, columnas=6, semilla=15)

    primero = con_cache.extract(datos, etiquetas, dataset_id="prueba")
    archivos = list(tmp_path.iterdir())
    segundo = con_cache.extract(datos, etiquetas, dataset_id="prueba")

    assert len(archivos) == 1
    assert iguales(primero, segundo)


def test_the_cache_preserves_non_finite_values(tmp_path) -> None:
    """El round-trip por JSON debe conservar ``NaN``: no puede volver como cero."""
    con_cache = MetaFeatureExtractor(random_state=42, cache_dir=tmp_path)
    datos, etiquetas = tabla(n=250, columnas=6, semilla=16)

    primero = con_cache.extract(datos, etiquetas, dataset_id="prueba")
    segundo = con_cache.extract(datos, etiquetas, dataset_id="prueba")

    assert columnas_no_finitas(primero) == columnas_no_finitas(segundo)
    assert columnas_no_finitas(segundo)


def test_a_different_configuration_does_not_reuse_the_cache(tmp_path) -> None:
    """Cambiar la semilla o el tamaño de muestra cambia el vector, no puede reusarse."""
    uno = MetaFeatureExtractor(random_state=42, cache_dir=tmp_path)
    otro = MetaFeatureExtractor(random_state=99, cache_dir=tmp_path)

    assert uno._ruta_cache("d") != otro._ruta_cache("d")


def test_without_a_dataset_id_nothing_is_cached(tmp_path) -> None:
    con_cache = MetaFeatureExtractor(random_state=42, cache_dir=tmp_path)
    datos, etiquetas = tabla(n=250, columnas=6, semilla=17)

    con_cache.extract(datos, etiquetas)

    assert not list(tmp_path.iterdir())


def test_without_a_cache_dir_no_path_is_computed() -> None:
    assert MetaFeatureExtractor()._ruta_cache("d") is None


def test_the_cache_file_name_is_safe_for_the_filesystem(tmp_path) -> None:
    con_cache = MetaFeatureExtractor(cache_dir=tmp_path)

    ruta = con_cache._ruta_cache("openml/61: iris*?")

    assert ruta is not None
    assert not set(ruta.name) & set('/\\:*?"<>|')


# --------------------------------------------------------------------------- #
# Errores
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tamano", [0, -1])
def test_an_invalid_sample_size_raises(tamano: int) -> None:
    with pytest.raises(ValueError, match="sample_size"):
        MetaFeatureExtractor(sample_size=tamano)


@pytest.mark.parametrize("decimales", [0, -3])
def test_an_invalid_precision_raises(decimales: int) -> None:
    with pytest.raises(ValueError, match="decimals"):
        MetaFeatureExtractor(decimals=decimales)


def test_a_pymfe_failure_becomes_a_meta_feature_error(monkeypatch) -> None:
    """pymfe lanza ``Exception`` con mensaje vacío; sin envolverlo es indepurable."""

    def explotar(self, *args, **kwargs):
        raise Exception()          # noqa: TRY002 - reproduce la conducta de pymfe

    monkeypatch.setattr(modulo.MFE, "fit", explotar)
    datos, etiquetas = tabla(n=200, columnas=5, semilla=18)

    with pytest.raises(MetaFeatureError, match="sin mensaje"):
        MetaFeatureExtractor().extract(datos, etiquetas)


def test_a_name_collision_between_groups_raises(monkeypatch) -> None:
    """Dos grupos no pueden aportar el mismo nombre sobre datos distintos.

    Si ocurriera, el segundo pisaría al primero en silencio y el vector mezclaría
    un valor del dataset completo con otro de la submuestra.
    """
    monkeypatch.setattr(
        MetaFeatureExtractor, "_extraer", lambda self, X, y, grupos: {"repetida": 1.0}
    )
    datos, etiquetas = tabla(n=200, columnas=5, semilla=19)

    with pytest.raises(MetaFeatureError, match="repetida"):
        MetaFeatureExtractor().extract(datos, etiquetas)


# --------------------------------------------------------------------------- #
# Meta-características no finitas
# --------------------------------------------------------------------------- #
def test_non_finite_columns_are_reported_not_dropped(vector_base) -> None:
    """Descartarlas es una decisión del corpus, no de un dataset suelto."""
    no_finitas = columnas_no_finitas(vector_base)

    assert no_finitas
    assert no_finitas < set(vector_base)


def test_non_finite_helper_detects_infinities() -> None:
    vector = {"a": 1.0, "b": float("nan"), "c": float("inf"), "d": float("-inf")}

    assert columnas_no_finitas(vector) == {"b", "c", "d"}
