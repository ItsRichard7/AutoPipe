# tests/test_meta_knowledge_harness.py
"""Pruebas del arnés que construye la base de meta-conocimiento.

Verifican las dos propiedades que la hoja de ruta pide del arnés: **completitud**
—que cada entrada salga con todos sus campos y con una configuración
interpretable— y **reproducibilidad** —que la misma semilla produzca la misma
configuración ganadora—.

No tocan la red: la descarga se sustituye por un dataset sintético, de modo que
la prueba ejercita el camino completo (meta-características, búsqueda, armado de
la entrada) sin depender de OpenML.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

pytest.importorskip("openml", reason="el arnés importa openml")

import build_meta_knowledge as arnes  # noqa: E402

from autopipe.meta_features import MetaFeatureExtractor  # noqa: E402
from autopipe.search_space import TABULAR_SPACE_STEPS, TABULAR_SPACE_VERSION  # noqa: E402


@pytest.fixture
def dataset_sintetico():
    """Tabla pequeña con tipos mixtos, faltantes y tres clases."""
    rng = np.random.default_rng(0)
    n = 180
    clase = np.tile([0, 1, 2], n // 3)
    tabla = pd.DataFrame(
        {
            "senal": clase * 1.2 + rng.normal(0, 1.0, n),
            "ruido": rng.normal(0, 1.0, n),
            "escala": clase * 90.0 + rng.normal(0, 150.0, n),
            "region": np.where(rng.random(n) < 0.6, clase, rng.integers(0, 3, n)),
        }
    )
    tabla["region"] = tabla["region"].map({0: "norte", 1: "sur", 2: "este"})
    tabla.loc[rng.choice(n, 8, replace=False), "ruido"] = np.nan
    return tabla, clase


@pytest.fixture
def sin_red(monkeypatch, dataset_sintetico, tmp_path):
    """Sustituye la descarga y acorta la búsqueda, para no tocar la red."""
    tabla, clase = dataset_sintetico
    monkeypatch.setattr(
        arnes, "descargar", lambda did: (tabla, clase, "Public", [])
    )
    monkeypatch.setattr(arnes, "ENSAYOS", 12)
    monkeypatch.setattr(arnes, "CACHE_ENTRADAS", tmp_path / "entradas")
    return tmp_path


# --------------------------------------------------------------------------- #
# Lectura del corpus
# --------------------------------------------------------------------------- #
def test_the_corpus_table_is_parsed(tmp_path) -> None:
    documento = tmp_path / "corpus.md"
    documento.write_text(
        "# Corpus\n\n"
        "| # | OpenML ID | Nombre | IQR |\n"
        "|---:|---:|---|---:|\n"
        "| 1 | 40983 | `wilt` | 0.3239 |\n"
        "| 2 | 38 | `sick` | 0.3064 |\n",
        encoding="utf-8",
    )

    assert arnes.leer_corpus(documento) == [(40983, "wilt"), (38, "sick")]


def test_a_missing_corpus_file_points_at_the_screening(tmp_path) -> None:
    with pytest.raises(SystemExit, match="screen_corpus"):
        arnes.leer_corpus(tmp_path / "no-existe.md")


def test_a_corpus_without_rows_is_rejected(tmp_path) -> None:
    documento = tmp_path / "vacio.md"
    documento.write_text("# Corpus\n\nSin tabla.\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="ningún dataset"):
        arnes.leer_corpus(documento)


# --------------------------------------------------------------------------- #
# Caché y reanudación
# --------------------------------------------------------------------------- #
def test_the_cache_name_carries_the_space_version_and_the_seed() -> None:
    """Cambiar cualquiera de los dos invalida lo cacheado."""
    nombre = arnes._cache(123).name

    assert "123" in nombre
    assert TABULAR_SPACE_VERSION in nombre
    assert f"s{arnes.SEMILLA}" in nombre


def test_an_entry_survives_the_cache_round_trip(sin_red) -> None:
    extractor = MetaFeatureExtractor(random_state=arnes.SEMILLA)
    entrada = arnes.construir_entrada(7, "sintetico", extractor)

    arnes._guardar_cache(entrada)
    recuperada = arnes._leer_cache(7)

    assert recuperada is not None
    assert recuperada.configuration == entrada.configuration
    assert recuperada.metric_value == entrada.metric_value


def test_reading_an_absent_cache_entry_returns_none(sin_red) -> None:
    assert arnes._leer_cache(999) is None


# --------------------------------------------------------------------------- #
# Completitud de la entrada
# --------------------------------------------------------------------------- #
def test_the_entry_is_complete(sin_red) -> None:
    extractor = MetaFeatureExtractor(random_state=arnes.SEMILLA)

    entrada = arnes.construir_entrada(7, "sintetico", extractor)

    assert entrada.dataset_id == "7"
    assert entrada.dataset_name == "sintetico"
    assert entrada.metric == "macro_f1"
    assert 0.0 <= entrada.metric_value <= 1.0
    assert entrada.seed == arnes.SEMILLA
    assert entrada.n_trials == arnes.ENSAYOS
    assert 0 <= entrada.n_failed_trials <= entrada.n_trials
    assert entrada.space_version == TABULAR_SPACE_VERSION
    assert entrada.seconds > 0
    assert len(entrada.meta_features) > 50


def test_the_configuration_has_one_index_per_canonical_step(sin_red) -> None:
    """Sin un índice por paso, la configuración no se puede volver a inyectar."""
    extractor = MetaFeatureExtractor(random_state=arnes.SEMILLA)

    entrada = arnes.construir_entrada(7, "sintetico", extractor)

    assert set(entrada.configuration) == set(TABULAR_SPACE_STEPS)
    assert all(isinstance(indice, int) for indice in entrada.configuration.values())


def test_the_readable_parameters_match_the_kept_steps(sin_red) -> None:
    """Los pasos omitidos no aparecen en el pipeline ni en sus parámetros."""
    from autopipe.search_space import build_tabular_search_space

    espacio = build_tabular_search_space()
    extractor = MetaFeatureExtractor(random_state=arnes.SEMILLA)

    entrada = arnes.construir_entrada(7, "sintetico", extractor)

    conservados = sum(
        1
        for paso, indice in entrada.configuration.items()
        if indice < len(espacio[paso])
    )
    assert len(entrada.pipeline_params) == conservados


# --------------------------------------------------------------------------- #
# Reproducibilidad
# --------------------------------------------------------------------------- #
def test_the_same_seed_gives_the_same_winner(sin_red) -> None:
    """Criterio de reproducibilidad del anteproyecto, Sección 4.3.5."""
    extractor = MetaFeatureExtractor(random_state=arnes.SEMILLA)

    primera = arnes.construir_entrada(7, "sintetico", extractor)
    segunda = arnes.construir_entrada(7, "sintetico", extractor)

    assert primera.configuration == segunda.configuration
    assert primera.metric_value == segunda.metric_value
    assert primera.pipeline_params == segunda.pipeline_params


def test_a_search_without_a_winner_is_reported(sin_red, monkeypatch) -> None:
    """Si ningún ensayo fuera evaluable, callarlo dejaría una base incompleta."""

    class MotorSinGanador:
        best_configuration_ = None
        tied_configurations_ = None

        def __init__(self, *args, **kwargs):
            pass

        def run(self, *args, **kwargs):
            from autopipe.pipeline.pipeline import Pipeline

            return Pipeline([]), {"macro_f1": float("nan")}, []

    monkeypatch.setattr(arnes, "OptunaSearchEngine", MotorSinGanador)
    extractor = MetaFeatureExtractor(random_state=arnes.SEMILLA)

    with pytest.raises(RuntimeError, match="no dejó configuración ganadora"):
        arnes.construir_entrada(7, "sintetico", extractor)


# --------------------------------------------------------------------------- #
# Desempate entre configuraciones que empatan
# --------------------------------------------------------------------------- #
def test_the_entry_stores_the_canonical_configuration(sin_red, monkeypatch) -> None:
    """Ante un empate, la entrada guarda la canónica y no la primera visitada.

    Es lo que impide que la base registre una propiedad de la corrida en lugar de
    una del dataset. Se simula un motor cuyo ganador omite dos pasos y que empata
    con otra configuración que no omite ninguno: la entrada debe quedarse con la
    segunda, porque es la que se puede ejecutar en un dataset receptor con
    faltantes o con columnas categóricas.
    """
    omite_dos = {"imputation": 2, "scaling": 0, "encoding": 2, "selection": 3}
    completa = {"imputation": 0, "scaling": 0, "encoding": 0, "selection": 3}

    class MotorConEmpate:
        best_configuration_ = omite_dos
        tied_configurations_ = [omite_dos, completa]

        def __init__(self, *args, **kwargs):
            pass

        def run(self, *args, **kwargs):
            from autopipe.pipeline.pipeline import Pipeline

            return Pipeline([]), {"macro_f1": 0.75}, [{"objective": 0.75}]

    monkeypatch.setattr(arnes, "OptunaSearchEngine", MotorConEmpate)
    extractor = MetaFeatureExtractor(random_state=arnes.SEMILLA)

    entrada = arnes.construir_entrada(7, "sintetico", extractor)

    assert entrada.configuration == completa
    assert entrada.configuration != MotorConEmpate.best_configuration_


def test_the_readable_parameters_follow_the_canonical_configuration(
    sin_red, monkeypatch
) -> None:
    """Los parámetros guardados describen la canónica, no el pipeline devuelto.

    Si se guardara el pipeline que el motor construyó, los índices y los
    parámetros de la misma entrada describirían pipelines distintos, y nada
    fallaría al leerlos.
    """
    omite_todo = {"imputation": 2, "scaling": 3, "encoding": 2, "selection": 3}
    completa = {"imputation": 1, "scaling": 2, "encoding": 1, "selection": 0}

    class MotorConEmpate:
        best_configuration_ = omite_todo
        tied_configurations_ = [omite_todo, completa]

        def __init__(self, *args, **kwargs):
            pass

        def run(self, *args, **kwargs):
            from autopipe.pipeline.pipeline import Pipeline

            return Pipeline([]), {"macro_f1": 0.5}, [{"objective": 0.5}]

    monkeypatch.setattr(arnes, "OptunaSearchEngine", MotorConEmpate)
    extractor = MetaFeatureExtractor(random_state=arnes.SEMILLA)

    entrada = arnes.construir_entrada(7, "sintetico", extractor)

    assert entrada.configuration == completa
    assert len(entrada.pipeline_params) == 4
    assert [tipo.split("_", 1)[1] for tipo in entrada.pipeline_params] == [
        "ImputeMissing",
        "ScaleNumeric",
        "EncodeCategorical",
        "SelectFeatures",
    ]


def test_the_cache_key_records_the_tie_breaking_rule() -> None:
    """La regla de desempate viaja en el nombre del archivo de caché.

    Cambiarla cambia qué configuración guarda cada entrada. Sin la regla en la
    clave, una caché construida con otro criterio se reutilizaría en silencio y
    la base mezclaría dos criterios distintos.
    """
    ruta = arnes._cache(7)

    assert arnes.DESEMPATE in ruta.name
    assert arnes.TABULAR_SPACE_VERSION in ruta.name
    assert f"s{arnes.SEMILLA}" in ruta.name
