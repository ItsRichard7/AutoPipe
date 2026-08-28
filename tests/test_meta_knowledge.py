# tests/test_meta_knowledge.py
"""Pruebas de la base de meta-conocimiento.

Los dos invariantes que verifica la base no son formalidades: sin el conjunto de
nombres común, el recomendador compararía vectores desalineados; sin la versión
única del espacio, una configuración inyectada designaría transformaciones
distintas de las previstas. En ambos casos la búsqueda correría igual y daría un
número que no mide lo que se cree.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from autopipe.meta_knowledge import MetaKnowledgeBase, MetaKnowledgeEntry
from autopipe.search_space import TABULAR_SPACE_VERSION


def entrada(
    identificador: str,
    meta: dict[str, float] | None = None,
    version: str = TABULAR_SPACE_VERSION,
    valor: float = 0.8,
) -> MetaKnowledgeEntry:
    """Arma una entrada sintética."""
    return MetaKnowledgeEntry(
        dataset_id=identificador,
        dataset_name=f"dataset-{identificador}",
        meta_features=meta if meta is not None else {"a": 1.0, "b": 2.0, "c": 3.0},
        configuration={"imputation": 0, "scaling": 1, "encoding": 0, "selection": 3},
        pipeline_params={"0_ImputeMissing": {"numeric_strategy": "mean"}},
        metric="macro_f1",
        metric_value=valor,
        seed=42,
        n_trials=50,
        n_failed_trials=0,
        space_version=version,
        seconds=1.5,
    )


@pytest.fixture
def base() -> MetaKnowledgeBase:
    return MetaKnowledgeBase(
        [
            entrada("1", {"a": 1.0, "b": 2.0, "c": float("nan")}),
            entrada("2", {"a": 4.0, "b": 5.0, "c": 6.0}),
            entrada("3", {"a": 7.0, "b": float("inf"), "c": 9.0}),
        ]
    )


# --------------------------------------------------------------------------- #
# Invariantes
# --------------------------------------------------------------------------- #
def test_an_empty_base_is_valid() -> None:
    assert len(MetaKnowledgeBase()) == 0
    assert MetaKnowledgeBase().feature_names() == []


def test_entries_must_share_the_same_feature_names() -> None:
    """Sin esto el recomendador compararía vectores desalineados en silencio."""
    with pytest.raises(ValueError, match="desalineados"):
        MetaKnowledgeBase(
            [
                entrada("1", {"a": 1.0, "b": 2.0}),
                entrada("2", {"a": 1.0, "z": 2.0}),
            ]
        )


def test_the_offending_dataset_is_named_in_the_error() -> None:
    with pytest.raises(ValueError, match="dataset-2"):
        MetaKnowledgeBase(
            [entrada("1", {"a": 1.0}), entrada("2", {"a": 1.0, "b": 2.0})]
        )


def test_the_base_cannot_mix_search_space_versions() -> None:
    """Un índice solo significa algo respecto de una versión concreta."""
    with pytest.raises(ValueError, match="versiones del espacio"):
        MetaKnowledgeBase([entrada("1"), entrada("2", version="tabular-v0")])


def test_duplicate_dataset_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="repetidos"):
        MetaKnowledgeBase([entrada("1"), entrada("1")])


# --------------------------------------------------------------------------- #
# Regla de meta-características no finitas
# --------------------------------------------------------------------------- #
def test_a_feature_non_finite_in_any_dataset_does_not_survive(base) -> None:
    """Es la regla del pre-registro: descartar, no imputar al nivel meta."""
    assert base.surviving_features() == ["a"]


def test_all_features_survive_when_every_value_is_finite() -> None:
    completa = MetaKnowledgeBase([entrada("1"), entrada("2")])

    assert completa.surviving_features() == ["a", "b", "c"]


def test_infinities_are_discarded_like_nan(base) -> None:
    assert "b" not in base.surviving_features()
    assert "c" not in base.surviving_features()


def test_the_full_vector_is_kept_even_when_a_feature_is_discarded(base) -> None:
    """Guardar solo las supervivientes volvería la regla irreversible."""
    assert base.feature_names() == ["a", "b", "c"]
    assert math.isnan(base.entry_for("1").meta_features["c"])


# --------------------------------------------------------------------------- #
# Matriz de meta-características
# --------------------------------------------------------------------------- #
def test_the_matrix_uses_the_surviving_features_by_default(base) -> None:
    matriz, columnas = base.feature_matrix()

    assert columnas == ["a"]
    np.testing.assert_allclose(matriz, [[1.0], [4.0], [7.0]])


def test_the_matrix_rows_follow_the_entry_order(base) -> None:
    matriz, _ = base.feature_matrix(names=["a"])

    assert [fila[0] for fila in matriz] == [1.0, 4.0, 7.0]


def test_the_matrix_can_be_asked_for_explicit_columns(base) -> None:
    matriz, columnas = base.feature_matrix(names=["b", "a"])

    assert columnas == ["b", "a"]
    assert matriz.shape == (3, 2)


# --------------------------------------------------------------------------- #
# Consulta y pliegues
# --------------------------------------------------------------------------- #
def test_the_base_is_iterable(base) -> None:
    assert [entrada.dataset_id for entrada in base] == ["1", "2", "3"]


def test_an_entry_can_be_looked_up_by_id(base) -> None:
    assert base.entry_for("2").dataset_name == "dataset-2"


def test_looking_up_an_absent_dataset_raises(base) -> None:
    with pytest.raises(KeyError, match="ausente"):
        base.entry_for("ausente")


def test_a_fold_excludes_exactly_one_dataset(base) -> None:
    """Es el pliegue del leave-one-dataset-out."""
    pliegue = base.without("2")

    assert len(pliegue) == 2
    assert pliegue.dataset_ids == ["1", "3"]


def test_excluding_a_dataset_does_not_modify_the_original(base) -> None:
    base.without("2")

    assert len(base) == 3


def test_excluding_an_absent_dataset_raises(base) -> None:
    with pytest.raises(KeyError):
        base.without("ausente")


def test_removing_a_dataset_can_change_which_features_survive() -> None:
    """El descarte depende del corpus, y por eso se recalcula por pliegue."""
    base = MetaKnowledgeBase(
        [
            entrada("1", {"a": 1.0, "b": float("nan")}),
            entrada("2", {"a": 2.0, "b": 5.0}),
        ]
    )

    assert base.surviving_features() == ["a"]
    assert base.without("1").surviving_features() == ["a", "b"]


# --------------------------------------------------------------------------- #
# Persistencia
# --------------------------------------------------------------------------- #
def test_the_base_survives_a_round_trip_through_disk(base, tmp_path) -> None:
    ruta = base.save(tmp_path / "base.json")

    recuperada = MetaKnowledgeBase.load(ruta)

    assert recuperada.dataset_ids == base.dataset_ids
    assert recuperada.feature_names() == base.feature_names()
    assert recuperada.surviving_features() == base.surviving_features()


def test_non_finite_values_survive_the_round_trip(base, tmp_path) -> None:
    """Si volvieran como cero, una columna descartada pasaría a sobrevivir."""
    recuperada = MetaKnowledgeBase.load(base.save(tmp_path / "base.json"))

    assert math.isnan(recuperada.entry_for("1").meta_features["c"])
    assert math.isinf(recuperada.entry_for("3").meta_features["b"])


def test_the_winning_configuration_survives_the_round_trip(base, tmp_path) -> None:
    recuperada = MetaKnowledgeBase.load(base.save(tmp_path / "base.json"))

    assert recuperada.entry_for("1").configuration == {
        "imputation": 0,
        "scaling": 1,
        "encoding": 0,
        "selection": 3,
    }


def test_loading_a_base_from_another_space_version_is_rejected(tmp_path) -> None:
    """Se rechaza en lugar de reinterpretar índices que significan otra cosa."""
    ruta = MetaKnowledgeBase([entrada("1", version="tabular-v0")]).save(
        tmp_path / "vieja.json"
    )

    with pytest.raises(ValueError, match="tabular-v0"):
        MetaKnowledgeBase.load(ruta)


def test_an_old_base_can_be_inspected_without_version_check(tmp_path) -> None:
    ruta = MetaKnowledgeBase([entrada("1", version="tabular-v0")]).save(
        tmp_path / "vieja.json"
    )

    recuperada = MetaKnowledgeBase.load(ruta, check_version=False)

    assert recuperada.entries[0].space_version == "tabular-v0"


def test_an_empty_base_can_be_saved_and_loaded(tmp_path) -> None:
    ruta = MetaKnowledgeBase().save(tmp_path / "vacia.json")

    assert len(MetaKnowledgeBase.load(ruta)) == 0


def test_save_creates_the_parent_directory(base, tmp_path) -> None:
    ruta = base.save(tmp_path / "nueva" / "carpeta" / "base.json")

    assert ruta.is_file()
