# tests/test_verify_reproducibility.py
"""Pruebas del verificador de reproducibilidad de punta a punta.

Las reejecuciones reales —de los experimentos y del caso de imágenes— no se prueban
aquí: cuestan minutos y las ejercita el propio guion al correr. Lo que sí se fija es la
**lógica de comparación**, porque un error en ella haría que el documento declarara
«idéntico» algo que difiere, que es exactamente lo que un verificador no puede hacer:
que ``NaN`` cuente como igual a ``NaN``, que la comparación de curvas no tenga
tolerancia, que una curva más corta cuente como diferencia, que la única línea ignorada
de un documento sea su fecha y que una verificación omitida no se lea como un éxito.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

pytest.importorskip("openml", reason="el verificador importa los arneses de los experimentos")

import run_lodo  # noqa: E402
import run_lodo_e2  # noqa: E402
import verify_reproducibility as verificador  # noqa: E402

from autopipe.meta_knowledge import MetaKnowledgeBase, MetaKnowledgeEntry  # noqa: E402
from autopipe.meta_recommender import MetaRecommender  # noqa: E402
from autopipe.search_space import TABULAR_SPACE_VERSION  # noqa: E402

RAIZ_REPO = Path(__file__).resolve().parents[1]
NAN, INF = float("nan"), float("inf")


# --------------------------------------------------------------------------- #
# Comparación de valores y curvas
# --------------------------------------------------------------------------- #
def test_nan_counts_as_equal_to_nan() -> None:
    """Un ensayo inevaluable guardado como NaN tiene que coincidir consigo mismo."""
    assert verificador.iguales(NAN, NAN)
    assert not verificador.iguales(NAN, 0.5)
    assert verificador.iguales(-INF, -INF)


def test_non_numeric_values_are_compared_as_they_are() -> None:
    assert verificador.iguales({"a": 1}, {"a": 1})
    assert not verificador.iguales({"a": 1}, {"a": 2})


def test_identical_curves_have_no_differences() -> None:
    assert verificador.diferencias_de_curva([0.1, -INF, NAN], [0.1, -INF, NAN]) == 0


def test_the_comparison_has_no_tolerance() -> None:
    """Prueba por sabotaje de la exactitud.

    El documento afirma que las corridas se comparan bit a bit. Si alguien introdujera
    una tolerancia, una diferencia en la séptima cifra dejaría de contarse y el
    verificador declararía idéntico algo que no lo es.
    """
    assert verificador.diferencias_de_curva([0.1, 0.2, 0.3], [0.1, 0.2000001, 0.3]) == 1


def test_a_shorter_curve_counts_its_missing_trials() -> None:
    """Una corrida truncada no puede pasar por idéntica solo porque su prefijo coincide."""
    assert verificador.diferencias_de_curva([0.1, 0.2, 0.3], [0.1]) == 2


# --------------------------------------------------------------------------- #
# Documentos regenerados
# --------------------------------------------------------------------------- #
def _documento(fecha: str, cuerpo: str = "| a | 1 |") -> str:
    return f"# Título\n\n- **Generado:** {fecha}\n- **Duración:** 1.0 min\n\n{cuerpo}\n"


def test_only_the_generation_date_is_ignored() -> None:
    lineas = verificador._sin_fecha(_documento("2026-09-10"))

    assert not any("Generado" in linea for linea in lineas)
    assert "- **Duración:** 1.0 min" in lineas


def test_a_document_that_differs_only_in_its_date_is_identical(tmp_path: Path) -> None:
    ruta = tmp_path / "doc.md"
    ruta.write_text(_documento("2026-09-01"), encoding="utf-8")

    registro = verificador.verificar_documento("doc", _documento("2026-09-10"), ruta)

    assert registro["estado"] == verificador.IDENTICO


def test_a_document_that_differs_beyond_its_date_is_reported(tmp_path: Path) -> None:
    ruta = tmp_path / "doc.md"
    ruta.write_text(_documento("2026-09-01", "| a | 1 |"), encoding="utf-8")

    registro = verificador.verificar_documento("doc", _documento("2026-09-01", "| a | 2 |"), ruta)

    assert registro["estado"] == verificador.DIFIERE
    assert registro["detalle"] == "1 líneas distintas"


def test_a_missing_document_is_omitted_and_not_passed(tmp_path: Path) -> None:
    registro = verificador.verificar_documento("doc", "texto", tmp_path / "no-existe.md")

    assert registro["estado"] == verificador.OMITIDO


# --------------------------------------------------------------------------- #
# Recomendaciones
# --------------------------------------------------------------------------- #
CONFIGS = (
    {"imputation": 0, "scaling": 0, "encoding": 0, "selection": 0},
    {"imputation": 1, "scaling": 1, "encoding": 1, "selection": 1},
    {"imputation": 0, "scaling": 2, "encoding": 1, "selection": 2},
)


def _base() -> MetaKnowledgeBase:
    return MetaKnowledgeBase(
        [
            MetaKnowledgeEntry(
                dataset_id=str(i),
                dataset_name=f"juguete{i}",
                meta_features={"a": float(3 - i), "b": float(2 * (3 - i))},
                configuration=config,
                pipeline_params={},
                metric="macro_f1",
                metric_value=0.8,
                seed=42,
                n_trials=50,
                n_failed_trials=0,
                space_version=TABULAR_SPACE_VERSION,
            )
            for i, config in enumerate(CONFIGS)
        ]
    )


def _pliegue_con_lo_que_recomienda(base: MetaKnowledgeBase, estrategias: tuple) -> dict:
    """Pliegue cuyas recomendaciones guardadas son las que el recomendador produce hoy."""
    entrada = base.entries[0]
    return {
        "dataset_id": entrada.dataset_id,
        "dataset_name": entrada.dataset_name,
        "recomendaciones": {
            e: {
                "configuraciones": MetaRecommender(
                    n_configurations=5, strategy=e, random_state=202
                ).fit(base.without(entrada.dataset_id)).recommend(entrada.meta_features)
            }
            for e in estrategias
        },
    }


def test_recommendations_that_reproduce_are_identical() -> None:
    base = _base()
    pliegue = _pliegue_con_lo_que_recomienda(base, ("nearest", "frequent"))

    registro = verificador.verificar_recomendaciones("E", base, pliegue, ("nearest", "frequent"))

    assert registro["estado"] == verificador.IDENTICO


def test_a_changed_recommendation_names_its_strategy() -> None:
    """Es la comprobación que demuestra que el recomendador no cambió desde el experimento 1."""
    base = _base()
    pliegue = _pliegue_con_lo_que_recomienda(base, ("nearest", "frequent"))
    pliegue["recomendaciones"]["frequent"]["configuraciones"] = [dict(CONFIGS[2])]

    registro = verificador.verificar_recomendaciones("E", base, pliegue, ("nearest", "frequent"))

    assert registro["estado"] == verificador.DIFIERE
    assert "frequent" in registro["detalle"]
    assert "nearest" not in registro["detalle"]


def test_a_missing_fold_is_refused() -> None:
    with pytest.raises(KeyError, match="balance-scale"):
        verificador._pliegue({"pliegues": []}, "balance-scale")


# --------------------------------------------------------------------------- #
# Caso de imágenes y veredicto
# --------------------------------------------------------------------------- #
def test_the_image_check_is_omitted_without_data(tmp_path: Path) -> None:
    """Sin datos no hay evidencia: se marca omitido, nunca idéntico."""
    assert verificador.verificar_imagenes(None, {"categorias": []})["estado"] == verificador.OMITIDO
    assert verificador.verificar_imagenes(tmp_path, {"categorias": []})["estado"] == (
        verificador.OMITIDO
    )
    assert verificador.verificar_imagenes(tmp_path, None)["estado"] == verificador.OMITIDO


def _registro(estado: str) -> dict:
    return {"verificacion": "v", "compara": "c", "estado": estado, "detalle": "d"}


def test_the_verdict_passes_only_when_nothing_differs() -> None:
    texto = verificador.redactar([_registro(verificador.IDENTICO)] * 3)

    assert "se reproducen exactamente" in texto


def test_a_single_difference_turns_the_verdict() -> None:
    texto = verificador.redactar(
        [_registro(verificador.IDENTICO), _registro(verificador.DIFIERE)]
    )

    assert "1 verificaciones no se reproducen" in texto
    assert "se reproducen exactamente" not in texto


def test_omissions_are_counted_and_not_hidden() -> None:
    texto = verificador.redactar(
        [_registro(verificador.IDENTICO), _registro(verificador.OMITIDO)]
    )

    assert "**Omitidas:** 1" in texto
    assert "no es un fallo, pero tampoco una evidencia" in texto


# --------------------------------------------------------------------------- #
# Las parejas elegidas existen en cada experimento
# --------------------------------------------------------------------------- #
def test_the_seeds_of_each_pair_exist_in_its_experiment() -> None:
    assert {s for _, s in verificador.PARES_E1} <= set(run_lodo.SEMILLAS)
    assert {s for _, s in verificador.PARES_E2} <= set(run_lodo_e2.SEMILLAS)


def test_seed_606_exists_only_in_experiment_two() -> None:
    """El documento lo afirma: es lo que hace que esa pareja ejercite la ampliación."""
    assert 606 in run_lodo_e2.SEMILLAS
    assert 606 not in run_lodo.SEMILLAS


def test_the_datasets_of_each_pair_are_in_its_corpus() -> None:
    base_e1 = {e.dataset_name for e in MetaKnowledgeBase.load(RAIZ_REPO / run_lodo.BASE)}
    base_e2 = {e.dataset_name for e in MetaKnowledgeBase.load(RAIZ_REPO / run_lodo_e2.BASE)}

    assert {n for n, _ in verificador.PARES_E1} <= base_e1
    assert {n for n, _ in verificador.PARES_E2} <= base_e2
    assert verificador.PAR_SENSIBILIDAD[0] in base_e2


def test_the_sensitivity_pair_is_one_of_experiment_two() -> None:
    """La sensibilidad reutiliza las semillas y las inyecciones del experimento 2."""
    assert verificador.PAR_SENSIBILIDAD in verificador.PARES_E2


def test_nan_is_not_silently_equal_to_itself_by_identity_only() -> None:
    """``float('nan') == float('nan')`` es falso en Python; la función lo corrige a propósito."""
    nan_a, nan_b = float("nan"), float("nan")

    assert not nan_a == nan_b  # noqa: PLR0124
    assert verificador.iguales(nan_a, nan_b)
    assert math.isnan(nan_a)
