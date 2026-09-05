# tests/test_evaluate_recommender.py
"""Pruebas de la evaluación preliminar del recomendador.

La comprobación que más importa de este script no es que calcule bien una
mediana, sino **dónde mide**. El corpus son 25 de los 52 candidatos cribados, y
esos 25 son los pliegues del *leave-one-dataset-out*: medir sobre ellos sería
medir sobre el conjunto de prueba. La evaluación tiene que restringirse a los 27
que quedaron fuera, y hay una prueba escrita para fallar si algún día alguno del
corpus se cuela en el conjunto evaluado.

El resto de las pruebas fijan que el veredicto que el informe redacta se derive
de los números y no esté escrito a mano, que es lo que impide que el documento
afirme algo distinto de lo que la ejecución produjo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

pytest.importorskip("openml", reason="la evaluación importa el descargador del corpus")

import evaluate_recommender as evaluacion  # noqa: E402

from autopipe.meta_knowledge import MetaKnowledgeBase, MetaKnowledgeEntry  # noqa: E402
from autopipe.meta_recommender import ESTRATEGIAS  # noqa: E402
from autopipe.search_space import TABULAR_SPACE_VERSION  # noqa: E402

CONFIG_A = {"imputation": 0, "scaling": 0, "encoding": 0, "selection": 0}
CONFIG_B = {"imputation": 1, "scaling": 1, "encoding": 1, "selection": 1}

CORPUS_DE_PRUEBA = """# Corpus seleccionado (RE4)

## Corpus

| # | OpenML ID | Nombre | IQR | Instancias |
|---:|---:|---|---:|---:|
| 1 | 11 | `en-el-corpus` | 0.3000 | 500 |

## Candidatos no seleccionados

| OpenML ID | Nombre | IQR | Motivo |
|---:|---|---:|---|
| 22 | `mfeat-zernike` | 0.0577 | familia |
| 23 | `cmc` | 0.0433 | orden |
"""


def entrada(dataset_id: str, valor: float, configuracion: dict[str, int]):
    """Arma una entrada mínima de la base."""
    return MetaKnowledgeEntry(
        dataset_id=dataset_id,
        dataset_name=f"d{dataset_id}",
        meta_features={"a": valor, "b": valor * 2},
        configuration=configuracion,
        pipeline_params={},
        metric="macro_f1",
        metric_value=0.7,
        seed=42,
        n_trials=50,
        n_failed_trials=0,
        space_version=TABULAR_SPACE_VERSION,
    )


@pytest.fixture
def base() -> MetaKnowledgeBase:
    """Base con dos configuraciones ganadoras distintas."""
    return MetaKnowledgeBase(
        [
            entrada("0", 0.0, CONFIG_A),
            entrada("1", 1.0, CONFIG_A),
            entrada("2", 2.0, CONFIG_B),
        ]
    )


# --------------------------------------------------------------------------- #
# Dónde se mide
# --------------------------------------------------------------------------- #
def test_only_the_non_selected_candidates_are_read(tmp_path) -> None:
    """Se leen los retenidos, nunca los del corpus.

    Es la prueba que sostiene la validez de todo el script: si algún dataset del
    corpus se colara en el conjunto evaluado, se estaría midiendo sobre los
    pliegues del experimento y nada fallaría al hacerlo.
    """
    ruta = tmp_path / "CORPUS-seleccionado.md"
    ruta.write_text(CORPUS_DE_PRUEBA, encoding="utf-8")

    retenidos = evaluacion.leer_retenidos(ruta)

    nombres = [r["dataset_name"] for r in retenidos]
    assert nombres == ["mfeat-zernike", "cmc"]
    assert "en-el-corpus" not in nombres


def test_the_reason_for_exclusion_is_kept(tmp_path) -> None:
    """Los dos grupos sesgan en direcciones opuestas y se reportan por separado."""
    ruta = tmp_path / "CORPUS-seleccionado.md"
    ruta.write_text(CORPUS_DE_PRUEBA, encoding="utf-8")

    retenidos = evaluacion.leer_retenidos(ruta)

    assert {r["motivo"] for r in retenidos} == {"familia", "orden"}
    assert retenidos[0]["iqr"] == pytest.approx(0.0577)


def test_a_missing_corpus_document_is_reported(tmp_path) -> None:
    with pytest.raises(SystemExit, match="No existe"):
        evaluacion.leer_retenidos(tmp_path / "no-existe.md")


def test_a_corpus_without_the_expected_section_is_reported(tmp_path) -> None:
    ruta = tmp_path / "CORPUS-seleccionado.md"
    ruta.write_text("# Corpus\n\nsin la sección esperada\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="no tiene la sección"):
        evaluacion.leer_retenidos(ruta)


# --------------------------------------------------------------------------- #
# Medición sobre un dataset
# --------------------------------------------------------------------------- #
def _rejilla_con(mejor_en: dict[str, int], valor: float, resto: float) -> list[float]:
    """Rejilla sintética donde una configuración concreta es la mejor."""
    objetivos = [resto] * 144
    objetivos[evaluacion.indice_desde_configuracion(mejor_en)] = valor
    return objetivos


def test_the_distance_to_the_optimum_is_measured(monkeypatch, base) -> None:
    """La distancia es el óptimo del espacio menos la mejor de las cinco."""
    monkeypatch.setattr(
        evaluacion, "rejilla_exhaustiva", lambda *a, **k: _rejilla_con(CONFIG_B, 1.0, 0.4)
    )
    monkeypatch.setattr(evaluacion, "descargar", lambda *a, **k: (None, None, None, None))
    extractor = type("E", (), {"extract": lambda self, *a, **k: {"a": 0.0, "b": 0.0}})()
    recomendadores = evaluacion._recomendadores(base)

    resultado = evaluacion.evaluar_dataset(
        {"dataset_id": "7", "dataset_name": "d7", "iqr": 0.1, "motivo": "orden"},
        recomendadores,
        extractor,
    )

    assert resultado["optimo"] == pytest.approx(1.0)
    # La consulta cae junto a los datasets cuya ganadora es CONFIG_A, pero con
    # cinco configuraciones se acaba recogiendo también CONFIG_B, que es óptima.
    assert resultado["estrategias"]["nearest"]["distancia_al_optimo"] == pytest.approx(0.0)
    assert resultado["estrategias"]["nearest"]["alcanza_el_objetivo"] is True


def test_a_recommendation_that_misses_the_optimum_is_reported(monkeypatch, base) -> None:
    """Si ninguna de las cinco es óptima, la distancia lo dice."""
    fuera = {"imputation": 2, "scaling": 3, "encoding": 2, "selection": 3}
    monkeypatch.setattr(
        evaluacion, "rejilla_exhaustiva", lambda *a, **k: _rejilla_con(fuera, 1.0, 0.4)
    )
    monkeypatch.setattr(evaluacion, "descargar", lambda *a, **k: (None, None, None, None))
    extractor = type("E", (), {"extract": lambda self, *a, **k: {"a": 0.0, "b": 0.0}})()

    resultado = evaluacion.evaluar_dataset(
        {"dataset_id": "7", "dataset_name": "d7", "iqr": 0.1, "motivo": "orden"},
        evaluacion._recomendadores(base),
        extractor,
    )

    assert resultado["estrategias"]["nearest"]["distancia_al_optimo"] == pytest.approx(0.6)
    assert resultado["estrategias"]["nearest"]["alcanza_el_objetivo"] is False


def test_unevaluable_recommendations_are_counted(monkeypatch, base) -> None:
    """Una recomendación que no corre en el dataset receptor se cuenta aparte."""
    objetivos = [float("nan")] * 144
    objetivos[evaluacion.indice_desde_configuracion(CONFIG_A)] = 0.8
    monkeypatch.setattr(evaluacion, "rejilla_exhaustiva", lambda *a, **k: objetivos)
    monkeypatch.setattr(evaluacion, "descargar", lambda *a, **k: (None, None, None, None))
    extractor = type("E", (), {"extract": lambda self, *a, **k: {"a": 0.0, "b": 0.0}})()

    resultado = evaluacion.evaluar_dataset(
        {"dataset_id": "7", "dataset_name": "d7", "iqr": 0.1, "motivo": "orden"},
        evaluacion._recomendadores(base),
        extractor,
    )

    datos = resultado["estrategias"]["nearest"]
    assert datos["n_inevaluables"] >= 1
    assert datos["mejor"] == pytest.approx(0.8)


def test_every_strategy_is_evaluated(monkeypatch, base) -> None:
    monkeypatch.setattr(
        evaluacion, "rejilla_exhaustiva", lambda *a, **k: _rejilla_con(CONFIG_A, 1.0, 0.4)
    )
    monkeypatch.setattr(evaluacion, "descargar", lambda *a, **k: (None, None, None, None))
    extractor = type("E", (), {"extract": lambda self, *a, **k: {"a": 0.0, "b": 0.0}})()

    resultado = evaluacion.evaluar_dataset(
        {"dataset_id": "7", "dataset_name": "d7", "iqr": 0.1, "motivo": "orden"},
        evaluacion._recomendadores(base),
        extractor,
    )

    assert set(resultado["estrategias"]) == set(ESTRATEGIAS)


def test_the_recommenders_are_fitted_on_the_whole_base(base) -> None:
    """No se excluye nada: los datasets consultados están fuera del corpus."""
    recomendadores = evaluacion._recomendadores(base)

    assert set(recomendadores) == set(ESTRATEGIAS)
    for recomendador in recomendadores.values():
        assert recomendador.n_configurations == evaluacion.N_CONFIGURACIONES
        assert len(recomendador._base) == len(base)


# --------------------------------------------------------------------------- #
# Agregación y veredicto
# --------------------------------------------------------------------------- #
def _dataset(motivo: str, distancias: dict[str, float]) -> dict:
    """Resultado sintético de un dataset ya evaluado."""
    return {
        "dataset_id": "1",
        "dataset_name": "d",
        "iqr": 0.1,
        "motivo": motivo,
        "optimo": 1.0,
        "umbral": 0.95,
        "estrategias": {
            estrategia: {
                "distancia_al_optimo": valor,
                "alcanza_el_objetivo": valor <= 0.05,
                "n_inevaluables": 0,
                "n_recomendadas": 5,
                "mejor": 1.0 - valor,
                "vecinos": [],
            }
            for estrategia, valor in distancias.items()
        },
    }


def test_the_summary_filters_by_reason() -> None:
    """Cada grupo se agrega por separado, porque sesgan en direcciones opuestas."""
    datasets = [
        _dataset("orden", dict.fromkeys(ESTRATEGIAS, 0.10)),
        _dataset("familia", dict.fromkeys(ESTRATEGIAS, 0.02)),
    ]

    assert evaluacion.resumir(datasets)["n_datasets"] == 2
    assert evaluacion.resumir(datasets, "orden")["n_datasets"] == 1
    assert evaluacion.resumir(datasets, "orden")["nearest"][
        "distancia_mediana"
    ] == pytest.approx(0.10)
    assert evaluacion.resumir(datasets, "familia")["nearest"][
        "n_alcanza_el_objetivo"
    ] == 1


def test_an_exact_optimum_is_counted() -> None:
    datasets = [_dataset("orden", dict.fromkeys(ESTRATEGIAS, 0.0))]

    assert evaluacion.resumir(datasets)["nearest"]["n_optimo_exacto"] == 1


def test_the_verdict_names_the_strategy_that_leads() -> None:
    """El veredicto se deriva de los números, no está escrito a mano.

    Es lo que impide que el documento afirme algo distinto de lo que la ejecución
    produjo. Se construye un caso donde la aleatoria gana y se comprueba que el
    texto lo dice.
    """
    distancias = {"nearest": 0.05, "diverse": 0.06, "frequent": 0.07, "random": 0.01}
    datasets = [_dataset("orden", distancias), _dataset("familia", distancias)]
    total = evaluacion.resumir(datasets)
    orden = evaluacion.resumir(datasets, "orden")
    familia = evaluacion.resumir(datasets, "familia")

    texto = evaluacion._veredicto(total, orden, familia)

    assert "`random`" in texto
    assert "la ablación aleatoria queda" in texto
    assert "0.0100" in texto


def test_the_verdict_says_when_the_primary_leads() -> None:
    """El caso contrario también se redacta desde los datos."""
    distancias = {"nearest": 0.01, "diverse": 0.02, "frequent": 0.03, "random": 0.09}
    datasets = [_dataset("orden", distancias), _dataset("familia", distancias)]

    texto = evaluacion._veredicto(
        evaluacion.resumir(datasets),
        evaluacion.resumir(datasets, "orden"),
        evaluacion.resumir(datasets, "familia"),
    )

    assert "la estrategia primaria queda" in texto


def test_the_report_states_where_it_measures_and_what_it_does_not() -> None:
    """El documento tiene que decir por qué mide fuera del corpus.

    Sin eso, un lector podría tomarlo por el experimento, que es exactamente la
    confusión que este punto del plan estuvo a punto de causar.
    """
    distancias = dict.fromkeys(ESTRATEGIAS, 0.01)
    datasets = [_dataset("orden", distancias), _dataset("familia", distancias)]

    texto = evaluacion._informe(
        datasets,
        evaluacion.resumir(datasets),
        evaluacion.resumir(datasets, "orden"),
        evaluacion.resumir(datasets, "familia"),
    )

    assert "no entran nunca al" in texto
    assert "conjunto de prueba" in texto
    assert "No se ejecuta ninguna búsqueda de Optuna" in texto
    assert "No se usó para elegir nada" in texto
    for linea in texto.splitlines():
        if not linea.startswith("|"):
            assert len(linea) <= 100, f"línea larga en el informe: {linea!r}"


def test_the_grid_is_read_with_the_mapping_it_was_built_with() -> None:
    """La rejilla y el índice con que se lee son literalmente los del análisis.

    Es el riesgo silencioso de este script: lee valores de una tabla de 144
    posiciones que construyó otro módulo. Si alguien reimplantara aquí la
    traducción de configuración a índice y las dos versiones se separaran, cada
    recomendación se leería en la casilla equivocada y el informe daría valores
    que no le corresponden, sin que nada fallara. Compartir la función lo
    impide, y esta prueba fija que se comparta.
    """
    import analyze_winner_diversity as analisis

    assert evaluacion.indice_desde_configuracion is analisis.indice_desde_configuracion
    assert evaluacion.rejilla_exhaustiva is analisis.rejilla_exhaustiva
    assert evaluacion.FRACCION_OBJETIVO == analisis.FRACCION_OBJETIVO

    valores = np.arange(144, dtype=float)
    assert valores[evaluacion.indice_desde_configuracion(CONFIG_A)] == 0.0
