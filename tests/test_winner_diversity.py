# tests/test_winner_diversity.py
"""Pruebas del análisis exploratorio de diversidad de los ganadores.

El análisis no cambia el comportamiento de la librería: produce números que
después sostienen decisiones de diseño del recomendador y quedan citados en el
informe. El riesgo, por tanto, no es que reviente sino que **cuente mal en
silencio**, y esa es la clase de fallo que estas pruebas persiguen.

La traducción entre el índice de un pipeline en la rejilla y su configuración por
paso es la parte más frágil, porque reproduce por fuera el orden que
``itertools.product`` impone dentro del motor de búsqueda. Se verifica contra el
propio motor y en ambos sentidos: si alguien reordena los pasos del espacio, la
correspondencia se rompe y la prueba cae.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

pytest.importorskip("openml", reason="el análisis importa el descargador del corpus")

import analyze_winner_diversity as analisis  # noqa: E402

from autopipe.meta_knowledge import MetaKnowledgeBase, MetaKnowledgeEntry  # noqa: E402
from autopipe.search_engine import GridSearchEngine  # noqa: E402
from autopipe.search_space import (  # noqa: E402
    TABULAR_SPACE_STEPS,
    build_tabular_search_space,
    pipeline_from_configuration,
)

N_PIPELINES = 144
"""Tamaño del espacio canónico contando la omisión de cada paso."""


def entrada(
    dataset_id: str,
    configuracion: dict[str, int],
    meta_features: dict[str, float],
    metric_value: float = 0.5,
) -> MetaKnowledgeEntry:
    """Arma una entrada mínima de la base.

    Args:
        dataset_id: Identificador del dataset.
        configuracion: Índice ganador por paso.
        meta_features: Vector de meta-características.
        metric_value: Valor de la métrica del ganador.

    Returns:
        La entrada construida.
    """
    return MetaKnowledgeEntry(
        dataset_id=dataset_id,
        dataset_name=f"d{dataset_id}",
        meta_features=meta_features,
        configuration=configuracion,
        pipeline_params={},
        metric="macro_f1",
        metric_value=metric_value,
        seed=42,
        n_trials=50,
        n_failed_trials=0,
    )


# --------------------------------------------------------------------------
# Traducción entre índice de la rejilla y configuración
# --------------------------------------------------------------------------


def test_el_indice_decodifica_al_pipeline_que_genera_el_motor() -> None:
    """Cada índice de la rejilla designa el mismo pipeline que el motor construye.

    Es la prueba que sostiene todo el análisis de meseta: si la decodificación se
    desalineara, los valores seguirían siendo números válidos y se atribuirían a
    configuraciones que no son.
    """
    motor = GridSearchEngine(build_tabular_search_space(), task="classification")
    pipelines = motor._build_pipelines()
    assert len(pipelines) == N_PIPELINES
    for indice, pipeline in enumerate(pipelines):
        configuracion = analisis.configuracion_desde_indice(indice)
        reconstruido = pipeline_from_configuration(configuracion)
        assert reconstruido.get_params() == pipeline.get_params()


def test_indice_y_configuracion_son_inversas() -> None:
    """Ir de índice a configuración y volver devuelve el índice original."""
    for indice in range(N_PIPELINES):
        configuracion = analisis.configuracion_desde_indice(indice)
        assert analisis.indice_desde_configuracion(configuracion) == indice


def test_la_decodificacion_cubre_todos_los_pasos() -> None:
    """La configuración decodificada nombra los cuatro pasos del espacio."""
    configuracion = analisis.configuracion_desde_indice(0)
    assert set(configuracion) == set(TABULAR_SPACE_STEPS)
    assert all(valor == 0 for valor in configuracion.values())


# --------------------------------------------------------------------------
# Diversidad estructural
# --------------------------------------------------------------------------


def _base_con_configuraciones(configuraciones: list[dict[str, int]]) -> MetaKnowledgeBase:
    """Arma una base con las configuraciones dadas y meta-características finitas."""
    return MetaKnowledgeBase(
        [
            entrada(str(numero), configuracion, {"a": 1.0, "b": 2.0})
            for numero, configuracion in enumerate(configuraciones)
        ]
    )


def test_cuenta_las_configuraciones_distintas() -> None:
    """Dos entradas con el mismo ganador cuentan como una sola configuración."""
    ganador = {"imputation": 0, "scaling": 1, "encoding": 0, "selection": 3}
    otro = {"imputation": 1, "scaling": 1, "encoding": 0, "selection": 3}
    resultado = analisis.diversidad_estructural(
        _base_con_configuraciones([ganador, dict(ganador), otro])
    )
    assert resultado["n_datasets"] == 3
    assert resultado["n_configuraciones_distintas"] == 2


def test_la_entropia_es_cero_cuando_el_paso_no_varia() -> None:
    """Un paso que siempre gana con el mismo índice tiene entropía nula.

    Es el caso que importa detectar: un paso de entropía nula no distingue un
    dataset de otro, así que dos configuraciones que solo difieran en él son la
    misma recomendación a efectos prácticos.
    """
    configuraciones = [
        {"imputation": 0, "scaling": indice, "encoding": 0, "selection": 0}
        for indice in range(4)
    ]
    resultado = analisis.diversidad_estructural(_base_con_configuraciones(configuraciones))
    assert resultado["pasos"]["imputation"]["entropia_normalizada"] == 0.0
    assert resultado["pasos"]["scaling"]["entropia_normalizada"] == 1.0


def test_la_distancia_de_hamming_cuenta_los_pasos_que_difieren() -> None:
    """Dos ganadores que difieren en un solo paso quedan a distancia uno."""
    uno = {"imputation": 0, "scaling": 0, "encoding": 0, "selection": 0}
    dos = {"imputation": 0, "scaling": 1, "encoding": 0, "selection": 0}
    resultado = analisis.diversidad_estructural(_base_con_configuraciones([uno, dos]))
    assert resultado["n_pares"] == 1
    assert resultado["hamming_histograma"][1] == 1
    assert resultado["hamming_media"] == 1.0


# --------------------------------------------------------------------------
# Estabilidad de las meta-características supervivientes
# --------------------------------------------------------------------------


CONFIG = {"imputation": 0, "scaling": 0, "encoding": 0, "selection": 0}


def test_sin_no_finitas_ningun_pliegue_difiere() -> None:
    """Si todo es finito, la regla por pliegue y la del corpus coinciden."""
    base = MetaKnowledgeBase(
        [entrada(str(n), CONFIG, {"a": 1.0, "b": 2.0}) for n in range(3)]
    )
    resultado = analisis.estabilidad_supervivientes(base)
    assert resultado["n_supervivientes_corpus"] == 2
    assert resultado["n_pliegues_que_difieren"] == 0


def test_detecta_el_pliegue_que_conserva_una_columna_de_mas() -> None:
    """Una columna no finita en un solo dataset reaparece al excluirlo.

    Es exactamente la discrepancia entre las dos reglas: el corpus completo
    descarta la columna, y el pliegue que excluye al culpable la conserva. La
    prueba fija además que se cuenta como descartable al intersecar, porque el
    dataset excluido es justo el que la tiene no finita.
    """
    base = MetaKnowledgeBase(
        [
            entrada("0", CONFIG, {"a": 1.0, "b": float("nan")}),
            entrada("1", CONFIG, {"a": 1.0, "b": 2.0}),
            entrada("2", CONFIG, {"a": 1.0, "b": 3.0}),
        ]
    )
    resultado = analisis.estabilidad_supervivientes(base)
    assert resultado["n_supervivientes_corpus"] == 1
    assert resultado["n_pliegues_que_difieren"] == 1
    pliegue = next(p for p in resultado["pliegues"] if p["dataset_id"] == "0")
    assert pliegue["de_mas"] == ["b"]
    assert pliegue["n_descartadas_al_intersecar"] == 1


def test_una_columna_no_finita_en_dos_datasets_no_reaparece() -> None:
    """Excluir a uno de dos culpables no basta para recuperar la columna."""
    base = MetaKnowledgeBase(
        [
            entrada("0", CONFIG, {"a": 1.0, "b": float("inf")}),
            entrada("1", CONFIG, {"a": 1.0, "b": float("nan")}),
            entrada("2", CONFIG, {"a": 1.0, "b": 3.0}),
        ]
    )
    resultado = analisis.estabilidad_supervivientes(base)
    assert resultado["n_pliegues_que_difieren"] == 0


# --------------------------------------------------------------------------
# Anchura de la meseta
# --------------------------------------------------------------------------


def test_la_meseta_separa_el_empate_estructural_del_metrico(monkeypatch) -> None:
    """El empate se clasifica por los pares (escalado, selección) que abarca.

    Se construye una rejilla en la que solo empatan configuraciones que comparten
    escalado y selección: es la degeneración estructural que produce un dataset
    numérico y sin faltantes, donde la imputación y la codificación son la
    identidad. Debe contarse como un único par.
    """
    objetivos = [0.0] * N_PIPELINES
    optimo = {"imputation": 0, "scaling": 2, "encoding": 0, "selection": 1}
    empatadas = [
        {**optimo, "imputation": i, "encoding": e} for i in range(3) for e in range(3)
    ]
    for configuracion in empatadas:
        objetivos[analisis.indice_desde_configuracion(configuracion)] = 0.9

    monkeypatch.setattr(analisis, "rejilla_exhaustiva", lambda *a, **k: objetivos)
    base = MetaKnowledgeBase([entrada("0", optimo, {"a": 1.0}, metric_value=0.9)])
    resultado = analisis.anchura_de_meseta(base)

    dato = resultado["datasets"][0]
    assert dato["n_empatadas_en_el_optimo"] == 9
    assert dato["n_pares_escalado_seleccion_empatados"] == 1
    assert dato["configuracion_de_la_base_es_optima"] is True
    assert resultado["n_empate_solo_estructural"] == 1


def test_la_meseta_cuenta_las_que_alcanzan_el_95_por_ciento(monkeypatch) -> None:
    """En meseta entran las configuraciones que llegan al 95 % del óptimo."""
    objetivos = [0.5] * N_PIPELINES
    ganadora = {"imputation": 1, "scaling": 0, "encoding": 1, "selection": 2}
    objetivos[analisis.indice_desde_configuracion(ganadora)] = 1.0
    # 0.96 alcanza el umbral de 0.95; 0.94 no.
    objetivos[0], objetivos[1] = 0.96, 0.94

    monkeypatch.setattr(analisis, "rejilla_exhaustiva", lambda *a, **k: objetivos)
    base = MetaKnowledgeBase([entrada("0", ganadora, {"a": 1.0}, metric_value=1.0)])
    resultado = analisis.anchura_de_meseta(base)

    assert resultado["datasets"][0]["n_en_meseta"] == 2
    assert resultado["datasets"][0]["n_empatadas_en_el_optimo"] == 1
    assert resultado["n_con_optimo_unico"] == 1


def test_los_inevaluables_no_cuentan_ni_ganan(monkeypatch) -> None:
    """Un pipeline con métrica ``NaN`` ni entra en la meseta ni resulta óptimo."""
    objetivos = [float("nan")] * N_PIPELINES
    ganadora = {"imputation": 0, "scaling": 0, "encoding": 0, "selection": 0}
    objetivos[analisis.indice_desde_configuracion(ganadora)] = 0.7

    monkeypatch.setattr(analisis, "rejilla_exhaustiva", lambda *a, **k: objetivos)
    base = MetaKnowledgeBase([entrada("0", ganadora, {"a": 1.0}, metric_value=0.7)])
    resultado = analisis.anchura_de_meseta(base)

    dato = resultado["datasets"][0]
    assert dato["n_evaluables"] == 1
    assert dato["n_en_meseta"] == 1
    assert dato["optimo_rejilla"] == pytest.approx(0.7)


def test_detecta_que_la_base_no_guarda_el_optimo(monkeypatch) -> None:
    """Si el ganador guardado no alcanza el óptimo de la rejilla, se reporta.

    Es la comprobación de que 50 ensayos de Optuna no garantizan el óptimo del
    espacio: la base puede guardar una aproximación, y el recomendador estaría
    transfiriendo esa aproximación.
    """
    objetivos = [0.1] * N_PIPELINES
    guardada = {"imputation": 0, "scaling": 0, "encoding": 0, "selection": 0}
    mejor = {"imputation": 2, "scaling": 3, "encoding": 2, "selection": 3}
    objetivos[analisis.indice_desde_configuracion(guardada)] = 0.6
    objetivos[analisis.indice_desde_configuracion(mejor)] = 0.8

    monkeypatch.setattr(analisis, "rejilla_exhaustiva", lambda *a, **k: objetivos)
    base = MetaKnowledgeBase([entrada("0", guardada, {"a": 1.0}, metric_value=0.6)])
    resultado = analisis.anchura_de_meseta(base)

    dato = resultado["datasets"][0]
    assert dato["configuracion_de_la_base_es_optima"] is False
    assert dato["brecha"] == pytest.approx(0.2)
    assert resultado["n_base_es_optima"] == 0


# --------------------------------------------------------------------------
# Canonización de los empates
#
# La regla en sí vive en la librería y se prueba en tests/test_search_space.py.
# Aquí se prueba solo cómo la aplica el análisis sobre el corpus.
# --------------------------------------------------------------------------


def test_el_impacto_de_canonizar_cuenta_las_entradas_que_cambian(monkeypatch) -> None:
    """Se reportan las entradas que cambian y las omisiones que desaparecen."""
    guardada = {"imputation": 2, "scaling": 0, "encoding": 2, "selection": 0}
    alternativa = {"imputation": 0, "scaling": 0, "encoding": 0, "selection": 0}
    monkeypatch.setattr(
        analisis,
        "empates_en_el_presupuesto",
        lambda *a, **k: [guardada, alternativa],
    )
    base = MetaKnowledgeBase([entrada("0", guardada, {"a": 1.0})])
    resultado = analisis.impacto_de_canonizar(base)

    assert resultado["n_cambian"] == 1
    assert resultado["omiten_imputacion_actual"] == 1
    assert resultado["omiten_imputacion_canonica"] == 0
    assert resultado["omiten_codificacion_actual"] == 1
    assert resultado["omiten_codificacion_canonica"] == 0


def test_sin_empate_la_canonizacion_no_cambia_nada(monkeypatch) -> None:
    """Si el ganador guardado es el único con su valor, canonizar lo respeta."""
    guardada = {"imputation": 1, "scaling": 2, "encoding": 0, "selection": 1}
    monkeypatch.setattr(analisis, "empates_en_el_presupuesto", lambda *a, **k: [guardada])
    base = MetaKnowledgeBase([entrada("0", guardada, {"a": 1.0})])
    resultado = analisis.impacto_de_canonizar(base)

    assert resultado["n_cambian"] == 0
    assert resultado["n_distintas_actual"] == resultado["n_distintas_canonica"] == 1


def test_los_empates_del_presupuesto_se_leen_de_la_cache(tmp_path, monkeypatch) -> None:
    """Con los empates cacheados no se descarga ni se vuelve a buscar."""

    def _explota(*_args, **_kwargs):
        raise AssertionError("no debía descargarse el dataset")

    monkeypatch.setattr(analisis, "descargar", _explota)
    monkeypatch.setattr(analisis, "CACHE_ENSAYOS", tmp_path)
    empatadas = [{"imputation": 0, "scaling": 1, "encoding": 0, "selection": 2}]
    analisis._cache_ensayos("77").write_text(
        json.dumps({"dataset_id": "77", "empatadas": empatadas}), encoding="utf-8"
    )
    assert analisis.empates_en_el_presupuesto("77") == empatadas


def test_el_informe_nombra_los_hallazgos(monkeypatch) -> None:
    """El documento generado incluye los tres análisis y sus cifras."""
    objetivos = [0.5] * N_PIPELINES
    ganadora = {"imputation": 0, "scaling": 0, "encoding": 0, "selection": 0}
    objetivos[analisis.indice_desde_configuracion(ganadora)] = 1.0

    monkeypatch.setattr(analisis, "rejilla_exhaustiva", lambda *a, **k: objetivos)
    monkeypatch.setattr(analisis, "empates_en_el_presupuesto", lambda *a, **k: [ganadora])
    base = MetaKnowledgeBase(
        [
            entrada("0", ganadora, {"a": 1.0, "b": float("nan")}, metric_value=1.0),
            entrada("1", ganadora, {"a": 2.0, "b": 1.0}, metric_value=1.0),
        ]
    )
    texto = analisis._informe(
        analisis.diversidad_estructural(base),
        analisis.anchura_de_meseta(base),
        analisis.estabilidad_supervivientes(base),
        analisis.impacto_de_canonizar(base),
    )
    assert "Forma de la diversidad" in texto
    assert "Anchura de la meseta" in texto
    assert "Estabilidad de las meta-características supervivientes" in texto
    assert "Las dos reglas no coinciden" in texto
    assert "Efecto piso" in texto
    assert "Desempate entre configuraciones que empatan" in texto
    assert "La base ya es canónica" in texto
    assert "exploratorio" in texto
    for linea in texto.splitlines():
        if not linea.startswith("|"):
            assert len(linea) <= 100, f"línea larga en el informe: {linea!r}"


def test_el_informe_avisa_si_la_base_no_es_canonica(monkeypatch) -> None:
    """Una base con empates sin desempatar se reporta como tal, no en silencio.

    El desempate lo aplica el arnés de construcción y el informe lo comprueba por
    separado sobre el artefacto guardado. Si los dos se separaran, la base
    mezclaría dos criterios sin que nada fallara, y este aviso es lo único que
    lo haría visible.
    """
    objetivos = [0.5] * N_PIPELINES
    guardada = {"imputation": 2, "scaling": 0, "encoding": 2, "selection": 0}
    mejor = {"imputation": 0, "scaling": 0, "encoding": 0, "selection": 0}
    monkeypatch.setattr(analisis, "rejilla_exhaustiva", lambda *a, **k: objetivos)
    monkeypatch.setattr(
        analisis, "empates_en_el_presupuesto", lambda *a, **k: [guardada, mejor]
    )
    base = MetaKnowledgeBase([entrada("0", guardada, {"a": 1.0}, metric_value=0.5)])
    texto = analisis._informe(
        analisis.diversidad_estructural(base),
        analisis.anchura_de_meseta(base),
        analisis.estabilidad_supervivientes(base),
        analisis.impacto_de_canonizar(base),
    )
    assert "La base no es canónica" in texto
    assert "build_meta_knowledge.py" in texto


def test_el_informe_declara_la_coincidencia_cuando_no_hay_discrepancia(
    monkeypatch,
) -> None:
    """Sin pliegues discrepantes, el informe lo dice de forma explícita."""
    objetivos = [0.5] * N_PIPELINES
    ganadora = {"imputation": 0, "scaling": 0, "encoding": 0, "selection": 0}
    monkeypatch.setattr(analisis, "rejilla_exhaustiva", lambda *a, **k: objetivos)
    monkeypatch.setattr(analisis, "empates_en_el_presupuesto", lambda *a, **k: [ganadora])
    base = MetaKnowledgeBase(
        [entrada(str(n), ganadora, {"a": float(n + 1)}, metric_value=0.5) for n in range(2)]
    )
    texto = analisis._informe(
        analisis.diversidad_estructural(base),
        analisis.anchura_de_meseta(base),
        analisis.estabilidad_supervivientes(base),
        analisis.impacto_de_canonizar(base),
    )
    assert "las dos reglas coinciden" in texto


def test_la_rejilla_se_lee_de_la_cache(tmp_path, monkeypatch) -> None:
    """Con la rejilla cacheada no se descarga ni se evalúa nada.

    Lo verifica sustituyendo el descargador por uno que falla: si se llegara a
    invocar, la prueba caería.
    """

    def _explota(*_args, **_kwargs):
        raise AssertionError("no debía descargarse el dataset")

    monkeypatch.setattr(analisis, "descargar", _explota)
    monkeypatch.setattr(analisis, "CACHE_REJILLA", tmp_path)
    objetivos = [float(v) for v in np.linspace(0.0, 1.0, N_PIPELINES)]
    analisis._cache_rejilla("77").write_text(
        json.dumps({"dataset_id": "77", "dataset_name": "d", "objetivos": objetivos}),
        encoding="utf-8",
    )
    assert analisis.rejilla_exhaustiva("77", "d") == pytest.approx(objetivos)
