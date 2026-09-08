# tests/test_lodo_harness.py
"""Pruebas del arnés de validación *leave-one-dataset-out*.

El arnés produce el número del que depende la conclusión del TFG, y sus errores
posibles no son excepciones: son valores plausibles y equivocados. Un conteo
desplazado en uno, un ensayo inevaluable descontado, un pliegue que se ajusta con
el corpus completo. Nada de eso falla al ejecutarse.

Por eso las pruebas se dividen en dos clases. Las primeras fijan la **definición**
de la variable medida contra curvas construidas a mano, donde la respuesta se
puede calcular mentalmente. Las últimas recorren el protocolo entero sobre un
**corpus sintético de juguete**, comprobando las propiedades que el pre-registro
declara: que el arranque en frío no puede censurarse, que cada pliegue excluye su
dataset, y que las cinco semillas producen cinco observaciones.

Nada aquí toca el corpus real ni la red: el cargador de datasets entra por
parámetro precisamente para eso.
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

pytest.importorskip("openml", reason="el arnés importa el descargador del corpus")

import run_lodo as arnes  # noqa: E402

from autopipe.meta_knowledge import MetaKnowledgeBase, MetaKnowledgeEntry  # noqa: E402
from autopipe.search_space import TABULAR_SPACE_VERSION  # noqa: E402

CONFIG_A = {"imputation": 0, "scaling": 0, "encoding": 0, "selection": 0}
CONFIG_B = {"imputation": 1, "scaling": 1, "encoding": 1, "selection": 1}
CONFIG_C = {"imputation": 0, "scaling": 2, "encoding": 0, "selection": 2}
NAN = float("nan")


# --------------------------------------------------------------------------- #
# La definición de la variable medida
# --------------------------------------------------------------------------- #
def test_the_count_starts_at_one() -> None:
    """El primer ensayo de la búsqueda es el número uno, no el cero.

    Es lo coherente con que la censura valga 31, es decir presupuesto más uno, y
    un desplazamiento de una unidad aquí sesgaría toda la comparación.
    """
    assert arnes.ensayos_hasta_el_objetivo([0.9, 0.5, 0.5], 0.8) == 1
    assert arnes.ensayos_hasta_el_objetivo([0.5, 0.9, 0.5], 0.8) == 2


def test_reaching_exactly_the_target_counts() -> None:
    """El umbral es «al menos», no «más que»."""
    assert arnes.ensayos_hasta_el_objetivo([0.8], 0.8) == 1


def test_never_reaching_the_target_is_censored() -> None:
    assert arnes.ensayos_hasta_el_objetivo([0.1, 0.2, 0.3], 0.8) == arnes.CENSURA


def test_the_censoring_value_can_be_replaced() -> None:
    """El chequeo de robustez recalcula la variable con castigos más severos."""
    for censura in arnes.CENSURAS_ROBUSTEZ:
        assert arnes.ensayos_hasta_el_objetivo([0.1], 0.8, censura) == censura


def test_unevaluable_trials_consume_their_turn() -> None:
    """Un ensayo inevaluable gasta presupuesto y nunca alcanza el objetivo.

    Es la precisión del pre-registro que más fácil sería implementar mal: saltar
    los ``NaN`` en lugar de contarlos daría a ambas condiciones un número menor
    que el real, y a la caliente más que a la fría, porque el frío cae más en la
    región inevaluable.
    """
    assert arnes.ensayos_hasta_el_objetivo([NAN, NAN, 0.9], 0.8) == 3


def test_a_run_that_is_entirely_unevaluable_is_censored() -> None:
    assert arnes.ensayos_hasta_el_objetivo([NAN, NAN], 0.8) == arnes.CENSURA


def test_the_best_at_reduced_budget_ignores_later_trials() -> None:
    """Medida descriptiva del anteproyecto: mejor valor tras quince ensayos."""
    curva = [0.1] * 14 + [0.5] + [0.99] * 15

    assert arnes.mejor_hasta(curva, arnes.PRESUPUESTO_REDUCIDO) == pytest.approx(0.5)
    assert arnes.mejor_hasta(curva, arnes.PRESUPUESTO) == pytest.approx(0.99)


def test_the_unevaluable_fraction_is_a_covariate() -> None:
    """Primera covariable declarada en el pre-registro v1.0."""
    assert arnes.fraccion_inevaluable([NAN, 0.5, NAN, 0.5]) == pytest.approx(0.5)
    assert arnes.fraccion_inevaluable([0.5, 0.5]) == 0.0


# --------------------------------------------------------------------------- #
# Agregación y contraste
# --------------------------------------------------------------------------- #
def _pliegue(nombre: str, frio: list[int], nearest: list[int]) -> dict:
    """Arma un pliegue sintético cuyas curvas dan los ensayos indicados.

    Se construye una curva que alcanza el objetivo exactamente en la posición
    pedida, de modo que la prueba pueda razonar sobre números y no sobre
    búsquedas.
    """

    def curva(posicion: int) -> list[float]:
        if posicion > arnes.PRESUPUESTO:
            return [0.1] * arnes.PRESUPUESTO
        return [0.1] * (posicion - 1) + [1.0] * (arnes.PRESUPUESTO - posicion + 1)

    semillas = []
    for indice, semilla in enumerate(arnes.SEMILLAS):
        condiciones = {
            "frio": {
                "curva": curva(frio[indice]),
                "fraccion_inevaluable": 0.0,
                "a_presupuesto_reducido": 1.0,
            }
        }
        for estrategia in arnes.ESTRATEGIAS_CALIENTE:
            posicion = nearest[indice] if estrategia == "nearest" else frio[indice]
            condiciones[estrategia] = {
                "curva": curva(posicion),
                "fraccion_inevaluable": 0.0,
                "a_presupuesto_reducido": 1.0,
            }
        semillas.append({"semilla": semilla, "objetivo": 0.95, **condiciones})
    return {"dataset_id": nombre, "dataset_name": nombre, "semillas": semillas}


def test_the_paired_observation_is_the_median_of_five_seeds() -> None:
    """Una observación por dataset, mediana de las cinco semillas."""
    pliegue = _pliegue("d", frio=[1, 2, 3, 4, 5], nearest=[10, 10, 10, 10, 10])

    assert arnes.observacion_pareada(pliegue, "frio") == 3
    assert arnes.observacion_pareada(pliegue, "nearest") == 10


def test_a_clear_reduction_confirms_the_hypothesis() -> None:
    """Con el caliente sistemáticamente por delante, el criterio se cumple.

    Verifica los dos requisitos de la Sección 4.3.4 del anteproyecto a la vez: la
    reducción de al menos el 30 % y la significancia del Wilcoxon de una cola.
    """
    pliegues = [
        _pliegue(f"d{i}", frio=[10] * 5, nearest=[2] * 5) for i in range(12)
    ]

    resultado = arnes.contrastar(pliegues, "nearest")

    assert resultado["mediana_frio"] == 10
    assert resultado["mediana_caliente"] == 2
    assert resultado["reduccion"] == pytest.approx(0.8)
    assert resultado["p_una_cola"] < 0.05
    assert resultado["confirma"] is True


def test_no_difference_does_not_confirm() -> None:
    """Si las dos condiciones empatan, el criterio no se cumple."""
    pliegues = [_pliegue(f"d{i}", frio=[5] * 5, nearest=[5] * 5) for i in range(12)]

    resultado = arnes.contrastar(pliegues, "nearest")

    assert resultado["reduccion"] == pytest.approx(0.0)
    assert resultado["confirma"] is False


def test_a_reduction_below_thirty_percent_does_not_confirm() -> None:
    """El umbral del anteproyecto es del 30 %, y se respeta aunque haya señal.

    Con 10 contra 8 la reducción es del 20 %: la prueba puede salir
    significativa, y aun así el criterio no se cumple. Es la comprobación de que
    el arnés exige **las dos** condiciones y no solo el valor $p$.
    """
    pliegues = [_pliegue(f"d{i}", frio=[10] * 5, nearest=[8] * 5) for i in range(12)]

    resultado = arnes.contrastar(pliegues, "nearest")

    assert resultado["reduccion"] == pytest.approx(0.2)
    assert resultado["p_una_cola"] < 0.05
    assert resultado["confirma"] is False


def test_both_p_values_are_reported() -> None:
    """La lateralidad no puede ser un grado de libertad oculto."""
    pliegues = [_pliegue(f"d{i}", frio=[10] * 5, nearest=[2] * 5) for i in range(12)]

    resultado = arnes.contrastar(pliegues, "nearest")

    assert resultado["p_una_cola"] < resultado["p_dos_colas"]


def test_the_analysis_recomputes_the_robustness_rules() -> None:
    """El primario se recalcula íntegro con las censuras alternativas."""
    pliegues = [_pliegue(f"d{i}", frio=[10] * 5, nearest=[2] * 5) for i in range(12)]

    analisis = arnes.analizar(pliegues)

    assert [r["censura"] for r in analisis["robustez"]] == list(arnes.CENSURAS_ROBUSTEZ)
    assert [s["estrategia"] for s in analisis["secundarios"]] == ["random", "frequent"]


def test_the_per_dataset_table_carries_both_covariates() -> None:
    """La tabla por dataset lleva las dos covariables declaradas."""
    pliegues = [_pliegue("d0", frio=[10] * 5, nearest=[2] * 5)]

    analisis = arnes.analizar(pliegues, meseta={"d0": 36})

    fila = analisis["por_dataset"][0]
    assert fila["meseta"] == 36
    assert fila["frio_inevaluable"] == 0.0
    assert "frio_a_15" in fila and "nearest_a_15" in fila


# --------------------------------------------------------------------------- #
# El protocolo entero, sobre un corpus de juguete
# --------------------------------------------------------------------------- #
FILAS_JUGUETE = 40
"""Tamaño de cada tabla del corpus de juguete.

Deliberadamente pequeño. Estas tablas existen para ejercitar la **estructura** del
protocolo —pliegues, semillas, condiciones, conteo y agregación—, no para
parecerse a un dataset real, y el costo de la suite crece con el cuadrado de sus
filas porque el modelo base es de vecinos más cercanos por fuerza bruta.
"""


def _tabla(semilla: int, separacion: float) -> tuple[pd.DataFrame, np.ndarray]:
    """Tabla sintética pequeña, con señal, ruido y una columna categórica."""
    rng = np.random.default_rng(semilla)
    clase = np.tile([0, 1], FILAS_JUGUETE // 2)
    tabla = pd.DataFrame(
        {
            "senal": clase * separacion + rng.normal(0, 0.5, size=FILAS_JUGUETE),
            "ruido": rng.normal(0, 1.0, size=FILAS_JUGUETE),
            "cat": np.where(
                rng.random(FILAS_JUGUETE) < 0.8, clase, 1 - clase
            ).astype(str),
        }
    )
    return tabla, clase


@pytest.fixture(scope="module")
def resultado_juguete(corpus_de_juguete) -> dict:
    """Ejecuta el LODO de juguete **una sola vez** para todas las pruebas.

    Recorrer el protocolo entero cuesta 3 datasets x 5 semillas x 4 condiciones x
    30 ensayos, y relanzarlo por prueba multiplicaba por cinco el tiempo de la
    suite sin comprobar nada nuevo: las propiedades estructurales se verifican
    sobre la misma corrida.
    """
    base, datos = corpus_de_juguete
    return arnes.ejecutar_lodo(base, lambda i: datos[i])


@pytest.fixture(scope="module")
def corpus_de_juguete() -> tuple[MetaKnowledgeBase, dict]:
    """Tres datasets sintéticos con sus entradas en la base.

    Tres es el mínimo que permite que cada pliegue conserve dos vecinos, que es
    lo que hace falta para que el recomendador tenga entre qué elegir.
    """
    filas = [("0", 3.0, CONFIG_A), ("1", 2.0, CONFIG_B), ("2", 1.0, CONFIG_C)]
    entradas = [
        MetaKnowledgeEntry(
            dataset_id=i,
            dataset_name=f"juguete{i}",
            meta_features={"a": sep, "b": sep * 2},
            configuration=cfg,
            pipeline_params={},
            metric="macro_f1",
            metric_value=0.8,
            seed=42,
            n_trials=50,
            n_failed_trials=0,
            space_version=TABULAR_SPACE_VERSION,
        )
        for i, sep, cfg in filas
    ]
    datos = {i: _tabla(int(i), sep) for i, sep, _ in filas}
    return MetaKnowledgeBase(entradas), datos


def test_the_whole_protocol_runs_on_a_toy_corpus(corpus_de_juguete, resultado_juguete) -> None:
    """Recorre el LODO entero: pliegues, semillas, condiciones y análisis.

    No comprueba ningún valor concreto, porque no hay ninguno que deba salir: lo
    que fija son las propiedades estructurales que el pre-registro declara.
    """
    base, _ = corpus_de_juguete
    resultado = resultado_juguete

    assert len(resultado["pliegues"]) == len(base)
    for pliegue in resultado["pliegues"]:
        assert len(pliegue["semillas"]) == len(arnes.SEMILLAS)
        for semilla in pliegue["semillas"]:
            for condicion in ("frio", *arnes.ESTRATEGIAS_CALIENTE):
                assert len(semilla[condicion]["curva"]) == arnes.PRESUPUESTO
    assert resultado["analisis"]["primario"]["n_pares"] == len(base)


def test_the_cold_start_can_never_be_censored(resultado_juguete) -> None:
    """La propiedad que el pre-registro demuestra, comprobada sobre datos reales.

    El objetivo es el 95 % del mejor valor que el propio frío alcanza, de modo
    que cruzarlo no es probable sino inevitable. Si alguna corrida lo violara
    sería un error de implementación, y el arnés aborta en vez de seguir.
    """
    for pliegue in resultado_juguete["pliegues"]:
        for semilla in pliegue["semillas"]:
            ensayos = arnes.ensayos_hasta_el_objetivo(
                semilla["frio"]["curva"], semilla["objetivo"]
            )
            assert ensayos < arnes.CENSURA


def test_a_censored_cold_start_aborts(corpus_de_juguete, monkeypatch) -> None:
    """Si esa propiedad se violara, el arnés falla ruidosamente.

    Se fuerza el caso imposible devolviendo una curva enteramente inevaluable, y
    se comprueba que el arnés lo trate como error y no como resultado.
    """
    base, datos = corpus_de_juguete
    monkeypatch.setattr(arnes, "ejecutar_condicion", lambda *a, **k: [NAN] * arnes.PRESUPUESTO)

    with pytest.raises(RuntimeError, match="imposible por construcción"):
        arnes.ejecutar_lodo(base, lambda i: datos[i])


def test_each_fold_excludes_its_own_dataset(resultado_juguete) -> None:
    """Ausencia de fuga en el nivel meta, comprobada pliegue a pliegue.

    Ninguna de las configuraciones recomendadas puede provenir del dataset que el
    pliegue excluye. Es la garantía que el pre-registro exige y la que, de
    romperse, inflaría la ventaja del arranque en caliente sin que nada fallara.
    """
    for pliegue in resultado_juguete["pliegues"]:
        propio = pliegue["dataset_name"]
        for recomendacion in pliegue["recomendaciones"].values():
            assert propio not in recomendacion["vecinos"]


def test_every_declared_strategy_is_executed(resultado_juguete) -> None:
    """Las tres condiciones calientes del pre-registro se ejecutan."""
    for pliegue in resultado_juguete["pliegues"]:
        assert set(pliegue["recomendaciones"]) == set(arnes.ESTRATEGIAS_CALIENTE)


def test_a_fold_is_reproducible(corpus_de_juguete) -> None:
    """Criterio de reproducibilidad del anteproyecto, Sección 4.3.5.

    Se comprueba sobre **un pliegue** y no sobre el LODO entero: la
    determinación es la misma y cuesta un tercio, que en una prueba que
    necesariamente ejecuta dos veces sí se nota.

    No se puede recortar el corpus de juguete a dos datasets para abaratarla,
    porque entonces cada pliegue dejaría un solo vecino y el recomendador
    rechaza —con razón— una base cuyas meta-características no varían.
    """
    base, datos = corpus_de_juguete
    entrada = base.entries[0]
    features, labels = datos[entrada.dataset_id]

    corridas = [
        arnes.ejecutar_pliegue(
            entrada.dataset_id,
            entrada.dataset_name,
            base,
            entrada.meta_features,
            features,
            labels,
        )
        for _ in range(2)
    ]

    assert corridas[0] == corridas[1]


# --------------------------------------------------------------------------- #
# Constantes del protocolo
# --------------------------------------------------------------------------- #
def test_the_protocol_constants_match_the_pre_registration() -> None:
    """Fija los números que la v1.0 congela.

    Si alguien los cambiara, el arnés dejaría de medir lo pre-registrado sin que
    ninguna prueba de comportamiento se enterara.
    """
    assert arnes.SEMILLAS == (101, 202, 303, 404, 505)
    assert arnes.PRESUPUESTO == 30
    assert arnes.CENSURA == 31
    assert arnes.CENSURAS_ROBUSTEZ == (60, 300)
    assert arnes.FRACCION_OBJETIVO == 0.95
    assert arnes.PRESUPUESTO_REDUCIDO == 15
    assert arnes.N_CONFIGURACIONES == 5
    assert arnes.ESTRATEGIAS_CALIENTE[0] == "nearest"


def test_the_plateau_covariate_is_read_from_the_diversity_analysis(tmp_path) -> None:
    """La segunda covariable se lee del análisis de la semana 7, no se recalcula."""
    ruta = tmp_path / "DIVERSIDAD-valores.json"
    ruta.write_text(
        '{"meseta": {"datasets": [{"dataset_id": "40983", "n_en_meseta": 18}]}}',
        encoding="utf-8",
    )

    assert arnes.leer_meseta(ruta) == {"40983": 18}


def test_a_missing_diversity_file_leaves_the_covariate_empty(tmp_path, caplog) -> None:
    """Sin el análisis, la covariable queda vacía y se avisa, no se inventa."""
    with caplog.at_level("WARNING"):
        assert arnes.leer_meseta(tmp_path / "no-existe.json") == {}

    assert "quedará vacía" in caplog.text
