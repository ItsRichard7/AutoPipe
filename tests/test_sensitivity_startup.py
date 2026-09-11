# tests/test_sensitivity_startup.py
"""Pruebas de la sensibilidad del experimento 2 a ``n_startup_trials``.

La propiedad central es que **lo único que cambia es el parámetro**. Se protege desde tres
lados: con 10, la función de ejecución reproduce las curvas del arnés ensayo por ensayo;
las medidas derivadas de una curva coinciden con las que el experimento 2 guardó; y las
configuraciones inyectadas se leen de ese experimento en vez de recalcularse.

Las demás protegen lo que el spike A advirtió ---que el efecto se mida también en el
frío--- y el signo del beneficio.
"""

from __future__ import annotations

import inspect
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
SCRIPTS = RAIZ / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

pytest.importorskip("openml", reason="la sensibilidad importa el arnés del experimento 2")

import run_lodo_e2 as arnes  # noqa: E402
import run_sensitivity_startup as sensibilidad  # noqa: E402

from autopipe.search_engine import OptunaSearchEngine  # noqa: E402

CONFIG_A = {"imputation": 0, "scaling": 0, "encoding": 0, "selection": 0}
CONFIG_B = {"imputation": 1, "scaling": 1, "encoding": 1, "selection": 1}
SEMILLAS = (101, 202, 303)


def _iguales(una: list[float], otra: list[float]) -> bool:
    """Compara dos curvas tratando dos NaN en la misma posición como iguales."""
    return len(una) == len(otra) and all(
        a == b or (math.isnan(a) and math.isnan(b)) for a, b in zip(una, otra)
    )


def _medida(regret: float, ensayos: int = 5) -> dict:
    return {
        "curva": [0.9],
        "regret": regret,
        "ensayos": ensayos,
        "mejor": 0.9,
        "a_presupuesto_reducido": {"5": 0.9, "15": 0.9, "30": 0.9},
        "fraccion_inevaluable": 0.0,
    }


def _pliegue(nombre: str, regrets: dict[str, float], ensayos: int = 5) -> dict:
    """Pliegue con la forma del experimento 2 y el mismo regret en todas las semillas."""
    return {
        "dataset_id": nombre,
        "dataset_name": f"d{nombre}",
        "recomendaciones": {
            "nearest": {"configuraciones": [CONFIG_A], "vecinos": []},
            "frequent": {"configuraciones": [CONFIG_B], "vecinos": []},
        },
        "semillas": [
            {"semilla": s, "optimo": 1.0, "objetivo": 0.99,
             **{c: _medida(r, ensayos) for c, r in regrets.items()}}
            for s in SEMILLAS
        ],
    }


def _e2(pliegues: list[dict]) -> dict:
    """Documento del experimento 2 con sus confirmatorios calculados por el arnés."""
    return {
        "pliegues": pliegues,
        "analisis": {
            "confirmatorios": arnes.holm(
                [arnes.contrastar(pliegues, c, r) for c, r in arnes.CONFIRMATORIOS]
            )
        },
    }


# --------------------------------------------------------------------------- #
# Los valores y las condiciones
# --------------------------------------------------------------------------- #
def test_the_values_are_the_ones_the_spike_fixed() -> None:
    """Prueba por sabotaje de los dos valores.

    El 10 no es una elección del script: es el valor por defecto del motor, con el que se
    ejecutó el experimento 2. Si alguien cambiara el motor, la «sensibilidad» dejaría de
    comparar contra lo que realmente se ejecutó.
    """
    por_defecto = inspect.signature(OptunaSearchEngine).parameters["n_startup_trials"].default

    assert sensibilidad.N_STARTUP_SENSIBILIDAD == 5
    assert sensibilidad.N_STARTUP_PRIMARIO == 10
    assert por_defecto == sensibilidad.N_STARTUP_PRIMARIO


def test_the_conditions_cover_both_confirmatory_contrasts_and_the_cold_start() -> None:
    """El spike A exigió medir el efecto también en el frío."""
    assert sensibilidad.CONDICIONES[0] == "frio"
    for condicion, referencia in arnes.CONFIRMATORIOS:
        assert condicion in sensibilidad.CONDICIONES
        assert referencia in sensibilidad.CONDICIONES


# --------------------------------------------------------------------------- #
# Lo único que cambia es el parámetro
# --------------------------------------------------------------------------- #
def _tabla(filas: int = 40) -> tuple[pd.DataFrame, np.ndarray]:
    """Tabla sintética pequeña, con señal, ruido y una columna categórica."""
    rng = np.random.default_rng(7)
    clase = np.tile([0, 1], filas // 2)
    tabla = pd.DataFrame(
        {
            "senal": clase * 2.0 + rng.normal(0, 0.5, size=filas),
            "ruido": rng.normal(0, 1.0, size=filas),
            "cat": np.where(rng.random(filas) < 0.8, clase, 1 - clase).astype(str),
        }
    )
    return tabla, clase


@pytest.mark.parametrize("inyectadas", [None, [CONFIG_A, CONFIG_B]])
def test_with_ten_it_reproduces_the_harness_trial_by_trial(inyectadas) -> None:
    """Prueba de equivalencia con el arnés que produjo el experimento 2."""
    features, labels = _tabla()

    propia = sensibilidad.ejecutar_condicion(features, labels, 101, inyectadas, 10, 12)
    del_arnes = arnes.ejecutar_condicion(features, labels, 101, inyectadas, 12)

    assert _iguales(propia, del_arnes)


def test_the_engine_receives_the_sensitivity_value(monkeypatch: pytest.MonkeyPatch) -> None:
    registro: list[dict] = []

    class MotorFalso:
        def __init__(self, espacio: object, **opciones: object) -> None:
            registro.append(opciones)
            self.n_trials = opciones["n_trials"]

        def run(self, *args: object, **kwargs: object) -> tuple:
            return None, None, [{"objective": 0.5}] * self.n_trials

    monkeypatch.setattr(sensibilidad, "OptunaSearchEngine", MotorFalso)

    curva = sensibilidad.ejecutar_condicion(None, np.array([0, 1]), 101, [CONFIG_A], 5, 7)

    assert registro[0]["n_startup_trials"] == 5
    assert registro[0]["n_trials"] == 7
    assert registro[0]["warm_start"] == [CONFIG_A]
    assert curva == [0.5] * 7


@pytest.mark.skipif(
    not (RAIZ / sensibilidad.RESULTADOS_E2).is_file(),
    reason="requiere los resultados del experimento 2",
)
def test_the_measures_match_what_the_harness_stored() -> None:
    """Las medidas derivadas de una curva guardada son exactamente las guardadas."""
    e2 = json.loads((RAIZ / sensibilidad.RESULTADOS_E2).read_text(encoding="utf-8"))
    semilla = e2["pliegues"][0]["semillas"][0]

    for condicion in sensibilidad.CONDICIONES:
        guardada = semilla[condicion]
        medida = sensibilidad.medir(
            guardada["curva"], semilla["optimo"], semilla["objetivo"], arnes.PRESUPUESTO
        )
        for clave in ("regret", "ensayos", "mejor", "a_presupuesto_reducido",
                      "fraccion_inevaluable"):
            assert medida[clave] == guardada[clave], (condicion, clave)


def test_the_injected_configurations_are_read_from_experiment_two(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llamadas: list[tuple] = []

    def falsa(features, labels, semilla, inyectadas, n_startup, presupuesto):
        llamadas.append((semilla, inyectadas, n_startup, presupuesto))
        return [0.9] * presupuesto

    monkeypatch.setattr(sensibilidad, "ejecutar_condicion", falsa)
    pliegue_e2 = _pliegue("a", {"frio": 0.1, "nearest": 0.05, "frequent": 0.04})

    pliegue = sensibilidad.ejecutar_pliegue(pliegue_e2, None, None, presupuesto=4)

    inyectadas = [llamada[1] for llamada in llamadas[:3]]
    assert inyectadas == [None, [CONFIG_A], [CONFIG_B]]
    assert {llamada[2] for llamada in llamadas} == {5}
    assert [s["semilla"] for s in pliegue["semillas"]] == list(SEMILLAS)
    assert pliegue["semillas"][0]["objetivo"] == 0.99


# --------------------------------------------------------------------------- #
# El pareo
# --------------------------------------------------------------------------- #
def test_a_different_target_is_refused() -> None:
    diez = [_pliegue("a", {"frio": 0.1, "nearest": 0.05, "frequent": 0.04})]
    cinco = [_pliegue("a", {"frio": 0.1, "nearest": 0.05, "frequent": 0.04})]
    cinco[0]["semillas"][1]["objetivo"] = 0.98

    with pytest.raises(ValueError, match="no se comparan objetivos distintos"):
        sensibilidad.combinar(diez, cinco)


def test_a_seed_missing_from_experiment_two_is_refused() -> None:
    diez = [_pliegue("a", {"frio": 0.1, "nearest": 0.05, "frequent": 0.04})]
    cinco = [_pliegue("a", {"frio": 0.1, "nearest": 0.05, "frequent": 0.04})]
    cinco[0]["semillas"][0]["semilla"] = 999

    with pytest.raises(KeyError, match="999"):
        sensibilidad.combinar(diez, cinco)


# --------------------------------------------------------------------------- #
# El análisis
# --------------------------------------------------------------------------- #
def _caso(n: int = 8) -> tuple[list[dict], dict]:
    """Con 5 el frío mejora poco y los calientes mucho, distinto en cada conjunto."""
    diez = [
        _pliegue(str(i), {"frio": 0.10 + 0.01 * i, "nearest": 0.06 + 0.005 * i,
                          "frequent": 0.05 + 0.004 * i})
        for i in range(n)
    ]
    cinco = [
        _pliegue(str(i), {"frio": 0.09 + 0.01 * i, "nearest": 0.02 + 0.001 * i,
                          "frequent": 0.03 + 0.002 * i})
        for i in range(n)
    ]
    return cinco, _e2(diez)


def test_the_confirmatory_contrasts_are_recomputed_with_holm() -> None:
    cinco, e2 = _caso()

    analisis = sensibilidad.analizar(cinco, e2)

    pares = [(c["condicion"], c["referencia"]) for c in analisis["confirmatorios_5"]]
    assert pares == list(arnes.CONFIRMATORIOS)
    assert all("p_ajustado" in c and "cumple_criterio" in c for c in analisis["confirmatorios_5"])


def test_the_effect_is_measured_in_every_arm_including_the_cold_start() -> None:
    cinco, e2 = _caso()

    brazos = {b["condicion"]: b for b in sensibilidad.analizar(cinco, e2)["por_brazo"]}

    assert list(brazos) == ["frio", "nearest", "frequent"]
    assert brazos["frio"]["gana"] == 8
    assert brazos["frio"]["mediana_5"] < brazos["frio"]["mediana_10"]


def test_the_benefit_is_cold_minus_the_condition_each_with_its_own_cold_start() -> None:
    """Prueba por sabotaje del signo y del pareo del frío.

    Con 10, `nearest` ahorra 0,04 + 0,005 i; con 5, 0,07 + 0,009 i. El beneficio es
    positivo y crece en los ocho conjuntos. Con la resta al revés, o restando el frío de
    10 a la condición con 5, las cifras serían otras.
    """
    cinco, e2 = _caso()

    beneficios = {b["condicion"]: b for b in sensibilidad.analizar(cinco, e2)["beneficio"]}
    nearest = beneficios["nearest"]

    assert "frio" not in beneficios
    assert nearest["beneficio_mediano_10"] == pytest.approx(0.04 + 0.005 * 3.5)
    assert nearest["beneficio_mediano_5"] == pytest.approx(0.07 + 0.009 * 3.5)
    assert nearest["crece"] == 8


def test_the_contrasts_with_ten_are_checked_against_experiment_two() -> None:
    cinco, e2 = _caso()

    assert sensibilidad.analizar(cinco, e2)["coincide_con_e2"] is True

    e2["analisis"]["confirmatorios"][0]["p_ajustado"] = 0.5
    assert sensibilidad.analizar(cinco, e2)["coincide_con_e2"] is False


def test_a_partial_run_is_compared_only_on_its_datasets() -> None:
    cinco, e2 = _caso()

    analisis = sensibilidad.analizar(cinco[:6], e2)

    assert analisis["n_conjuntos"] == 6
    assert analisis["coincide_con_e2"] is None


def test_the_censored_runs_are_counted_in_both_versions() -> None:
    diez = [_pliegue(str(i), {"frio": 0.1 + 0.01 * i, "nearest": 0.05, "frequent": 0.04},
                     ensayos=61) for i in range(4)]
    cinco = [_pliegue(str(i), {"frio": 0.1 + 0.01 * i, "nearest": 0.05, "frequent": 0.04})
             for i in range(4)]

    censurados = sensibilidad.analizar(cinco, _e2(diez))["censurados"]

    assert censurados["frio"] == {"con_10": 12, "con_5": 0}


# --------------------------------------------------------------------------- #
# El documento y el punto de entrada
# --------------------------------------------------------------------------- #
def _resultado() -> tuple[dict, dict]:
    cinco, e2 = _caso()
    return {
        "n_startup_trials": 5,
        "condiciones": list(sensibilidad.CONDICIONES),
        "pliegues": cinco,
        "minutos": 1.5,
        "analisis": sensibilidad.analizar(cinco, e2),
    }, e2


def test_the_document_says_it_does_not_replace_the_confirmatory_contrasts() -> None:
    texto = sensibilidad.redactar(_resultado()[0])

    assert "No\n> sustituye" in texto
    assert "también en el frío" in texto
    assert "cifra por cifra" in texto


def test_the_document_puts_both_values_side_by_side_with_derived_verdicts() -> None:
    resultado, _ = _resultado()

    texto = sensibilidad.redactar(resultado)

    assert "| `nearest` vs `frio` | 10 |" in texto
    assert "| `nearest` vs `frio` | 5 |" in texto
    # El veredicto lo deriva la función del reporte del experimento 2, no una copia.
    for contraste in resultado["analisis"]["confirmatorios_5"]:
        assert sensibilidad.redactar_veredicto(contraste) in texto


def test_it_refuses_to_run_without_experiment_two(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sensibilidad, "RESULTADOS_E2", tmp_path / "no-existe.json")
    monkeypatch.setattr(sys, "argv", ["run_sensitivity_startup.py"])

    with pytest.raises(SystemExit, match="run_lodo_e2"):
        sensibilidad.main()


def test_the_report_alone_needs_the_sensitivity_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    e2 = tmp_path / "e2.json"
    e2.write_text(json.dumps(_resultado()[1]), encoding="utf-8")
    monkeypatch.setattr(sensibilidad, "RESULTADOS_E2", e2)
    monkeypatch.setattr(sys, "argv", ["x", "--solo-reporte",
                                      "--resultados", str(tmp_path / "no-existe.json")])

    with pytest.raises(SystemExit, match="sin --solo-reporte"):
        sensibilidad.main()


def test_it_runs_analyses_and_writes_both_documents(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, e2 = _caso()
    ruta_e2 = tmp_path / "e2.json"
    ruta_e2.write_text(json.dumps(e2), encoding="utf-8")

    def falsa(features, labels, semilla, inyectadas, n_startup, presupuesto):
        valor = 0.8 if inyectadas is None else 0.95
        return [valor] * presupuesto

    monkeypatch.setattr(sensibilidad, "RESULTADOS_E2", ruta_e2)
    monkeypatch.setattr(sensibilidad, "ejecutar_condicion", falsa)
    monkeypatch.setattr(sensibilidad, "_cargar_de_openml", lambda i: (None, None))
    monkeypatch.setattr(sensibilidad, "_silenciar", lambda: None)
    json_salida, md_salida = tmp_path / "s.json", tmp_path / "s.md"
    monkeypatch.setattr(sys, "argv", ["x", "--limit", "3", "--resultados", str(json_salida),
                                      "--resumen", str(md_salida)])

    assert sensibilidad.main() == 0
    guardado = json.loads(json_salida.read_text(encoding="utf-8"))
    assert guardado["analisis"]["n_conjuntos"] == 3
    assert "Sensibilidad del experimento 2" in md_salida.read_text(encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["x", "--solo-reporte", "--resultados", str(json_salida),
                                      "--resumen", str(tmp_path / "otra.md")])
    assert sensibilidad.main() == 0
    assert (tmp_path / "otra.md").read_text(encoding="utf-8") == md_salida.read_text(
        encoding="utf-8"
    )
