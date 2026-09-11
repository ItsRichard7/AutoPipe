# tests/test_by_dataset.py
"""Pruebas de la tabla por conjunto de datos y de la relación entre meseta y beneficio.

Protegen las tres exigencias del pre-registro que un resumen de medianas pierde en
silencio: que los conjuntos planos se vean **en su propia fila con su sensibilidad**, que
la censura sea **la de cada experimento** y que un pipeline vacío inevaluable no aparezca
como un margen nulo. Y, en la parte exploratoria, el **signo** del beneficio.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

pytest.importorskip("openml", reason="la tabla importa los arneses de los experimentos")

import tabulate_by_dataset as tabla  # noqa: E402

E1, E2 = tabla.EXPERIMENTO_1, tabla.EXPERIMENTO_2


def _corrida(ensayos: int, regret: float | None, valor: float = 0.8, e2: bool = True) -> dict:
    """Corrida con la forma que guardan los arneses."""
    corrida = {
        "curva": [valor],
        "ensayos": ensayos,
        "mejor": valor,
        "a_presupuesto_reducido": {"5": valor, "15": valor, "30": valor} if e2 else valor,
        "fraccion_inevaluable": 0.1,
    }
    if regret is not None:
        corrida["regret"] = regret
    return corrida


def _pliegue(
    nombre: str,
    experimento: tabla.Experimento,
    ensayos: tuple[int, ...] = (2, 3, 4),
    regrets: dict[str, float] | None = None,
) -> dict:
    """Pliegue con todas las condiciones del experimento."""
    e2 = experimento is E2
    regrets = regrets or {}
    return {
        "dataset_id": nombre,
        "dataset_name": nombre,
        "recomendaciones": {},
        "semillas": [
            {
                "semilla": 101 * (i + 1),
                **{
                    c: _corrida(e, regrets.get(c, 0.01) if e2 else None, e2=e2)
                    for c in experimento.condiciones
                },
            }
            for i, e in enumerate(ensayos)
        ],
    }


def _resultado(pliegues: list[dict], mesetas: dict[str, float] | None = None) -> dict:
    mesetas = mesetas or {}
    return {
        "pliegues": pliegues,
        "analisis": {
            "por_dataset": [
                {"dataset_id": p["dataset_id"], "meseta": mesetas.get(p["dataset_id"], 12)}
                for p in pliegues
            ]
        },
    }


def _cribado(**medidas: tuple[float, float | None]) -> dict[str, dict]:
    return {
        nombre: {"iqr": iqr, "margen": margen, "categoricas": 0 if margen else 3,
                 "faltantes": False}
        for nombre, (iqr, margen) in medidas.items()
    }


# --------------------------------------------------------------------------- #
# Las filas
# --------------------------------------------------------------------------- #
def test_the_flat_datasets_keep_their_own_row_with_their_sensitivity_visible() -> None:
    """Es literalmente lo que pide el pre-registro, y lo que un resumen de medianas pierde."""
    resultado = _resultado([_pliegue("plano", E1), _pliegue("sensible", E1)])
    cribado = _cribado(plano=(0.005, 0.01), sensible=(0.3, 0.2))

    filas = tabla.tabular(resultado, E1, cribado)
    texto = tabla.redactar(E1, filas)

    assert [f["dataset_name"] for f in filas] == ["sensible", "plano"]
    assert "| `plano` | 0.0050 † |" in texto
    assert "| `sensible` | 0.3000 |" in texto


def test_the_trials_are_listed_seed_by_seed_in_order() -> None:
    filas = tabla.tabular(
        _resultado([_pliegue("a", E1, ensayos=(2, 31, 4))]), E1, _cribado(a=(0.1, 0.1))
    )

    assert filas[0]["condiciones"][0]["ensayos"] == [2, 31, 4]
    assert "| 2, 31, 4 |" in tabla.redactar(E1, filas)


def test_censoring_uses_each_experiments_own_value() -> None:
    """Prueba por sabotaje de la censura.

    Un 31 es una corrida censurada en el experimento 1 y un ensayo cualquiera en el 2,
    cuya censura es 61. Con una sola constante compartida, uno de los dos contaría mal.
    """
    cribado = _cribado(a=(0.1, 0.1))
    uno = tabla.tabular(_resultado([_pliegue("a", E1, ensayos=(31, 31, 5))]), E1, cribado)
    dos = tabla.tabular(_resultado([_pliegue("a", E2, ensayos=(31, 61, 5))]), E2, cribado)

    assert uno[0]["condiciones"][0]["censurados"] == 2
    assert dos[0]["condiciones"][0]["censurados"] == 1


def test_a_dataset_missing_from_the_screening_is_refused() -> None:
    resultado = _resultado([_pliegue("conocido", E1), _pliegue("huerfano", E1)])

    with pytest.raises(KeyError, match="huerfano"):
        tabla.tabular(resultado, E1, _cribado(conocido=(0.1, 0.1)))


def test_an_unevaluable_empty_pipeline_is_not_shown_as_a_zero_margin() -> None:
    """Un margen nulo diría que el preprocesamiento no ayuda; «n/e» dice que no se pudo medir."""
    filas = tabla.tabular(_resultado([_pliegue("a", E1)]), E1, _cribado(a=(0.1, None)))

    texto = tabla.redactar(E1, filas)

    assert "| `a` | 0.1000 | n/e |" in texto
    assert "1 conjuntos en que ocurre, 1 tienen columnas categóricas" in texto


def test_experiment_two_carries_regret_and_its_three_budgets() -> None:
    cribado = _cribado(a=(0.1, 0.1))
    uno = tabla.redactar(E1, tabla.tabular(_resultado([_pliegue("a", E1)]), E1, cribado))
    dos = tabla.redactar(E2, tabla.tabular(_resultado([_pliegue("a", E2)]), E2, cribado),
                         relacion=None)

    assert "Mejor a 15 | Inevaluables" in uno
    assert "Regret" not in uno.split("## Tabla")[1]
    assert "| Regret | Mejor a 5 | Mejor a 15 | Mejor a 30 |" in dos
    assert "hereda" in dos


# --------------------------------------------------------------------------- #
# La meseta y el beneficio
# --------------------------------------------------------------------------- #
def _con_meseta() -> list[dict]:
    """Cinco conjuntos: cuanto más estrecha la meseta, más ayuda `nearest`."""
    pliegues, mesetas, cribado = [], {}, {}
    for i, meseta in enumerate((2, 5, 10, 40, 100)):
        nombre = f"d{i}"
        # El frío varía para que ninguna correlación se calcule sobre una constante.
        frio = 0.1 + 0.01 * i
        pliegues.append(
            _pliegue(nombre, E2, regrets={"frio": frio, "nearest": frio - 0.08 / (i + 1)})
        )
        mesetas[nombre] = meseta
        cribado[nombre] = {"iqr": 0.1, "margen": 0.1, "categoricas": 0, "faltantes": False}
    return tabla.tabular(_resultado(pliegues, mesetas), E2, cribado)


def test_the_benefit_is_cold_minus_the_condition() -> None:
    """Prueba por sabotaje del signo.

    `nearest` reduce el regret en todos, y más donde la meseta es estrecha. El beneficio
    tiene que ser positivo y su correlación con la meseta negativa; con la resta al revés
    saldrían los dos signos invertidos.
    """
    relacion = tabla.meseta_y_beneficio(_con_meseta())
    nearest = next(c for c in relacion["condiciones"] if c["condicion"] == "nearest")

    assert nearest["beneficio_mediano"] > 0
    assert nearest["con_beneficio"] == 5
    assert nearest["spearman"] == pytest.approx(-1.0)
    assert nearest["pearson"] < 0


def test_the_cold_start_is_the_reference_and_not_a_condition() -> None:
    relacion = tabla.meseta_y_beneficio(_con_meseta())

    assert "frio" not in [c["condicion"] for c in relacion["condiciones"]]
    assert relacion["referencia"]["n"] == 5


def test_the_relation_requires_the_regret() -> None:
    filas = tabla.tabular(_resultado([_pliegue("a", E1)]), E1, _cribado(a=(0.1, 0.1)))

    with pytest.raises(ValueError, match="no lo midió"):
        tabla.meseta_y_beneficio(filas)


def test_the_relation_is_marked_as_exploratory() -> None:
    filas = _con_meseta()

    texto = tabla.redactar(E2, filas, tabla.meseta_y_beneficio(filas))

    assert "Exploratorio, no confirmatorio" in texto
    assert "Meseta y beneficio de `nearest`" in texto


def test_only_experiment_two_declared_the_relation() -> None:
    assert E2.explorar_meseta
    assert not E1.explorar_meseta


# --------------------------------------------------------------------------- #
# Las fuentes y el punto de entrada
# --------------------------------------------------------------------------- #
def test_the_screening_is_indexed_by_openml_id_as_text(tmp_path: Path) -> None:
    ruta = tmp_path / "cribado.json"
    ruta.write_text(
        json.dumps({"resultados": [
            {"did": 40983, "iqr": 0.3, "margen": 0.1, "categoricas": 0, "faltantes": False},
            {"did": 38, "iqr": 0.2, "margen": None, "categoricas": 23, "faltantes": True},
        ]}),
        encoding="utf-8",
    )

    cribado = tabla.leer_cribado(ruta)

    assert cribado["40983"]["iqr"] == 0.3
    assert cribado["38"]["margen"] is None


def test_a_missing_screening_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="screen_corpus"):
        tabla.leer_cribado(tmp_path / "no-existe.json")


def _preparar(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, con_e2: bool = True) -> None:
    cribado = tmp_path / "cribado.json"
    cribado.write_text(
        json.dumps({"resultados": [
            {"did": i, "iqr": 0.1, "margen": 0.1, "categoricas": 0, "faltantes": False}
            for i in range(5)
        ]}),
        encoding="utf-8",
    )
    uno = tmp_path / "e1.json"
    uno.write_text(json.dumps(_resultado([_pliegue(str(i), E1) for i in range(5)])), "utf-8")
    dos = tmp_path / "e2.json"
    if con_e2:
        pliegues = [
            _pliegue(str(i), E2, regrets={"frio": 0.1 + 0.01 * i, "nearest": 0.02 * i})
            for i in range(5)
        ]
        dos.write_text(
            json.dumps(_resultado(pliegues, {str(i): 2 + 10 * i for i in range(5)})), "utf-8"
        )
    monkeypatch.setattr(tabla, "CRIBADO", cribado)
    monkeypatch.setattr(tabla, "TABULADOS", (
        dataclasses.replace(E1, resultados=uno, salida=tmp_path / "E1.md"),
        dataclasses.replace(E2, resultados=dos, salida=tmp_path / "E2.md"),
    ))


def test_it_refuses_to_run_without_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _preparar(monkeypatch, tmp_path, con_e2=False)

    with pytest.raises(SystemExit, match="run_lodo_e2.py"):
        tabla.main()
    assert not (tmp_path / "E1.md").exists(), "no debe escribir la mitad de los documentos"


def test_it_writes_both_documents(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _preparar(monkeypatch, tmp_path)

    assert tabla.main() == 0
    assert "Exploratorio" not in (tmp_path / "E1.md").read_text(encoding="utf-8")
    assert "Exploratorio" in (tmp_path / "E2.md").read_text(encoding="utf-8")
