# tests/test_stratified.py
"""Pruebas del análisis estratificado por sensibilidad al preprocesamiento.

Lo que más importa proteger aquí es que el **umbral no se mueva** y que ningún
conjunto se asigne a un estrato por omisión. El pre-registro fijó 0,02 antes de ver
ninguna distribución y prohíbe expresamente redefinirlo para forzar una partición; una
asignación silenciosa de un conjunto sin sensibilidad conocida tendría el mismo efecto
sin que nada fallara.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

pytest.importorskip("openml", reason="el análisis importa el arnés del experimento 2")

import analyze_stratified as estratificado  # noqa: E402


def _pliegue(nombre: str, frio: float, nearest: float, frequent: float = 0.02) -> dict:
    """Pliegue mínimo con las medianas que el contraste necesita."""
    semillas = []
    for i in range(3):
        semilla = {"semilla": 100 + i, "optimo": 1.0, "objetivo": 0.99, "n_en_meseta": 5}
        for condicion, valor in (("frio", frio), ("nearest", nearest),
                                 ("frequent", frequent)):
            semilla[condicion] = {
                "curva": [valor],
                "regret": valor,
                "ensayos": 1,
                "fraccion_inevaluable": 0.0,
            }
        semillas.append(semilla)
    return {
        "dataset_id": nombre,
        "dataset_name": f"d{nombre}",
        "meseta_mediana": 5,
        "recomendaciones": {},
        "semillas": semillas,
    }


# --------------------------------------------------------------------------- #
# El umbral y el reparto
# --------------------------------------------------------------------------- #
def test_the_threshold_is_the_one_the_pre_registration_fixed() -> None:
    """Prueba por sabotaje del umbral.

    El pre-registro v1.0 lo fijó en 0,02 antes de observar la distribución y prohíbe
    redefinirlo para forzar una partición del corpus. Si alguien lo moviera, el
    estrato cambiaría de composición y ninguna prueba de comportamiento se enteraría.
    """
    assert estratificado.UMBRAL == 0.02


def test_the_threshold_is_inclusive_on_the_sensitive_side() -> None:
    """«IQR mayor o igual a 0,02», dice el documento: el borde entra en sensibles."""
    pliegues = [_pliegue("justo", 0.05, 0.01)]

    estratos = estratificado.estratificar(pliegues, {"justo": 0.02})

    assert len(estratos["sensibles"]) == 1
    assert estratos["insensibles"] == []


def test_a_value_below_the_threshold_goes_to_the_insensitive_stratum() -> None:
    pliegues = [_pliegue("bajo", 0.05, 0.01)]

    estratos = estratificado.estratificar(pliegues, {"bajo": 0.0199})

    assert len(estratos["insensibles"]) == 1


def test_a_dataset_without_a_known_sensitivity_is_refused() -> None:
    """Asignarlo por omisión falsearía el contraste sin que nada fallara."""
    pliegues = [_pliegue("conocido", 0.05, 0.01), _pliegue("huerfano", 0.05, 0.01)]

    with pytest.raises(KeyError, match="huerfano"):
        estratificado.estratificar(pliegues, {"conocido": 0.05})


def test_the_split_is_reported_in_the_analysis() -> None:
    pliegues = [_pliegue("a", 0.05, 0.01), _pliegue("b", 0.05, 0.01)]

    analisis = estratificado.analizar(pliegues, {"a": 0.05, "b": 0.001})

    assert analisis["reparto"] == {"sensibles": 1, "insensibles": 1}
    assert analisis["umbral"] == 0.02


# --------------------------------------------------------------------------- #
# Los contrastes salen del arnés, no de una copia
# --------------------------------------------------------------------------- #
def test_the_contrasts_are_the_two_confirmatory_ones() -> None:
    pliegues = [_pliegue(str(i), 0.05, 0.01) for i in range(6)]

    analisis = estratificado.analizar(pliegues, dict.fromkeys("012345", 0.05))

    pares = [
        (c["condicion"], c["referencia"]) for c in analisis["contrastes"]["sensibles"]
    ]
    assert pares == [("nearest", "frio"), ("nearest", "frequent")]


def test_an_empty_stratum_produces_no_contrast_instead_of_a_spurious_one() -> None:
    pliegues = [_pliegue(str(i), 0.05, 0.01) for i in range(4)]

    analisis = estratificado.analizar(pliegues, dict.fromkeys("0123", 0.05))

    assert analisis["contrastes"]["insensibles"] == []
    assert analisis["reparto"]["insensibles"] == 0


def test_each_stratum_is_contrasted_only_against_itself() -> None:
    """Un estrato con regret claramente menor no debe contaminar al otro."""
    pliegues = [_pliegue(f"s{i}", 0.20, 0.02) for i in range(5)]
    pliegues += [_pliegue(f"i{i}", 0.002, 0.001) for i in range(5)]
    sensibilidad = {**dict.fromkeys([f"s{i}" for i in range(5)], 0.05),
                    **dict.fromkeys([f"i{i}" for i in range(5)], 0.001)}

    analisis = estratificado.analizar(pliegues, sensibilidad)

    sensibles = analisis["contrastes"]["sensibles"][0]
    insensibles = analisis["contrastes"]["insensibles"][0]
    assert sensibles["mediana_referencia"] == pytest.approx(0.20)
    assert insensibles["mediana_referencia"] == pytest.approx(0.002)
    assert sensibles["n_pares"] == 5
    assert insensibles["n_pares"] == 5


# --------------------------------------------------------------------------- #
# El documento
# --------------------------------------------------------------------------- #
def _analisis() -> dict:
    pliegues = [_pliegue(f"s{i}", 0.05, 0.01) for i in range(6)]
    pliegues += [_pliegue("i0", 0.002, 0.001)]
    sensibilidad = {**dict.fromkeys([f"s{i}" for i in range(6)], 0.05), "i0": 0.001}
    return estratificado.analizar(pliegues, sensibilidad)


def test_the_document_says_it_is_declared_and_not_exploratory() -> None:
    """Es la diferencia que decide cuánto peso tiene el resultado."""
    texto = estratificado.redactar(_analisis())

    assert "secundario declarado" in texto
    assert "no exploratorio" in texto
    assert "no decide sobre la hipótesis" in texto


def test_the_document_explains_why_it_was_empty_in_experiment_one() -> None:
    texto = estratificado.redactar(_analisis())

    assert "vacío por construcción" in texto
    assert "sin haber tocado el umbral" in texto


def test_the_document_lists_the_insensitive_datasets() -> None:
    """Sin la lista, el lector no puede juzgar si el estrato es representativo."""
    texto = estratificado.redactar(_analisis())

    assert "`di0`" in texto


def test_an_empty_insensitive_stratum_is_reported_as_such(monkeypatch) -> None:
    """Es el caso del experimento 1, y el pre-registro pide reportarlo así."""
    pliegues = [_pliegue(f"s{i}", 0.05, 0.01) for i in range(4)]
    analisis = estratificado.analizar(pliegues, dict.fromkeys(["s0", "s1", "s2", "s3"], 0.05))

    texto = estratificado.redactar(analisis)

    assert "Estrato vacío" in texto


# --------------------------------------------------------------------------- #
# La lectura del corpus y el punto de entrada
# --------------------------------------------------------------------------- #
def test_the_sensitivity_is_read_from_the_corpus_document(tmp_path: Path) -> None:
    documento = tmp_path / "CORPUS-E2.md"
    documento.write_text(
        "| # | OpenML ID | Nombre | IQR | Procedencia |\n"
        "|---:|---:|---|---:|---|\n"
        "| 1 | 40983 | `wilt` | 0.1234 | corpus del experimento 1 |\n"
        "| 2 | 15 | `breast-w` | 0.0049 | incorporado en el experimento 2 |\n",
        encoding="utf-8",
    )

    valores = estratificado.leer_sensibilidad(documento)

    assert valores == {"40983": 0.1234, "15": 0.0049}


def test_a_missing_corpus_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="build_corpus_e2"):
        estratificado.leer_sensibilidad(tmp_path / "no-existe.md")


def test_it_refuses_to_run_without_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(estratificado, "RESULTADOS", tmp_path / "no-existe.json")

    with pytest.raises(SystemExit, match="run_lodo_e2"):
        estratificado.main()


def test_it_writes_the_document(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Los identificadores tienen que ser los de OpenML, que es como los indexa el
    # documento del corpus: la sensibilidad se cruza por ese identificador.
    pliegues = [_pliegue(str(i), 0.05, 0.01) for i in range(4)]
    datos = tmp_path / "LODO-E2-resultados.json"
    datos.write_text(json.dumps({"pliegues": pliegues}), encoding="utf-8")
    corpus = tmp_path / "CORPUS-E2.md"
    corpus.write_text(
        "| 1 | 0 | `ds0` | 0.05 |\n| 2 | 1 | `ds1` | 0.05 |\n"
        "| 3 | 2 | `ds2` | 0.05 |\n| 4 | 3 | `ds3` | 0.001 |\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(estratificado, "RESULTADOS", datos)
    monkeypatch.setattr(estratificado, "CORPUS", corpus)
    monkeypatch.setattr(estratificado, "SALIDA", tmp_path / "ESTRATIFICADO-E2.md")

    assert estratificado.main() == 0
    assert (tmp_path / "ESTRATIFICADO-E2.md").is_file()
