# tests/test_comparative_report.py
"""Pruebas del reporte comparativo del LODO.

El reporte es el documento del que va a salir el capítulo de resultados, así que
lo que se prueba aquí no es que se genere HTML sino que **no pueda mentir**: que el
veredicto salga de las cifras y no de una bandera, que las estrategias no estén
escritas a mano, y que las covariables no se confundan con condiciones.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from autopipe.reporter import ComparativeReportGenerator, redactar_veredicto
from autopipe.reporter import report as report_module


def _contraste(
    estrategia: str = "nearest",
    reduccion: float | None = 0.50,
    p_una_cola: float = 0.001,
    confirma: bool | None = None,
    censura: int = 31,
) -> dict:
    """Arma un contraste con la forma que produce el arnés.

    Args:
        estrategia: Condición caliente contrastada.
        reduccion: Reducción relativa de la mediana, o ``None``.
        p_una_cola: Valor p de una cola.
        confirma: Bandera del arnés. Si es ``None``, se deriva de las cifras.
        censura: Valor de censura del contraste.

    Returns:
        El contraste.
    """
    if confirma is None:
        confirma = reduccion is not None and reduccion >= 0.30 and p_una_cola < 0.05
    return {
        "estrategia": estrategia,
        "censura": censura,
        "n_pares": 25,
        "mediana_frio": 8.0,
        "mediana_caliente": 4.0,
        "reduccion": reduccion,
        "p_una_cola": p_una_cola,
        "p_dos_colas": p_una_cola * 2,
        "confirma": confirma,
        "censurados_frio": 0,
        "censurados_caliente": 2,
    }


def _fila(nombre: str, frio: float, nearest: float, meseta: int = 12) -> dict:
    """Arma una fila de la tabla por conjunto de datos."""
    fila = {"dataset_id": nombre, "dataset_name": nombre, "meseta": meseta}
    for condicion, valor in (("frio", frio), ("nearest", nearest), ("random", 6.0),
                             ("frequent", 5.0)):
        fila[condicion] = valor
        fila[f"{condicion}_inevaluable"] = 0.1
        fila[f"{condicion}_a_15"] = 0.8
    return fila


@pytest.fixture
def analisis() -> dict:
    """Análisis completo, con la forma exacta que entrega ``run_lodo.analizar``."""
    return {
        "primario": _contraste(),
        "secundarios": [_contraste("random", 0.10, 0.4), _contraste("frequent", 0.25, 0.2)],
        "robustez": [
            _contraste(censura=60, reduccion=0.55),
            _contraste(censura=300, reduccion=0.60),
        ],
        "por_dataset": [
            _fila("wilt", 8.0, 3.0),
            _fila("sick", 4.0, 9.0),
            _fila("car", 5.0, 5.0),
        ],
    }


# --------------------------------------------------------------------------- #
# El veredicto sale de las cifras
# --------------------------------------------------------------------------- #
def test_the_verdict_confirms_when_both_conditions_hold() -> None:
    texto = redactar_veredicto(_contraste(reduccion=0.50, p_una_cola=0.001))

    assert "se confirma" in texto
    assert "no se confirma" not in texto
    assert "50.0 %" in texto


def test_the_verdict_names_the_magnitude_when_only_it_falls_short() -> None:
    texto = redactar_veredicto(_contraste(reduccion=0.12, p_una_cola=0.001))

    assert "no se confirma" in texto
    assert "12.0 %" in texto
    assert "0,05" not in texto, "No debe achacar una falta de significancia que no hubo."


def test_the_verdict_names_significance_when_only_it_falls_short() -> None:
    texto = redactar_veredicto(_contraste(reduccion=0.80, p_una_cola=0.30))

    assert "no se confirma" in texto
    assert "0.3000" in texto
    assert "el criterio exige un 30 %" not in texto


def test_the_verdict_names_both_when_both_fall_short() -> None:
    texto = redactar_veredicto(_contraste(reduccion=0.05, p_una_cola=0.60))

    assert "5.0 %" in texto
    assert "0.6000" in texto


def test_the_verdict_reports_an_undefined_reduction_without_claiming_a_result() -> None:
    """La mediana en frío puede ser cero y entonces la reducción no existe."""
    contraste = _contraste(reduccion=None, p_una_cola=0.01)
    contraste["mediana_frio"] = 0.0

    texto = redactar_veredicto(contraste)

    assert "no se confirma" in texto
    assert "no está definida" in texto


def test_the_verdict_denounces_a_flag_that_contradicts_the_numbers() -> None:
    """Prueba por sabotaje: es la garantía de que el reporte no copia el veredicto.

    Si el reporte se limitara a leer ``confirma``, este análisis —cuyas cifras no
    cumplen ninguna de las dos condiciones— produciría un veredicto favorable sin
    nada que lo sostenga. La prueba falla si alguien vuelve a esa implementación.
    """
    saboteado = _contraste(reduccion=0.01, p_una_cola=0.90, confirma=True)

    texto = redactar_veredicto(saboteado)

    assert "discrepancia" in texto
    assert "se confirma:" not in texto


def test_the_verdict_also_denounces_the_flag_in_the_opposite_direction() -> None:
    saboteado = _contraste(reduccion=0.90, p_una_cola=0.001, confirma=False)

    texto = redactar_veredicto(saboteado)

    assert "discrepancia" in texto


def test_a_missing_flag_is_not_treated_as_a_contradiction() -> None:
    """El arnés siempre la trae, pero el reporte no debe exigirla."""
    sin_bandera = _contraste()
    del sin_bandera["confirma"]

    assert "se confirma" in redactar_veredicto(sin_bandera)


# --------------------------------------------------------------------------- #
# La generación del documento
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("clave", ["primario", "secundarios", "robustez", "por_dataset"])
def test_an_incomplete_analysis_is_refused(analisis: dict, tmp_path: Path, clave: str) -> None:
    del analisis[clave]

    with pytest.raises(KeyError, match=clave):
        ComparativeReportGenerator(tmp_path).generate_comparison(analisis, pdf=False)


def test_the_report_is_written_and_carries_the_verdict(analisis: dict, tmp_path: Path) -> None:
    rutas = ComparativeReportGenerator(tmp_path).generate_comparison(analisis, pdf=False)

    assert rutas["html"] == tmp_path / "lodo.html"
    assert rutas["pdf"] is None
    html = rutas["html"].read_text(encoding="utf-8")
    assert redactar_veredicto(analisis["primario"]) in html
    for nombre in ("wilt", "sick", "car"):
        assert nombre in html


def test_the_protocol_block_shows_exactly_what_it_receives(
    analisis: dict, tmp_path: Path
) -> None:
    protocolo = {"Semillas": "101, 202, 303, 404, 505", "Espacio": "tabular-v1"}

    rutas = ComparativeReportGenerator(tmp_path).generate_comparison(
        analisis, protocolo=protocolo, pdf=False
    )

    html = rutas["html"].read_text(encoding="utf-8")
    assert "101, 202, 303, 404, 505" in html
    assert "tabular-v1" in html


def test_the_file_name_is_configurable(analisis: dict, tmp_path: Path) -> None:
    rutas = ComparativeReportGenerator(tmp_path).generate_comparison(
        analisis, pdf=False, nombre="lodo-robustez"
    )

    assert rutas["html"].name == "lodo-robustez.html"


# --------------------------------------------------------------------------- #
# Las condiciones se deducen, no se escriben a mano
# --------------------------------------------------------------------------- #
def test_a_new_strategy_appears_without_touching_the_report(
    analisis: dict, tmp_path: Path
) -> None:
    """Si el arnés añadiera una estrategia, el reporte tiene que mostrarla sola."""
    for fila in analisis["por_dataset"]:
        fila["inventada"] = 2.0
        fila["inventada_inevaluable"] = 0.0
        fila["inventada_a_15"] = 0.9

    rutas = ComparativeReportGenerator(tmp_path).generate_comparison(analisis, pdf=False)

    assert "Ensayos (inventada)" in rutas["html"].read_text(encoding="utf-8")


def test_the_covariate_columns_are_not_mistaken_for_conditions(
    analisis: dict, tmp_path: Path
) -> None:
    """``frio_a_15`` y ``frio_inevaluable`` acompañan a cada condición, no son una."""
    condiciones = ComparativeReportGenerator._condiciones(analisis["por_dataset"])

    assert condiciones == ["frio", "nearest", "random", "frequent"]


def test_an_empty_table_does_not_break_the_report(analisis: dict, tmp_path: Path) -> None:
    analisis["por_dataset"] = []

    rutas = ComparativeReportGenerator(tmp_path).generate_comparison(analisis, pdf=False)

    assert rutas["html"].is_file()


# --------------------------------------------------------------------------- #
# Lo que la tabla dice de un vistazo
# --------------------------------------------------------------------------- #
def test_each_row_is_coloured_by_which_condition_won(analisis: dict, tmp_path: Path) -> None:
    rutas = ComparativeReportGenerator(tmp_path).generate_comparison(analisis, pdf=False)
    html = rutas["html"].read_text(encoding="utf-8")

    fila_wilt = next(linea for linea in html.split("<tr") if "wilt" in linea)
    fila_sick = next(linea for linea in html.split("<tr") if "sick" in linea)
    fila_car = next(linea for linea in html.split("<tr") if ">car<" in linea)

    assert 'class="mejor"' in fila_wilt, "El caliente ganó 8 contra 3."
    assert 'class="peor"' in fila_sick, "El caliente perdió 4 contra 9."
    assert 'class=""' in fila_car, "Empataron a 5."


def test_the_censoring_column_only_appears_in_the_robustness_table(
    analisis: dict, tmp_path: Path
) -> None:
    html = (
        ComparativeReportGenerator(tmp_path)
        .generate_comparison(analisis, pdf=False)["html"]
        .read_text(encoding="utf-8")
    )

    assert html.count("<th>Censura</th>") == 1
    assert "<td class=\"num\">60</td>" in html
    assert "<td class=\"num\">300</td>" in html


def test_the_secondary_controls_are_tabulated_next_to_the_primary(
    analisis: dict, tmp_path: Path
) -> None:
    html = (
        ComparativeReportGenerator(tmp_path)
        .generate_comparison(analisis, pdf=False)["html"]
        .read_text(encoding="utf-8")
    )

    assert "<td>random</td>" in html
    assert "<td>frequent</td>" in html


def test_a_control_that_meets_the_criterion_is_not_read_as_a_confirmation(
    analisis: dict, tmp_path: Path
) -> None:
    """El caso real del experimento: un control cumple el criterio y la primaria no.

    Sin marcar el papel de cada fila, un lector concluiría que la hipótesis se
    confirmó. La prueba falla si la tabla deja de distinguirlos.
    """
    analisis["primario"].update({"reduccion": 0.33, "p_una_cola": 0.31, "confirma": False})
    analisis["secundarios"][1].update(
        {"reduccion": 0.33, "p_una_cola": 0.02, "confirma": True}
    )

    html = (
        ComparativeReportGenerator(tmp_path)
        .generate_comparison(analisis, pdf=False)["html"]
        .read_text(encoding="utf-8")
    )

    assert "<th>Papel</th>" in html
    assert "<td>primario</td>" in html
    assert html.count("<td>control</td>") == 2
    assert "se juega\núnicamente en la fila primaria" in html
    assert "no se confirma" in html


def test_the_robustness_table_carries_no_role_column(analisis: dict, tmp_path: Path) -> None:
    """Todas sus filas son la misma condición primaria: la columna sobraría."""
    html = (
        ComparativeReportGenerator(tmp_path)
        .generate_comparison(analisis, pdf=False)["html"]
        .read_text(encoding="utf-8")
    )

    assert html.count("<th>Papel</th>") == 1


def test_the_pdf_takes_the_name_of_its_report(
    analisis: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """El generador heredado siempre escribe ``report.pdf``; aquí hay varios.

    Sin el renombrado, dos reportes comparativos del mismo directorio se pisarían
    el PDF entre sí y el segundo se llevaría el contenido del primero.
    """
    def _fingir(origen: str, destino: str) -> None:
        Path(destino).write_bytes(b"%PDF-1.4 fingido")

    monkeypatch.setattr(report_module.pdfkit, "from_file", _fingir)

    rutas = ComparativeReportGenerator(tmp_path).generate_comparison(
        analisis, nombre="lodo-robustez"
    )

    assert rutas["pdf"] == tmp_path / "lodo-robustez.pdf"
    assert rutas["pdf"].is_file()
    assert not (tmp_path / "report.pdf").exists()


def test_a_pdf_that_cannot_be_generated_does_not_sink_the_report(
    analisis: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sin ``wkhtmltopdf`` el HTML tiene que quedar igual."""
    def _fallar(origen: str, destino: str) -> None:
        raise OSError("no wkhtmltopdf")

    monkeypatch.setattr(report_module.pdfkit, "from_file", _fallar)

    rutas = ComparativeReportGenerator(tmp_path).generate_comparison(analisis)

    assert rutas["pdf"] is None
    assert rutas["html"].is_file()


def test_a_text_value_is_shown_as_it_comes(analisis: dict, tmp_path: Path) -> None:
    """La meseta puede faltar y la censura puede venir como texto."""
    analisis["por_dataset"][0] = {**analisis["por_dataset"][0], "meseta": None}
    analisis["robustez"][0] = {**analisis["robustez"][0], "censura": "sin censura"}

    html = (
        ComparativeReportGenerator(tmp_path)
        .generate_comparison(analisis, pdf=False)["html"]
        .read_text(encoding="utf-8")
    )

    assert "—" in html
    assert "sin censura" in html


def test_the_charts_are_embedded_and_no_figure_is_left_open(
    analisis: dict, tmp_path: Path
) -> None:
    """Una figura sin cerrar se acumula en el estado global de matplotlib."""
    plt.close("all")

    html = (
        ComparativeReportGenerator(tmp_path)
        .generate_comparison(analisis, pdf=False)["html"]
        .read_text(encoding="utf-8")
    )

    assert html.count("data:image/png;base64,") == 2
    assert plt.get_fignums() == []
