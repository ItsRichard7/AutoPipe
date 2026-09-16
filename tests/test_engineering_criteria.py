# tests/test_engineering_criteria.py
"""Pruebas de la verificación con evidencia de los criterios de ingeniería.

Un verificador que declara «cumple» cuando no debería es peor que no tener ninguno. Las
pruebas por sabotaje fijan los cuatro modos de fallo que lo harían mentir: dar por no
editada una prueba cuya aserción cambió, aprobar pruebas ejecutadas contra una copia de
``autopipe`` que no es la del repositorio, dejar pasar un módulo al 69,96 % porque se
muestra redondeado, y contar como reproducible un documento sin ninguna verificación
idéntica.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import verify_engineering_criteria as criterios  # noqa: E402

PRUEBA = '''
import pytest


def auxiliar():
    return 1


def test_suma():
    assert 1 + 1 == 2


@pytest.mark.parametrize("valor", [1, 2])
def test_parametrizada(valor):
    assert valor > 0


class TestGrupo:
    def test_metodo(self):
        assert True

    def ayuda(self):
        pass
'''


# --------------------------------------------------------------------------- #
# Los umbrales
# --------------------------------------------------------------------------- #
def test_the_thresholds_are_the_ones_the_proposal_fixed() -> None:
    """Prueba por sabotaje de los umbrales del anteproyecto, Sección 4.3.5."""
    assert criterios.PRUEBAS_BASE == 134
    assert criterios.UMBRAL_COBERTURA == 70.0
    assert criterios.LIMITE_INSTRUCCIONES == 20
    assert criterios.COMMIT_MIGRACION == "35f1db5"


# --------------------------------------------------------------------------- #
# (a) Las funciones de prueba y su comparación
# --------------------------------------------------------------------------- #
def test_test_functions_and_methods_are_found_and_helpers_are_not() -> None:
    funciones = criterios.funciones_de_prueba(PRUEBA)

    assert set(funciones) == {"test_suma", "test_parametrizada", "TestGrupo::test_metodo"}


def test_reformatting_or_commenting_a_test_does_not_count_as_editing_it() -> None:
    reformateada = PRUEBA.replace(
        "    assert 1 + 1 == 2", "    # comentario nuevo\n    assert (1 + 1\n            == 2)"
    )

    comparacion = criterios.comparar_funciones(
        criterios.funciones_de_prueba(PRUEBA), criterios.funciones_de_prueba(reformateada)
    )

    assert "test_suma" in comparacion["identicas"]
    assert comparacion["modificadas"] == []


def test_changing_an_assertion_counts_as_editing_the_test() -> None:
    """Prueba por sabotaje: es exactamente la edición que el criterio prohíbe ocultar."""
    editada = PRUEBA.replace("assert 1 + 1 == 2", "assert 1 + 1 >= 2")

    comparacion = criterios.comparar_funciones(
        criterios.funciones_de_prueba(PRUEBA), criterios.funciones_de_prueba(editada)
    )

    assert comparacion["modificadas"] == ["test_suma"]


def test_changing_a_parametrization_counts_as_editing_the_test() -> None:
    editada = PRUEBA.replace("[1, 2]", "[1]")

    comparacion = criterios.comparar_funciones(
        criterios.funciones_de_prueba(PRUEBA), criterios.funciones_de_prueba(editada)
    )

    assert comparacion["modificadas"] == ["test_parametrizada"]


def test_a_renamed_test_is_reported_as_missing() -> None:
    renombrada = PRUEBA.replace("def test_suma", "def test_adicion")

    comparacion = criterios.comparar_funciones(
        criterios.funciones_de_prueba(PRUEBA), criterios.funciones_de_prueba(renombrada)
    )

    assert comparacion["ausentes"] == ["test_suma"]


@pytest.mark.parametrize(
    ("linea", "esperado"),
    [
        ("134 passed in 16.86s", {"passed": 134}),
        ("1 failed, 133 passed, 38 warnings in 3.01s", {"failed": 1, "passed": 133,
                                                         "warnings": 38}),
        ("134 tests collected in 5.80s", {"collected": 134}),
        ("2 errors in 1.00s", {"errors": 2}),
        ("1 test collected in 0.10s", {"collected": 1}),
    ],
)
def test_the_pytest_summary_is_read_from_its_last_line(linea: str, esperado: dict) -> None:
    recuentos = criterios.resumen_pytest(f"....\nalgo 99 passed en medio\n{linea}\n")

    assert {k: v for k, v in recuentos.items() if v} == esperado


def test_an_empty_pytest_output_counts_nothing() -> None:
    assert not any(criterios.resumen_pytest("").values())


def _comparacion(identicas: int = 111) -> dict:
    return {"identicas": [f"t{i}" for i in range(identicas)], "modificadas": [], "ausentes": []}


def test_the_regression_criterion_passes_with_the_whole_base_suite_green() -> None:
    resultado = criterios.evaluar_regresion(
        _comparacion(), {"collected": 134}, {"passed": 134, "failed": 0, "errors": 0},
        criterios.RAIZ / "autopipe" / "__init__.py",
    )

    assert resultado["cumple"]
    assert resultado["funciones"] == 111


def test_tests_run_against_another_copy_of_the_library_prove_nothing() -> None:
    """Prueba por sabotaje del origen: pasar contra una copia instalada no dice nada."""
    resultado = criterios.evaluar_regresion(
        _comparacion(), {"collected": 134}, {"passed": 134, "failed": 0, "errors": 0},
        Path("C:/otro/site-packages/autopipe/__init__.py"),
    )

    assert not resultado["contra_el_repositorio"]
    assert not resultado["cumple"]


@pytest.mark.parametrize(
    ("recoleccion", "ejecucion"),
    [
        ({"collected": 134}, {"passed": 133, "failed": 1, "errors": 0}),
        ({"collected": 134}, {"passed": 134, "failed": 0, "errors": 1}),
        ({"collected": 120}, {"passed": 120, "failed": 0, "errors": 0}),
    ],
)
def test_a_failure_an_error_or_a_shrunken_suite_does_not_pass(recoleccion, ejecucion) -> None:
    resultado = criterios.evaluar_regresion(
        _comparacion(), recoleccion, ejecucion, criterios.RAIZ / "autopipe" / "__init__.py"
    )

    assert not resultado["cumple"]


def test_the_migrated_tests_are_extracted_as_they_were(tmp_path: Path) -> None:
    """Se lee del commit real: la carpeta extraída es la de la migración, no la de hoy."""
    fuentes = criterios.extraer_pruebas(criterios.COMMIT_MIGRACION, tmp_path)

    assert "tests/conftest.py" in fuentes
    assert (tmp_path / "tests" / "test_pipeline.py").is_file()
    assert not (tmp_path / "tests" / "test_engineering_criteria.py").exists()


# --------------------------------------------------------------------------- #
# (b) Cobertura
# --------------------------------------------------------------------------- #
def _cobertura(archivos: dict[str, float]) -> dict:
    return {
        "meta": {"timestamp": "2026-09-27T10:00:00"},
        "totals": {"percent_covered": 99.0, "num_statements": 100},
        "files": {
            nombre: {"summary": {"percent_covered": valor, "num_statements": 10}}
            for nombre, valor in archivos.items()
        },
    }


def test_a_new_module_just_below_the_threshold_fails_even_if_it_rounds_to_it() -> None:
    """Prueba por sabotaje del redondeo: 69,96 % se muestra como 70,0 % y no cumple."""
    resultado = criterios.criterio_cobertura(
        _cobertura({"autopipe\\nuevo.py": 69.96, "autopipe\\otro.py": 100.0}), set()
    )

    assert resultado["por_debajo"] == ["autopipe/nuevo.py"]
    assert not resultado["cumple"]


def test_migrated_modules_are_outside_the_criterion() -> None:
    """El criterio es sobre los módulos nuevos; un migrado bajo no lo hace fallar."""
    resultado = criterios.criterio_cobertura(
        _cobertura({"autopipe\\viejo.py": 10.0, "autopipe\\nuevo.py": 95.0}),
        {"autopipe/viejo.py"},
    )

    assert [m["modulo"] for m in resultado["modulos"]] == ["autopipe/nuevo.py"]
    assert resultado["cumple"]


def test_the_lowest_new_module_is_reported_first() -> None:
    resultado = criterios.criterio_cobertura(
        _cobertura({"autopipe/a.py": 100.0, "autopipe/b.py": 88.0}), set()
    )

    assert resultado["minimo"]["modulo"] == "autopipe/b.py"


def test_without_new_modules_the_criterion_is_not_met() -> None:
    assert not criterios.criterio_cobertura(_cobertura({"autopipe/a.py": 100.0}),
                                            {"autopipe/a.py"})["cumple"]


# --------------------------------------------------------------------------- #
# (c) Reproducibilidad
# --------------------------------------------------------------------------- #
def _documento(*estados: str) -> str:
    filas = "\n".join(f"| Verificación {i} | algo | **{e}** | detalle |"
                      for i, e in enumerate(estados))
    cabecera = "| Verificación | Qué compara | Estado | Detalle |\n|---|---|---|---|"
    return f"# Reproducibilidad\n\n{cabecera}\n{filas}\n"


def test_reproducibility_passes_when_nothing_differs() -> None:
    resultado = criterios.criterio_reproducibilidad(_documento("idéntico", "idéntico", "omitido"))

    assert resultado == {"verificaciones": 3, "identicas": 2, "difieren": 0, "omitidas": 1,
                         "cumple": True}


def test_a_single_difference_fails_reproducibility() -> None:
    assert not criterios.criterio_reproducibilidad(_documento("idéntico", "difiere"))["cumple"]


def test_only_omissions_are_not_evidence() -> None:
    """Prueba por sabotaje: sin ninguna verificación idéntica no hay reproducibilidad que citar."""
    assert not criterios.criterio_reproducibilidad(_documento("omitido", "omitido"))["cumple"]


def test_the_states_match_the_ones_the_reproducibility_script_writes() -> None:
    pytest.importorskip("openml", reason="el verificador importa los arneses")
    import verify_reproducibility as verificador

    assert (criterios.IDENTICO, criterios.DIFIERE, criterios.OMITIDO) == (
        verificador.IDENTICO, verificador.DIFIERE, verificador.OMITIDO,
    )
    registro = {"verificacion": "v", "compara": "c", "estado": verificador.IDENTICO,
                "detalle": "d"}
    assert criterios.criterio_reproducibilidad(verificador.redactar([registro]))["identicas"] == 1


# --------------------------------------------------------------------------- #
# (d) Instrucciones
# --------------------------------------------------------------------------- #
def test_the_count_matches_the_one_in_the_examples_test() -> None:
    """Dos implementaciones del mismo conteo tienen que dar lo mismo sobre los ejemplos."""
    from tests.test_examples import _contar_instrucciones

    for ejemplo in criterios.EJEMPLOS:
        assert criterios.contar_instrucciones(ejemplo) == _contar_instrucciones(ejemplo)


def test_a_flow_above_the_limit_fails(tmp_path: Path) -> None:
    largo = tmp_path / "largo.py"
    largo.write_text('"""Docstring."""\n' + "x = 1\n" * 21, encoding="utf-8")
    corto = tmp_path / "corto.py"
    corto.write_text("x = 1\n" * 20, encoding="utf-8")

    assert criterios.criterio_instrucciones([corto])["cumple"]
    resultado = criterios.criterio_instrucciones([corto, largo])
    assert [f["instrucciones"] for f in resultado["ejemplos"]] == [20, 21]
    assert not resultado["cumple"]


# --------------------------------------------------------------------------- #
# El documento y el punto de entrada
# --------------------------------------------------------------------------- #
def _mediciones(regresion_cumple: bool = True) -> tuple[dict, dict, dict, dict]:
    origen = criterios.RAIZ / "autopipe" / "__init__.py"
    ejecucion = {"passed": 134 if regresion_cumple else 133, "failed": 0 if regresion_cumple
                 else 1, "errors": 0}
    comparacion = _comparacion(110)
    comparacion["modificadas"] = [] if regresion_cumple else ["tests/test_x.py::test_y"]
    regresion = criterios.evaluar_regresion(comparacion, {"collected": 134}, ejecucion, origen)
    cobertura = criterios.criterio_cobertura(_cobertura({"autopipe/nuevo.py": 99.4}), set())
    reproducibilidad = criterios.criterio_reproducibilidad(_documento("idéntico"))
    instrucciones = {"ejemplos": [{"ejemplo": "examples/a.py", "instrucciones": 13}],
                     "cumple": True}
    return regresion, cobertura, reproducibilidad, instrucciones


def test_the_document_states_the_verdict_and_names_edited_tests() -> None:
    texto = criterios.redactar(*_mediciones(regresion_cumple=False), suite=None)

    assert "**3 de 4 criterios cumplidos**" in texto
    assert "`tests/test_x.py::test_y`" in texto
    assert "no se ejecutó en esta corrida" in texto


def test_the_document_reports_all_four_when_they_hold() -> None:
    texto = criterios.redactar(*_mediciones(), suite={"passed": 837, "failed": 0, "errors": 0})

    assert "**4 de 4 criterios cumplidos**" in texto
    assert "837 pasan, 0 fallan, 0 errores" in texto
    assert "99.4 %" in texto


def test_it_refuses_to_run_without_the_reproducibility_document(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(criterios, "REPRODUCIBILIDAD", tmp_path / "no-existe.md")
    monkeypatch.setattr(sys, "argv", ["x", "--sin-suite"])

    with pytest.raises(SystemExit, match="verify_reproducibility"):
        criterios.main()


@pytest.mark.parametrize("regresion_cumple", [True, False])
def test_main_writes_the_document_and_exits_by_the_verdict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, regresion_cumple: bool
) -> None:
    regresion, _, _, _ = _mediciones(regresion_cumple)
    documento = tmp_path / "REPRODUCIBILIDAD.md"
    documento.write_text(_documento("idéntico"), encoding="utf-8")
    monkeypatch.setattr(criterios, "REPRODUCIBILIDAD", documento)
    monkeypatch.setattr(criterios, "SALIDA", tmp_path / "CRITERIOS.md")
    monkeypatch.setattr(criterios, "criterio_regresion", lambda: regresion)
    monkeypatch.setattr(
        criterios, "medir_cobertura",
        lambda ejecutar_suite: (None, _cobertura({"autopipe/nuevo.py": 99.4})),
    )
    monkeypatch.setattr(criterios, "modulos_migrados", lambda: set())
    monkeypatch.setattr(sys, "argv", ["x", "--sin-suite"])

    assert criterios.main() == (0 if regresion_cumple else 1)
    cumplidos = 4 if regresion_cumple else 3
    texto = (tmp_path / "CRITERIOS.md").read_text(encoding="utf-8")
    assert f"**{cumplidos} de 4 criterios cumplidos**" in texto
