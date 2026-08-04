# tests/test_reporter.py
"""Pruebas unitarias para autopipe.reporter.report.ReportGenerator."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from autopipe.pipeline.pipeline import Pipeline
from autopipe.pipeline.transforms import (
    CLAHETransform,
    GammaCorrection,
    GaussianBlur,
    GrayscaleConversion,
    Normalize,
)
from autopipe.reporter import report as report_module
from autopipe.reporter.report import ReportGenerator


@pytest.fixture
def best_pipeline() -> Pipeline:
    return Pipeline([GaussianBlur(kernel_size=3)])


@pytest.fixture
def results(best_pipeline: Pipeline) -> list[dict]:
    losing = Pipeline([GaussianBlur(kernel_size=5)])
    metrics_best = {
        "auc_roc": 1.0,
        "f1": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "false_positive_rate": 0.0,
    }
    metrics_worse = {
        "auc_roc": 0.5,
        "f1": 0.4,
        "precision": 0.5,
        "recall": 0.3,
        "false_positive_rate": 0.6,
    }
    return [
        {
            "pipeline_index": 0,
            "params": best_pipeline.get_params(),
            "metrics": metrics_best,
            "objective": 1.0,
        },
        {
            "pipeline_index": 1,
            "params": losing.get_params(),
            "metrics": metrics_worse,
            "objective": 0.5,
        },
    ]


def test_generate_html_always(tmp_path: Path, results, best_pipeline) -> None:
    generator = ReportGenerator(output_dir=tmp_path)
    paths = generator.generate(results, best_pipeline, results[0]["metrics"], pdf=False)

    assert paths["html"].exists()
    assert paths["pdf"] is None
    assert paths["config"].exists()


def test_html_contains_all_sections(tmp_path: Path, results, best_pipeline) -> None:
    generator = ReportGenerator(output_dir=tmp_path)
    generator.generate(results, best_pipeline, results[0]["metrics"], metric="auc_roc", pdf=False)
    html = (tmp_path / "report.html").read_text(encoding="utf-8")

    assert "Tabla comparativa" in html
    assert "Evolución de la métrica" in html
    assert "Pipeline ganador" in html
    assert "data:image/png;base64," in html  # gráfica de evolución embebida
    assert "kernel_size" in html  # configuración del ganador
    assert 'class="best"' in html  # fila ganadora destacada


def test_anomaly_gallery_included(tmp_path: Path, results, best_pipeline) -> None:
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, size=(6, 8, 8, 3), dtype=np.uint8)
    scores = np.array([0.1, 0.9, 0.5, 0.8, 0.2, 0.7])
    generator = ReportGenerator(output_dir=tmp_path, top_anomalies=3)
    generator.generate(
        results, best_pipeline, results[0]["metrics"], images=images, scores=scores, pdf=False
    )
    html = (tmp_path / "report.html").read_text(encoding="utf-8")

    assert "Imágenes anómalas destacadas" in html
    assert html.count("<figure>") == 3


def test_no_gallery_without_images(tmp_path: Path, results, best_pipeline) -> None:
    generator = ReportGenerator(output_dir=tmp_path)
    generator.generate(results, best_pipeline, results[0]["metrics"], pdf=False)
    html = (tmp_path / "report.html").read_text(encoding="utf-8")

    assert "Imágenes anómalas destacadas" not in html


def test_pdf_generated_when_available(tmp_path: Path, results, best_pipeline, monkeypatch) -> None:
    def fake_from_file(html_path: str, pdf_path: str) -> None:
        Path(pdf_path).write_bytes(b"%PDF-1.4 fake")

    monkeypatch.setattr(report_module.pdfkit, "from_file", fake_from_file)
    generator = ReportGenerator(output_dir=tmp_path)
    paths = generator.generate(results, best_pipeline, results[0]["metrics"], pdf=True)

    assert paths["pdf"] is not None
    assert paths["pdf"].exists()


def test_pdf_tolerant_when_missing(
    tmp_path: Path, results, best_pipeline, monkeypatch, caplog
) -> None:
    def raise_missing(html_path: str, pdf_path: str) -> None:
        raise OSError("No wkhtmltopdf executable found")

    monkeypatch.setattr(report_module.pdfkit, "from_file", raise_missing)
    generator = ReportGenerator(output_dir=tmp_path)
    with caplog.at_level(logging.WARNING, logger="autopipe.reporter.report"):
        paths = generator.generate(results, best_pipeline, results[0]["metrics"], pdf=True)

    assert paths["pdf"] is None
    assert paths["html"].exists()
    assert any("PDF" in record.message for record in caplog.records)


@pytest.mark.parametrize("config_format", ["json", "yaml", "yml"])
def test_save_load_and_rebuild_roundtrip(tmp_path: Path, config_format: str) -> None:
    original = Pipeline(
        [
            GaussianBlur(kernel_size=3),
            Normalize(mean=0.5, std=0.5),
            CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8)),
        ]
    )
    generator = ReportGenerator(output_dir=tmp_path)
    config_path = generator.save_pipeline_config(original, config_format=config_format)

    loaded = ReportGenerator.load_pipeline_config(config_path)
    rebuilt = ReportGenerator.rebuild_pipeline(loaded)
    assert rebuilt.get_params() == original.get_params()


def test_rebuild_roundtrip_with_gamma_and_grayscale(tmp_path: Path) -> None:
    original = Pipeline(
        [
            GammaCorrection(gamma=2.2),
            GrayscaleConversion(),
        ]
    )
    generator = ReportGenerator(output_dir=tmp_path)
    config_path = generator.save_pipeline_config(original, config_format="json")

    loaded = ReportGenerator.load_pipeline_config(config_path)
    rebuilt = ReportGenerator.rebuild_pipeline(loaded)
    assert rebuilt.get_params() == original.get_params()


@pytest.mark.parametrize("config_format", ["json", "yaml"])
def test_skipped_steps_roundtrip(tmp_path: Path, config_format: str) -> None:
    """Un pipeline con pasos omitidos por la búsqueda serializa y reconstruye igual.

    Los pasos omitidos no aparecen en la configuración; los índices de los
    presentes quedan contiguos (re-enumerados), y rebuild_pipeline los respeta.
    """
    # Simula un ganador donde la búsqueda omitió el paso intermedio (blur).
    winner = Pipeline([GaussianBlur(kernel_size=3), Normalize(mean=0.5, std=0.5)])
    generator = ReportGenerator(output_dir=tmp_path)
    config_path = generator.save_pipeline_config(winner, config_format=config_format)

    loaded = ReportGenerator.load_pipeline_config(config_path)
    assert list(loaded) == ["0_GaussianBlur", "1_Normalize"]  # índices contiguos

    rebuilt = ReportGenerator.rebuild_pipeline(loaded)
    assert rebuilt.get_params() == winner.get_params()


@pytest.mark.parametrize("config_format", ["json", "yaml"])
def test_empty_pipeline_roundtrip(tmp_path: Path, config_format: str) -> None:
    """El pipeline vacío (todos los pasos omitidos) también hace roundtrip."""
    generator = ReportGenerator(output_dir=tmp_path)
    config_path = generator.save_pipeline_config(Pipeline([]), config_format=config_format)

    loaded = ReportGenerator.load_pipeline_config(config_path)
    rebuilt = ReportGenerator.rebuild_pipeline(loaded or {})

    assert rebuilt.steps == []
    assert rebuilt.get_params() == {}


def test_save_config_invalid_format_raises(tmp_path: Path, best_pipeline) -> None:
    generator = ReportGenerator(output_dir=tmp_path)
    with pytest.raises(ValueError):
        generator.save_pipeline_config(best_pipeline, config_format="txt")


def test_load_config_invalid_extension_raises(tmp_path: Path) -> None:
    bogus = tmp_path / "config.txt"
    bogus.write_text("nope", encoding="utf-8")
    with pytest.raises(ValueError):
        ReportGenerator.load_pipeline_config(bogus)


def test_rebuild_unknown_transform_raises() -> None:
    with pytest.raises(ValueError):
        ReportGenerator.rebuild_pipeline({"0_MysteryTransform": {"foo": 1}})


# --------------------------------------------------------------------------- #
# Resolución de la carpeta de salida por defecto
# --------------------------------------------------------------------------- #
def test_default_output_dir_next_to_invoking_script(tmp_path: Path) -> None:
    """Sin output_dir, los archivos van a reports/ junto al script, no al CWD."""
    script = tmp_path / "mi_script.py"
    script.write_text(
        "from autopipe.pipeline.pipeline import Pipeline\n"
        "from autopipe.pipeline.transforms import GaussianBlur\n"
        "from autopipe.reporter.report import ReportGenerator\n"
        "ReportGenerator().save_pipeline_config(Pipeline([GaussianBlur(kernel_size=3)]))\n",
        encoding="utf-8",
    )
    other_cwd = tmp_path / "otro_directorio"
    other_cwd.mkdir()

    subprocess.run([sys.executable, str(script)], cwd=other_cwd, check=True)

    assert (tmp_path / "reports" / "winning_pipeline.json").exists()
    assert not (other_cwd / "reports").exists()


def test_default_output_dir_interactive_uses_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Sin __main__.__file__ (Jupyter/REPL) se usa el CWD del proceso."""
    monkeypatch.delattr(sys.modules["__main__"], "__file__", raising=False)
    monkeypatch.chdir(tmp_path)

    generator = ReportGenerator()

    assert generator.output_dir == tmp_path / "reports"


def test_default_output_dir_ignores_site_packages_main(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Un __main__ dentro de site-packages (python -m herramienta) no se usa."""
    fake_main = tmp_path / "site-packages" / "tool" / "__main__.py"
    monkeypatch.setattr(sys.modules["__main__"], "__file__", str(fake_main), raising=False)
    monkeypatch.chdir(tmp_path)

    assert ReportGenerator._default_output_dir() == tmp_path / "reports"


def test_explicit_output_dir_is_respected(tmp_path: Path) -> None:
    generator = ReportGenerator(output_dir=tmp_path / "custom")
    assert generator.output_dir == tmp_path / "custom"

@pytest.mark.parametrize(
    "array",
    [
        np.linspace(-1.0, 2.0, 64).reshape(8, 8),  # float con rango -> min-máx
        np.full((8, 8), 0.5, dtype=np.float64),  # float constante -> ceros
        np.zeros((8, 8, 3), dtype=np.uint8),  # uint8 se usa tal cual
    ],
)
def test_array_to_data_uri_handles_dtypes(array: np.ndarray) -> None:
    uri = ReportGenerator._array_to_data_uri(array)
    assert uri.startswith("data:image/png;base64,")
