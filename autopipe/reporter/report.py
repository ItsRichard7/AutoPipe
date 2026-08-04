# autopipe/reporter/report.py
"""Generación de reportes HTML/PDF y reproducibilidad del pipeline ganador."""

from __future__ import annotations

import base64
import io
import json
import logging
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pdfkit  # noqa: E402
import yaml  # noqa: E402
from jinja2 import Template  # noqa: E402
from PIL import Image  # noqa: E402

from autopipe.pipeline.pipeline import Pipeline  # noqa: E402
from autopipe.pipeline.transforms import (  # noqa: E402
    CLAHETransform,
    GammaCorrection,
    GaussianBlur,
    GrayscaleConversion,
    MedianFilter,
    Normalize,
    Resize,
)

logger = logging.getLogger(__name__)

_TRANSFORM_REGISTRY = {
    "GaussianBlur": GaussianBlur,
    "Normalize": Normalize,
    "Resize": Resize,
    "CLAHETransform": CLAHETransform,
    "MedianFilter": MedianFilter,
    "GammaCorrection": GammaCorrection,
    "GrayscaleConversion": GrayscaleConversion,
}

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>{{ title }}</title>
<style>
body { font-family: Arial, sans-serif; margin: 2rem; color: #222; }
h1 { color: #1a3c5e; }
h2 { color: #1a3c5e; border-bottom: 2px solid #dde; padding-bottom: .3rem; }
table { border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; }
th, td { border: 1px solid #ccc; padding: .4rem .6rem; text-align: left; font-size: .9rem; }
th { background: #f0f4f8; }
tr.best { background: #e6f7e6; font-weight: bold; }
pre { background: #f7f7f7; padding: 1rem; border-radius: 6px; overflow-x: auto; }
code { font-family: Consolas, monospace; }
.gallery { display: flex; flex-wrap: wrap; gap: 1rem; }
figure { margin: 0; text-align: center; }
figure img { max-width: 160px; border: 1px solid #ccc; }
figcaption { font-size: .8rem; color: #555; }
img.chart { max-width: 100%; }
</style>
</head>
<body>
<h1>{{ title }}</h1>
<p>Generado: {{ generated_at }} &middot; Métrica objetivo: <strong>{{ metric }}</strong></p>

<h2>Pipeline ganador</h2>
<table>
<thead><tr>{% for col in metric_columns %}<th>{{ col }}</th>{% endfor %}</tr></thead>
<tbody><tr>
{% for col in metric_columns %}<td>{{ '%.4f'|format(best_metrics[col]) }}</td>{% endfor %}
</tr></tbody>
</table>
<pre>{{ best_params_str }}</pre>

<h2>Evolución de la métrica objetivo</h2>
<img class="chart" src="{{ chart_uri }}" alt="Evolución de {{ metric }}">

<h2>Tabla comparativa de pipelines</h2>
<table>
<thead><tr><th>#</th><th>Parámetros</th>
{% for col in metric_columns %}<th>{{ col }}</th>{% endfor %}
</tr></thead>
<tbody>
{% for row in table %}
<tr class="{{ 'best' if row.is_best else '' }}">
<td>{{ row.index }}</td>
<td><code>{{ row.params }}</code></td>
{% for col in metric_columns %}<td>{{ '%.4f'|format(row.metrics[col]) }}</td>{% endfor %}
</tr>
{% endfor %}
</tbody>
</table>

{% if anomalies %}
<h2>Imágenes anómalas destacadas</h2>
<div class="gallery">
{% for item in anomalies %}
<figure>
<img src="{{ item.uri }}" alt="Anomalía {{ item.index }}">
<figcaption>#{{ item.index }} &middot; score {{ '%.4f'|format(item.score) }}</figcaption>
</figure>
{% endfor %}
</div>
{% endif %}
</body>
</html>
"""


class ReportGenerator:
    """Genera reportes reproducibles de la búsqueda de pipelines.

    El reporte HTML se genera siempre con Jinja2 e incluye la tabla comparativa
    de todos los pipelines, la gráfica de evolución de la métrica objetivo, la
    configuración del pipeline ganador y las imágenes anómalas de mayor score.
    La exportación a PDF (pdfkit + wkhtmltopdf) es opcional y tolerante a la
    ausencia del binario ``wkhtmltopdf``.

    Attributes:
        output_dir (Path): Directorio de salida de los reportes.
        top_anomalies (int): Número de imágenes anómalas a destacar.
    """

    def __init__(self, output_dir: str | Path | None = None, top_anomalies: int = 5) -> None:
        """Inicializa el generador de reportes.

        Args:
            output_dir: Directorio donde se escriben los reportes. Si es
                ``None`` (por defecto) se usa una subcarpeta ``reports/`` junto
                al script que invoca la librería; en entornos interactivos
                (Jupyter, REPL) se usa el directorio de trabajo, que Jupyter
                fija al directorio del cuaderno. La carpeta se crea
                automáticamente al generar el primer archivo.
            top_anomalies: Número de imágenes anómalas de mayor score a destacar.
        """
        self.output_dir = (
            Path(output_dir) if output_dir is not None else self._default_output_dir()
        )
        self.top_anomalies = top_anomalies

    @staticmethod
    def _default_output_dir() -> Path:
        """Resuelve la carpeta de reportes por defecto de forma portable.

        Devuelve ``reports/`` junto al script principal
        (``__main__.__file__``). Si el proceso no tiene script principal
        (Jupyter, REPL) o este vive dentro de ``site-packages`` (ejecutores
        tipo ``python -m herramienta``), se usa el directorio de trabajo
        actual, que en Jupyter corresponde al directorio del cuaderno.

        Returns:
            Ruta absoluta de la subcarpeta ``reports/`` por defecto.
        """
        main_file = getattr(sys.modules.get("__main__"), "__file__", None)
        if main_file is not None:
            main_path = Path(main_file).resolve()
            if "site-packages" not in main_path.parts:
                return main_path.parent / "reports"
        return Path.cwd() / "reports"

    def generate(
        self,
        results: list[dict],
        best_pipeline: Pipeline,
        best_metrics: dict,
        metric: str = "auc_roc",
        images: Sequence[np.ndarray] | None = None,
        scores: np.ndarray | None = None,
        pdf: bool = True,
        config_format: str = "json",
    ) -> dict:
        """Genera el reporte HTML (siempre) y, opcionalmente, el PDF.

        Args:
            results: Tabla de resultados de la búsqueda (lista de diccionarios).
            best_pipeline: Pipeline ganador.
            best_metrics: Métricas del pipeline ganador.
            metric: Métrica objetivo optimizada.
            images: Imágenes del dataset, para destacar las anómalas (opcional).
            scores: Puntajes de anomalía alineados con ``images`` (opcional).
            pdf: Si se intenta exportar a PDF.
            config_format: Formato de la configuración del ganador (``json`` o ``yaml``).

        Returns:
            Diccionario con las rutas ``html``, ``pdf`` (``None`` si no se generó)
            y ``config``.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        context = self._build_context(results, best_pipeline, best_metrics, metric, images, scores)
        html = Template(_HTML_TEMPLATE).render(**context)

        html_path = self.output_dir / "report.html"
        html_path.write_text(html, encoding="utf-8")

        config_path = self.save_pipeline_config(best_pipeline, config_format)
        pdf_path = self._export_pdf(html_path) if pdf else None
        logger.info("Reporte HTML generado en %s.", html_path)
        return {"html": html_path, "pdf": pdf_path, "config": config_path}

    def save_pipeline_config(self, pipeline: Pipeline, config_format: str = "json") -> Path:
        """Guarda la configuración del pipeline ganador para re-ejecución.

        Args:
            pipeline: Pipeline a serializar.
            config_format: Formato de salida (``json`` o ``yaml``).

        Returns:
            Ruta del archivo de configuración escrito.

        Raises:
            ValueError: Si ``config_format`` no es ``json``, ``yaml`` ni ``yml``.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        params = pipeline.get_params()
        config_format = config_format.lower()
        if config_format == "json":
            path = self.output_dir / "winning_pipeline.json"
            path.write_text(json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8")
        elif config_format in ("yaml", "yml"):
            path = self.output_dir / f"winning_pipeline.{config_format}"
            path.write_text(yaml.safe_dump(params), encoding="utf-8")
        else:
            raise ValueError(
                f"Formato no soportado: {config_format!r}. Use 'json', 'yaml' o 'yml'."
            )
        return path

    @staticmethod
    def load_pipeline_config(path: str | Path) -> dict:
        """Carga la configuración de un pipeline desde JSON o YAML.

        Args:
            path: Ruta del archivo de configuración.

        Returns:
            Diccionario de parámetros del pipeline.

        Raises:
            ValueError: Si la extensión no es ``.json``, ``.yaml`` ni ``.yml``.
        """
        path = Path(path)
        content = path.read_text(encoding="utf-8")
        suffix = path.suffix.lower()
        if suffix == ".json":
            return json.loads(content)
        if suffix in (".yaml", ".yml"):
            return yaml.safe_load(content)
        raise ValueError(
            f"Extensión no soportada: {path.suffix!r}. Use .json, .yaml o .yml."
        )

    @staticmethod
    def rebuild_pipeline(config: dict) -> Pipeline:
        """Reconstruye un pipeline de transformaciones predefinidas desde su config.

        Permite re-ejecutar la búsqueda ganadora a partir de un archivo guardado.
        Solo soporta las transformaciones predefinidas de la librería.

        Args:
            config: Diccionario de parámetros tal como lo produce
                :meth:`~autopipe.pipeline.pipeline.Pipeline.get_params`.

        Returns:
            El pipeline reconstruido.

        Raises:
            ValueError: Si alguna transformación no está registrada.
        """
        steps = []
        for key in sorted(config, key=lambda item: int(item.split("_", 1)[0])):
            class_name = key.split("_", 1)[1]
            if class_name not in _TRANSFORM_REGISTRY:
                raise ValueError(f"Transformación no registrada: {class_name!r}.")
            params = dict(config[key])
            if "tile_grid_size" in params:
                params["tile_grid_size"] = tuple(params["tile_grid_size"])
            steps.append(_TRANSFORM_REGISTRY[class_name](**params))
        return Pipeline(steps)

    def _build_context(
        self,
        results: list[dict],
        best_pipeline: Pipeline,
        best_metrics: dict,
        metric: str,
        images: Sequence[np.ndarray] | None,
        scores: np.ndarray | None,
    ) -> dict:
        """Construye el contexto de renderizado de la plantilla HTML.

        Args:
            results: Tabla de resultados de la búsqueda.
            best_pipeline: Pipeline ganador.
            best_metrics: Métricas del pipeline ganador.
            metric: Métrica objetivo.
            images: Imágenes del dataset (opcional).
            scores: Puntajes de anomalía (opcional).

        Returns:
            Diccionario con todas las variables de la plantilla.
        """
        best_params = best_pipeline.get_params()
        metric_columns = sorted({key for record in results for key in record["metrics"]})
        table = [
            {
                "index": record.get("pipeline_index", record.get("trial", index)),
                "params": json.dumps(record["params"], ensure_ascii=False),
                "metrics": record["metrics"],
                "is_best": record["params"] == best_params,
            }
            for index, record in enumerate(results)
        ]
        return {
            "title": "AutoPipe — Reporte de búsqueda de pipelines",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "metric": metric,
            "metric_columns": metric_columns or sorted(best_metrics),
            "best_metrics": best_metrics,
            "best_params_str": json.dumps(best_params, indent=2, ensure_ascii=False),
            "chart_uri": self._build_evolution_chart(results, metric),
            "table": table,
            "anomalies": self._build_anomaly_gallery(images, scores),
        }

    def _build_evolution_chart(self, results: list[dict], metric: str) -> str:
        """Genera la gráfica de evolución de la métrica como data URI PNG.

        Args:
            results: Tabla de resultados de la búsqueda.
            metric: Métrica objetivo.

        Returns:
            Data URI con la imagen PNG de la gráfica codificada en base64.
        """
        objectives = [record["objective"] for record in results]
        positions = list(range(len(objectives)))
        figure, axis = plt.subplots(figsize=(7, 3.5))
        axis.plot(positions, objectives, marker="o", color="#1a3c5e")
        axis.set_xlabel("Pipeline evaluado")
        axis.set_ylabel(metric)
        axis.set_title(f"Evolución de {metric}")
        axis.grid(True, linestyle="--", alpha=0.5)
        figure.tight_layout()

        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=100)
        plt.close(figure)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def _build_anomaly_gallery(
        self, images: Sequence[np.ndarray] | None, scores: np.ndarray | None
    ) -> list[dict]:
        """Selecciona y codifica las imágenes de mayor score de anomalía.

        Args:
            images: Imágenes del dataset.
            scores: Puntajes de anomalía alineados con ``images``.

        Returns:
            Lista de diccionarios con ``index``, ``score`` y ``uri`` por imagen,
            o una lista vacía si no se proporcionan imágenes y puntajes.
        """
        if images is None or scores is None:
            return []
        scores = np.asarray(scores)
        order = np.argsort(scores)[::-1][: self.top_anomalies]
        return [
            {
                "index": int(idx),
                "score": float(scores[idx]),
                "uri": self._array_to_data_uri(images[idx]),
            }
            for idx in order
        ]

    @staticmethod
    def _array_to_data_uri(array: np.ndarray) -> str:
        """Convierte una imagen NumPy en un data URI PNG para incrustar en HTML.

        Las imágenes que no son ``uint8`` se reescalan a [0, 255] por min-máx.

        Args:
            array: Imagen como arreglo NumPy (2D escala de grises o 3D RGB).

        Returns:
            Data URI con la imagen PNG codificada en base64.
        """
        array = np.asarray(array)
        if array.dtype != np.uint8:
            array = array.astype(np.float64)
            minimum, maximum = array.min(), array.max()
            if maximum > minimum:
                array = (array - minimum) / (maximum - minimum) * 255.0
            else:
                array = np.zeros_like(array)
            array = array.astype(np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(array).save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def _export_pdf(self, html_path: Path) -> Path | None:
        """Intenta exportar el reporte HTML a PDF de forma tolerante a fallos.

        Args:
            html_path: Ruta del reporte HTML ya generado.

        Returns:
            Ruta del PDF, o ``None`` si no se pudo generar (por ejemplo, si
            ``wkhtmltopdf`` no está disponible).
        """
        pdf_path = self.output_dir / "report.pdf"
        try:
            pdfkit.from_file(str(html_path), str(pdf_path))
        except (OSError, ValueError) as exc:
            logger.warning(
                "No se pudo generar el PDF (¿wkhtmltopdf instalado?): %s. "
                "Se conserva el reporte HTML.",
                exc,
            )
            return None
        logger.info("Reporte PDF generado en %s.", pdf_path)
        return pdf_path
