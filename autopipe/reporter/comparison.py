# autopipe/reporter/comparison.py
"""Reporte comparativo del experimento *leave-one-dataset-out* (RE6).

:class:`~autopipe.reporter.report.ReportGenerator` documenta **una** búsqueda: qué
pipelines se probaron sobre un dataset y cuál ganó. El experimento del TFG pregunta
otra cosa —si el arranque en caliente encuentra el objetivo antes que el frío, sobre
veinticinco datasets y cinco semillas— y esa pregunta no cabe en aquella plantilla.
Este módulo la responde extendiendo el generador existente en lugar de duplicarlo:
hereda la resolución del directorio de salida, la exportación tolerante a PDF y la
codificación de gráficas, y añade solo lo comparativo.

**Qué recibe.** El diccionario de análisis que produce ``scripts/run_lodo.py``. El
reporte no calcula ningún estadístico: los lee. Es deliberado —el análisis lo fija
el pre-registro y vive junto al arnés que lo ejecuta— y evita que existan dos
implementaciones del mismo contraste que puedan discrepar.

**Qué no hace.** No decide si la hipótesis se confirma. Lee el veredicto que ya
viene calculado contra el criterio del anteproyecto y lo enuncia, favorable o no.
Un reporte que suavizara un resultado adverso no serviría para lo único que tiene
que servir: sostener el capítulo de resultados ante el tribunal.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from jinja2 import Template

from autopipe.reporter.report import ReportGenerator

logger = logging.getLogger(__name__)

Contraste = Mapping[str, Any]
"""Un contraste estadístico tal como lo produce el arnés del LODO."""

Fila = Mapping[str, Any]
"""Una fila de la tabla por conjunto de datos."""

CONDICION_PRIMARIA = "nearest"
"""Condición caliente que produce el contraste primario del pre-registro."""

_COLORES = {
    "frio": "#1a3c5e",
    "nearest": "#c0392b",
    "random": "#7f8c8d",
    "frequent": "#e67e22",
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
td.num { text-align: right; font-variant-numeric: tabular-nums; }
tr.mejor { background: #e6f7e6; }
tr.peor { background: #fdeaea; }
p.veredicto { border-left: 6px solid {{ veredicto_color }}; background: #f7f7f7;
              padding: .8rem 1rem; font-size: 1.05rem; }
dl.protocolo { display: grid; grid-template-columns: max-content 1fr; gap: .2rem 1rem; }
dl.protocolo dt { font-weight: bold; color: #1a3c5e; }
dl.protocolo dd { margin: 0; }
img.chart { max-width: 100%; }
small { color: #555; }
</style>
</head>
<body>
<h1>{{ title }}</h1>
<p>Generado: {{ generated_at }} &middot; Pares: <strong>{{ primario.n_pares }}</strong>
&middot; Variable: <strong>ensayos hasta el objetivo</strong></p>

<h2>Protocolo ejecutado</h2>
<dl class="protocolo">
{% for clave, valor in protocolo %}<dt>{{ clave }}</dt><dd>{{ valor }}</dd>{% endfor %}
</dl>

<h2>Contraste primario: {{ primario.estrategia }} contra frío</h2>
<p class="veredicto">{{ veredicto }}</p>
<table>
<thead><tr><th>Magnitud</th><th>Valor</th></tr></thead>
<tbody>
{% for etiqueta, valor in resumen %}
<tr><td>{{ etiqueta }}</td><td class="num">{{ valor }}</td></tr>
{% endfor %}
</tbody>
</table>

<h2>Diferencia por conjunto de datos</h2>
<img class="chart" src="{{ chart_diferencias }}"
     alt="Diferencia de ensayos por conjunto de datos">
<p><small>Barras a la izquierda del cero: el arranque en caliente alcanzó el objetivo
antes. A la derecha: lo alcanzó después.</small></p>

<h2>Medianas por condición</h2>
<img class="chart" src="{{ chart_condiciones }}" alt="Mediana de ensayos por condición">

<h2>Contraste primario y controles secundarios</h2>
<p><small>Las estrategias de control comparten todo con la primaria salvo cómo eligen
las cinco configuraciones. Si la primaria no se separa de ellas, la similitud entre
conjuntos de datos no está aportando nada. <strong>La hipótesis del trabajo se juega
únicamente en la fila primaria</strong>: que un control cumpla el criterio no la
confirma, y es en sí mismo un hallazgo sobre el control, no sobre el
meta-aprendizaje.</small></p>
{{ tabla_contrastes }}

<h2>Robustez al valor de censura</h2>
<p><small>El análisis primario castiga a las corridas que no alcanzan el objetivo con
{{ primario.censura }}. Estas filas lo recalculan con castigos más severos, fijados
antes de ejecutar.</small></p>
{{ tabla_robustez }}

<h2>Resultado por conjunto de datos</h2>
<p><small>Cada celda es la mediana de las cinco semillas. «Meseta» y «inevaluables»
son las dos covariables declaradas en el pre-registro, medidas antes de ejecutar.</small></p>
{{ tabla_datasets }}
</body>
</html>
"""


def _formatear(valor: object, decimales: int = 2) -> str:
    """Presenta un número con un ancho estable, o un guion si no hay valor.

    Args:
        valor: Número a presentar, o ``None``.
        decimales: Cifras decimales.

    Returns:
        La representación textual.
    """
    if valor is None:
        return "—"
    if isinstance(valor, bool):
        return "sí" if valor else "no"
    if isinstance(valor, (int, float)):
        return f"{valor:.{decimales}f}"
    return str(valor)


REDUCCION_EXIGIDA = 0.30
"""Reducción mínima de la mediana que exige el anteproyecto en su Sección 4.3.4."""

ALFA = 0.05
"""Nivel de significancia comprometido, de una cola."""


def _cumple(primario: Contraste) -> tuple[bool, bool]:
    """Comprueba por separado las dos condiciones del criterio comprometido.

    Args:
        primario: Contraste primario.

    Returns:
        Par ``(cumple_magnitud, cumple_significancia)``.
    """
    reduccion = primario.get("reduccion")
    p_valor = primario.get("p_una_cola")
    magnitud = isinstance(reduccion, (int, float)) and float(reduccion) >= REDUCCION_EXIGIDA
    significancia = isinstance(p_valor, (int, float)) and float(p_valor) < ALFA
    return magnitud, significancia


def redactar_veredicto(primario: Contraste) -> str:
    """Enuncia el resultado del contraste primario contra el criterio comprometido.

    El criterio del anteproyecto exige **las dos** condiciones a la vez: una
    reducción de al menos el 30 % de la mediana y un valor p de una cola por debajo
    de 0,05. Cuando falla, el texto dice cuál de las dos falló, porque una reducción
    grande sin significancia y una significancia sin magnitud son hallazgos
    distintos y se discuten distinto.

    El veredicto se **deriva de los números**, no se copia de la bandera
    ``confirma`` que trae el análisis. Si ambas discrepan, el texto lo denuncia en
    vez de elegir una: un reporte que presentara un veredicto sin respaldo en sus
    propias cifras es exactamente lo que no puede pasar en el capítulo de
    resultados.

    Args:
        primario: Contraste primario, tal como lo produce el arnés.

    Returns:
        El párrafo de veredicto.
    """
    reduccion = primario.get("reduccion")
    p_valor = primario.get("p_una_cola")
    frio = float(primario["mediana_frio"])  # type: ignore[arg-type]
    caliente = float(primario["mediana_caliente"])  # type: ignore[arg-type]
    cabecera = f"Mediana de {frio:.1f} ensayos en frío contra {caliente:.1f} en caliente."

    magnitud, significancia = _cumple(primario)
    confirma = magnitud and significancia
    porcentaje = (
        f"{float(reduccion) * 100:.1f} %" if isinstance(reduccion, (int, float)) else "—"
    )

    declarado = primario.get("confirma")
    if declarado is not None and bool(declarado) != confirma:
        return (
            f"{cabecera} El análisis declara «confirma={bool(declarado)}» pero sus "
            f"propias cifras dicen lo contrario: reducción "
            f"{_formatear(reduccion, 4)} y valor p de una cola "
            f"{_formatear(p_valor, 4)}. No se enuncia ningún veredicto hasta resolver "
            f"la discrepancia."
        )

    if confirma:
        return (
            f"{cabecera} La hipótesis se confirma: la reducción es del "
            f"{porcentaje} —el criterio exige un 30 %— con un valor p "
            f"de una cola de {_formatear(p_valor, 4)}, por debajo de 0,05."
        )

    if reduccion is None:
        return (
            f"{cabecera} La hipótesis no se confirma y el contraste tampoco es "
            f"interpretable en magnitud: la mediana del arranque en frío es cero y la "
            f"reducción relativa no está definida."
        )

    faltas = [
        f"la reducción observada es del {porcentaje} y el criterio exige un 30 %"
    ] if not magnitud else []
    if not significancia:
        faltas.append(
            f"el valor p de una cola es {_formatear(p_valor, 4)} y el criterio exige "
            f"menos de 0,05"
        )
    return (
        f"{cabecera} La hipótesis no se confirma: " + "; ".join(faltas) + ". "
        "El resultado se reporta tal como salió, que es lo que el pre-registro obliga."
    )


class ComparativeReportGenerator(ReportGenerator):
    """Genera el reporte comparativo del *leave-one-dataset-out*.

    Extiende :class:`~autopipe.reporter.report.ReportGenerator`: reutiliza la
    resolución del directorio de salida, la codificación de figuras y la
    exportación tolerante a PDF, y añade la plantilla comparativa. El generador
    original queda intacto, de modo que ningún consumidor suyo cambia.

    Attributes:
        output_dir (Path): Directorio de salida de los reportes.
    """

    def generate_comparison(
        self,
        analisis: Mapping[str, Any],
        protocolo: Mapping[str, Any] | None = None,
        pdf: bool = True,
        nombre: str = "lodo",
    ) -> dict:
        """Escribe el reporte comparativo en HTML y, si se puede, en PDF.

        Args:
            analisis: Diccionario de análisis del arnés, con las claves
                ``primario``, ``secundarios``, ``robustez`` y ``por_dataset``.
            protocolo: Pares de identificación del protocolo ejecutado (semillas,
                presupuesto, modelo base, versión del espacio). Se muestran tal
                como llegan.
            pdf: Si se intenta exportar a PDF.
            nombre: Base del nombre de archivo.

        Returns:
            Diccionario con las rutas ``html`` y ``pdf`` (``None`` si no se generó).

        Raises:
            KeyError: Si falta alguna de las claves obligatorias del análisis.
        """
        for clave in ("primario", "secundarios", "robustez", "por_dataset"):
            if clave not in analisis:
                raise KeyError(f"El análisis no trae la clave obligatoria {clave!r}.")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        contexto = self._build_comparison_context(analisis, protocolo)
        html = Template(_HTML_TEMPLATE).render(**contexto)
        html_path = self.output_dir / f"{nombre}.html"
        html_path.write_text(html, encoding="utf-8")

        pdf_path = self._export_pdf(html_path) if pdf else None
        logger.info("Reporte comparativo generado en %s.", html_path)
        return {"html": html_path, "pdf": pdf_path}

    def _export_pdf(self, html_path: Path) -> Path | None:
        """Exporta a PDF conservando el nombre base del reporte comparativo.

        Args:
            html_path: Reporte HTML ya escrito.

        Returns:
            La ruta del PDF, o ``None`` si no se pudo generar.
        """
        original = self.output_dir / "report.pdf"
        destino = html_path.with_suffix(".pdf")
        generado = super()._export_pdf(html_path)
        if generado is None:
            return None
        original.replace(destino)
        return destino

    def _build_comparison_context(
        self, analisis: Mapping[str, Any], protocolo: Mapping[str, Any] | None
    ) -> dict:
        """Arma el contexto de la plantilla comparativa.

        Args:
            analisis: Diccionario de análisis del arnés.
            protocolo: Pares de identificación del protocolo.

        Returns:
            Las variables de la plantilla.
        """
        primario = analisis["primario"]
        por_dataset = analisis["por_dataset"]
        confirma = all(_cumple(primario))
        return {
            "title": "AutoPipe — Arranque en frío contra caliente (LODO)",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "primario": primario,
            "protocolo": list((protocolo or {}).items()),
            "veredicto": redactar_veredicto(primario),
            "veredicto_color": "#2e7d32" if confirma else "#c0392b",
            "resumen": self._resumen(primario),
            "chart_diferencias": self._chart_diferencias(por_dataset, primario["estrategia"]),
            "chart_condiciones": self._chart_condiciones(por_dataset),
            "tabla_contrastes": self._tabla_contrastes(
                [
                    {**primario, "papel": "primario"},
                    *({**c, "papel": "control"} for c in analisis["secundarios"]),
                ]
            ),
            "tabla_robustez": self._tabla_contrastes(analisis["robustez"], con_censura=True),
            "tabla_datasets": self._tabla_datasets(por_dataset),
        }

    @staticmethod
    def _resumen(primario: Contraste) -> list[tuple[str, str]]:
        """Traduce el contraste primario a filas legibles.

        Args:
            primario: Contraste primario.

        Returns:
            Pares ``(etiqueta, valor)``.
        """
        reduccion = primario.get("reduccion")
        return [
            ("Mediana en frío (ensayos)", _formatear(primario["mediana_frio"], 1)),
            ("Mediana en caliente (ensayos)", _formatear(primario["mediana_caliente"], 1)),
            (
                "Reducción de la mediana",
                "—" if reduccion is None else f"{reduccion * 100:.1f} %",
            ),
            ("Valor p (una cola)", _formatear(primario.get("p_una_cola"), 4)),
            ("Valor p (dos colas)", _formatear(primario.get("p_dos_colas"), 4)),
            ("Corridas censuradas en frío", str(primario.get("censurados_frio", 0))),
            ("Corridas censuradas en caliente", str(primario.get("censurados_caliente", 0))),
            ("Criterio del anteproyecto", _formatear(primario.get("confirma"))),
        ]

    @staticmethod
    def _condiciones(por_dataset: Sequence[Fila]) -> list[str]:
        """Deduce las condiciones ejecutadas a partir de la tabla por dataset.

        Se deducen en vez de fijarse para que añadir una estrategia al arnés no
        obligue a tocar el reporte.

        Args:
            por_dataset: Filas por conjunto de datos.

        Returns:
            ``frio`` seguido de las condiciones calientes, en orden de aparición.
        """
        if not por_dataset:
            return []
        excluidas = {"dataset_id", "dataset_name", "meseta"}
        condiciones = [
            clave
            for clave in por_dataset[0]
            if clave not in excluidas
            and not clave.endswith(("_inevaluable", "_a_15"))
        ]
        return condiciones

    def _chart_diferencias(
        self, por_dataset: Sequence[Fila], estrategia: str
    ) -> str:
        """Dibuja la diferencia de ensayos por conjunto de datos.

        Es la gráfica que corresponde a la prueba: la de Wilcoxon opera sobre estas
        mismas diferencias pareadas, así que verlas ordenadas dice de un vistazo
        cuántos conjuntos van en cada dirección y con qué magnitud.

        Args:
            por_dataset: Filas por conjunto de datos.
            estrategia: Condición caliente del contraste primario.

        Returns:
            Data URI de la gráfica.
        """
        filas = sorted(por_dataset, key=lambda f: f[estrategia] - f["frio"])
        nombres = [str(f["dataset_name"])[:26] for f in filas]
        diferencias = [float(f[estrategia]) - float(f["frio"]) for f in filas]
        colores = ["#2e7d32" if d < 0 else "#c0392b" if d > 0 else "#95a5a6" for d in diferencias]

        figure, axis = plt.subplots(figsize=(8, max(3.0, 0.28 * len(filas) + 1.0)))
        axis.barh(nombres, diferencias, color=colores)
        axis.axvline(0, color="#222", linewidth=1)
        axis.set_xlabel(f"Ensayos en caliente ({estrategia}) menos ensayos en frío")
        axis.tick_params(axis="y", labelsize=8)
        axis.grid(True, axis="x", linestyle="--", alpha=0.5)
        figure.tight_layout()
        return self._figure_to_data_uri(figure)

    def _chart_condiciones(self, por_dataset: Sequence[Fila]) -> str:
        """Dibuja la mediana de ensayos de cada condición sobre todos los datasets.

        Args:
            por_dataset: Filas por conjunto de datos.

        Returns:
            Data URI de la gráfica.
        """
        condiciones = self._condiciones(por_dataset)
        figure, axis = plt.subplots(figsize=(7, 3.5))
        if condiciones and por_dataset:
            datos = [[float(f[c]) for f in por_dataset] for c in condiciones]
            cajas = axis.boxplot(datos, tick_labels=condiciones, patch_artist=True)
            for caja, condicion in zip(cajas["boxes"], condiciones):
                caja.set_facecolor(_COLORES.get(condicion, "#bdc3c7"))
                caja.set_alpha(0.65)
        axis.set_ylabel("Ensayos hasta el objetivo")
        axis.grid(True, axis="y", linestyle="--", alpha=0.5)
        figure.tight_layout()
        return self._figure_to_data_uri(figure)

    @staticmethod
    def _tabla_contrastes(
        contrastes: Sequence[Contraste], con_censura: bool = False
    ) -> str:
        """Arma la tabla HTML de una lista de contrastes.

        Args:
            contrastes: Contrastes a tabular.
            con_censura: Si se muestra la columna del valor de censura.

        Returns:
            El fragmento HTML de la tabla.
        """
        con_papel = any("papel" in c for c in contrastes)
        cabeceras = ["Estrategia"]
        if con_papel:
            cabeceras.append("Papel")
        if con_censura:
            cabeceras.append("Censura")
        cabeceras += [
            "Mediana frío", "Mediana caliente", "Reducción", "p (una cola)",
            "p (dos colas)", "Censurados", "Cumple el criterio",
        ]
        filas = []
        for c in contrastes:
            reduccion = c.get("reduccion")
            celdas = [f"<td>{c['estrategia']}</td>"]
            if con_papel:
                celdas.append(f"<td>{c.get('papel', '—')}</td>")
            if con_censura:
                celdas.append(f"<td class=\"num\">{c['censura']}</td>")
            celdas += [
                f"<td class=\"num\">{_formatear(c['mediana_frio'], 1)}</td>",
                f"<td class=\"num\">{_formatear(c['mediana_caliente'], 1)}</td>",
                f"<td class=\"num\">"
                f"{'—' if reduccion is None else f'{reduccion * 100:.1f} %'}</td>",
                f"<td class=\"num\">{_formatear(c.get('p_una_cola'), 4)}</td>",
                f"<td class=\"num\">{_formatear(c.get('p_dos_colas'), 4)}</td>",
                f"<td class=\"num\">{c.get('censurados_caliente', 0)}</td>",
                f"<td>{_formatear(c.get('confirma'))}</td>",
            ]
            clase = "mejor" if c.get("confirma") else ""
            filas.append(f"<tr class=\"{clase}\">{''.join(celdas)}</tr>")
        encabezado = "".join(f"<th>{h}</th>" for h in cabeceras)
        return (
            f"<table><thead><tr>{encabezado}</tr></thead>"
            f"<tbody>{''.join(filas)}</tbody></table>"
        )

    def _tabla_datasets(self, por_dataset: Sequence[Fila]) -> str:
        """Arma la tabla HTML con una fila por conjunto de datos.

        Cada fila se colorea según haya ganado el frío o el caliente, para que la
        dispersión del resultado se vea antes de leer un solo número.

        Args:
            por_dataset: Filas por conjunto de datos.

        Returns:
            El fragmento HTML de la tabla.
        """
        condiciones = self._condiciones(por_dataset)
        cabeceras = (
            ["Conjunto de datos", "Meseta"]
            + [f"Ensayos ({c})" for c in condiciones]
            + ["Inevaluables (frío)", f"Mejor a 15 ensayos ({CONDICION_PRIMARIA})"]
        )
        filas = []
        for fila in por_dataset:
            caliente = fila.get(CONDICION_PRIMARIA)
            clase = ""
            if caliente is not None:
                diferencia = float(caliente) - float(fila["frio"])
                clase = "mejor" if diferencia < 0 else "peor" if diferencia > 0 else ""
            celdas = [
                f"<td>{fila['dataset_name']}</td>",
                f"<td class=\"num\">{_formatear(fila.get('meseta'), 0)}</td>",
            ]
            celdas += [f"<td class=\"num\">{_formatear(fila[c], 1)}</td>" for c in condiciones]
            celdas += [
                f"<td class=\"num\">{_formatear(fila.get('frio_inevaluable'), 3)}</td>",
                f"<td class=\"num\">"
                f"{_formatear(fila.get(f'{CONDICION_PRIMARIA}_a_15'), 4)}</td>",
            ]
            filas.append(f"<tr class=\"{clase}\">{''.join(celdas)}</tr>")
        encabezado = "".join(f"<th>{h}</th>" for h in cabeceras)
        return (
            f"<table><thead><tr>{encabezado}</tr></thead>"
            f"<tbody>{''.join(filas)}</tbody></table>"
        )
