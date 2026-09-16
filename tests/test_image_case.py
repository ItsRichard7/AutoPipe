# tests/test_image_case.py
"""Pruebas del caso de anomalías en imágenes.

Se ejecutan sobre una categoría **sintética** con la misma estructura de carpetas que
MVTec AD, generada al vuelo: las pruebas no dependen de datos externos ni de su
licencia.

Las dos que sostienen el caso son las que comprueban que Optuna nunca supera a la
rejilla exhaustiva —si lo hiciera, las dos búsquedas no habrían usado la misma
partición— y que los puntajes reproducidos para la galería dan el mismo AUC-ROC que
informó el motor —si no, la galería mostraría una partición distinta de la evaluada—.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from sklearn.metrics import roc_auc_score

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_image_case as caso  # noqa: E402

from autopipe import GridSearchEngine, ImageDataLoader  # noqa: E402

RAIZ_REPO = Path(__file__).resolve().parents[1]


def _categoria(
    raiz: Path,
    nombre: str = "sintetica",
    lado: int = 80,
    normales: int = 10,
    por_defecto: int = 6,
) -> Path:
    """Crea una categoría con la estructura de MVTec AD y anomalías separables.

    Las normales son oscuras y las anómalas claras, repartidas en dos tipos de
    defecto, para que el caso tenga una respuesta clara y las comprobaciones de
    igualdad no dependan de empates. Por omisión las anómalas son **mayoría**, como en
    la partición ``test/`` de MVTec AD.
    """
    prueba = raiz / "mvtec_ad" / nombre / "test"
    rng = np.random.default_rng(0)
    for carpeta, cantidad, rango in (
        ("good", normales, (0, 70)),
        ("rayon", por_defecto, (170, 256)),
        ("mancha", por_defecto, (150, 256)),
    ):
        destino = prueba / carpeta
        destino.mkdir(parents=True)
        for indice in range(cantidad):
            arreglo = rng.integers(*rango, size=(lado, lado, 3), dtype=np.uint8)
            Image.fromarray(arreglo).save(destino / f"{indice:03d}.png")
    return raiz / "mvtec_ad" / nombre


@pytest.fixture(scope="module")
def preparado(tmp_path_factory: pytest.TempPathFactory) -> tuple[np.ndarray, np.ndarray, dict]:
    """Prepara la categoría sintética una sola vez y la carga con el cargador real."""
    raiz = tmp_path_factory.mktemp("datos")
    destino = tmp_path_factory.mktemp("preparado") / "sintetica"
    recuento = caso.preparar_categoria(_categoria(raiz), destino)
    imagenes, etiquetas = ImageDataLoader(destino).load()
    return imagenes, etiquetas, recuento


@pytest.fixture(scope="module")
def registro(preparado) -> dict:
    """Corre la búsqueda una sola vez para todas las pruebas que la necesitan."""
    imagenes, etiquetas, _ = preparado
    return caso.buscar(imagenes, etiquetas, semilla=101, presupuesto=20)


# --------------------------------------------------------------------------- #
# El espacio y la preparación
# --------------------------------------------------------------------------- #
def test_the_space_has_the_eighteen_candidates_of_the_base_project() -> None:
    """CLAHE (2 + omitir) x suavizado (2 + omitir) x normalización (1 + omitir)."""
    pipelines = GridSearchEngine(
        caso.construir_espacio(), task="anomaly_detection"
    )._build_pipelines()

    assert len(pipelines) == 18


def test_the_resolution_is_not_a_searched_step() -> None:
    """Prueba por sabotaje de la decisión de preparación.

    Con ``Resize`` dentro del espacio, el motor añadiría la opción de omitirlo y la
    mitad de los pipelines aplanaría imágenes de hasta 1024 x 1024 x 3. La prueba
    falla si alguien vuelve a meterlo en el espacio.
    """
    pipelines = GridSearchEngine(
        caso.construir_espacio(), task="anomaly_detection"
    )._build_pipelines()

    for pipeline in pipelines:
        assert not any("Resize" in clave for clave in pipeline.get_params())


def test_every_image_is_prepared_at_the_working_resolution(tmp_path: Path) -> None:
    destino = tmp_path / "preparado"

    caso.preparar_categoria(_categoria(tmp_path), destino)

    rutas = [*(destino / "normal").glob("*.png"), *(destino / "anomaly").glob("*.png")]
    assert rutas
    for ruta in rutas:
        with Image.open(ruta) as imagen:
            assert imagen.size == (caso.RESOLUCION, caso.RESOLUCION)


def test_the_preparation_counts_each_class_and_defect(preparado) -> None:
    _, etiquetas, recuento = preparado

    assert recuento["normal"] == 10
    assert recuento["anomaly"] == 12
    assert recuento["defectos"] == {"mancha": 6, "rayon": 6}
    assert int((etiquetas == 0).sum()) == 10
    assert int((etiquetas == 1).sum()) == 12


def test_a_missing_category_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="mvtec_ad"):
        caso.preparar_categoria(tmp_path / "no-existe", tmp_path / "destino")


# --------------------------------------------------------------------------- #
# La búsqueda
# --------------------------------------------------------------------------- #
def test_the_grid_evaluates_the_whole_space(registro) -> None:
    assert registro["n_candidatos"] == 18


def test_optuna_never_beats_the_exhaustive_grid(registro) -> None:
    """El invariante que sostiene la comparación entre los dos motores."""
    assert registro["optuna_mejor"] <= registro["optimo"] + caso.TOLERANCIA


def test_the_winner_metrics_carry_the_anomaly_task_metrics(registro) -> None:
    """Es la tarea de anomalías la que se resolvió, no la de clasificación."""
    assert "auc_roc" in registro["metricas"]
    assert "macro_f1" not in registro["metricas"]
    assert registro["metricas"]["auc_roc"] == pytest.approx(registro["optimo"])


def test_the_detector_beats_chance_even_when_anomalies_are_the_majority(registro) -> None:
    """Aquí las anómalas son mayoría (12 de 22), como en la partición ``test/`` de MVTec AD.

    Isolation Forest supone que las anomalías son escasas, y el protocolo heredado lo
    entrena sobre una partición donde no lo son. Por eso no se exige un valor alto, sino
    que el detector no quede al azar ni invierta los puntajes. Es la misma limitación
    que explica en buena parte el resultado de ``capsule``.
    """
    assert registro["optimo"] > 0.5


def test_a_rare_and_separable_anomaly_is_detected(tmp_path: Path) -> None:
    """Cuando el supuesto del modelo se cumple, la maquinaria detecta la anomalía."""
    destino = tmp_path / "preparado"
    caso.preparar_categoria(_categoria(tmp_path, normales=18, por_defecto=2), destino)
    imagenes, etiquetas = ImageDataLoader(destino).load()

    resultado = caso.buscar(imagenes, etiquetas, semilla=101, presupuesto=20)

    assert resultado["optimo"] >= 0.95


# --------------------------------------------------------------------------- #
# La galería reproduce la partición evaluada
# --------------------------------------------------------------------------- #
def test_the_reproduced_scores_give_the_engine_auc(preparado, registro) -> None:
    """Prueba por sabotaje de la reproducción del protocolo.

    Si ``puntuar_validacion`` usara otra partición u otro ajuste que el motor, el
    AUC-ROC reproducido no coincidiría con el informado.
    """
    imagenes, etiquetas, _ = preparado
    mejor, metricas, _ = registro["_objetos"]

    validacion, puntajes = caso.puntuar_validacion(mejor, imagenes, etiquetas, 101)

    assert math.isclose(
        roc_auc_score(etiquetas[validacion], puntajes), metricas["auc_roc"], rel_tol=1e-9
    )


def test_a_report_whose_scores_disagree_with_the_engine_is_refused(
    preparado, registro, tmp_path: Path
) -> None:
    imagenes, etiquetas, _ = preparado
    mejor, metricas, resultados = registro["_objetos"]
    alterado = {
        **registro,
        "_objetos": (mejor, {**metricas, "auc_roc": metricas["auc_roc"] - 0.25}, resultados),
    }

    with pytest.raises(RuntimeError, match="partición evaluada"):
        caso.generar_reporte(imagenes, etiquetas, alterado, tmp_path)


def test_the_report_is_generated_with_its_gallery(preparado, registro, tmp_path: Path) -> None:
    imagenes, etiquetas, _ = preparado

    html = caso.generar_reporte(imagenes, etiquetas, registro, tmp_path)

    assert html.is_file()
    assert "Imágenes anómalas destacadas" in html.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Agregación y documento
# --------------------------------------------------------------------------- #
def test_the_first_trial_that_reaches_a_value_counts_from_one() -> None:
    nan = float("nan")

    assert caso._primer_ensayo_que_alcanza([0.1, nan, 0.9, 0.9], 0.9) == 3
    assert caso._primer_ensayo_que_alcanza([0.1, 0.2], 0.9) is None


def test_the_summary_reports_the_spread_and_not_a_single_value() -> None:
    corridas = [
        {"optimo": v, "ganador": {"0_G": {}} if i else {}, "optuna_ensayo_del_optimo": e}
        for i, (v, e) in enumerate([(0.6, 3), (0.8, None), (0.7, 5)])
    ]

    resumen = caso.resumir(corridas)

    assert resumen["auc_mediana"] == pytest.approx(0.7)
    assert (resumen["auc_minimo"], resumen["auc_maximo"]) == (0.6, 0.8)
    assert resumen["ganadores_distintos"] == 2
    assert resumen["optuna_alcanza_el_optimo"] == 2


def _resultado() -> dict:
    corrida = {
        "semilla": 101, "optimo": 0.5, "ganador": {"0_CLAHETransform": {}},
        "optuna_ensayo_del_optimo": None,
    }
    return {
        "minutos": 1.0,
        "categorias": [
            {
                "categoria": "sintetica",
                "recuento": {"normal": 10, "anomaly": 12, "defectos": {}},
                "semillas": [corrida],
                "resumen": caso.resumir([corrida]),
            }
        ],
    }


def test_the_document_states_what_it_does_not_verify() -> None:
    """Sin estas frases, el caso podría leerse como un contraste de la hipótesis."""
    texto = caso.redactar(_resultado())

    assert "No es un contraste de la hipótesis" in texto
    assert "no hay arranque en caliente" in texto


def test_the_document_explains_why_the_html_reports_are_not_versioned() -> None:
    texto = caso.redactar(_resultado())

    assert "No se versionan" in texto
    assert "licencia" in texto


def test_the_html_reports_are_ignored_by_git() -> None:
    """Prueba por sabotaje de la licencia.

    Los reportes embeben miniaturas de MVTec AD, que el proyecto decidió no
    redistribuir. Si alguien quitara la regla de ``.gitignore``, un ``git add`` los
    publicaría.
    """
    reglas = (RAIZ_REPO / ".gitignore").read_text(encoding="utf-8")

    assert "docs/experimentos/reportes/imagenes/" in reglas


def test_the_protocol_constants_are_the_declared_ones() -> None:
    assert caso.CATEGORIAS == ("bottle", "cable", "capsule")
    assert caso.RESOLUCION == 64
    assert caso.SEMILLAS == (101, 202, 303, 404, 505)
    assert caso.VAL_SIZE == 0.3
    assert caso.PRESUPUESTO_OPTUNA == 20
    assert caso.TAREA == "anomaly_detection"
