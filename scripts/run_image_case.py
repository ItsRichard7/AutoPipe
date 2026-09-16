# scripts/run_image_case.py
"""Caso completo de detección de anomalías en imágenes (RE6).

**Qué verifica.** La generalización del framework por tarea: que el mismo cargador,
el mismo registro de tareas, los mismos dos motores de búsqueda, el mismo evaluador
y el mismo generador de reportes que sostienen los experimentos tabulares resuelven
también la detección de anomalías en imágenes, con su métrica propia (AUC-ROC).
**No es un contraste de la hipótesis** del trabajo, que se decidió en los dos
experimentos tabulares.

**Datos.** Tres categorías de MVTec AD —``bottle``, ``cable`` y ``capsule``—, usando
su partición ``test/``, que ya mezcla imágenes normales (``good``) y anómalas (una
carpeta por tipo de defecto). La raíz se toma de ``--datos``, de la variable de
entorno ``AUTOPIPE_DATASETS`` o, en último caso, de ``datasets/``.

**Decisiones declaradas antes de ejecutar**, y por qué:

* **Resolución de trabajo de 64 x 64, fijada en la preparación y no buscada.** El
  motor añade a cada paso del espacio la opción de omitirlo, de modo que un
  ``Resize`` dentro del espacio haría que la mitad de los pipelines aplanara imágenes
  de hasta 1024 x 1024 x 3, más de tres millones de características por imagen. Fijar
  la resolución antes iguala además tres categorías que llegan a 900, 1000 y 1024
  píxeles. Se usa el mismo ``cv2.resize`` que la transformación ``Resize``.
* **El espacio es el del proyecto base sin el paso de redimensionado**: CLAHE con dos
  límites de recorte, suavizado gaussiano con dos núcleos y normalización, cada uno
  omitible. Son exactamente los pipelines «con redimensionado» del espacio que usan
  ``scripts/check_image_regression.py`` y el notebook de MVTec AD: 18 candidatos.
* **``IsolationForest(random_state=0)`` como modelo**, por continuidad con la
  librería base. Su semilla queda fija para que la única fuente de variación entre
  corridas sea la partición.
* **Cinco semillas de partición**, las del pre-registro v1.0. Con 20 imágenes normales
  en ``bottle``, la partición de validación deja unas seis normales, así que el AUC-ROC
  es inestable entre semillas; se reporta la dispersión en lugar de un valor único.

**Limitación que se declara.** No hay arranque en caliente sobre imágenes: exigiría un
corpus de datasets de imágenes, su base de meta-conocimiento y su propio
pre-registro, y queda fuera del alcance comprometido.

**Licencia.** Los reportes HTML embeben miniaturas de MVTec AD, cuya licencia
---CC BY-NC-SA 4.0--- condiciona la redistribución a un uso no comercial, con atribución
y bajo la misma licencia. El proyecto opta por no redistribuirlas: los reportes se
generan en local y no se versionan. El JSON y el resumen en Markdown no llevan imágenes.

Ejecutar con:
    .\\.venv\\Scripts\\python.exe scripts\\run_image_case.py --datos C:\\datasets
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import shutil
import statistics
import tempfile
import time
import warnings
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import optuna
from check_image_regression import raiz_datasets
from PIL import Image
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

from autopipe import (
    CLAHETransform,
    DataSplitter,
    GaussianBlur,
    GridSearchEngine,
    ImageDataLoader,
    Normalize,
    OptunaSearchEngine,
    ReportGenerator,
    get_task,
)
from autopipe.pipeline.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SALIDA_JSON = Path("docs/experimentos/IMAGENES-resultados.json")
SALIDA_MD = Path("docs/experimentos/IMAGENES-resultados.md")
REPORTES = Path("docs/experimentos/reportes/imagenes")

CATEGORIAS = ("bottle", "cable", "capsule")
"""Categorías de MVTec AD sobre las que se verifica el caso."""

RESOLUCION = 64
"""Lado de la resolución de trabajo, la misma que usa el proyecto base."""

SEMILLAS = (101, 202, 303, 404, 505)
"""Semillas de partición, las del pre-registro v1.0."""

VAL_SIZE = 0.3
"""Fracción de validación, la del notebook y el guion de no-regresión."""

PRESUPUESTO_OPTUNA = 20
"""Ensayos de Optuna por semilla, sobre un espacio de 18 candidatos."""

TAREA = "anomaly_detection"
METRICA = "auc_roc"

TOLERANCIA = 1e-12
"""Margen numérico con que se comprueba que Optuna no supere a la rejilla."""


def _modelo() -> IsolationForest:
    """Devuelve el modelo base, el de la librería base con semilla fija."""
    return IsolationForest(random_state=0)


def construir_espacio() -> dict[str, list]:
    """Construye el espacio de búsqueda del caso de imágenes.

    Devuelve instancias nuevas en cada llamada, por la misma razón que el espacio
    tabular: compartirlas entre dos búsquedas arrastraría estado de una a otra.

    Returns:
        Diccionario de paso a transformaciones candidatas. Con la omisión que añade
        el motor, contiene 3 x 3 x 2 = 18 pipelines.
    """
    return {
        "clahe": [
            CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8)),
            CLAHETransform(clip_limit=4.0, tile_grid_size=(8, 8)),
        ],
        "blur": [GaussianBlur(kernel_size=3), GaussianBlur(kernel_size=5)],
        "normalize": [Normalize(mean=0.5, std=0.5)],
    }


def preparar_categoria(origen: Path, destino: Path, resolucion: int = RESOLUCION) -> dict:
    """Convierte ``<categoría>/test/`` en ``normal/`` + ``anomaly/`` a resolución fija.

    Args:
        origen: Carpeta de la categoría de MVTec AD.
        destino: Carpeta donde escribir las imágenes preparadas.
        resolucion: Lado de la resolución de trabajo.

    Returns:
        Recuento de imágenes escritas por clase y por tipo de defecto.

    Raises:
        SystemExit: Si la categoría no está descargada.
    """
    prueba = origen / "test"
    if not prueba.is_dir():
        raise SystemExit(
            f"No se encontró {prueba}. Descargue la categoría de MVTec AD y pase "
            f"--datos con la carpeta que contiene mvtec_ad/. Véase datasets/README.md."
        )
    recuento: dict = {"normal": 0, "anomaly": 0, "defectos": {}}
    for clase in ("normal", "anomaly"):
        (destino / clase).mkdir(parents=True, exist_ok=True)
    for sub in sorted(p for p in prueba.iterdir() if p.is_dir()):
        clase = "normal" if sub.name == "good" else "anomaly"
        rutas = sorted(sub.glob("*.png"))
        for indice, ruta in enumerate(rutas):
            with Image.open(ruta) as imagen:
                arreglo = np.asarray(imagen.convert("RGB"))
            reducida = cv2.resize(arreglo, (resolucion, resolucion))
            Image.fromarray(reducida).save(destino / clase / f"{sub.name}_{indice:03d}.png")
        recuento[clase] += len(rutas)
        if clase == "anomaly":
            recuento["defectos"][sub.name] = len(rutas)
    return recuento


def _matriz(muestras: Sequence[np.ndarray]) -> np.ndarray:
    """Aplana y apila las muestras, igual que hace el motor antes de entrenar."""
    return np.vstack([np.asarray(muestra).ravel() for muestra in muestras])


def _primer_ensayo_que_alcanza(curva: Sequence[float], valor: float) -> int | None:
    """Posición, empezando en uno, del primer ensayo que alcanza un valor."""
    for posicion, objetivo in enumerate(curva, start=1):
        if math.isfinite(objetivo) and objetivo >= valor - TOLERANCIA:
            return posicion
    return None


def buscar(imagenes: np.ndarray, etiquetas: np.ndarray, semilla: int, presupuesto: int) -> dict:
    """Corre la rejilla exhaustiva y Optuna sobre la misma partición.

    La rejilla da el óptimo exacto del espacio sobre esa partición, de modo que acota
    por arriba cualquier valor que Optuna alcance. Es el mismo invariante que sostiene
    la vara del experimento 2, y se comprueba igual.

    Args:
        imagenes: Imágenes preparadas.
        etiquetas: Etiquetas binarias.
        semilla: Semilla de la partición y del muestreador.
        presupuesto: Ensayos de Optuna.

    Returns:
        Diccionario con el registro serializable y, bajo ``_objetos``, el pipeline
        ganador, sus métricas y la tabla de resultados, para el reporte.

    Raises:
        RuntimeError: Si Optuna supera al óptimo de la rejilla, lo que solo puede
            indicar que las dos búsquedas no usaron la misma partición.
    """
    inicio = time.perf_counter()
    rejilla = GridSearchEngine(
        construir_espacio(), val_size=VAL_SIZE, random_state=semilla, task=TAREA
    )
    mejor, metricas, resultados = rejilla.run(imagenes, etiquetas, _modelo(), metric=METRICA)
    objetivos = [float(r["objective"]) for r in resultados]
    finitos = [v for v in objetivos if math.isfinite(v)]
    optimo = max(finitos)

    motor = OptunaSearchEngine(
        construir_espacio(),
        n_trials=presupuesto,
        val_size=VAL_SIZE,
        random_state=semilla,
        task=TAREA,
    )
    _, _, ensayos = motor.run(imagenes, etiquetas, _modelo(), metric=METRICA)
    curva = [float(r["objective"]) for r in ensayos]
    mejor_optuna = max(v for v in curva if math.isfinite(v))
    if mejor_optuna > optimo + TOLERANCIA:
        raise RuntimeError(
            f"Optuna alcanzó {mejor_optuna!r} y la rejilla exhaustiva da {optimo!r} con "
            f"la semilla {semilla}. La rejilla evalúa todo el espacio sobre la misma "
            f"partición, así que esto indica particiones distintas, no un resultado."
        )

    return {
        "semilla": semilla,
        "optimo": optimo,
        "ganador": mejor.get_params(),
        "metricas": {clave: float(valor) for clave, valor in metricas.items()},
        "n_candidatos": len(objetivos),
        "n_inevaluables": len(objetivos) - len(finitos),
        "optuna_mejor": mejor_optuna,
        "optuna_ensayo_del_optimo": _primer_ensayo_que_alcanza(curva, optimo),
        "segundos": round(time.perf_counter() - inicio, 2),
        "_objetos": (mejor, metricas, resultados),
    }


def puntuar_validacion(
    pipeline: Pipeline, imagenes: np.ndarray, etiquetas: np.ndarray, semilla: int
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce la evaluación del ganador para obtener los puntajes de validación.

    Sigue el protocolo del motor paso a paso: la misma partición, el pipeline clonado
    y ajustado solo con entrenamiento, y el modelo clonado. El motor no devuelve los
    puntajes por imagen, y el reporte los necesita para su galería.

    Args:
        pipeline: Pipeline ganador, sin ajustar.
        imagenes: Imágenes preparadas.
        etiquetas: Etiquetas binarias.
        semilla: Semilla de la partición.

    Returns:
        Tupla ``(índices_de_validación, puntajes)``.
    """
    entrenamiento, validacion = DataSplitter(val_size=VAL_SIZE, random_state=semilla).split(
        etiquetas
    )
    ajustado = copy.deepcopy(pipeline)
    x_entrenamiento = _matriz(
        ajustado.fit_transform(imagenes[entrenamiento], etiquetas[entrenamiento])
    )
    x_validacion = _matriz(ajustado.transform(imagenes[validacion]))
    modelo = _modelo().fit(x_entrenamiento, etiquetas[entrenamiento])
    puntajes = get_task(TAREA).predict_scores(modelo, x_validacion)
    return validacion, np.asarray(puntajes)


def generar_reporte(
    imagenes: np.ndarray, etiquetas: np.ndarray, registro: dict, destino: Path
) -> Path:
    """Genera el reporte HTML con la galería de las imágenes más anómalas.

    Antes de generarlo comprueba que los puntajes reproducidos dan el mismo AUC-ROC
    que el motor informó: si no coincidiera, la galería mostraría una partición
    distinta de la evaluada.

    Args:
        imagenes: Imágenes preparadas.
        etiquetas: Etiquetas binarias.
        registro: Resultado de :func:`buscar`, con sus objetos.
        destino: Carpeta del reporte.

    Returns:
        La ruta del HTML generado.

    Raises:
        RuntimeError: Si el AUC-ROC reproducido no coincide con el del motor.
    """
    mejor, metricas, resultados = registro["_objetos"]
    validacion, puntajes = puntuar_validacion(mejor, imagenes, etiquetas, registro["semilla"])
    reproducido = float(roc_auc_score(etiquetas[validacion], puntajes))
    if not math.isclose(reproducido, float(metricas[METRICA]), rel_tol=1e-9):
        raise RuntimeError(
            f"El AUC-ROC reproducido ({reproducido}) no coincide con el del motor "
            f"({metricas[METRICA]}): la galería no correspondería a la partición evaluada."
        )
    rutas = ReportGenerator(output_dir=destino).generate(
        resultados, mejor, metricas, metric=METRICA,
        images=imagenes[validacion], scores=puntajes, pdf=False,
    )
    return Path(rutas["html"])


def resumir(por_semilla: list[dict]) -> dict:
    """Agrega las corridas de una categoría.

    Args:
        por_semilla: Registros de :func:`buscar`, sin sus objetos.

    Returns:
        Mediana, mínimo y máximo del óptimo, ganadores distintos y cuántas veces
        Optuna alcanzó el óptimo exacto.
    """
    optimos = [r["optimo"] for r in por_semilla]
    ganadores = {json.dumps(r["ganador"], sort_keys=True) for r in por_semilla}
    return {
        "auc_mediana": statistics.median(optimos),
        "auc_minimo": min(optimos),
        "auc_maximo": max(optimos),
        "ganadores_distintos": len(ganadores),
        "optuna_alcanza_el_optimo": sum(
            1 for r in por_semilla if r["optuna_ensayo_del_optimo"] is not None
        ),
        "n_semillas": len(por_semilla),
    }


def ejecutar_categoria(
    categoria: str, raiz: Path, semillas: Sequence[int], con_reporte: bool
) -> dict:
    """Prepara una categoría y corre todas sus semillas.

    Args:
        categoria: Nombre de la categoría de MVTec AD.
        raiz: Carpeta que contiene ``mvtec_ad/``.
        semillas: Semillas de partición.
        con_reporte: Si se genera el reporte HTML de la primera semilla.

    Returns:
        Diccionario serializable con el recuento, las corridas y el resumen.
    """
    temporal = Path(tempfile.mkdtemp())
    try:
        recuento = preparar_categoria(raiz / "mvtec_ad" / categoria, temporal / categoria)
        imagenes, etiquetas = ImageDataLoader(temporal / categoria).load()
    finally:
        shutil.rmtree(temporal, ignore_errors=True)

    corridas = [buscar(imagenes, etiquetas, s, PRESUPUESTO_OPTUNA) for s in semillas]
    reporte = None
    if con_reporte:
        reporte = generar_reporte(imagenes, etiquetas, corridas[0], REPORTES / categoria)
    registros = [{k: v for k, v in c.items() if k != "_objetos"} for c in corridas]
    logger.info(
        "%-8s %3d imágenes  AUC óptimo mediana %.4f [%.4f, %.4f]",
        categoria, len(etiquetas),
        statistics.median(r["optimo"] for r in registros),
        min(r["optimo"] for r in registros),
        max(r["optimo"] for r in registros),
    )
    return {
        "categoria": categoria,
        "recuento": recuento,
        "semillas": registros,
        "resumen": resumir(registros),
        "reporte_local": None if reporte is None else reporte.as_posix(),
    }


def _fila_categoria(dato: dict) -> str:
    """Arma la fila resumen de una categoría."""
    r, c = dato["resumen"], dato["recuento"]
    return (
        f"| `{dato['categoria']}` | {c['normal']} | {c['anomaly']} "
        f"| {r['auc_mediana']:.4f} | {r['auc_minimo']:.4f} – {r['auc_maximo']:.4f} "
        f"| {r['ganadores_distintos']} de {r['n_semillas']} "
        f"| {r['optuna_alcanza_el_optimo']} de {r['n_semillas']} |"
    )


def _describir_pipeline(parametros: dict) -> str:
    """Resume un pipeline serializado en una línea legible."""
    if not parametros:
        return "sin transformaciones"
    return " → ".join(clave.split("_", 1)[1] for clave in sorted(parametros))


def redactar(resultado: dict) -> str:
    """Redacta el resumen en Markdown del caso de imágenes.

    Args:
        resultado: Documento completo producido por :func:`main`.

    Returns:
        El documento Markdown.
    """
    filas = "\n".join(_fila_categoria(d) for d in resultado["categorias"])
    ganadores = "\n".join(
        f"| `{d['categoria']}` | {r['semilla']} | {r['optimo']:.4f} "
        f"| {_describir_pipeline(r['ganador'])} | {r['optuna_ensayo_del_optimo'] or '—'} |"
        for d in resultado["categorias"]
        for r in d["semillas"]
    )
    return f"""# Caso de anomalías en imágenes (RE6)

> **Archivo generado.** No editar a mano: se regenera con
> `.\\.venv\\Scripts\\python.exe scripts\\run_image_case.py --datos <carpeta>`.

- **Generado:** {date.today().isoformat()}
- **Duración:** {resultado['minutos']:.1f} min

## Qué verifica y qué no

Verifica la **generalización del framework por tarea**: el mismo cargador, el mismo
registro de tareas, los mismos dos motores, el mismo evaluador y el mismo generador de
reportes que sostienen los experimentos tabulares resuelven la detección de anomalías
en imágenes con su métrica propia, AUC-ROC.

**No es un contraste de la hipótesis**, y **no hay arranque en caliente sobre
imágenes**: exigiría un corpus de datasets de imágenes, su base de meta-conocimiento y su
propio pre-registro, fuera del alcance comprometido.

## Protocolo

- **Datos:** partición `test/` de MVTec AD, categorías {', '.join(f'`{c}`' for c in CATEGORIAS)}.
- **Resolución de trabajo:** {RESOLUCION} x {RESOLUCION}, fijada en la preparación y no buscada.
- **Espacio:** el del proyecto base sin redimensionado —CLAHE (2,0 / 4,0), suavizado
  gaussiano (3 / 5) y normalización, cada uno omitible—: 18 candidatos.
- **Modelo:** `IsolationForest(random_state=0)`, por continuidad con la librería base.
- **Validación:** {int(VAL_SIZE * 100)} %, cinco semillas de partición:
  {', '.join(map(str, SEMILLAS))}.
- **Motores:** rejilla exhaustiva (óptimo exacto) y Optuna con {PRESUPUESTO_OPTUNA} ensayos,
  sobre la misma partición.

## Resultado por categoría

El AUC-ROC es el óptimo exacto del espacio sobre cada partición. Se reporta la
**dispersión entre semillas** y no un valor único: con pocas imágenes normales, la
partición de validación deja muy pocas y el AUC-ROC varía de una semilla a otra.

| Categoría | Normales | Anómalas | AUC mediano | Rango | Ganadores | Óptimo en Optuna |
|---|---:|---:|---:|---|---:|---:|
{filas}

## Ganador y convergencia por semilla

«Ensayo» es la posición en que Optuna alcanzó por primera vez el óptimo de la rejilla;
un guion indica que no lo alcanzó dentro del presupuesto.

| Categoría | Semilla | AUC-ROC | Pipeline ganador | Ensayo |
|---|---:|---:|---|---:|
{ganadores}

## Invariante comprobado

En ninguna corrida Optuna superó el óptimo de la rejilla exhaustiva. Como ambos motores
evalúan el mismo espacio sobre la misma partición, superarlo solo podría indicar
particiones distintas; el guion aborta si ocurre.

## Reportes HTML

Cada categoría genera en local un reporte con la galería de las imágenes más anómalas.
**No se versionan**: embeben miniaturas de MVTec AD, cuya licencia (CC BY-NC-SA 4.0)
condiciona su redistribución a un uso no comercial, y el proyecto opta por no
redistribuirlas. Antes de generarlos, el guion comprueba que los puntajes reproducidos
dan el mismo AUC-ROC que informó el motor.
"""


def main() -> int:
    """Punto de entrada.

    Returns:
        ``0`` si el caso se ejecutó sobre todas las categorías.
    """
    parser = argparse.ArgumentParser(description="Caso de anomalías en imágenes.")
    parser.add_argument("--datos", type=Path, help="Carpeta que contiene mvtec_ad/.")
    parser.add_argument("--sin-reportes", action="store_true", help="No generar HTML.")
    args = parser.parse_args()

    # OpenCV multihilo puede variar bits entre ejecuciones; un solo hilo la hace exacta.
    cv2.setNumThreads(1)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logging.getLogger("autopipe").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning)

    raiz = args.datos or raiz_datasets()
    inicio = time.perf_counter()
    categorias = [
        ejecutar_categoria(c, raiz, SEMILLAS, con_reporte=not args.sin_reportes)
        for c in CATEGORIAS
    ]
    resultado = {
        "categorias": categorias,
        "resolucion": RESOLUCION,
        "val_size": VAL_SIZE,
        "presupuesto_optuna": PRESUPUESTO_OPTUNA,
        "minutos": round((time.perf_counter() - inicio) / 60, 2),
    }
    SALIDA_JSON.write_text(
        json.dumps(resultado, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8"
    )
    SALIDA_MD.write_text(redactar(resultado), encoding="utf-8")
    logger.info("Resultados en %s y %s", SALIDA_JSON, SALIDA_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
