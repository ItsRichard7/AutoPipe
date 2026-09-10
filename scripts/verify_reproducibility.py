# scripts/verify_reproducibility.py
"""Reproducibilidad de punta a punta de los resultados reportados (RE6).

El anteproyecto exige como criterio de ingeniería que **la misma semilla produzca el
mismo resultado**. Hasta ahora estaba verificado a nivel de componente; este guion lo
verifica sobre **los resultados que el informe cita**, en tres niveles:

1. **Los documentos generados se regeneran idénticos** desde los datos versionados.
   Si un resumen no pudiera reproducirse de su JSON, lo que dice no tendría respaldo.
   Se ignora solo la línea de fecha de generación.
2. **Las recomendaciones se reproducen.** Para cada pliegue elegido se vuelve a ajustar
   el recomendador con la base sin ese conjunto y se comparan las configuraciones con
   las guardadas. En el experimento 1 esto comprueba además que la refactorización
   posterior del recomendador no cambió sus recomendaciones.
3. **Las corridas se reproducen ensayo por ensayo.** Se reejecutan parejas
   (conjunto, semilla) de ambos experimentos y del caso de imágenes y se comparan las
   curvas completas con las guardadas, valor por valor, sin tolerancia.

**Por qué parejas y no todo.** Reejecutar los dos experimentos completos costaría más
de dos horas y no probaría nada que una muestra no pruebe: si una pareja se reproduce
bit a bit, el procedimiento es determinista. Las parejas se eligieron entre los
conjuntos más pequeños, por costo, y con semillas presentes en cada experimento.

Ejecutar con:
    .\\.venv\\Scripts\\python.exe scripts\\verify_reproducibility.py --datos C:\\datasets
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import tempfile
import warnings
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path

import analyze_e2_mechanism
import analyze_stratified
import cv2
import openml
import optuna
import report_lodo
import report_lodo_e2
import run_image_case
import run_lodo
import run_lodo_e2

from autopipe import ImageDataLoader
from autopipe.meta_knowledge import MetaKnowledgeBase
from autopipe.meta_recommender import MetaRecommender

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

EXPERIMENTOS = Path("docs/experimentos")
SALIDA = EXPERIMENTOS / "REPRODUCIBILIDAD.md"

PARES_E1 = (("balance-scale", 101), ("diabetes", 303))
"""Parejas del experimento 1: conjuntos pequeños y semillas de sus cinco."""

PARES_E2 = (("balance-scale", 101), ("blood-transfusion-service-center", 606))
"""Parejas del experimento 2. La semilla 606 solo existe en este experimento."""

PAR_IMAGENES = ("bottle", 101)
"""Categoría y semilla del caso de imágenes."""

IDENTICO, DIFIERE, OMITIDO = "idéntico", "difiere", "omitido"


def iguales(uno: object, otro: object) -> bool:
    """Compara dos valores tratando ``NaN`` como igual a ``NaN``.

    Args:
        uno: Primer valor.
        otro: Segundo valor.

    Returns:
        ``True`` si son el mismo valor, o ambos ``NaN``.
    """
    if isinstance(uno, float) and isinstance(otro, float):
        return (math.isnan(uno) and math.isnan(otro)) or uno == otro
    return uno == otro


def diferencias_de_curva(guardada: Sequence[float], nueva: Sequence[float]) -> int:
    """Cuenta las posiciones en que dos curvas difieren, incluida su longitud.

    Args:
        guardada: Curva leída del documento de resultados.
        nueva: Curva recién ejecutada.

    Returns:
        Número de posiciones distintas; una diferencia de longitud cuenta todas las
        posiciones sobrantes.
    """
    comunes = sum(1 for a, b in zip(guardada, nueva) if not iguales(float(a), float(b)))
    return comunes + abs(len(guardada) - len(nueva))


def _sin_fecha(texto: str) -> list[str]:
    """Quita la línea de fecha de generación, la única que varía legítimamente."""
    return [linea for linea in texto.splitlines() if not linea.startswith("- **Generado:**")]


def _verificacion(nombre: str, compara: str, estado: str, detalle: str) -> dict:
    """Arma el registro de una verificación."""
    return {"verificacion": nombre, "compara": compara, "estado": estado, "detalle": detalle}


def verificar_documento(nombre: str, generado: str, ruta: Path) -> dict:
    """Comprueba que un documento regenerado coincide con el versionado.

    Args:
        nombre: Nombre de la verificación.
        generado: Texto recién generado.
        ruta: Documento versionado.

    Returns:
        El registro de la verificación.
    """
    if not ruta.is_file():
        return _verificacion(nombre, ruta.as_posix(), OMITIDO, "el documento no existe")
    antes, despues = _sin_fecha(ruta.read_text(encoding="utf-8")), _sin_fecha(generado)
    distintas = sum(1 for a, b in zip(antes, despues) if a != b) + abs(len(antes) - len(despues))
    estado = IDENTICO if distintas == 0 else DIFIERE
    return _verificacion(nombre, ruta.as_posix(), estado, f"{distintas} líneas distintas")


def verificar_documentos(e1: dict, e2: dict, imagenes: dict | None) -> list[dict]:
    """Regenera cada documento desde su JSON y lo compara con el versionado.

    Args:
        e1: Resultados del experimento 1.
        e2: Resultados del experimento 2.
        imagenes: Resultados del caso de imágenes, o ``None`` si no existen.

    Returns:
        Un registro por documento.
    """
    registros = [
        verificar_documento(
            "Resumen del experimento 1",
            report_lodo.redactar_resumen(e1, Path("docs/experimentos/reportes/lodo.html")),
            EXPERIMENTOS / "LODO-resultados.md",
        ),
        verificar_documento(
            "Resumen del experimento 2",
            report_lodo_e2.redactar_resumen(e2),
            EXPERIMENTOS / "LODO-E2-resultados.md",
        ),
        verificar_documento(
            "Mecanismo del experimento 2",
            analyze_e2_mechanism.redactar(
                analyze_e2_mechanism.analizar(e2), len(e2["pliegues"])
            ),
            EXPERIMENTOS / "MECANISMO-E2.md",
        ),
        verificar_documento(
            "Análisis estratificado",
            analyze_stratified.redactar(
                analyze_stratified.analizar(
                    e2["pliegues"], analyze_stratified.leer_sensibilidad(analyze_stratified.CORPUS)
                )
            ),
            EXPERIMENTOS / "ESTRATIFICADO-E2.md",
        ),
    ]
    if imagenes is not None:
        registros.append(
            verificar_documento(
                "Resumen del caso de imágenes",
                run_image_case.redactar(imagenes),
                EXPERIMENTOS / "IMAGENES-resultados.md",
            )
        )
    return registros


def _pliegue(resultados: dict, nombre: str) -> dict:
    """Busca el pliegue de un conjunto por su nombre."""
    for pliegue in resultados["pliegues"]:
        if pliegue["dataset_name"] == nombre:
            return pliegue
    raise KeyError(f"No hay ningún pliegue de {nombre!r} en los resultados.")


def verificar_recomendaciones(
    experimento: str, base: MetaKnowledgeBase, pliegue: dict, estrategias: Sequence[str]
) -> dict:
    """Reajusta el recomendador del pliegue y compara lo que recomienda.

    Args:
        experimento: Etiqueta del experimento, para el registro.
        base: Base de meta-conocimiento completa del experimento.
        pliegue: Pliegue guardado.
        estrategias: Estrategias calientes del experimento.

    Returns:
        El registro de la verificación.
    """
    entrada = next(e for e in base if e.dataset_id == pliegue["dataset_id"])
    distintas = []
    for estrategia in estrategias:
        nuevas = MetaRecommender(
            n_configurations=5, strategy=estrategia, random_state=202
        ).fit(base.without(pliegue["dataset_id"])).recommend(entrada.meta_features)
        if nuevas != pliegue["recomendaciones"][estrategia]["configuraciones"]:
            distintas.append(estrategia)
    return _verificacion(
        f"Recomendaciones {experimento}, {pliegue['dataset_name']}",
        ", ".join(estrategias),
        IDENTICO if not distintas else DIFIERE,
        "todas coinciden" if not distintas else f"difieren: {', '.join(distintas)}",
    )


def verificar_corridas(
    experimento: str,
    pliegue: dict,
    semilla: int,
    condiciones: Sequence[str],
    ejecutar: Callable[[object, object, int, list | None], list[float]],
    cargar: Callable[[str], tuple[object, object]],
) -> dict:
    """Reejecuta todas las condiciones de una pareja y compara sus curvas.

    Args:
        experimento: Etiqueta del experimento.
        pliegue: Pliegue guardado.
        semilla: Semilla a reejecutar.
        condiciones: Condiciones del experimento, empezando por ``frio``.
        ejecutar: Función que corre una condición y devuelve su curva.
        cargar: Función que carga el conjunto de datos.

    Returns:
        El registro de la verificación.
    """
    guardada = next(s for s in pliegue["semillas"] if s["semilla"] == semilla)
    features, labels = cargar(pliegue["dataset_id"])
    total, distintas, ensayos = 0, 0, 0
    for condicion in condiciones:
        inyectadas = (
            None
            if condicion == "frio"
            else pliegue["recomendaciones"][condicion]["configuraciones"]
        )
        curva = ejecutar(features, labels, semilla, inyectadas)
        diferencias = diferencias_de_curva(guardada[condicion]["curva"], curva)
        distintas += diferencias
        ensayos += len(curva)
        total += 1
    return _verificacion(
        f"Corridas {experimento}, {pliegue['dataset_name']}, semilla {semilla}",
        f"{total} condiciones, {ensayos} ensayos",
        IDENTICO if distintas == 0 else DIFIERE,
        f"{distintas} ensayos distintos",
    )


def verificar_imagenes(datos: Path | None, imagenes: dict | None) -> dict:
    """Reejecuta una pareja del caso de imágenes y compara su resultado.

    Args:
        datos: Carpeta que contiene ``mvtec_ad/``, o ``None``.
        imagenes: Resultados guardados del caso, o ``None``.

    Returns:
        El registro de la verificación.
    """
    categoria, semilla = PAR_IMAGENES
    nombre = f"Caso de imágenes, {categoria}, semilla {semilla}"
    origen = None if datos is None else datos / "mvtec_ad" / categoria
    if imagenes is None or origen is None or not (origen / "test").is_dir():
        return _verificacion(
            nombre, "rejilla y Optuna", OMITIDO, "datos de MVTec AD no disponibles"
        )

    guardada = next(
        s
        for c in imagenes["categorias"]
        if c["categoria"] == categoria
        for s in c["semillas"]
        if s["semilla"] == semilla
    )
    temporal = Path(tempfile.mkdtemp())
    try:
        run_image_case.preparar_categoria(origen, temporal / categoria)
        cargadas, etiquetas = ImageDataLoader(temporal / categoria).load()
    finally:
        shutil.rmtree(temporal, ignore_errors=True)
    nueva = run_image_case.buscar(cargadas, etiquetas, semilla, run_image_case.PRESUPUESTO_OPTUNA)

    campos = ("optimo", "optuna_mejor", "optuna_ensayo_del_optimo", "n_inevaluables")
    distintos = [c for c in campos if not iguales(guardada[c], nueva[c])]
    if json.loads(json.dumps(nueva["ganador"])) != guardada["ganador"]:
        distintos.append("ganador")
    distintos += [
        f"metricas.{c}" for c in guardada["metricas"]
        if not iguales(guardada["metricas"][c], nueva["metricas"][c])
    ]
    return _verificacion(
        nombre,
        "óptimo, ganador, métricas y convergencia de Optuna",
        IDENTICO if not distintos else DIFIERE,
        "todo coincide" if not distintos else f"difieren: {', '.join(distintos)}",
    )


def redactar(verificaciones: list[dict]) -> str:
    """Redacta el documento de reproducibilidad.

    Args:
        verificaciones: Registros de todas las verificaciones.

    Returns:
        El documento Markdown.
    """
    filas = "\n".join(
        f"| {v['verificacion']} | {v['compara']} | **{v['estado']}** | {v['detalle']} |"
        for v in verificaciones
    )
    fallos = sum(1 for v in verificaciones if v["estado"] == DIFIERE)
    omitidas = sum(1 for v in verificaciones if v["estado"] == OMITIDO)
    veredicto = (
        "**Todas las verificaciones ejecutadas se reproducen exactamente.**"
        if fallos == 0
        else f"**{fallos} verificaciones no se reproducen.** Revisar antes de citar."
    )
    return f"""# Reproducibilidad de punta a punta

> **Archivo generado.** No editar a mano: se regenera con
> `.\\.venv\\Scripts\\python.exe scripts\\verify_reproducibility.py --datos <carpeta>`.

- **Generado:** {date.today().isoformat()}
- **Verificaciones:** {len(verificaciones)} · **Omitidas:** {omitidas}

## Veredicto

{veredicto}

La comparación de corridas es **exacta, sin tolerancia**: cada ensayo recién ejecutado
tiene que valer lo mismo que el guardado, bit a bit. Los documentos se comparan línea
por línea ignorando únicamente su fecha de generación.

| Verificación | Qué compara | Estado | Detalle |
|---|---|---|---|
{filas}

## Qué cubre y qué no

Reejecuta **parejas** de cada experimento y no los experimentos enteros: si una pareja se
reproduce bit a bit, el procedimiento es determinista, y reejecutarlo todo costaría más
de dos horas sin probar nada adicional. Las parejas se eligieron entre los conjuntos más
pequeños, por costo, antes de ejecutar la verificación.

Una verificación **omitida** no es un fallo, pero tampoco una evidencia: indica que los
datos que necesitaba no estaban disponibles en la máquina donde se ejecutó.
"""


def _leer(ruta: Path) -> dict | None:
    """Lee un documento JSON si existe."""
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.is_file() else None


def main() -> int:
    """Punto de entrada.

    Returns:
        ``0`` si todas las verificaciones ejecutadas se reprodujeron, ``1`` si alguna
        difiere.
    """
    parser = argparse.ArgumentParser(description="Reproducibilidad de punta a punta.")
    parser.add_argument("--datos", type=Path, help="Carpeta que contiene mvtec_ad/.")
    args = parser.parse_args()

    cv2.setNumThreads(1)
    openml.config.set_console_log_level(logging.WARNING)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logging.getLogger("autopipe").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")

    e1 = json.loads((EXPERIMENTOS / "LODO-resultados.json").read_text(encoding="utf-8"))
    e2 = json.loads((EXPERIMENTOS / "LODO-E2-resultados.json").read_text(encoding="utf-8"))
    imagenes = _leer(EXPERIMENTOS / "IMAGENES-resultados.json")
    base_e1 = MetaKnowledgeBase.load(run_lodo.BASE)
    base_e2 = MetaKnowledgeBase.load(run_lodo_e2.BASE)

    verificaciones = verificar_documentos(e1, e2, imagenes)
    for nombre, semilla in PARES_E1:
        pliegue = _pliegue(e1, nombre)
        verificaciones.append(
            verificar_recomendaciones("E1", base_e1, pliegue, run_lodo.ESTRATEGIAS_CALIENTE)
        )
        verificaciones.append(
            verificar_corridas(
                "E1", pliegue, semilla, ("frio", *run_lodo.ESTRATEGIAS_CALIENTE),
                run_lodo.ejecutar_condicion, run_lodo._cargar_de_openml,
            )
        )
    for nombre, semilla in PARES_E2:
        pliegue = _pliegue(e2, nombre)
        verificaciones.append(
            verificar_recomendaciones("E2", base_e2, pliegue, run_lodo_e2.CONDICIONES_CALIENTES)
        )
        verificaciones.append(
            verificar_corridas(
                "E2", pliegue, semilla, ("frio", *run_lodo_e2.CONDICIONES_CALIENTES),
                run_lodo_e2.ejecutar_condicion, run_lodo_e2._cargar_de_openml,
            )
        )
    verificaciones.append(verificar_imagenes(args.datos, imagenes))

    SALIDA.write_text(redactar(verificaciones), encoding="utf-8")
    for v in verificaciones:
        logger.info("%-10s %s — %s", v["estado"], v["verificacion"], v["detalle"])
    logger.info("Documento en %s", SALIDA)
    return 1 if any(v["estado"] == DIFIERE for v in verificaciones) else 0


if __name__ == "__main__":
    raise SystemExit(main())
