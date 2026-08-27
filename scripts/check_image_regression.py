# scripts/check_image_regression.py
"""Verifica que el camino de imágenes no haya cambiado de resultados (RE1).

Corre la misma búsqueda de pipelines sobre MVTec AD dos veces —con el código de
un commit de referencia y con el del árbol de trabajo actual— y compara el
pipeline ganador, todas las métricas y la tabla completa de objetivos.

**Por qué existe.** La generalización del framework hacia tareas y modalidades
nuevas no puede alterar el comportamiento de la librería base. Afirmarlo no
basta: este script lo convierte en una comparación numérica reproducible, que es
lo que pide el criterio de reproducibilidad de la Sección 4.3.5 del anteproyecto.

**Cómo fuerza la comparación.** El paquete se instala en modo editable con un
*finder* en ``sys.meta_path``, que gana sobre ``sys.path``. Sin desactivarlo,
ambas ejecuciones importarían el mismo código y la comparación no verificaría
nada. Cada ejecución informa desde qué archivo cargó ``autopipe``, y el script
aborta si no es el esperado.

**Datos.** Requiere la categoría ``bottle`` de MVTec AD, que no se versiona por
tamaño y licencia. La raíz se toma de la variable de entorno
``AUTOPIPE_DATASETS`` y, si no está definida, de la carpeta ``datasets/`` del
repositorio. Véase ``datasets/README.md``.

Ejecutar con:
    .\\.venv\\Scripts\\python.exe scripts\\check_image_regression.py
    .\\.venv\\Scripts\\python.exe scripts\\check_image_regression.py --base <commit>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

RAIZ_REPO = Path(__file__).resolve().parents[1]
COMMIT_BASE = "229c810"
"""Estado anterior al trabajo de generalización de la semana 4."""

CATEGORIA = Path("mvtec_ad/bottle")
SEMILLA = 42
VAL_SIZE = 0.3
METRICA = "auc_roc"


def raiz_datasets() -> Path:
    """Resuelve la raíz de los datasets locales.

    Returns:
        La ruta de ``AUTOPIPE_DATASETS`` si está definida; si no, la carpeta
        ``datasets/`` del repositorio.
    """
    variable = os.environ.get("AUTOPIPE_DATASETS")
    return Path(variable) if variable else RAIZ_REPO / "datasets"


def preparar_dataset(origen: Path, destino: Path) -> Path:
    """Convierte ``<categoría>/test/`` en la estructura ``normal/`` + ``anomaly/``.

    El split de prueba de MVTec AD ya mezcla imágenes normales (carpeta ``good``)
    y anómalas (el resto de carpetas, una por tipo de defecto), que es lo que
    ``ImageDataLoader`` necesita etiquetado.

    Args:
        origen: Carpeta de la categoría de MVTec AD.
        destino: Carpeta donde escribir la estructura convertida.

    Returns:
        La carpeta de destino.

    Raises:
        SystemExit: Si la categoría no está descargada.
    """
    prueba = origen / "test"
    if not prueba.is_dir():
        raise SystemExit(
            f"No se encontró {prueba}. Descargue la categoría 'bottle' de MVTec AD "
            f"y apunte AUTOPIPE_DATASETS a la carpeta que contiene mvtec_ad/. "
            f"Véase datasets/README.md."
        )
    (destino / "normal").mkdir(parents=True)
    (destino / "anomaly").mkdir(parents=True)
    for sub in sorted(p for p in prueba.iterdir() if p.is_dir()):
        carpeta = "normal" if sub.name == "good" else "anomaly"
        for indice, imagen in enumerate(sorted(sub.glob("*.png"))):
            shutil.copy(imagen, destino / carpeta / f"{sub.name}_{indice:03d}.png")
    return destino


def _importar_autopipe_desde(raiz: Path) -> object:
    """Carga el paquete ``autopipe`` desde una raíz concreta.

    Args:
        raiz: Raíz del repositorio de la que debe cargarse el paquete.

    Returns:
        El módulo ``autopipe`` cargado.

    Raises:
        SystemExit: Si el paquete terminó cargándose desde otra ruta.
    """
    sys.meta_path = [
        finder
        for finder in sys.meta_path
        if "__editable__" not in getattr(type(finder), "__module__", "")
    ]
    for nombre in [n for n in sys.modules if n == "autopipe" or n.startswith("autopipe.")]:
        del sys.modules[nombre]
    sys.path.insert(0, str(raiz))

    import autopipe

    cargado = Path(autopipe.__file__).resolve().parent.parent
    if cargado != raiz.resolve():
        raise SystemExit(f"Se cargó autopipe desde {cargado} y se esperaba {raiz}.")
    return autopipe


def ejecutar(raiz: Path) -> dict:
    """Corre la búsqueda con el código de una raíz concreta.

    Args:
        raiz: Raíz del repositorio cuyo código debe ejercitarse.

    Returns:
        Diccionario con el archivo del que se cargó el paquete, el pipeline
        ganador, sus métricas y los objetivos de todos los candidatos.
    """
    paquete = _importar_autopipe_desde(raiz)

    import cv2
    from sklearn.ensemble import IsolationForest

    # OpenCV multihilo puede variar bits entre ejecuciones; un solo hilo la hace
    # exacta (véase tests/test_end_to_end.py).
    cv2.setNumThreads(1)

    temporal = Path(tempfile.mkdtemp())
    try:
        carpeta = preparar_dataset(raiz_datasets() / CATEGORIA, temporal / "listo")
        imagenes, etiquetas = paquete.ImageDataLoader(carpeta).load()
        espacio = {
            "resize": [paquete.Resize(width=64, height=64)],
            "clahe": [
                paquete.CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8)),
                paquete.CLAHETransform(clip_limit=4.0, tile_grid_size=(8, 8)),
            ],
            "blur": [
                paquete.GaussianBlur(kernel_size=3),
                paquete.GaussianBlur(kernel_size=5),
            ],
            "normalize": [paquete.Normalize(mean=0.5, std=0.5)],
        }
        motor = paquete.GridSearchEngine(
            espacio, val_size=VAL_SIZE, random_state=SEMILLA
        )
        mejor, metricas, resultados = motor.run(
            imagenes, etiquetas, IsolationForest(random_state=0), metric=METRICA
        )
    finally:
        shutil.rmtree(temporal, ignore_errors=True)

    return {
        "autopipe_file": str(Path(paquete.__file__).resolve()),
        "n_imagenes": int(len(etiquetas)),
        "n_candidatos": len(resultados),
        "ganador": mejor.get_params(),
        "metricas": metricas,
        "objetivos": [registro["objective"] for registro in resultados],
    }


def ejecutar_aparte(raiz: Path) -> dict:
    """Corre :func:`ejecutar` en un subproceso limpio.

    Cada raíz necesita su propio intérprete: desactivar el *finder* del
    instalador editable y purgar ``sys.modules`` no basta si el paquete ya se
    importó en este proceso.

    Args:
        raiz: Raíz del repositorio cuyo código debe ejercitarse.

    Returns:
        El diccionario de resultados de esa ejecución.
    """
    salida = Path(tempfile.mkdtemp()) / "resultado.json"
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--ejecutar-en", str(raiz),
         "--salida", str(salida)],
        check=True,
    )
    resultado = json.loads(salida.read_text(encoding="utf-8"))
    shutil.rmtree(salida.parent, ignore_errors=True)
    return resultado


def _iguales(uno: float, otro: float) -> bool:
    """Compara dos objetivos tratando ``NaN`` como igual a ``NaN``.

    Args:
        uno: Primer valor.
        otro: Segundo valor.

    Returns:
        ``True`` si son el mismo valor o ambos son ``NaN``.
    """
    return (math.isnan(uno) and math.isnan(otro)) or uno == otro


def comparar(base: dict, actual: dict) -> bool:
    """Compara dos ejecuciones e informa el resultado por pantalla.

    Args:
        base: Resultados del commit de referencia.
        actual: Resultados del árbol de trabajo.

    Returns:
        ``True`` si no hay ninguna diferencia.
    """
    print(f"\ncódigo base   : {base['autopipe_file']}")
    print(f"código actual : {actual['autopipe_file']}")
    print(f"imágenes      : {base['n_imagenes']}   candidatos: {base['n_candidatos']}")

    mismo_ganador = base["ganador"] == actual["ganador"]
    print(f"\nganador idéntico: {mismo_ganador}")
    print(f"  {json.dumps(actual['ganador'], ensure_ascii=False)}")

    print("\nmétricas")
    for clave in sorted(base["metricas"]):
        anterior, nueva = base["metricas"][clave], actual["metricas"][clave]
        estado = "IGUAL" if _iguales(anterior, nueva) else "DIFIERE"
        print(f"  {clave:>20}: {anterior:.10f}  ->  {nueva:.10f}   {estado}")

    distintos = [
        indice
        for indice, (anterior, nueva) in enumerate(
            zip(base["objetivos"], actual["objetivos"])
        )
        if not _iguales(anterior, nueva)
    ]
    print(f"\ntabla completa: {len(distintos)} de {len(base['objetivos'])} difieren")

    sin_regresion = mismo_ganador and not distintos and all(
        _iguales(base["metricas"][c], actual["metricas"][c]) for c in base["metricas"]
    )
    print("\nSIN REGRESIÓN" if sin_regresion else "\nHAY DIFERENCIAS: revisar")
    return sin_regresion


def main() -> int:
    """Punto de entrada.

    Returns:
        ``0`` si no hay regresión, ``1`` si la hay.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=COMMIT_BASE, help="Commit de referencia.")
    parser.add_argument("--ejecutar-en", help="Uso interno: raíz a ejercitar.")
    parser.add_argument("--salida", help="Uso interno: archivo JSON de salida.")
    args = parser.parse_args()

    if args.ejecutar_en:
        Path(args.salida).write_text(
            json.dumps(ejecutar(Path(args.ejecutar_en)), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return 0

    arbol = Path(tempfile.mkdtemp()) / "base"
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(arbol), args.base],
        cwd=RAIZ_REPO,
        check=True,
    )
    try:
        base = ejecutar_aparte(arbol)
        actual = ejecutar_aparte(RAIZ_REPO)
    finally:
        # El borrado del árbol auxiliar puede fallar si OneDrive tiene tomado
        # alguno de sus archivos; ``prune`` deja el registro consistente igual.
        for orden in (["remove", "--force", str(arbol)], ["prune"]):
            subprocess.run(["git", "worktree", *orden], cwd=RAIZ_REPO, check=False)
        shutil.rmtree(arbol.parent, ignore_errors=True)

    return 0 if comparar(base, actual) else 1


if __name__ == "__main__":
    raise SystemExit(main())
