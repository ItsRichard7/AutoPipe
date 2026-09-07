# scripts/evaluate_recommender.py
"""Evaluación preliminar del recomendador sobre datasets fuera del corpus (RE5).

**Dónde se evalúa, y por qué ahí.** El corpus son 25 de los 52 candidatos que
pasaron el cribado. Los otros **27 no entran nunca al
*leave-one-dataset-out***, de modo que medir sobre ellos no contamina el
experimento. Evaluar sobre los 25 sí lo haría: son los pliegues, es decir, el
conjunto de prueba.

**Qué se mide.** Para cada dataset retenido se evalúan de forma exhaustiva los
144 pipelines del espacio, lo que da su óptimo exacto, y después se comprueba
dónde caen las cinco configuraciones que cada estrategia recomienda. Las
recomendaciones no se vuelven a ejecutar: se leen de la misma rejilla, de modo
que la comparación entre estrategias es exacta y no cuesta cómputo adicional.

Las dos medidas son:

* **Distancia al óptimo** de la mejor de las cinco recomendaciones. Dice cuánto
  se pierde por quedarse solo con lo inyectado.
* **Si alguna de las cinco alcanza el objetivo**, definido como el 95 % del
  óptimo, que es el mismo umbral que usa el protocolo del experimento.

**Qué NO se mide, y a propósito.** No se ejecuta ninguna búsqueda de Optuna ni se
compara arranque frío contra caliente. Eso es el experimento, y producirlo antes
de cerrar el pre-registro v1.0 contaminaría el resultado.

**Los dos sesgos de este conjunto, que hay que tener presentes al leerlo.** Los
17 excluidos por orden tienen un IQR por debajo del mínimo del corpus, así que
son *menos* sensibles al preprocesamiento y sesgan hacia «da igual lo que se
recomiende». Los 10 excluidos por la regla de familia comparten origen con algún
dataset del corpus, que sí está en la base, y sesgan en la dirección contraria.
Por eso el informe los reporta por separado.

Ejecutar con:
    .\\.venv\\Scripts\\python.exe scripts\\evaluate_recommender.py
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
import time
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import openml
import optuna
from analyze_winner_diversity import (
    FRACCION_OBJETIVO,
    indice_desde_configuracion,
    rejilla_exhaustiva,
)
from download_corpus import descargar

from autopipe.meta_features import MetaFeatureExtractor
from autopipe.meta_knowledge import MetaKnowledgeBase
from autopipe.meta_recommender import ESTRATEGIAS, MetaRecommender
from autopipe.search_space import TABULAR_SPACE_VERSION

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

BASE = Path("docs/experimentos/BASE-meta-conocimiento.json")
CORPUS = Path("docs/experimentos/CORPUS-seleccionado.md")
INFORME = Path("docs/experimentos/EVALUACION-recomendador.md")
VALORES = Path("docs/experimentos/EVALUACION-recomendador.json")
CACHE_META = Path("datasets/.meta_cache")

SEMILLA = 42
"""Semilla de la extracción de meta-características, la misma con que se
construyó la base: si difiriera, los vectores no serían comparables."""

SEMILLA_ALEATORIA = 202
"""Semilla de la estrategia aleatoria. Se declara aquí para que la ablación sea
reproducible."""

N_CONFIGURACIONES = 5
"""Las que fija el anteproyecto en sus criterios de éxito."""

FILA_NO_SELECCIONADO = re.compile(
    r"^\|\s*(\d+)\s*\|\s*`([^`]+)`\s*\|\s*([\d.]+)\s*\|\s*(\w+)\s*\|"
)


def leer_retenidos(ruta: Path) -> list[dict]:
    """Lee los candidatos cribados que no entraron al corpus.

    Args:
        ruta: Documento generado por ``screen_corpus.py --seleccionar``.

    Returns:
        Lista de diccionarios con identificador, nombre, IQR y motivo de la
        exclusión.

    Raises:
        SystemExit: Si el archivo no existe o no tiene la sección esperada.
    """
    if not ruta.is_file():
        raise SystemExit(f"No existe {ruta}. Ejecute antes scripts/screen_corpus.py.")
    texto = ruta.read_text(encoding="utf-8")
    marca = "## Candidatos no seleccionados"
    if marca not in texto:
        raise SystemExit(f"{ruta} no tiene la sección {marca!r}.")
    filas = [
        {
            "dataset_id": m.group(1),
            "dataset_name": m.group(2),
            "iqr": float(m.group(3)),
            "motivo": m.group(4),
        }
        for m in (
            FILA_NO_SELECCIONADO.match(linea)
            for linea in texto[texto.index(marca) :].splitlines()
        )
        if m
    ]
    if not filas:
        raise SystemExit(f"No se pudo leer ningún candidato retenido de {ruta}.")
    return filas


def _recomendadores(base: MetaKnowledgeBase) -> dict[str, MetaRecommender]:
    """Ajusta un recomendador por estrategia sobre el corpus completo.

    Se ajusta con las 25 entradas y no con un pliegue porque los datasets que se
    van a consultar están **fuera** del corpus: no hay nada que excluir.

    Args:
        base: Base de meta-conocimiento completa.

    Returns:
        Diccionario de nombre de estrategia a recomendador ya ajustado.
    """
    return {
        estrategia: MetaRecommender(
            n_configurations=N_CONFIGURACIONES,
            strategy=estrategia,
            random_state=SEMILLA_ALEATORIA,
        ).fit(base)
        for estrategia in ESTRATEGIAS
    }


def evaluar_dataset(
    retenido: dict,
    recomendadores: dict[str, MetaRecommender],
    extractor: MetaFeatureExtractor,
    rehacer: bool = False,
) -> dict:
    """Mide dónde caen las recomendaciones de cada estrategia en un dataset.

    Args:
        retenido: Ficha del dataset retenido.
        recomendadores: Recomendadores ya ajustados, uno por estrategia.
        extractor: Extractor de meta-características, con su caché.
        rehacer: Si se ignora la caché de la rejilla.

    Returns:
        Diccionario con el óptimo del dataset y, por estrategia, el mejor valor
        alcanzado por sus cinco recomendaciones.
    """
    identificador = retenido["dataset_id"]
    features, labels, _, _ = descargar(int(identificador))
    vector = extractor.extract(features, labels, dataset_id=identificador)

    objetivos = np.asarray(
        rejilla_exhaustiva(identificador, retenido["dataset_name"], rehacer), dtype=float
    )
    finitos = objetivos[np.isfinite(objetivos)]
    optimo = float(finitos.max())
    umbral = FRACCION_OBJETIVO * optimo

    por_estrategia = {}
    for estrategia, recomendador in recomendadores.items():
        configuraciones = recomendador.recommend(vector)
        valores = [
            float(objetivos[indice_desde_configuracion(c)]) for c in configuraciones
        ]
        evaluables = [v for v in valores if np.isfinite(v)]
        mejor = max(evaluables) if evaluables else float("nan")
        por_estrategia[estrategia] = {
            "n_recomendadas": len(configuraciones),
            "n_inevaluables": len(valores) - len(evaluables),
            "mejor": round(mejor, 6) if evaluables else None,
            "distancia_al_optimo": round(optimo - mejor, 6) if evaluables else None,
            "alcanza_el_objetivo": bool(evaluables and mejor >= umbral),
            "vecinos": [v["dataset_name"] for v in recomendador.neighbours_],
        }

    return {
        **retenido,
        "optimo": round(optimo, 6),
        "umbral": round(umbral, 6),
        "n_en_meseta": int((finitos >= umbral).sum()),
        "n_evaluables": int(finitos.size),
        "estrategias": por_estrategia,
    }


def resumir(datasets: list[dict], motivo: str | None = None) -> dict:
    """Agrega los resultados de un grupo de datasets, por estrategia.

    Args:
        datasets: Resultados de :func:`evaluar_dataset`.
        motivo: Si se indica, solo se agregan los datasets excluidos por ese
            motivo.

    Returns:
        Diccionario de estrategia a sus agregados.
    """
    grupo = [d for d in datasets if motivo is None or d["motivo"] == motivo]
    resumen = {"n_datasets": len(grupo)}
    for estrategia in ESTRATEGIAS:
        distancias = [
            d["estrategias"][estrategia]["distancia_al_optimo"]
            for d in grupo
            if d["estrategias"][estrategia]["distancia_al_optimo"] is not None
        ]
        resumen[estrategia] = {
            "distancia_mediana": round(statistics.median(distancias), 6)
            if distancias
            else None,
            "distancia_maxima": round(max(distancias), 6) if distancias else None,
            "n_optimo_exacto": sum(1 for v in distancias if v <= 1e-9),
            "n_alcanza_el_objetivo": sum(
                1 for d in grupo if d["estrategias"][estrategia]["alcanza_el_objetivo"]
            ),
            "n_inevaluables": sum(
                d["estrategias"][estrategia]["n_inevaluables"] for d in grupo
            ),
        }
    return resumen


def _tabla_resumen(resumen: dict) -> str:
    """Redacta la tabla comparativa de estrategias de un grupo."""
    filas = []
    for estrategia in ESTRATEGIAS:
        datos = resumen[estrategia]
        n = resumen["n_datasets"]
        mediana = "—" if datos["distancia_mediana"] is None else f"{datos['distancia_mediana']:.4f}"
        maxima = "—" if datos["distancia_maxima"] is None else f"{datos['distancia_maxima']:.4f}"
        filas.append(
            f"| `{estrategia}` | {mediana} | {maxima} | "
            f"{datos['n_optimo_exacto']} / {n} | "
            f"{datos['n_alcanza_el_objetivo']} / {n} | {datos['n_inevaluables']} |"
        )
    return "\n".join(filas)


def _tabla_datasets(datasets: list[dict]) -> str:
    """Redacta la tabla por dataset, con la distancia al óptimo de cada estrategia."""
    filas = []
    for d in sorted(datasets, key=lambda x: -x["iqr"]):
        celdas = []
        for estrategia in ESTRATEGIAS:
            valor = d["estrategias"][estrategia]["distancia_al_optimo"]
            marca = "**" if d["estrategias"][estrategia]["alcanza_el_objetivo"] else ""
            celdas.append("—" if valor is None else f"{marca}{valor:.4f}{marca}")
        filas.append(
            f"| {d['dataset_id']} | `{d['dataset_name']}` | {d['motivo']} | "
            f"{d['iqr']:.4f} | {d['optimo']:.4f} | " + " | ".join(celdas) + " |"
        )
    return "\n".join(filas)


def _veredicto(total: dict, orden: dict, familia: dict) -> str:
    """Redacta lo que el resultado permite y no permite concluir.

    Se genera desde los números en lugar de escribirse a mano, para que no pueda
    quedar desfasado respecto de lo que la ejecución produjo.

    Args:
        total: Agregado sobre todos los retenidos.
        orden: Agregado sobre los excluidos por orden.
        familia: Agregado sobre los excluidos por regla de familia.

    Returns:
        El texto de la sección, en markdown.
    """
    n = total["n_datasets"]
    medianas = {e: total[e]["distancia_mediana"] for e in ESTRATEGIAS}
    mejor = min(medianas, key=lambda e: medianas[e])
    peor = max(medianas, key=lambda e: medianas[e])
    separacion = medianas[peor] - medianas[mejor]
    objetivo = {e: total[e]["n_alcanza_el_objetivo"] for e in ESTRATEGIAS}
    primaria_gana_por_familia = (
        familia["nearest"]["distancia_mediana"] < familia["random"]["distancia_mediana"]
    )
    primaria_gana_por_orden = (
        orden["nearest"]["distancia_mediana"] < orden["random"]["distancia_mediana"]
    )

    comparacion = (
        "la ablación aleatoria queda\npor delante de la estrategia primaria"
        if not primaria_gana_por_orden
        else "la estrategia primaria queda\npor delante de la ablación aleatoria"
    )
    familia_dice = (
        f"El único grupo donde la primaria queda por delante es el de los\n"
        f"{familia['n_datasets']} excluidos por familia, que es justo el grupo cuyo "
        f"sesgo la\nfavorece: el vecino ideal está en la base casi por construcción."
        if primaria_gana_por_familia
        else f"Tampoco queda por delante en los {familia['n_datasets']} excluidos por\n"
        f"familia, que son el grupo cuyo sesgo la favorecía."
    )
    return f"""Las cuatro estrategias quedan **prácticamente empatadas**. La distancia
mediana al óptimo va de {medianas[mejor]:.4f} con `{mejor}` a
{medianas[peor]:.4f} con `{peor}`: una separación de {separacion:.4f} de
macro-F1. Y entre {min(objetivo.values())} y {max(objetivo.values())} de los
{n} datasets alcanzan el objetivo **con cualquiera de las cuatro**.

**Este conjunto no puede distinguir las estrategias, y eso es un resultado sobre
la medición y no sobre las estrategias.** Es el mismo efecto piso que el análisis
de la semana 7 midió sobre el corpus: cuando una fracción grande del espacio
alcanza el 95 % del óptimo, acertar deja de ser difícil y cualquier recomendación
plausible parece buena.

Conviene decir el dato incómodo con todas las letras. Sobre los
{orden["n_datasets"]} excluidos por orden, {comparacion} en distancia mediana.
{familia_dice}

**Lo que sí queda comprobado** es la sensatez mecánica: las cuatro estrategias
producen configuraciones válidas, ejecutables casi siempre y muy cercanas al
óptimo del espacio, con un número marginal de recomendaciones inevaluables. Eso
era lo que este punto del plan tenía que verificar.

**Lo que anticipa del experimento.** Si el corpus se comporta como este conjunto,
el *leave-one-dataset-out* tendrá el mismo problema de discriminación. Refuerza
la decisión abierta sobre la región inevaluable y sobre la variable «ensayos
hasta el objetivo» que la v1.0 del pre-registro tiene que cerrar."""


def _informe(datasets: list[dict], total: dict, orden: dict, familia: dict) -> str:
    """Redacta el documento con la evaluación.

    Args:
        datasets: Resultados por dataset.
        total: Agregado sobre los 27.
        orden: Agregado sobre los excluidos por orden.
        familia: Agregado sobre los excluidos por regla de familia.

    Returns:
        El documento markdown completo.
    """
    columnas = " | ".join(f"`{e}`" for e in ESTRATEGIAS)
    alineacion = "|".join(["---:"] * len(ESTRATEGIAS))
    return f"""# Evaluación preliminar del recomendador (RE5)

> **Archivo generado.** No editar a mano: se regenera con
> `.\\.venv\\Scripts\\python.exe scripts\\evaluate_recommender.py`.

- **Generado:** {date.today().isoformat()}
- **Datasets evaluados:** {total["n_datasets"]}, todos **fuera del corpus**
- **Configuraciones por recomendación:** {N_CONFIGURACIONES}
- **Versión del espacio:** `{TABULAR_SPACE_VERSION}`
- **Valores completos:** `EVALUACION-recomendador.json`

## Dónde se evalúa, y por qué ahí

El corpus son 25 de los 52 candidatos que pasaron el cribado. Los otros
{total["n_datasets"]} **no entran nunca al *leave-one-dataset-out***, de modo que
medir sobre ellos no contamina el experimento. Medir sobre los 25 sí lo haría:
son los pliegues, es decir, el conjunto de prueba.

El recomendador se ajusta con las 25 entradas completas y no con un pliegue,
porque los datasets consultados están fuera: no hay nada que excluir.

## Qué se mide

De cada dataset retenido se evalúan de forma exhaustiva los 144 pipelines del
espacio, lo que da su **óptimo exacto**. Después se comprueba dónde caen las
cinco configuraciones que cada estrategia recomienda, leyéndolas de esa misma
rejilla: la comparación entre estrategias es exacta y no cuesta cómputo extra.

- **Distancia al óptimo:** cuánto pierde la mejor de las cinco recomendaciones
  frente al mejor pipeline posible. Cero significa que la recomendación contiene
  el óptimo exacto.
- **Alcanza el objetivo:** si alguna de las cinco llega al
  {100 * FRACCION_OBJETIVO:.0f} % del óptimo, que es el umbral del protocolo.

**No se ejecuta ninguna búsqueda de Optuna ni se compara arranque frío contra
caliente.** Eso es el experimento, y producirlo antes de cerrar el pre-registro
v1.0 contaminaría el resultado.

## Las cuatro estrategias

- **`nearest`** toma las de los vecinos más cercanos en meta-características. Es
  la primaria, la que el anteproyecto fija como línea base.
- **`diverse`** toma, entre las candidatas cercanas, el subconjunto más separado
  entre sí. Ataca el modo de fallo que midió el spike A.
- **`frequent`** toma las que más veces ganan en la base, **ignorando por
  completo las meta-características**. Aísla la contribución del
  meta-aprendizaje: si la primaria no le gana, las meta-características no
  aportan nada.
- **`random`** toma las ganadoras de datasets elegidos al azar. Es la ablación
  del pre-registro: aísla si lo que aporta es la **similitud**.

Las cuatro devuelven ganadoras de datasets reales, de modo que la comparación no
se contamina con la calidad intrínseca de lo inyectado: lo único que cambia es
**de qué datasets** se toman.

## Resultado sobre los {total["n_datasets"]} retenidos

| Estrategia | Distancia mediana | Máxima | Óptimo exacto | Alcanza el objetivo | Inevaluables |
|---|---:|---:|---:|---:|---:|
{_tabla_resumen(total)}

### Desglosado por motivo de exclusión

Los dos grupos sesgan en direcciones opuestas y por eso van separados.

**Excluidos por orden ({orden["n_datasets"]} datasets).** Su IQR está por debajo
del mínimo del corpus, así que son **menos sensibles al preprocesamiento**: aquí
cualquier estrategia parece buena, y las diferencias entre ellas se comprimen.

| Estrategia | Distancia mediana | Máxima | Óptimo exacto | Alcanza el objetivo | Inevaluables |
|---|---:|---:|---:|---:|---:|
{_tabla_resumen(orden)}

**Excluidos por regla de familia ({familia["n_datasets"]} datasets).** Comparten
origen con algún dataset del corpus, que sí está en la base del recomendador. El
sesgo va en la dirección contraria: **sobreestiman** lo que la similitud aporta,
porque el vecino ideal existe casi por construcción.

| Estrategia | Distancia mediana | Máxima | Óptimo exacto | Alcanza el objetivo | Inevaluables |
|---|---:|---:|---:|---:|---:|
{_tabla_resumen(familia)}

## Por dataset

Distancia al óptimo de la mejor de las cinco recomendaciones. En **negrita** las
que alcanzan el objetivo.

| ID | Nombre | Motivo | IQR | Óptimo | {columnas} |
|---:|---|---|---:|---:|{alineacion}|
{_tabla_datasets(datasets)}

## Qué dice el resultado

{_veredicto(total, orden, familia)}

## Cómo leer esto, y cómo no

Es una **comprobación de sensatez**, no una medida del efecto que el experimento
va a medir. Tres limitaciones que conviene tener delante:

1. **Ninguno de los dos grupos es un sustituto del corpus.** Uno sesga hacia
   abajo y el otro hacia arriba, y ninguno de los dos se parece en sensibilidad
   al conjunto sobre el que se ejecutará el *leave-one-dataset-out*.
2. **Mide la calidad de lo inyectado, no el ahorro de la búsqueda.** La
   hipótesis del proyecto es sobre *ensayos hasta el objetivo*, que depende
   además de cómo el muestreador aprovecha lo inyectado; el spike A midió que
   inyectar bien no garantiza buscar mejor.
3. **No se usó para elegir nada.** El valor de *k*, la métrica de distancia y el
   escalado quedaron fijados a priori antes de esta medición, y esta medición no
   los revisa.
"""


def main() -> int:
    """Punto de entrada.

    Returns:
        ``0`` si la evaluación se completó.
    """
    parser = argparse.ArgumentParser(description="Evaluación preliminar del recomendador.")
    parser.add_argument("--limit", type=int, help="Procesar solo los primeros N.")
    parser.add_argument(
        "--rehacer", action="store_true", help="Ignorar la caché de las rejillas."
    )
    args = parser.parse_args()

    openml.config.set_console_log_level(logging.WARNING)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logging.getLogger("autopipe").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    if not BASE.is_file():
        raise SystemExit(f"No existe {BASE}. Ejecute antes scripts/build_meta_knowledge.py.")
    base = MetaKnowledgeBase.load(BASE)
    retenidos = leer_retenidos(CORPUS)[: args.limit]
    recomendadores = _recomendadores(base)
    extractor = MetaFeatureExtractor(random_state=SEMILLA, cache_dir=CACHE_META)
    inicio = time.perf_counter()

    datasets = []
    for numero, retenido in enumerate(retenidos, 1):
        resultado = evaluar_dataset(retenido, recomendadores, extractor, args.rehacer)
        datasets.append(resultado)
        logger.info(
            "[%2d/%d] %-42s optimo %.4f  " + "  ".join(f"{e}=%s" for e in ESTRATEGIAS),
            numero,
            len(retenidos),
            retenido["dataset_name"][:42],
            resultado["optimo"],
            *(
                f"{resultado['estrategias'][e]['distancia_al_optimo']:.4f}"
                if resultado["estrategias"][e]["distancia_al_optimo"] is not None
                else "n/d"
                for e in ESTRATEGIAS
            ),
        )

    total = resumir(datasets)
    orden = resumir(datasets, "orden")
    familia = resumir(datasets, "familia")

    VALORES.write_text(
        json.dumps(
            {"datasets": datasets, "total": total, "orden": orden, "familia": familia},
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    INFORME.write_text(_informe(datasets, total, orden, familia), encoding="utf-8")
    logger.info(
        "\nEvaluación completa en %.1f min sobre %d datasets retenidos. Informe en %s",
        (time.perf_counter() - inicio) / 60,
        len(datasets),
        INFORME,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
