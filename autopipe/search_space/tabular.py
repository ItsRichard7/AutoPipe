# autopipe/search_space/tabular.py
"""Espacio de búsqueda tabular canónico, único y versionado.

El espacio es **el mismo para todos los datasets** del corpus y se identifica con
un número de versión. Las dos propiedades son requisitos del arranque en
caliente, no comodidades:

* **Único.** El recomendador transfiere configuraciones halladas en unos datasets
  hacia otro que nunca vio. Si cada dataset tuviera su propio espacio, no habría
  nada que transferir.
* **Versionado.** Los parámetros que Optuna registra por ensayo son *índices
  enteros* sobre la lista de opciones de cada paso, más un índice adicional que
  representa omitirlo. Un índice no significa nada por sí mismo: depende del
  orden de la lista. Si el espacio cambiara de claves o de orden, una
  configuración inyectada como arranque en caliente designaría transformaciones
  distintas de las previstas **sin producir ningún error visible**. Comparar la
  versión convierte ese fallo mudo en un rechazo explícito.

Véase ``docs/experimentos/ESPACIO-tabular-v1.md`` para la composición congelada y
la tabla de índices, y la Sección 3 del pre-registro para la justificación.
"""

from __future__ import annotations

from collections.abc import Sequence

from autopipe.pipeline.base import BaseTransform
from autopipe.pipeline.pipeline import Pipeline
from autopipe.pipeline.tabular_transforms import (
    EncodeCategorical,
    ImputeMissing,
    ScaleNumeric,
    SelectFeatures,
)

TABULAR_SPACE_VERSION = "tabular-v1"
"""Identificador de la versión vigente del espacio tabular.

Se registra junto a cada entrada de la base de meta-conocimiento y junto a cada
resultado experimental. **Cambiar la composición del espacio obliga a cambiar
esta constante**: las configuraciones guardadas con la versión anterior dejan de
ser interpretables, porque sus índices apuntarían a otras transformaciones.
"""

TABULAR_SPACE_STEPS = ("imputation", "scaling", "encoding", "selection")
"""Pasos del espacio, en su orden canónico de aplicación.

El orden está congelado y es parte de la identidad de la versión. Motivo de este
orden en concreto: el escalado va **antes** de la codificación para que actúe
solo sobre las columnas numéricas originales y deje las indicadoras one-hot en
``0``/``1``. Estandarizar las indicadoras llevaría las categorías raras a valores
cercanos a ``±10``, que con un modelo base de vecinos más cercanos y distancia
euclídea dominaría la métrica. La selección va al final, cuando la tabla ya es
enteramente numérica.
"""


def build_tabular_search_space() -> dict[str, list[BaseTransform]]:
    """Construye el espacio de búsqueda tabular canónico.

    Devuelve **instancias nuevas** en cada invocación. Las transformaciones
    tabulares tienen estado, de modo que compartir instancias entre dos búsquedas
    arrastraría lo aprendido de una hacia la otra.

    Cada paso recibe además, de forma automática y dentro de los motores de
    búsqueda, la opción de **omitirse**. Con ella el espacio contiene
    ``3 x 4 x 3 x 4 = 144`` pipelines distintos.

    Returns:
        Diccionario de nombre de paso a lista de transformaciones candidatas, con
        las claves en el orden canónico de :data:`TABULAR_SPACE_STEPS`.
    """
    return {
        "imputation": [ImputeMissing("mean"), ImputeMissing("median")],
        "scaling": [
            ScaleNumeric("standard"),
            ScaleNumeric("minmax"),
            ScaleNumeric("robust"),
        ],
        "encoding": [EncodeCategorical("onehot"), EncodeCategorical("ordinal")],
        "selection": [
            SelectFeatures(0.25, "f_classif"),
            SelectFeatures(0.5, "f_classif"),
            SelectFeatures(0.75, "f_classif"),
        ],
    }


def validate_space_version(version: str) -> None:
    """Verifica que una configuración provenga de la versión vigente del espacio.

    Debe invocarse antes de inyectar como arranque en caliente cualquier
    configuración recuperada de la base de meta-conocimiento.

    Args:
        version: Identificador de versión con el que se guardó la configuración.

    Raises:
        ValueError: Si la versión no es la vigente. Se rechaza en lugar de
            ejecutarse: los índices de una versión distinta designarían
            transformaciones que no son las que se creyó recomendar.
    """
    if version != TABULAR_SPACE_VERSION:
        raise ValueError(
            f"La configuración proviene del espacio de búsqueda {version!r} y la "
            f"versión vigente es {TABULAR_SPACE_VERSION!r}. Sus índices designarían "
            f"transformaciones distintas de las previstas, así que se rechaza en "
            f"lugar de ejecutarse."
        )


def pipeline_from_configuration(
    configuration: dict[str, int],
    space: dict[str, list[BaseTransform]] | None = None,
) -> Pipeline:
    """Construye el pipeline que designa una configuración de índices.

    Es la operación inversa de lo que registran los motores de búsqueda: ellos
    guardan un índice por paso, y esto los resuelve a transformaciones. La
    necesitan tanto la validación de la base de meta-conocimiento como el
    recomendador, que recibe configuraciones ajenas y tiene que ejecutarlas.

    Args:
        configuration: Índice elegido por cada paso. El índice ``len(opciones)``
            significa omitir el paso.
        space: Espacio sobre el que resolver los índices. Si es ``None`` se
            construye el canónico.

    Returns:
        El pipeline correspondiente, sin los pasos omitidos.

    Raises:
        ValueError: Si falta algún paso del espacio o si un índice queda fuera de
            rango. Se rechaza en lugar de recortarlo: un índice fuera de rango
            indica que la configuración proviene de otro espacio, y ejecutarla
            aplicaría transformaciones que no son las que se creyó recomendar.
    """
    espacio = build_tabular_search_space() if space is None else space
    faltantes = [paso for paso in TABULAR_SPACE_STEPS if paso not in configuration]
    if faltantes:
        raise ValueError(
            f"A la configuración le faltan los pasos {faltantes}. Debe indicar un "
            f"índice por cada paso de {list(TABULAR_SPACE_STEPS)}."
        )
    pasos: list[BaseTransform] = []
    for paso in TABULAR_SPACE_STEPS:
        indice = configuration[paso]
        opciones = espacio[paso]
        if not 0 <= indice <= len(opciones):
            raise ValueError(
                f"El índice {indice} del paso {paso!r} está fuera de rango: el paso "
                f"admite de 0 a {len(opciones)}, donde {len(opciones)} significa "
                f"omitirlo. La configuración no proviene de este espacio."
            )
        if indice < len(opciones):
            pasos.append(opciones[indice])
    return Pipeline(pasos)


def omitted_steps(
    configuration: dict[str, int],
    space: dict[str, list[BaseTransform]] | None = None,
) -> list[str]:
    """Pasos que una configuración omite.

    Args:
        configuration: Índice elegido por cada paso.
        space: Espacio sobre el que interpretar los índices. Si es ``None`` se
            construye el canónico.

    Returns:
        Nombres de los pasos cuyo índice es el de omisión, en orden canónico.
    """
    espacio = build_tabular_search_space() if space is None else space
    return [
        paso
        for paso in TABULAR_SPACE_STEPS
        if configuration.get(paso) == len(espacio[paso])
    ]


def canonical_configuration(
    configurations: Sequence[dict[str, int]],
    space: dict[str, list[BaseTransform]] | None = None,
) -> dict[str, int]:
    """Elige un representante estable entre configuraciones que empatan.

    Sobre un dataset numérico y sin faltantes, la imputación y la codificación
    son la identidad: las combinaciones de esos dos pasos producen **el mismo
    pipeline** y empatan exactamente. La que una búsqueda guarda como ganadora es
    entonces la que el muestreador visitó primero, que es una propiedad de la
    corrida y no del dataset. Transferirla como arranque en caliente transfiere
    esa arbitrariedad.

    La regla es **la que omite menos pasos y, a igualdad, la de índices menores
    en el orden canónico**. No se elige por rendimiento —todas empatan— sino por
    transferibilidad: omitir la imputación o la codificación no cuesta nada donde
    esos pasos no tienen nada que hacer, pero vuelve el pipeline **inevaluable**
    en un dataset receptor que sí tenga valores faltantes o columnas
    categóricas, y ese ensayo se pierde sin producir una métrica.

    Args:
        configurations: Configuraciones empatadas. Debe tener al menos una.
        space: Espacio sobre el que interpretar los índices. Si es ``None`` se
            construye el canónico.

    Returns:
        La configuración canónica del empate.

    Raises:
        ValueError: Si no se recibe ninguna configuración.
    """
    if not configurations:
        raise ValueError("No hay configuraciones entre las que elegir una canónica.")
    espacio = build_tabular_search_space() if space is None else space
    return min(
        configurations,
        key=lambda c: (
            len(omitted_steps(c, espacio)),
            tuple(c[paso] for paso in TABULAR_SPACE_STEPS),
        ),
    )


def describe_tabular_search_space() -> list[dict]:
    """Enumera el espacio canónico paso por paso y por índice.

    Existe para que la documentación y los reportes se generen desde el propio
    espacio y no puedan quedar desfasados respecto de él.

    Returns:
        Lista de diccionarios con las claves ``step``, ``index``, ``transform`` y
        ``params``. El índice ``len(opciones)`` de cada paso, que representa
        omitirlo, no se enumera aquí: lo añaden los motores de búsqueda.
    """
    espacio = build_tabular_search_space()
    return [
        {
            "step": paso,
            "index": indice,
            "transform": type(transformacion).__name__,
            "params": transformacion.get_params(),
        }
        for paso in TABULAR_SPACE_STEPS
        for indice, transformacion in enumerate(espacio[paso])
    ]
