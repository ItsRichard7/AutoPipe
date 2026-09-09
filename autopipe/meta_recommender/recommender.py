# autopipe/meta_recommender/recommender.py
"""Recomendador de arranque en caliente por vecinos más cercanos.

Dado el vector de meta-características de un dataset nuevo, busca en la base de
meta-conocimiento los datasets más parecidos y devuelve sus configuraciones
ganadoras, listas para inyectarse en el motor de búsqueda con
``optuna.enqueue_trial``.

**Es una línea base deliberada, no un modelo entrenado.** El anteproyecto fija los
vecinos más cercanos como el recomendador del alcance mínimo y deja «un
meta-modelo más sofisticado» como extensión deseable.

Las tres decisiones de diseño, y por qué son las que son:

* **La normalización se ajusta con la base recibida y nada más.** En el
  *leave-one-dataset-out*, cada pliegue construye el recomendador con
  ``base.without(d)``, de modo que ni la mediana, ni el rango intercuartílico, ni
  el conjunto de meta-características supervivientes ven jamás el dataset
  excluido. Ajustarlos sobre el corpus completo contaminaría cada pliegue e
  inflaría la ventaja del arranque en caliente de un modo difícil de detectar a
  posteriori. Es el análogo, en el nivel meta, de ajustar las transformaciones
  con estado usando solo el subconjunto de entrenamiento.

* **La distancia es L1 sobre características escaladas de forma robusta.** La L1
  tiene precedente publicado para exactamente esta tarea: auto-sklearn ordena los
  datasets por su distancia L1 en el espacio de meta-características y usa los
  *k* más cercanos para arrancar en caliente su optimización bayesiana (Feurer
  et al., 2015, sección 3.1). El escalado, en cambio, **no** proviene de ahí: ese
  artículo no describe ninguno, y aquí hace falta porque el corpus mezcla
  meta-características acotadas con otras que van de 500 a 45 000; sin escalar,
  esas últimas dominarían la distancia por su magnitud. Se usan la mediana y el
  rango intercuartílico, y no la media y la desviación, porque con dos docenas de
  datasets un solo valor extremo desplaza ambos estadísticos.

* **El vector consultado se recorta al rango observado en el pliegue.** El
  escalado robusto controla los valores extremos *dentro* del conjunto de ajuste,
  pero no los de la consulta, y en el *leave-one-dataset-out* el dataset excluido
  puede caer muy fuera del corpus. Medido sobre el corpus real, sin recorte hay
  dos pliegues en los que **una sola meta-característica aporta más de la mitad
  de la distancia** —hasta el 71 %—, de modo que el vecino más parecido no lo
  elige un espacio de 87 meta-características sino una estadística de
  covarianza. El recorte acota lo que cualquier característica puede aportar sin
  necesidad de una constante arbitraria, y descansa en un argumento sencillo: el
  recomendador no puede medir *cuánto* fuera del corpus cae un dataset, solo que
  cae fuera. Afecta únicamente a la consulta, nunca al ajuste, así que no abre
  ninguna vía de fuga.

* **El número de vecinos no es un parámetro libre.** Lo que el anteproyecto fija
  son las **cinco configuraciones** que se inyectan, así que el recomendador
  camina hacia afuera desde el vecino más cercano hasta reunir cinco
  configuraciones **distintas**. Inyectar duplicados sería pérdida pura de
  presupuesto: dos ensayos idénticos gastan dos de los treinta y devuelven el
  mismo valor. La diversidad entre las cinco importa además por el modo de fallo
  que midió el spike A, donde cinco recomendaciones parecidas entre sí y
  equivocadas resultaron peores que no recomendar nada.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np

from autopipe.meta_knowledge import MetaKnowledgeBase
from autopipe.pipeline.base import NotFittedError
from autopipe.search_space import pipeline_from_configuration, validate_space_version

logger = logging.getLogger(__name__)

ESCALADOS = ("robust", "standard")
"""Formas de escalar el espacio de meta-características.

``robust`` (mediana y rango intercuartílico) es la primaria. ``standard`` (media
y desviación típica) existe para el análisis de sensibilidad declarado de
antemano, con el mismo patrón que ``n_startup_trials``: un valor primario y una
alternativa anunciada antes de ver los resultados.
"""

METRICAS = ("manhattan", "euclidean")
"""Distancias admitidas. ``manhattan`` (L1) es la primaria."""

ESTRATEGIAS = ("nearest", "diverse", "frequent", "random", "hybrid")
"""Cómo se eligen las configuraciones a recomendar.

Las cuatro devuelven configuraciones ganadoras de datasets reales de la base, de
modo que ninguna comparación entre ellas se contamina con la calidad intrínseca
de lo inyectado: lo único que cambia es **de qué datasets** se toman.

* ``nearest`` --- los vecinos más cercanos en el espacio de meta-características.
  Es la estrategia primaria y la que el anteproyecto fija como línea base.
* ``diverse`` --- también por cercanía, pero eligiendo entre las candidatas
  próximas el subconjunto más separado entre sí. El spike A midió que el modo de
  fallo del arranque en caliente son cinco recomendaciones parecidas entre sí y
  todas equivocadas, que TPE interpreta como una región buena; esta estrategia
  ataca ese modo de fallo directamente.
* ``frequent`` --- las configuraciones que más veces ganan en la base,
  **ignorando por completo las meta-características**. Es el control que aísla
  la contribución del meta-aprendizaje: si la estrategia primaria no le gana a
  esto, las meta-características no están aportando nada y bastaría recomendar
  siempre lo mismo.
* ``random`` --- las ganadoras de datasets elegidos al azar. Es la ablación que
  el pre-registro prevé: aísla si lo que aporta es la **similitud**, y no el
  mero hecho de arrancar en caliente con configuraciones plausibles.
* ``hybrid`` --- las dos primeras de ``frequent`` seguidas de las de ``nearest``.
  Nace del resultado del experimento 1, donde ``frequent`` superó a la estrategia
  primaria: si un buen valor por defecto vale más que la similitud, conviene
  probar si reservar parte del presupuesto a cada cosa vale más que cualquiera de
  las dos por separado. La define el pre-registro del experimento 2, y **no es la
  estrategia primaria de ningún contraste confirmatorio**: viaja en una condición
  aparte precisamente para que ``nearest`` siga siendo comparable con el
  experimento 1.
"""

N_HIBRIDO_FRECUENTES = 2
"""Cuántas configuraciones aporta ``frequent`` al arranque de ``hybrid``.

El reparto ---dos por frecuencia y el resto por cercanía--- lo fija el pre-registro
del experimento 2 y **no se explora**: probar varios repartos y quedarse con el
mejor sería ajustar sobre el resultado. La mayoría queda del lado de la similitud
porque es el método que el anteproyecto compromete, y las dos por frecuencia son el
mínimo que garantiza incluir las configuraciones más transferibles del pliegue.
"""

FACTOR_DIVERSIDAD = 2
"""Cuántas candidatas considera ``diverse`` por cada una que elige.

Acota lo lejos que la estrategia puede ir a buscar diversidad: con el doble de
candidatas, el subconjunto más separado sigue saliendo del entorno cercano del
dataset consultado. Es una decisión de diseño declarada, no un valor ajustado
sobre los datos.
"""


class MetaRecommender:
    """Recomienda configuraciones iniciales por vecinos más cercanos.

    Uso en un pliegue del *leave-one-dataset-out*::

        recomendador = MetaRecommender().fit(base.without(dataset_excluido))
        configuraciones = recomendador.recommend(vector_del_excluido)

    Attributes:
        n_configurations (int): Configuraciones distintas a devolver.
        scaling (str): Forma de escalar, de :data:`ESCALADOS`.
        metric (str): Distancia, de :data:`METRICAS`.
        strategy (str): Cómo se eligen las configuraciones, de
            :data:`ESTRATEGIAS`. La primaria es ``nearest``.
        random_state (int | None): Semilla de la estrategia ``random``.
        feature_names_ (list[str]): Meta-características con las que se ajustó,
            disponibles tras :meth:`fit`. Son las supervivientes **de la base
            recibida**, recalculadas por pliegue.
        center_ (np.ndarray): Centro de cada característica.
        scale_ (np.ndarray): Escala de cada característica.
        lower_ (np.ndarray): Mínimo observado en el pliegue por característica.
        upper_ (np.ndarray): Máximo observado en el pliegue por característica.
        clipped_features_ (list[str]): Meta-características en las que la última
            consulta cayó fuera del rango del pliegue y se recortó. Se expone
            porque un recorte es información: dice en qué se sale del corpus el
            dataset consultado.
        neighbours_ (list[dict]): Vecinos que aportaron cada configuración
            devuelta, disponibles tras :meth:`recommend`. Cada elemento lleva el
            identificador y el nombre del dataset, su distancia y su
            configuración. Existe para que una recomendación sea auditable: sin
            esto, el experimento registraría qué se inyectó pero no de dónde
            salió.
        used_features_ (list[str]): Meta-características que intervinieron de
            hecho en la última consulta.
    """

    def __init__(
        self,
        n_configurations: int = 5,
        scaling: str = "robust",
        metric: str = "manhattan",
        strategy: str = "nearest",
        random_state: int | None = None,
    ) -> None:
        """Inicializa el recomendador.

        Args:
            n_configurations: Cuántas configuraciones distintas devolver. El
                anteproyecto fija cinco en sus criterios de éxito.
            scaling: Forma de escalar el espacio de meta-características.
            metric: Distancia entre vectores escalados.
            strategy: Cómo se eligen las configuraciones, de :data:`ESTRATEGIAS`.
                La primaria es ``nearest``; las otras tres existen como controles
                declarados de antemano, no como alternativas a elegir después de
                ver los resultados.
            random_state: Semilla de la estrategia ``random``. Se ignora en las
                demás, que son deterministas.

        Raises:
            ValueError: Si el número de configuraciones no es positivo, o si el
                escalado, la distancia o la estrategia no están soportados.
        """
        if n_configurations < 1:
            raise ValueError(
                f"Hay que pedir al menos una configuración, no {n_configurations}."
            )
        if scaling not in ESCALADOS:
            raise ValueError(f"Escalado {scaling!r} no soportado. Use uno de {ESCALADOS}.")
        if metric not in METRICAS:
            raise ValueError(f"Distancia {metric!r} no soportada. Use una de {METRICAS}.")
        if strategy not in ESTRATEGIAS:
            raise ValueError(
                f"Estrategia {strategy!r} no soportada. Use una de {ESTRATEGIAS}."
            )
        self.n_configurations = n_configurations
        self.scaling = scaling
        self.metric = metric
        self.strategy = strategy
        self.random_state = random_state
        self.feature_names_: list[str] | None = None
        self.center_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.lower_: np.ndarray | None = None
        self.upper_: np.ndarray | None = None
        self.neighbours_: list[dict] | None = None
        self.used_features_: list[str] | None = None
        self.clipped_features_: list[str] | None = None
        self._base: MetaKnowledgeBase | None = None

    def fit(self, base: MetaKnowledgeBase) -> MetaRecommender:
        """Ajusta el recomendador con los datasets de un pliegue.

        Todo lo que aprende —qué meta-características sobreviven, y el centro y la
        escala de cada una— sale **exclusivamente** de las entradas recibidas. En
        el *leave-one-dataset-out* hay que pasarle ``base.without(d)``, nunca la
        base completa.

        Args:
            base: Base de meta-conocimiento del pliegue. No se modifica.

        Returns:
            El propio recomendador, para encadenar.

        Raises:
            ValueError: Si la base está vacía, si no deja ninguna
                meta-característica superviviente, o si sus entradas provienen de
                una versión del espacio de búsqueda que no es la vigente.
        """
        if not len(base):
            raise ValueError(
                "No se puede ajustar el recomendador con una base vacía: no habría "
                "vecinos entre los que buscar."
            )
        validate_space_version(base.entries[0].space_version)

        nombres = base.surviving_features()
        if not nombres:
            raise ValueError(
                "Ninguna meta-característica es finita en todas las entradas de la "
                "base, así que no queda espacio en el que medir distancias."
            )
        matriz, nombres = base.feature_matrix(nombres)
        centro, escala = self._centro_y_escala(matriz)

        # Una característica cuya escala es cero no varía entre los datasets del
        # pliegue: no aporta nada para ordenar vecinos, y dejarla sin escalar
        # haría que su magnitud en bruto dominara la distancia. Se descarta, que
        # es la misma operación conservadora que el pre-registro elige para las
        # meta-características no finitas.
        utiles = escala > 0
        descartadas = [n for n, util in zip(nombres, utiles) if not util]
        if descartadas:
            logger.info(
                "Se descartan %d meta-características sin variación en el pliegue: %s",
                len(descartadas),
                ", ".join(descartadas),
            )
        if not utiles.any():
            raise ValueError(
                "Ninguna meta-característica varía entre los datasets de la base: "
                "todas las distancias serían cero y el orden de vecinos sería "
                "arbitrario."
            )

        self.feature_names_ = [n for n, util in zip(nombres, utiles) if util]
        self.center_ = centro[utiles]
        self.scale_ = escala[utiles]
        self.lower_ = matriz.min(axis=0)[utiles]
        self.upper_ = matriz.max(axis=0)[utiles]
        self._base = base
        self.neighbours_ = None
        self.used_features_ = None
        self.clipped_features_ = None
        return self

    def recommend(self, meta_features: Mapping[str, float]) -> list[dict[str, int]]:
        """Devuelve las configuraciones a inyectar como arranque en caliente.

        Reúne :attr:`n_configurations` configuraciones **distintas** entre las
        ganadoras de la base, eligiendo de qué datasets tomarlas según
        :attr:`strategy`. Con la estrategia primaria camina desde el vecino más
        cercano hacia afuera, saltando las repetidas. Si la base no contiene
        tantas configuraciones distintas, devuelve las que haya y lo registra.

        Cada configuración devuelta se resuelve contra el espacio vigente antes de
        salir. No es una formalidad: el spike A midió que un índice fuera de rango
        pasado a ``enqueue_trial`` **aborta el estudio completo**, y que una clave
        desconocida se ignora en silencio.

        Args:
            meta_features: Vector del dataset consultado. Puede traer valores no
                finitos y meta-características que la base no tenga: se usa la
                intersección con las del ajuste.

        Returns:
            Lista de configuraciones, de más a menos parecida, sin repetidas.

        Raises:
            NotFittedError: Si no se llamó antes a :meth:`fit`.
            ValueError: Si el vector consultado no comparte ninguna
                meta-característica finita con las del ajuste.
        """
        base, ajuste = self._ajustado()

        columnas, indices = self._columnas_comunes(ajuste["names"], meta_features)
        matriz, _ = base.feature_matrix(columnas)
        centro = ajuste["center"][indices]
        escala = ajuste["scale"][indices]

        bruto = np.array([meta_features[n] for n in columnas], float)
        recortado = np.clip(bruto, ajuste["lower"][indices], ajuste["upper"][indices])
        consulta = (recortado - centro) / escala
        vecinos = (matriz - centro) / escala
        distancias = self._distancias(vecinos, consulta)

        self.used_features_ = columnas
        self.clipped_features_ = [
            nombre
            for nombre, antes, despues in zip(columnas, bruto, recortado)
            if antes != despues
        ]
        if self.clipped_features_:
            logger.info(
                "El dataset consultado cae fuera del rango del pliegue en %d de %d "
                "meta-características; se recortan al rango observado.",
                len(self.clipped_features_),
                len(columnas),
            )
        return self._caminar(base, distancias)

    # ------------------------------------------------------------------ #
    # Interno
    # ------------------------------------------------------------------ #
    def _ajustado(self) -> tuple[MetaKnowledgeBase, dict]:
        """Devuelve el estado del ajuste, o falla si no lo hay.

        Returns:
            Tupla ``(base, ajuste)``, donde el ajuste lleva los nombres, el
            centro, la escala y el rango observado de cada característica.

        Raises:
            NotFittedError: Si no se llamó antes a :meth:`fit`.
        """
        if (
            self._base is None
            or self.feature_names_ is None
            or self.center_ is None
            or self.scale_ is None
            or self.lower_ is None
            or self.upper_ is None
        ):
            raise NotFittedError(
                "El recomendador no está ajustado. Llame a fit(base) con los "
                "datasets del pliegue antes de pedir recomendaciones."
            )
        return self._base, {
            "names": self.feature_names_,
            "center": self.center_,
            "scale": self.scale_,
            "lower": self.lower_,
            "upper": self.upper_,
        }

    def _centro_y_escala(self, matriz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Calcula el centro y la escala de cada meta-característica.

        Args:
            matriz: Una fila por dataset del pliegue.

        Returns:
            Tupla ``(centro, escala)``, una entrada por columna.
        """
        if self.scaling == "robust":
            centro = np.median(matriz, axis=0)
            cuartiles = np.percentile(matriz, [25, 75], axis=0)
            return centro, cuartiles[1] - cuartiles[0]
        return matriz.mean(axis=0), matriz.std(axis=0)

    @staticmethod
    def _columnas_comunes(
        nombres: list[str], meta_features: Mapping[str, float]
    ) -> tuple[list[str], np.ndarray]:
        """Meta-características del ajuste que la consulta trae y son finitas.

        La regla del pre-registro descarta las no finitas en lugar de imputarlas,
        y la intersección la extiende a la consulta: una columna que el pliegue
        conserva puede no ser finita en el dataset excluido, y meterla en la
        distancia propagaría un ``NaN`` a todos los vecinos por igual.

        Args:
            nombres: Meta-características con las que se ajustó el recomendador.
            meta_features: Vector consultado.

        Returns:
            Tupla ``(nombres, posiciones)`` con las columnas utilizables y su
            posición dentro de :attr:`feature_names_`.

        Raises:
            ValueError: Si no queda ninguna columna utilizable.
        """
        pares = [
            (posicion, nombre)
            for posicion, nombre in enumerate(nombres)
            if nombre in meta_features and np.isfinite(meta_features[nombre])
        ]
        if not pares:
            raise ValueError(
                "El vector consultado no comparte ninguna meta-característica finita "
                "con las del ajuste, así que no hay espacio en el que medir su "
                "distancia a los vecinos."
            )
        if len(pares) < len(nombres):
            logger.info(
                "La consulta aporta %d de las %d meta-características del ajuste; "
                "el resto no es finito o no viene en el vector.",
                len(pares),
                len(nombres),
            )
        return [nombre for _, nombre in pares], np.array([p for p, _ in pares])

    def _distancias(self, vecinos: np.ndarray, consulta: np.ndarray) -> np.ndarray:
        """Distancia de la consulta a cada vecino.

        Args:
            vecinos: Matriz escalada, una fila por dataset.
            consulta: Vector escalado del dataset consultado.

        Returns:
            Una distancia por vecino.
        """
        diferencias = vecinos - consulta
        if self.metric == "manhattan":
            return np.abs(diferencias).sum(axis=1)
        return np.sqrt((diferencias**2).sum(axis=1))

    def _caminar(
        self, base: MetaKnowledgeBase, distancias: np.ndarray
    ) -> list[dict[str, int]]:
        """Reúne configuraciones distintas según la estrategia configurada.

        Todas las estrategias devuelven ganadoras de datasets reales de la base y
        saltan las repetidas, de modo que lo único que las distingue es **de qué
        datasets** toman las configuraciones. Eso es lo que permite compararlas
        sin que la comparación se contamine con la calidad de lo inyectado.

        Args:
            base: Base del pliegue, en su orden de inserción.
            distancias: Distancia de la consulta a cada entrada de la base.

        Returns:
            Las configuraciones elegidas, en el orden en que se inyectarán.
        """
        candidatos = self._candidatos(base, distancias)
        if self.strategy == "diverse":
            elegidos = self._diversificar(base, candidatos)
        else:
            elegidos = candidatos[: self.n_configurations]

        configuraciones: list[dict[str, int]] = []
        vecinos: list[dict] = []
        for posicion in elegidos:
            entrada = base.entries[posicion]
            self._validar(entrada.configuration, entrada.dataset_name)
            configuraciones.append(entrada.configuration)
            vecinos.append(
                {
                    "dataset_id": entrada.dataset_id,
                    "dataset_name": entrada.dataset_name,
                    "distance": float(distancias[posicion]),
                    "configuration": entrada.configuration,
                }
            )

        if len(configuraciones) < self.n_configurations:
            logger.warning(
                "Se pidieron %d configuraciones distintas y la base solo contiene "
                "%d: se devuelven las que hay.",
                self.n_configurations,
                len(configuraciones),
            )
        self.neighbours_ = vecinos
        return configuraciones

    def _candidatos(self, base: MetaKnowledgeBase, distancias: np.ndarray) -> list[int]:
        """Entradas con configuración distinta, en el orden que fija la estrategia.

        Args:
            base: Base del pliegue.
            distancias: Distancia de la consulta a cada entrada.

        Returns:
            Posiciones en la base, una por configuración distinta.
        """
        if self.strategy == "hybrid":
            frecuentes = self._candidatos_de("frequent", base, distancias)
            cercanos = self._candidatos_de("nearest", base, distancias)
            return self._sin_repetir(
                base,
                [
                    *frecuentes[:N_HIBRIDO_FRECUENTES],
                    *cercanos,
                    *frecuentes[N_HIBRIDO_FRECUENTES:],
                ],
            )
        return self._candidatos_de(self.strategy, base, distancias)

    def _candidatos_de(
        self, estrategia: str, base: MetaKnowledgeBase, distancias: np.ndarray
    ) -> list[int]:
        """Ordena las entradas según una estrategia concreta y quita las repetidas.

        Recibe la estrategia por parámetro y no la lee de :attr:`strategy` porque
        ``hybrid`` necesita las listas de otras dos para combinarlas.

        El desempate se resuelve siempre por la posición en la base, que es fija:
        sin un criterio determinista, dos ejecuciones con la misma semilla podrían
        inyectar configuraciones distintas y se rompería el criterio de
        reproducibilidad del anteproyecto.

        Args:
            estrategia: Estrategia con la que ordenar.
            base: Base del pliegue.
            distancias: Distancia de la consulta a cada entrada.

        Returns:
            Posiciones en la base, una por configuración distinta.
        """
        posiciones = range(len(base))
        if estrategia == "random":
            generador = np.random.default_rng(self.random_state)
            orden = [int(i) for i in generador.permutation(len(base))]
        elif estrategia == "frequent":
            conteo: dict[tuple, int] = {}
            for entrada in base:
                clave = self._clave(entrada.configuration)
                conteo[clave] = conteo.get(clave, 0) + 1
            orden = sorted(
                posiciones,
                key=lambda i: (
                    -conteo[self._clave(base.entries[i].configuration)],
                    self._clave(base.entries[i].configuration),
                    i,
                ),
            )
        else:
            orden = sorted(posiciones, key=lambda i: (float(distancias[i]), i))
        return self._sin_repetir(base, orden)

    @staticmethod
    def _sin_repetir(base: MetaKnowledgeBase, orden: list[int]) -> list[int]:
        """Quita de un orden las entradas cuya configuración ya apareció.

        Args:
            base: Base del pliegue.
            orden: Posiciones en el orden que fija la estrategia.

        Returns:
            Las posiciones que aportan una configuración nueva, en ese mismo orden.
        """
        vistas: list[dict[str, int]] = []
        candidatos: list[int] = []
        for posicion in orden:
            configuracion = base.entries[posicion].configuration
            if configuracion in vistas:
                continue
            vistas.append(configuracion)
            candidatos.append(posicion)
        return candidatos

    def _diversificar(self, base: MetaKnowledgeBase, candidatos: list[int]) -> list[int]:
        """Elige, entre las candidatas más cercanas, el subconjunto más separado.

        Ataca el modo de fallo que midió el spike A: cinco configuraciones
        parecidas entre sí y todas equivocadas forman, para el muestreador, una
        región aparentemente buena, y la búsqueda se concentra ahí en lugar de
        explorar. Un recomendador aleatorio no caía en la trampa precisamente
        porque sus puntos estaban dispersos.

        El procedimiento es voraz: se parte del vecino más cercano y se añade
        cada vez la candidata cuya distancia de Hamming mínima a las ya elegidas
        sea mayor, resolviendo empates por cercanía. Se consideran el **doble**
        de candidatas de las que se van a elegir, para que la diversidad no se
        compre yendo a buscar datasets arbitrariamente lejanos.

        Args:
            base: Base del pliegue.
            candidatos: Posiciones con configuración distinta, por cercanía.

        Returns:
            Las posiciones elegidas, empezando por la más cercana.
        """
        # La reserva nunca queda vacía: :meth:`fit` rechaza una base sin entradas,
        # de modo que siempre hay al menos una configuración candidata.
        reserva = candidatos[: FACTOR_DIVERSIDAD * self.n_configurations]
        elegidos = [reserva[0]]
        while len(elegidos) < min(self.n_configurations, len(reserva)):
            restantes = [p for p in reserva if p not in elegidos]
            elegidos.append(
                max(
                    restantes,
                    key=lambda p: (
                        min(
                            self._hamming(
                                base.entries[p].configuration,
                                base.entries[q].configuration,
                            )
                            for q in elegidos
                        ),
                        -reserva.index(p),
                    ),
                )
            )
        return elegidos

    @staticmethod
    def _clave(configuracion: dict[str, int]) -> tuple:
        """Clave ordenable y estable de una configuración."""
        return tuple(sorted(configuracion.items()))

    @staticmethod
    def _hamming(una: dict[str, int], otra: dict[str, int]) -> int:
        """Número de pasos en los que dos configuraciones difieren."""
        return sum(1 for paso, indice in una.items() if otra.get(paso) != indice)

    @staticmethod
    def _validar(configuracion: dict[str, int], dataset_name: str) -> None:
        """Comprueba que una configuración se resuelva contra el espacio vigente.

        Args:
            configuracion: Índice elegido por cada paso.
            dataset_name: Dataset del que proviene, para el mensaje de error.

        Raises:
            ValueError: Si la configuración no pertenece al espacio vigente.
        """
        try:
            pipeline_from_configuration(configuracion)
        except ValueError as exc:
            raise ValueError(
                f"La configuración ganadora de {dataset_name!r} no se puede resolver "
                f"contra el espacio de búsqueda vigente, así que no se inyecta: "
                f"pasársela a enqueue_trial abortaría la búsqueda entera. {exc}"
            ) from exc
