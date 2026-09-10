# Guía técnica de AutoPipe

- **Estado:** cubre el núcleo generalizado, las meta-características, la base de
  meta-conocimiento, el recomendador, el arranque en caliente y el arnés del
  experimento
- **Entregable:** RE7
- **Actualizada:** 2026-09-12

Documentación técnica del framework. Para el uso básico está el `README.md`; esta guía
explica **cómo funciona por dentro** y **qué garantiza**, que es lo que hace falta para
extenderlo o para confiar en sus resultados.

## Qué cambió respecto de AutoPipeAD

Quien conocía la librería base encontrará seis diferencias:

1. **El framework ya no asume una tarea.** Un objeto `Task` agrupa las métricas, la
   dirección de optimización, la normalización de predicciones y la obtención de puntajes.
   Se elige en el constructor de los motores.
2. **El contrato de transformación tiene ajuste.** `BaseTransform` gana `fit` y `transform`
   por lote, además del `apply` por muestra que ya existía.
3. **El pipeline se ajusta solo con el subconjunto de entrenamiento.** Antes se
   transformaba el dataset completo y después se partía.
4. **Hay soporte de datos tabulares**: un cargador y cuatro transformaciones con estado
   (imputación, escalado, codificación y selección de características).
5. **El espacio de búsqueda tabular es único y versionado.** No se arma a mano: se pide con
   `build_tabular_search_space()`, y lleva un identificador de versión.
6. **La búsqueda puede arrancar en caliente.** `OptunaSearchEngine` acepta
   configuraciones inyectadas, que un recomendador por vecinos más cercanos obtiene de
   la base de meta-conocimiento. Es la capa de meta-aprendizaje que da nombre al TFG.

Nada de lo anterior cambió una interfaz pública: el flujo de imágenes de la librería base
funciona sin modificar una línea, y sus 134 pruebas de regresión siguen en verde.

Eso último no es una afirmación de confianza. `scripts/check_image_regression.py` corre la
misma búsqueda sobre MVTec AD con el código anterior a la generalización y con el actual, y
compara el pipeline ganador, todas las métricas y la tabla completa de objetivos. El
resultado sobre la categoría `bottle` (83 imágenes reales) es idéntico: mismo ganador, las
cinco métricas iguales hasta el décimo decimal, cero de los 36 candidatos con objetivo
distinto.

## Las dos tareas

| | `anomaly_detection` | `classification` |
|---|---|---|
| Métricas | `auc_roc`, `f1`, `precision`, `recall`, `false_positive_rate` | `macro_f1`, `accuracy`, `balanced_accuracy`, `macro_precision`, `macro_recall` |
| Métrica primaria | `auc_roc` | `macro_f1` |
| Se minimiza | `false_positive_rate` | ninguna |
| Predicciones | se normalizan a `{0, 1}`, admitiendo la convención `-1`/`+1` de scikit-learn | se usan tal cual, sin reinterpretar |
| Puntajes continuos | `predict_proba[:, 1]`, `score_samples` o `decision_function` | no se usan |

Se selecciona en el constructor, nunca en `run(...)`:

```python
GridSearchEngine(espacio, task="classification")
OptunaSearchEngine(espacio, task="classification")
ParallelRunner(task="classification")
```

Por defecto es `anomaly_detection`, de modo que el código existente conserva su
comportamiento. La métrica objetivo de `run(...)` es opcional: si no se indica, se usa la
primaria de la tarea.

**Las cuatro responsabilidades viajan juntas a propósito.** Un registro que separara las
métricas de la normalización de predicciones permitiría que una tarea nueva declarara sus
métricas y heredara en silencio la normalización binaria. Ese fallo existía y está
documentado en `decisiones/AUDITORIA-acoplamiento.md`: la normalización binaria convierte
`[0, 1, 2, 3, 4]` en `[0, 1, 1, 1, 1]` sin levantar ninguna excepción.

## El contrato de transformación

Toda transformación implementa `apply` (una muestra) y `get_params`. Sobre eso hay dos
niveles:

**Sin estado.** Basta implementar `apply`; se heredan un `fit` que no aprende nada y un
`transform` que aplica `apply` muestra por muestra. Es el caso de las siete
transformaciones de imagen.

```python
class Duplicar(BaseTransform):
    def apply(self, image):
        return image * 2

    def get_params(self):
        return {}
```

**Con estado.** Se hereda de `StatefulTransform` y se implementan `_fit` y `_transform`.
Los métodos públicos `fit` y `transform` los aporta la clase base, que además registra el
ajuste y rechaza transformar antes de ajustar con `NotFittedError`. Así una subclase no
puede olvidarse de esa validación.

```python
class CentrarColumnas(StatefulTransform):
    def _fit(self, X, y=None):
        self.media_ = X.mean(axis=0)

    def _transform(self, X):
        return X - self.media_

    def get_params(self):
        return {}
```

Para transformaciones tabulares conviene heredar de `TabularTransform`, que exige una
tabla, recuerda las columnas del ajuste, valida que estén presentes al transformar y
resuelve el contrato por muestra.

### Los parámetros no pueden depender del dataset

Ninguna transformación debe recibir nombres ni posiciones de columnas, ni cantidades
absolutas ligadas al tamaño del dataset. La razón no es estética: el recomendador de
arranque en caliente inyecta en un dataset configuraciones halladas en otro, y un
parámetro como `columns=[0, 2, 5]` o `k=10` no significa lo mismo en dos datasets
distintos. Las transformaciones tabulares detectan por sí mismas sobre qué columnas actuar,
y las cantidades relativas se expresan como fracción.

Hay una prueba que lo vigila
(`test_tabular_transforms.py::test_parameters_do_not_mention_columns`): ningún parámetro
puede ser una colección ni coincidir con el nombre de una columna, y **todo parámetro
numérico debe ser una fracción en `(0, 1]`**. Un número fuera de ese intervalo delataría
una cantidad absoluta, que es justo lo que no se puede transferir entre datasets.

## El protocolo de evaluación

`evaluate_pipeline` es el corazón del framework y garantiza cuatro cosas:

1. **El pipeline se ajusta solo con `train_idx`.** Ajustarlo con el dataset completo
   filtraría las estadísticas de validación hacia el entrenamiento e invalidaría cualquier
   conclusión experimental.
2. **El pipeline se clona antes de ajustarse.** Los motores construyen cada candidato
   tomando instancias del espacio de búsqueda, de modo que la misma transformación aparece
   en muchos candidatos. Sin el clonado, una transformación con estado contaminaría a los
   candidatos siguientes, y bajo el backend `loky` habría además una condición de carrera.
3. **El modelo base se clona.** Cada candidato se evalúa sobre un modelo sin entrenar e
   idéntico.
4. **Todos los candidatos comparten la misma partición**, para que la comparación sea justa.

Cada garantía tiene su prueba, y las dos primeras están escritas para **fallar** si alguien
revierte el cambio.

### Por qué la primera garantía se volvió más severa

Con un paso **supervisado** en el pipeline, la fuga deja de ser sutil. Un escalador que
viera la validación filtraría una media; una selección que la viera elegiría **otras
columnas**, y el modelo entrenaría con una matriz distinta.

`tests/test_tabular_protocol.py` cubre ese caso con once pruebas: la selección sola, la
selección como último paso del orden canónico, el paso supervisado en cada posición —la
última recorre un camino distinto, porque `Pipeline.fit` ajusta el último paso sin
transformarlo—, la no herencia de ajuste entre candidatos y el cruce por los procesos de
`loky`.

El fixture es una tabla donde la columna informativa **cambia entre entrenamiento y
validación**, de modo que la elección delata con qué se ajustó: solo con entrenamiento elige
`a`, con el dataset completo elige `b`. Una de las pruebas fija esa discriminación, para que
las demás no puedan pasar por vacío.

Se verificó saboteando `evaluate_pipeline` a propósito para que ajustara con el dataset
completo: **siete de las once pruebas caen**. Una prueba anti-fuga que nunca se ha visto
fallar no prueba nada.

## Flujo tabular

El ejemplo ejecutable está en `examples/tabular_quickstart.py` (13 instrucciones) y el
recorrido comentado en `examples/tabular_walkthrough.ipynb`. En resumen:

```python
features, labels = TabularDataLoader(tabla).load()

motor = GridSearchEngine(
    build_tabular_search_space(), random_state=42, task="classification"
)
mejor, metricas, resultados = motor.run(features, labels, KNeighborsClassifier())
```

**El cargador devuelve una tabla, no una matriz numérica.** Conservar los tipos de columna
es lo que permite a las transformaciones distinguir solas las categóricas de las numéricas,
sin recibirlo como parámetro.

**El espacio no se arma a mano.** Se pide el canónico, que es el mismo para todos los
datasets; la siguiente sección explica por qué.

## El espacio de búsqueda canónico y versionado

La composición congelada, su tabla de índices y la justificación completa están en
`experimentos/ESPACIO-tabular-v1.md`. Lo esencial:

```python
from autopipe import (
    TABULAR_SPACE_VERSION,       # 'tabular-v1'
    TABULAR_SPACE_STEPS,         # ('imputation', 'scaling', 'encoding', 'selection')
    build_tabular_search_space,  # instancias nuevas en cada invocación
    validate_space_version,      # rechaza configuraciones de otra versión
    describe_tabular_search_space,
)
```

**Único**, porque el recomendador transfiere configuraciones halladas en unos datasets hacia
otro que nunca vio: si cada dataset tuviera su espacio, no habría nada que transferir.

**Versionado**, porque lo que Optuna registra por ensayo son *índices enteros* sobre la
lista de opciones de cada paso, más un índice extra que significa «omitir este paso». Un
índice no significa nada por sí mismo: depende del orden de la lista. Si el espacio cambiara
de claves o de orden, una configuración inyectada como arranque en caliente designaría
transformaciones distintas de las previstas **sin levantar ningún error**: la búsqueda
correría, daría un número, y ese número no mediría lo que se cree.
`validate_space_version` convierte ese fallo mudo en un rechazo explícito, y
`test_search_space.py::test_the_index_of_every_option_is_frozen` congela el índice de cada
opción para que un reordenamiento rompa una prueba en lugar de reinterpretar en silencio
todo lo ya guardado.

**Instancias nuevas en cada invocación.** Las transformaciones tabulares tienen estado, así
que compartir instancias entre dos búsquedas arrastraría lo aprendido de una hacia la otra.

### El orden está congelado

```
imputation → scaling → encoding → selection        3 × 4 × 3 × 4 = 144 pipelines
```

**El escalado va antes de la codificación**, y eso es deliberado: así actúa solo sobre las
columnas numéricas originales y deja las indicadoras *one-hot* en `0`/`1`. Si fuera después,
estandarizar la indicadora de una categoría rara la llevaría a valores cercanos a ±10, y con
el modelo base congelado —vecinos más cercanos con distancia euclídea— esa sola columna
dominaría la distancia: el resultado mediría la rareza de una categoría y no la calidad del
preprocesamiento. Es también la convención de `ColumnTransformer` en scikit-learn.

**La selección va al final**, cuando la tabla ya es enteramente numérica, de modo que la
fracción se aplica sobre el ancho ya codificado.

## Selección de características

`SelectFeatures(fraction, criterion)` conserva una **fracción** de las columnas numéricas.
Dos criterios: `f_classif` (ANOVA de una vía, supervisado, el único que entra al espacio
canónico) y `variance` (no supervisado, disponible para quien use la librería).

Las reglas de orden son explícitas porque el criterio de reproducibilidad exige que la misma
semilla produzca el mismo ganador, y nada puede quedar librado a la biblioteca:

| Situación | Regla |
|---|---|
| Empate de puntajes | Gana la columna de menor posición original |
| Puntaje `NaN` | El peor posible: el criterio **no pudo** evaluar la columna (constante bajo `f_classif`, sin valores finitos bajo `variance`) |
| Puntaje `+inf` | El **mejor** posible: la columna separa las clases a la perfección |
| Cuántas conserva | `ceil(fraction × n_numéricas)`, con un mínimo de una |
| Columnas no numéricas | Pasan intactas y nunca se descartan, igual que en `ScaleNumeric` |

La distinción entre `NaN` e `inf` no es un detalle: una primera versión trataba todo puntaje
no finito como el peor, y con eso descartaba justamente la mejor característica del dataset.
Lo detectó la prueba anti-fuga.

### Candidatos que fallan a propósito

El motor prueba también **omitir** cada paso. Sobre datos con columnas categóricas, los
candidatos que omiten la codificación no se pueden evaluar: la matriz de características no
puede ser numérica. Eso no interrumpe la búsqueda —`evaluate_pipeline_safe` registra una
advertencia y devuelve `NaN`, y un `NaN` nunca gana— pero explica por qué al ejecutar el
ejemplo aparecen mensajes como:

> El pipeline `Pipeline([ScaleNumeric(method='standard')])` no se pudo evaluar: la tabla
> transformada conserva columnas no numéricas; el pipeline necesita un paso de codificación
> antes de entrenar el modelo.

Son candidatos descartados, no errores del programa.

**Cuántos son.** La fracción está medida y depende del dataset:

| Dataset | Candidatos `NaN` |
|---|---|
| Solo numérico, sin faltantes | 0 de 144 |
| Solo numérico, con faltantes | 48 de 144 (33 %) |
| Con categóricas, sin faltantes | 48 de 144 (33 %) |
| Con categóricas **y** faltantes | 80 de 144 (56 %) |

Eso tiene consecuencias sobre el protocolo experimental, no solo sobre la legibilidad de la
salida: está discutido en `experimentos/ESPACIO-tabular-v1.md` y registrado como decisión
abierta en `Agenda.tex`.

**Si *todos* los candidatos fallan**, el motor de malla devuelve el primero con métricas
`NaN` y sin avisar de que no hubo comparación posible. Es deuda conocida; hay una prueba que
fija esa conducta para que cambiarla sea deliberado.

## Meta-características

`MetaFeatureExtractor` describe un dataset como un vector de 107 números
comparables con los de otro dataset. Es la entrada del recomendador.

```python
from autopipe.meta_features import MetaFeatureExtractor, columnas_no_finitas

extractor = MetaFeatureExtractor(random_state=42, cache_dir="datasets/.meta_cache")
vector = extractor.extract(features, labels, dataset_id="1590")
```

Cuatro decisiones lo gobiernan, todas medidas en el spike B y registradas en el
pre-registro v0.2:

| Decisión | Por qué |
|---|---|
| `general`, `statistical` e `info-theory` sobre el dataset completo; `complexity` sobre 500 instancias estratificadas | El costo de `complexity` crece de forma cuadrática (exponente 1,91): en 50 000 instancias serían 21,6 min por dataset |
| Submuestra de **tamaño fijo**, no fracción fija | Las medidas de complejidad cambian con el tamaño de la muestra; dos datasets solo son comparables si se extrajeron con el mismo. 500 es el único tamaño que todo dataset elegible alcanza |
| Imputación antes de extraer | Sin ella `complexity` falla con un `Exception` de mensaje vacío |
| Redondeo a nueve decimales | Dos extracciones difieren en `hubs.mean` y `hubs.sd` en torno a 3e-15 por orden de sumación |

**Las no finitas se reportan, no se descartan.** Qué posiciones lo son depende del
dataset, así que decidir cuáles se van es una propiedad del corpus completo.

## La base de meta-conocimiento

`MetaKnowledgeBase` es lo que el recomendador consulta: una entrada por dataset
del corpus con su vector, su pipeline ganador y la versión del espacio.

```python
from autopipe.meta_knowledge import MetaKnowledgeBase

base = MetaKnowledgeBase.load("docs/experimentos/BASE-meta-conocimiento.json")
pliegue = base.without("1590")               # leave-one-dataset-out
matriz, columnas = pliegue.feature_matrix()  # solo las supervivientes
```

**Verifica dos invariantes al construirse**, y ninguno es una formalidad:

1. **Todas las entradas comparten el conjunto de nombres.** Sin eso, el
   recomendador compararía vectores desalineados y nada fallaría.
2. **Todas provienen de la misma versión del espacio.** Los índices de una
   configuración solo significan algo respecto de una versión concreta.

`surviving_features()` aplica la regla del pre-registro: se descarta toda
meta-característica no finita en algún dataset **del conjunto en cuestión**. Se
recalcula por pliegue, porque el descarte depende de qué datasets hay.

**Se guarda el vector completo, no solo las supervivientes.** La regla depende del
corpus, de modo que guardar únicamente su resultado la volvería irreversible: no
se podría recalcular al cambiar el corpus ni auditar por qué se descartó algo.

La configuración ganadora se guarda de dos formas: como **índices por paso**, que
es lo que `optuna.enqueue_trial` consume, y como los parámetros legibles del
pipeline. `OptunaSearchEngine` expone los índices en `best_configuration_` tras
`run`.

### El ganador guardado es el canónico del empate

El óptimo casi nunca es único. Sobre un dataset numérico y sin faltantes la
imputación y la codificación son la identidad, de modo que varias combinaciones de
esos dos pasos son **el mismo pipeline** y empatan exactamente: la mediana medida
sobre el corpus es de nueve configuraciones empatadas.

Cuál de ellas guardaría una búsqueda sin más criterio es cuál visitó primero el
muestreador, que es una propiedad de la corrida y no del dataset. Transferirla como
arranque en caliente transferiría esa arbitrariedad, y no de forma inocua: un
ganador que **omite la codificación** es inevaluable en un dataset receptor con
columnas categóricas, y ese ensayo inyectado se gastaría sin producir métrica.

```python
from autopipe.search_space import canonical_configuration, omitted_steps

canonical_configuration(motor.tied_configurations_)  # la que omite menos pasos
```

La regla es **la que omite menos pasos y, a igualdad, la de índices menores**. No
elige por rendimiento —todas empatan— sino por transferibilidad. El motor expone el
empate completo en `tied_configurations_`, y el arnés de construcción aplica la
regla; la medición de su efecto está en `experimentos/DIVERSIDAD-ganadores.md`.

## El recomendador de arranque en caliente

`MetaRecommender` convierte la base en las configuraciones que se inyectan. Es
**vecinos más cercanos, no un modelo entrenado**: el anteproyecto lo fija como la
línea base del alcance mínimo.

```python
from autopipe.meta_recommender import MetaRecommender

recomendador = MetaRecommender().fit(base.without("1590"))   # un pliegue del LODO
configuraciones = recomendador.recommend(vector_del_excluido)
recomendador.neighbours_       # de qué dataset salió cada una, con su distancia
```

**Todo lo que aprende sale de la base que recibe y de nada más.** En el
*leave-one-dataset-out* hay que pasarle `base.without(d)`, nunca la base completa:
ajustar la normalización sobre el corpus entero contaminaría cada pliegue e
inflaría la ventaja del arranque en caliente de un modo difícil de detectar después.
Es el análogo, en el nivel meta, de ajustar las transformaciones con estado usando
solo el subconjunto de entrenamiento.

Cuatro decisiones, todas fijadas **a priori** y no midiendo sobre el corpus, porque
esos 25 datasets *son* los pliegues del experimento:

| Decisión | Valor | Por qué |
|---|---|---|
| Distancia | L1 sobre características escaladas | Precedente publicado: auto-sklearn ordena por distancia L1 en el espacio de meta-características (Feurer et al., 2015, §3.1) |
| Escalado | Mediana y rango intercuartílico | Decisión propia; ese artículo no describe ninguno. Con dos docenas de datasets, un solo valor extremo desplaza media y desviación |
| Número de vecinos | No es un parámetro | Lo que se fija son las **cinco configuraciones distintas**; se camina hacia afuera saltando las repetidas |
| Consulta fuera de rango | Se recorta al rango del pliegue | Sin recorte, en 2 de 25 pliegues una sola meta-característica aportaba más de la mitad de la distancia |

**Por qué se saltan las repetidas.** Inyectar dos veces la misma configuración
gasta dos de los treinta ensayos para devolver el mismo valor. **No** se exigen
distintas por diversidad: el spike A atribuyó el fallo del arranque en caliente a
recomendaciones parecidas entre sí, y el experimento 2 lo refutó midiéndolo —la
estrategia por similitud inyectaba los conjuntos más diversos y fue la peor—. Véase
`docs/experimentos/MECANISMO-E2.md`.

**Por qué se recorta la consulta.** El escalado robusto controla los valores
extremos *dentro* del conjunto de ajuste, pero el dataset excluido puede caer muy
fuera. El recorte no necesita ninguna constante arbitraria: el recomendador no puede
medir *cuánto* fuera del corpus cae un dataset, solo que cae fuera. `clipped_features_`
dice en qué columnas ocurrió.

### Las estrategias

`strategy=` decide **de qué datasets** se toman las configuraciones. Todas
devuelven ganadoras de datasets reales de la base y saltan las repetidas, de modo
que una comparación entre ellas no se contamina con la calidad intrínseca de lo
inyectado: es lo único que las hace comparables entre sí.

| Estrategia | De dónde toma | Para qué existe |
|---|---|---|
| `nearest` | Vecinos más cercanos | Primaria; la línea base del anteproyecto |
| `diverse` | El subconjunto más separado entre las cercanas | Diseñada contra un modo de fallo que el experimento 2 refutó; no se evaluó |
| `frequent` | Las que más ganan, **ignorando** meta-características | Aísla el meta-aprendizaje |
| `random` | Ganadoras de datasets al azar | Ablación: aísla la similitud |
| `hybrid` | 2 de `frequent` y el resto de `nearest` | Condición del experimento 2; no primaria |

**`frequent` es el control que más importa.** Si la primaria no le gana, las
meta-características no están aportando nada y bastaría recomendar siempre lo
mismo. Sin ese control, un resultado positivo tampoco sería atribuible: no se
podría distinguir «la similitud ayuda» de «inyectar lo que suele ganar ayuda».

**`diverse`** parte del vecino más cercano y añade cada vez la candidata cuya
distancia de Hamming mínima a las ya elegidas sea mayor, considerando el doble de
candidatas de las que va a elegir (`FACTOR_DIVERSIDAD`). El tope evita comprar
diversidad yendo a buscar datasets arbitrariamente lejanos.

**`random`** toma ganadoras de datasets al azar y **no** configuraciones al azar
del espacio. La diferencia es la pregunta que responde: así aísla si lo que aporta
es la **similitud**, y no el mero hecho de arrancar en caliente con
configuraciones plausibles. Necesita `random_state`.

### Dónde se puede evaluar el recomendador, y dónde no

El corpus son 25 de los 52 candidatos cribados, y esos 25 **son los pliegues** del
*leave-one-dataset-out*. Medir sobre ellos es medir sobre el conjunto de prueba.

`scripts/evaluate_recommender.py` evalúa sobre los **27 que quedaron fuera**, que
no entran nunca al experimento. Hay una prueba escrita para fallar si alguno del
corpus se cuela en ese conjunto.

Los dos grupos de retenidos sesgan en direcciones opuestas y se reportan por
separado: los excluidos por orden tienen un IQR por debajo del mínimo del corpus
y sesgan hacia «da igual lo que se recomiende»; los excluidos por familia
comparten origen con un dataset de la base y sesgan al contrario.

## El arnés del experimento

`scripts/run_lodo.py` ejecuta la validación *leave-one-dataset-out* tal como la
congela el pre-registro v1.0, y **solo** lo que ese documento dice. Cada constante
del protocolo está declarada arriba del archivo con la sección que la fija.

```python
resultado = ejecutar_lodo(base, cargar_dataset, meseta=leer_meseta(DIVERSIDAD))
```

Por cada dataset: lo excluye, ajusta el recomendador con `base.without(d)`, y
corre dos condiciones con las mismas cinco semillas. De cada corrida guarda la
**curva completa** de treinta ensayos, no el número resumen, porque de ahí salen
la variable primaria, la medida a quince ensayos y las reglas de censura
alternativas sin volver a ejecutar nada.

### Las tres precisiones del conteo

Son las que el pre-registro v1.0 añadió al implementar el arnés, y las tres son
fáciles de implementar mal sin que nada falle:

1. **El conteo empieza en uno**, coherente con que la censura valga 31.
2. **Los ensayos inevaluables consumen su turno.** Saltar los `NaN` daría a ambas
   condiciones un número menor que el real, y más a la caliente que a la fría,
   porque el frío cae más en la región inevaluable.
3. **Se guarda la curva completa**, no solo el ensayo en que se cruza el umbral.

### Una propiedad que se comprueba, no se supone

**El arranque en frío no puede censurarse.** El objetivo es el 95 % del mejor
valor que él mismo alcanza, de modo que si ese máximo ocurre en el ensayo *t*, el
frío cruzó el umbral en algún ensayo menor o igual que *t*. No es improbable: es
imposible.

El arnés lo verifica en cada corrida y **aborta** si ocurre, porque solo podría
deberse a un error de implementación. Hay una prueba que fuerza el caso imposible
y comprueba que falle ruidosamente.

### Por qué el cargador de datasets entra por parámetro

Para que las pruebas recorran el protocolo entero sobre un **corpus sintético de
juguete**, sin red y sin producir un solo número del experimento. `tests/test_lodo_harness.py`
verifica sobre él las propiedades que el pre-registro declara: que el frío nunca
se censura, que cada pliegue excluye su propio dataset de los vecinos, que las
cinco semillas dan cinco observaciones y que un pliegue es reproducible.

El corpus de juguete son tres datasets, que es el mínimo: con dos, cada pliegue
dejaría un solo vecino y el recomendador rechaza —con razón— una base cuyas
meta-características no varían.

**Esas pruebas cuestan unos 50 segundos**, y el costo no es el tamaño de las
tablas sino la sobrecarga de Optuna por ensayo: son 3000 ensayos reales. Se
aceptan tal cual porque son las únicas que verifican que el motor, el
recomendador y el conteo funcionan juntos.

### El reporte comparativo no calcula estadísticos

`ComparativeReportGenerator` extiende `ReportGenerator` —hereda el directorio de
salida, la codificación de figuras y la exportación tolerante a PDF— y añade la
plantilla del LODO: veredicto, contraste primario, controles secundarios, robustez
de la censura y tabla por dataset con las dos covariables.

```python
from autopipe.reporter import ComparativeReportGenerator

analisis = json.loads(Path("docs/experimentos/LODO-resultados.json").read_text())["analisis"]
ComparativeReportGenerator("reportes").generate_comparison(analisis, protocolo=PROTOCOLO)
```

**Todos los números vienen del arnés.** El reporte los transcribe. Es deliberado:
el pre-registro fija el análisis y este vive junto al arnés que lo ejecuta; si el
reporte recomputara los contrastes existirían dos implementaciones del mismo
estadístico y nada garantizaría que coincidan.

La única excepción es el veredicto, y lo es por seguridad: `redactar_veredicto`
**deriva** de las cifras si se cumplen las dos condiciones del criterio —reducción
≥30 % y *p* de una cola <0,05— en vez de copiar la bandera `confirma`. Si ambas
discrepan, el texto lo denuncia en lugar de elegir una. Hay dos pruebas por
sabotaje que fallan si alguien vuelve a la versión que copiaba la bandera: la
ingenua llega a redactar «la reducción es del 1,0 % … con un valor p de 0,9000, por
debajo de 0,05».

Las condiciones de la tabla **se deducen** de las columnas, así que añadir una
estrategia al arnés la hace aparecer en el reporte sin tocarlo. Las covariables
(`*_inevaluable`, `*_a_15`) acompañan a cada condición y no se confunden con una.

## El experimento 2 y en qué se diferencia del primero

Los gobiernan documentos distintos: el experimento 1 el pre-registro v1.0
(`prereg-v1.0`) y el 2 el suyo propio (`Preregistro-E2.tex`, `prereg-v2.0`). **Los dos
arneses conviven** —`run_lodo.py` y `run_lodo_e2.py`— y ninguno sustituye al otro,
porque cada uno sigue siendo el que produjo sus resultados.

| Pieza | Experimento 1 | Experimento 2 |
|---|---|---|
| Vara | 95 % del propio mejor del frío | 99 % del óptimo exhaustivo por (dataset, semilla) |
| Presupuesto | 30 ensayos | 60 |
| Variable primaria | Ensayos hasta el objetivo | *Regret* normalizado |
| Corpus | 25 | 42 |
| Semillas | 5 | 10 |
| Condiciones | 4 | 5 (`+hybrid`) |
| Contrastes | 1 | 2, con Holm |

### La vara externa cambia una propiedad del protocolo

En el experimento 1 el arranque en frío **no podía censurarse**: el objetivo era una
fracción de su propio máximo. El arnés lo comprobaba activamente y abortaba si ocurría.

Con vara externa esa demostración ya no vale, y su desaparición es intencionada: una
condición que no puede fallar tampoco puede informar. Por eso `run_lodo_e2.py`
**no lleva esa guarda**, y hay una prueba escrita para fallar si alguien la
reintroduce (`test_the_cold_start_is_no_longer_guarded_against_censoring`).

### El invariante que sostiene la vara

`reference_optima.py` evalúa las 144 configuraciones con el **mismo `val_size` y el
mismo `random_state`** que la búsqueda, de modo que su máximo acota por arriba
cualquier valor alcanzable en esa pareja. Es comprobable, y se comprueba.

Ese invariante ya cazó un error real: `resumir` redondeaba el óptimo a seis
decimales, lo que lo dejaba **por debajo** del valor alcanzable —0,8333333 pasaba a
0,833333— y una búsqueda que encontraba el óptimo lo superaba, con regret negativo.
El óptimo y el objetivo se guardan sin redondear.

### El regret, en una línea

Media sobre el presupuesto de `(óptimo − mejor_hasta_ahora) / óptimo`, con regret
pleno mientras no haya ningún ensayo evaluable. Menor es mejor, vive en `[0, 1]`, usa
los 60 ensayos y **no produce empates**: sobre los 25 conjuntos del experimento 1 la
variable entera daba 7 valores distintos y 18 colisiones, que Wilcoxon descarta.

El presupuesto se pasa explícito y no por omisión: los valores por defecto se enlazan
al importar el módulo, y promediar sobre un presupuesto distinto del ejecutado daría
un número silenciosamente equivocado.

### `hybrid`

Las dos primeras de `frequent` seguidas de las de `nearest` por cercanía, sin
repetir, hasta reunir cinco. El reparto lo fija el pre-registro y **no se explora**:
probar varios y quedarse con el mejor sería ajustar sobre el resultado.

Viaja en una condición **aparte** en vez de reemplazar a `nearest`, y esa es la
decisión que permite atribuir: si `nearest` mejora respecto del experimento 1, la
causa es el instrumento, porque el método es literalmente el mismo.

### Orden de ejecución

```powershell
.\.venv\Scripts\python.exe scripts\build_corpus_e2.py        # 42 conjuntos
.\.venv\Scripts\python.exe scripts\build_meta_knowledge.py `
    --corpus docs\experimentos\CORPUS-E2.md `
    --base docs\experimentos\BASE-E2.json `
    --informe docs\experimentos\BASE-E2.md
.\.venv\Scripts\python.exe scripts\reference_optima.py        # ~62 min, cacheado
.\.venv\Scripts\python.exe scripts\run_lodo_e2.py            # ~2 h
.\.venv\Scripts\python.exe scripts\report_lodo_e2.py
```

Las rejillas de referencia se cachean por pareja, así que reejecutar el resumen es
instantáneo. Las entradas de la base se cachean por dataset, de modo que los 25 del
experimento 1 se reutilizan sin recalcularse.

### Las tres capas de validación de una configuración

Una configuración recorre tres comprobaciones antes de llegar a `enqueue_trial`, y
ninguna sobra, porque cada una mira algo distinto:

| Capa | Qué verifica | Dónde |
|---|---|---|
| Base | Todas las entradas comparten nombres de meta-características y versión del espacio | `MetaKnowledgeBase.__post_init__` |
| Recomendador | La versión es la vigente y cada configuración devuelta **se resuelve** a un pipeline | `MetaRecommender.fit` / `recommend` |
| Motor | Las claves coinciden **exactamente** con las de *su* espacio y los índices están en rango | `validate_warm_start` |

La última no puede delegarse en la anterior: el motor también corre sobre el
espacio de imágenes, que no es el canónico tabular, así que valida contra el suyo.

## Arranque en caliente en el motor

```python
motor = OptunaSearchEngine(
    build_tabular_search_space(),
    n_trials=30,
    task="classification",
    warm_start=configuraciones,      # lo que devolvió el recomendador
)
```

El parámetro es **aditivo**: sin él, el motor se comporta exactamente como antes.

**Los ensayos inyectados se consumen primero, en orden, y descuentan del
presupuesto.** Con cinco configuraciones y `n_trials=30` la búsqueda ejecuta treinta
ensayos, no treinta y cinco, de modo que quedan veinticinco de exploración propia.
El pre-registro los cuenta dentro de «ensayos hasta el objetivo» por esa razón: no
contarlos daría al arranque en caliente una ventaja puramente contable.

**Las configuraciones se validan al construir el motor**, no al lanzar la búsqueda,
y la validación no es defensiva sino imprescindible. Los dos modos de fallo de
`enqueue_trial`, medidos en el spike A y vueltos a comprobar contra el código
integrado, van en direcciones opuestas:

| Configuración inyectada | Qué hace Optuna |
|---|---|
| Índice fuera de rango | `ValueError` que se propaga: **aborta el estudio completo** |
| Clave desconocida | **Se acepta en silencio**: se ignora, se muestrea el resto y el ensayo termina sin aviso |

Por eso las claves deben coincidir **exactamente** con las del espacio, ni sobrar ni
faltar, y se rechazan los índices no enteros y los booleanos, porque `True` valdría 1
sin que nada lo dijera.

Una configuración inyectada que resulta **inevaluable** en el dataset receptor es
otra cosa: es legítima, simplemente no corre sobre esos datos. Queda con métrica
`NaN`, nunca gana, y la búsqueda continúa.

## El caso de anomalías en imágenes

`scripts/run_image_case.py` verifica la **generalización por tarea** sobre tres categorías
de MVTec AD (`bottle`, `cable`, `capsule`): el mismo cargador, registro de tareas, motores,
evaluador y reporter que usan los experimentos tabulares resuelven la detección de
anomalías con AUC-ROC. **No es un contraste de la hipótesis**, y no hay arranque en
caliente sobre imágenes.

| Decisión | Valor | Por qué |
|---|---|---|
| Resolución | 64 × 64, fijada en la preparación | Con `Resize` dentro del espacio, el motor añadiría su omisión y la mitad de los pipelines aplanaría imágenes de hasta 1024 × 1024 × 3 |
| Espacio | CLAHE, suavizado y normalización: 18 candidatos | Es la mitad «con redimensionado» del espacio del proyecto base |
| Modelo | `IsolationForest(random_state=0)` | Continuidad con la librería base; semilla fija para que solo varíe la partición |
| Semillas | 101, 202, 303, 404, 505 | Con pocas normales el AUC-ROC es inestable: se reporta la dispersión, no un valor único |

Dos comprobaciones activas: Optuna **nunca puede superar** el óptimo de la rejilla sobre la
misma partición —el guion aborta si ocurre—, y los puntajes reproducidos para la galería
tienen que dar **el mismo AUC-ROC** que informó el motor.

**Limitación que conviene conocer.** La partición `test/` de MVTec AD tiene mayoría de
anomalías —109 de 132 en `capsule`— y *Isolation Forest* supone que son escasas. Es
herencia del protocolo base y explica en buena parte que `capsule` quede al azar.

**Los HTML no se versionan**: embeben miniaturas de MVTec AD, cuya licencia prohíbe
redistribuir las imágenes. `tests/test_image_case.py` falla si alguien quita la regla de
`.gitignore`.

```powershell
.\.venv\Scripts\python.exe scripts\run_image_case.py --datos C:\datasets
```

## Verificar que los resultados se reproducen

`scripts/verify_reproducibility.py` comprueba **los resultados que el informe cita**, en
tres niveles: los documentos se regeneran desde sus JSON y se comparan ignorando solo la
fecha; las recomendaciones se reajustan por pliegue y se comparan con las guardadas; y las
corridas se reejecutan y se comparan ensayo por ensayo, **sin tolerancia** —dos parejas de
cada experimento y una del caso de imágenes—. Escribe `docs/experimentos/REPRODUCIBILIDAD.md`
y termina con código 1 si algo difiere.

Una verificación **omitida** —por ejemplo, sin los datos de MVTec AD— no es un fallo, pero
tampoco una evidencia: el documento las cuenta aparte.

```powershell
.\.venv\Scripts\python.exe scripts\verify_reproducibility.py --datos C:\datasets
```

## Reproducibilidad

- `random_state` gobierna la partición y el muestreador de Optuna. La misma semilla produce
  el mismo pipeline ganador. **Dos semillas distintas evalúan sobre particiones distintas**,
  así que sus métricas no son comparables entre sí: es la razón por la que el protocolo del
  TFG repite cada condición con tres semillas y toma la mediana.
- `scripts/check_image_regression.py` compara el camino de imágenes contra un commit de
  referencia y sale con código 1 si algo cambió. Conviene correrlo antes de cada hito.
- El multihilo de OpenCV puede alterar bits; `cv2.setNumThreads(1)` hace la comparación
  exacta.
- La configuración ganadora se serializa y puede reconstruirse con
  `ReportGenerator.rebuild_pipeline`. **Un pipeline reconstruido viene sin ajustar**: el
  ajuste depende de los datos, no de la configuración, así que para datos tabulares hay que
  volver a llamar a `fit` antes de transformar.

## Errores frecuentes

| Mensaje | Causa |
|---|---|
| «la tabla transformada conserva columnas no numéricas» | Falta un paso de codificación en el pipeline |
| «no ha sido ajustada; invoque fit()» | Se transformó una transformación con estado sin ajustarla |
| «A la tabla le faltan columnas que sí estaban en el ajuste» | El conjunto a transformar no tiene las mismas columnas que el de entrenamiento |
| «Las transformaciones tabulares operan sobre un DataFrame» | Se pasó un arreglo de NumPy; use `TabularDataLoader` |
| «Métrica no soportada para la tarea» | La métrica pertenece a otra tarea |
| «El criterio 'f_classif' es supervisado y necesita las etiquetas» | Se llamó a `fit()` sin `y`; use `criterion='variance'` si no hay etiquetas |
| «No se pudo puntuar la tabla con 'f_classif'» | Quedan valores faltantes; falta el paso de imputación |
| «fraction debe estar en (0, 1]» | Se pasó un número absoluto de columnas en vez de una fracción |
| «La configuración proviene del espacio de búsqueda X» | La configuración es de otra versión del espacio y no se puede reinterpretar |
| «El recomendador no está ajustado» | Se llamó a `recommend()` sin haber llamado antes a `fit(base)` |
| «Ninguna meta-característica varía entre los datasets» | El pliegue no tiene variación: todas las distancias serían cero |
| «El vector consultado no comparte ninguna meta-característica finita» | La consulta no tiene ninguna columna finita en común con las del ajuste |
| «no nombra los pasos del espacio de búsqueda: faltan … y sobran …» | La configuración inyectada no encaja en el espacio; Optuna ignoraría las claves desconocidas sin avisar |
| «Inyectarlo abortaría el estudio completo» | Un índice del arranque en caliente está fuera del rango de su paso |
| «no llegarían a ejecutarse» | Se pidieron más configuraciones inyectadas que ensayos de presupuesto |

## Dónde está cada decisión

| Documento | Contenido |
|---|---|
| `decisiones/ADR-001-contrato-fit-transform.md` | Por qué el contrato se extendió y qué alternativas se rechazaron |
| `decisiones/ADR-002-interfaz-de-tarea.md` | Diseño del registro de tareas |
| `decisiones/ADR-003-alcance-agrupamiento.md` | Por qué el agrupamiento queda fuera del alcance, con las horas que lo respaldan |
| `decisiones/AUDITORIA-acoplamiento.md` | Qué dependía de imágenes y anomalías, con los fallos silenciosos encontrados |
| `experimentos/Preregistro.tex` | Protocolo experimental, fijado antes de ejecutar |
| `experimentos/ESPACIO-tabular-v1.md` | Composición congelada del espacio, tabla de índices y fracción inevaluable |
| `experimentos/CORPUS-seleccionado.md` | Los 25 datasets del corpus, con su IQR y el motivo de cada exclusión |
| `experimentos/BASE-meta-conocimiento.md` | Resumen de la base: supervivientes, diversidad de ganadores y tiempos |
| `experimentos/DIVERSIDAD-ganadores.md` | Análisis exploratorio: forma de la diversidad, anchura de la meseta y efecto piso |
| `experimentos/EVALUACION-recomendador.md` | Evaluación de las estrategias sobre los 27 datasets fuera del corpus |
| `experimentos/LODO-resultados.json` | Curvas y análisis del experimento, generado en la semana 10 |
| `experimentos/CORPUS-criterios-inclusion.md` | Cómo se selecciona el corpus |
| `experimentos/SPIKE-A-enqueue-trial.md` | Viabilidad del arranque en caliente |
| `experimentos/SPIKE-B-pymfe.md` | Costo y robustez de la extracción de meta-características |
| `Agenda.tex` | Cronograma planificado y bitácora real |
| `Informe/` | Informe de TFG y su bibliografía |
| `anteproyecto/` | Anteproyecto aprobado, que fija el alcance comprometido |
| `base/` | Documentación del proyecto base del que parte este TFG |

## Ejemplos ejecutables

| Archivo | Qué muestra |
|---|---|
| `examples/tabular_quickstart.py` | El flujo tabular completo en 13 instrucciones. Lo ejecuta la suite de pruebas |
| `examples/tabular_walkthrough.ipynb` | El mismo flujo comentado: espacio canónico, candidatos inevaluables, anti-fuga en vivo y reproducibilidad |
| `examples/basic_usage.ipynb` | Uso básico sobre imágenes |
| `examples/mvtec_anomaly_detection.ipynb` | Anomalías industriales sobre MVTec AD |
| `examples/cifar10_synthetic_anomalies.ipynb` | Anomalías sintéticas sobre CIFAR-10 |

Todos los notebooks corren **sin datos externos**: si no encuentran el dataset real generan
un sustituto sintético. Los datos reales van fuera del repositorio; véase
`datasets/README.md`.

## Pendiente en esta guía

Se amplía con los reportes comparativos del experimento y el análisis de causas
(semanas 10 y 11).
