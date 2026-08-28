# Guía técnica de AutoPipe

- **Estado:** completa para el núcleo generalizado (hito de la semana 4)
- **Entregable:** RE7
- **Actualizada:** 2026-08-27

Documentación técnica del framework. Para el uso básico está el `README.md`; esta guía
explica **cómo funciona por dentro** y **qué garantiza**, que es lo que hace falta para
extenderlo o para confiar en sus resultados.

## Qué cambió respecto de AutoPipeAD

Quien conocía la librería base encontrará cinco diferencias:

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

## Dónde está cada decisión

| Documento | Contenido |
|---|---|
| `decisiones/ADR-001-contrato-fit-transform.md` | Por qué el contrato se extendió y qué alternativas se rechazaron |
| `decisiones/ADR-002-interfaz-de-tarea.md` | Diseño del registro de tareas |
| `decisiones/AUDITORIA-acoplamiento.md` | Qué dependía de imágenes y anomalías, con los fallos silenciosos encontrados |
| `experimentos/Preregistro.tex` | Protocolo experimental, fijado antes de ejecutar |
| `experimentos/ESPACIO-tabular-v1.md` | Composición congelada del espacio, tabla de índices y fracción inevaluable |
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

Se amplía a medida que existan los módulos que faltan: el extractor de
meta-características (semana 5), la base de meta-conocimiento (semana 6) y el recomendador
con el arranque en caliente integrado en el motor de Optuna (semanas 7 y 8).
