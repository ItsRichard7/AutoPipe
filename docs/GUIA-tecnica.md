# Guía técnica de AutoPipe

- **Estado:** en construcción (se completa en la semana 4 del cronograma)
- **Entregable:** RE7
- **Actualizada:** 2026-08-19

Documentación técnica del framework. Para el uso básico está el `README.md`; esta guía
explica **cómo funciona por dentro** y **qué garantiza**, que es lo que hace falta para
extenderlo o para confiar en sus resultados.

## Qué cambió respecto de AutoPipeAD

Quien conocía la librería base encontrará cuatro diferencias:

1. **El framework ya no asume una tarea.** Un objeto `Task` agrupa las métricas, la
   dirección de optimización, la normalización de predicciones y la obtención de puntajes.
   Se elige en el constructor de los motores.
2. **El contrato de transformación tiene ajuste.** `BaseTransform` gana `fit` y `transform`
   por lote, además del `apply` por muestra que ya existía.
3. **El pipeline se ajusta solo con el subconjunto de entrenamiento.** Antes se
   transformaba el dataset completo y después se partía.
4. **Hay soporte de datos tabulares**: un cargador y tres transformaciones con estado.

Nada de lo anterior cambió una interfaz pública: el flujo de imágenes de la librería base
funciona sin modificar una línea, y sus 134 pruebas de regresión siguen en verde.

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

Hay una prueba que lo vigila: `get_params()` de las transformaciones tabulares solo puede
devolver cadenas.

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

## Flujo tabular

El ejemplo ejecutable está en `examples/tabular_quickstart.py`. En resumen:

```python
features, labels = TabularDataLoader(tabla).load()

espacio = {
    "imputacion": [ImputeMissing("mean"), ImputeMissing("median")],
    "codificacion": [EncodeCategorical("onehot"), EncodeCategorical("ordinal")],
    "escalado": [ScaleNumeric("standard"), ScaleNumeric("minmax")],
}

motor = GridSearchEngine(espacio, random_state=42, task="classification")
mejor, metricas, resultados = motor.run(features, labels, KNeighborsClassifier())
```

**El cargador devuelve una tabla, no una matriz numérica.** Conservar los tipos de columna
es lo que permite a las transformaciones distinguir solas las categóricas de las numéricas,
sin recibirlo como parámetro.

**El orden del espacio de búsqueda es el orden del pipeline.** Imputar antes de codificar y
codificar antes de escalar es el orden natural: escalar antes de codificar dejaría las
indicadoras sin escalar, y codificar antes de imputar convertiría los faltantes en una
categoría propia.

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

## Reproducibilidad

- `random_state` gobierna la partición y el muestreador de Optuna. La misma semilla produce
  el mismo pipeline ganador.
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

## Dónde está cada decisión

| Documento | Contenido |
|---|---|
| `decisiones/ADR-001-contrato-fit-transform.md` | Por qué el contrato se extendió y qué alternativas se rechazaron |
| `decisiones/ADR-002-interfaz-de-tarea.md` | Diseño del registro de tareas |
| `decisiones/AUDITORIA-acoplamiento.md` | Qué dependía de imágenes y anomalías, con los fallos silenciosos encontrados |
| `experimentos/Preregistro.tex` | Protocolo experimental, fijado antes de ejecutar |
| `experimentos/CORPUS-criterios-inclusion.md` | Cómo se selecciona el corpus |
| `experimentos/SPIKE-A-enqueue-trial.md` | Viabilidad del arranque en caliente |
| `experimentos/SPIKE-B-pymfe.md` | Costo y robustez de la extracción de meta-características |
| `Agenda.tex` | Cronograma planificado y bitácora real |
| `Informe/` | Informe de TFG y su bibliografía |
| `anteproyecto/` | Anteproyecto aprobado, que fija el alcance comprometido |
| `base/` | Documentación del proyecto base del que parte este TFG |

## Pendiente en esta guía

Se completa en la semana 4, cuando existan la selección de características y el espacio de
búsqueda canónico: falta documentar ese espacio y su versionado, y añadir el notebook
tabular equivalente a los tres de imágenes.
