# Manual de uso de AutoPipe

- **Entregable:** RE7
- **Versión de la librería:** 0.1.0
- **Actualizado:** 2026-09-27

Este manual explica cómo **usar** AutoPipe. Cómo funciona por dentro y qué garantiza está en
la guía técnica, [`GUIA-tecnica.md`](GUIA-tecnica.md). Todos los fragmentos de código de este
manual se ejecutaron contra la versión documentada; los dos flujos principales existen además
como ejemplos ejecutables en `examples/`, que la suite de pruebas corre en cada ejecución.

## Contenido

1. [Qué hace AutoPipe](#1-qué-hace-autopipe)
2. [Instalación](#2-instalación)
3. [Conceptos](#3-conceptos)
4. [Flujo 1: clasificación de datos tabulares](#4-flujo-1-clasificación-de-datos-tabulares)
5. [Flujo 2: arranque en caliente meta-aprendido](#5-flujo-2-arranque-en-caliente-meta-aprendido)
6. [Flujo 3: detección de anomalías en imágenes](#6-flujo-3-detección-de-anomalías-en-imágenes)
7. [Evaluación en paralelo](#7-evaluación-en-paralelo)
8. [Reportes y reutilización del ganador](#8-reportes-y-reutilización-del-ganador)
9. [Reproducibilidad](#9-reproducibilidad)
10. [Métricas por tarea](#10-métricas-por-tarea)
11. [Limitaciones conocidas](#11-limitaciones-conocidas)
12. [Solución de problemas](#12-solución-de-problemas)

## 1. Qué hace AutoPipe

Dado un conjunto de datos, una tarea de aprendizaje y un **modelo base**, AutoPipe busca el
**pipeline de preprocesamiento** que mejor desempeño le da a ese modelo: prueba
combinaciones de transformaciones, las evalúa todas sobre la misma partición y devuelve la
ganadora con su tabla completa de resultados.

Resuelve dos tareas:

| Tarea | Datos | Métrica primaria |
|---|---|---|
| `classification` | Tablas (CSV o `DataFrame`), binaria o multiclase | `macro_f1` |
| `anomaly_detection` | Imágenes en carpetas `normal/` y `anomaly/` | `auc_roc` |

Y ofrece, para la clasificación tabular, un **arranque en caliente meta-aprendido**: en lugar
de empezar la búsqueda a ciegas, describe el conjunto de datos con sus meta-características y
empieza por las configuraciones que ganaron en conjuntos de datos de una base de
meta-conocimiento.

**Lo que no hace:** no entrena ni ajusta los hiperparámetros del modelo base, que usted
elige y que solo necesita la interfaz `fit`/`predict` de scikit-learn. Tampoco trata texto,
series temporales, grafos ni agrupamiento.

## 2. Instalación

Requisitos:

- Python **3.10 o superior**. Se desarrolló y probó sobre 3.12.
- Opcional, solo para exportar reportes a PDF: el binario externo
  [`wkhtmltopdf`](https://wkhtmltopdf.org/) en el `PATH`. Sin él, los reportes HTML se generan
  igual.

```powershell
git clone https://github.com/ItsRichard7/AutoPipe.git
cd AutoPipe
python -m venv .venv
.\.venv\Scripts\activate          # en Linux o macOS: source .venv/bin/activate
pip install -e .                  # o pip install -e ".[dev]" para correr las pruebas
```

Para comprobar la instalación:

```powershell
python -c "import autopipe; print(autopipe.__version__)"      # 0.1.0
python examples\tabular_quickstart.py
```

## 3. Conceptos

| Concepto | Qué es |
|---|---|
| **Tarea** | Agrupa las métricas, su dirección de optimización y el tratamiento de las predicciones. Se elige con `task=` en el constructor de los motores; por defecto es `anomaly_detection` |
| **Modelo base** | El modelo que usted quiere preprocesar. AutoPipe lo clona para cada candidato y nunca modifica el original |
| **Espacio de búsqueda** | Un diccionario de nombre de paso a lista de transformaciones candidatas |
| **Candidato** | Un pipeline: una transformación por paso, en el orden del espacio |
| **Omisión de pasos** | Cada paso se prueba también **omitido**, sin que usted lo pida. Un paso omitido no aparece en el pipeline |
| **Candidato inevaluable** | Uno que no se puede evaluar sobre esos datos, por ejemplo porque omite la codificación y quedan columnas de texto. Se registra con métricas `NaN` y nunca gana |
| **Semilla** | `random_state`: gobierna la partición y el muestreador. La misma semilla produce el mismo ganador |

Dos garantías que conviene conocer: cada pipeline se **ajusta solo con el subconjunto de
entrenamiento**, de modo que ninguna estadística de validación se filtra al entrenamiento, y
**todos los candidatos comparten la misma partición**, de modo que la comparación es justa.

## 4. Flujo 1: clasificación de datos tabulares

### 4.1 Preparar los datos

```python
from autopipe import TabularDataLoader

cargador = TabularDataLoader("datos.csv")                 # la última columna es el objetivo
features, labels = cargador.load()
```

- `target_column` elige otra columna objetivo, por nombre o por posición:
  `TabularDataLoader("datos.csv", target_column="clase")`.
- `na_values` añade marcas de valor faltante al leer un CSV, por ejemplo `na_values=["?"]`,
  habitual en los conjuntos de UCI.
- También acepta un `DataFrame` en memoria: `TabularDataLoader(tabla)`.
- Devuelve las características como **tabla con sus tipos**, no como matriz: así las
  transformaciones distinguen solas las columnas numéricas de las categóricas.
- Las etiquetas quedan codificadas como enteros `0..k-1`; los valores originales están en
  `cargador.classes_`, y las columnas detectadas en `cargador.categorical_columns_` y
  `cargador.numeric_columns_`.

Se rechaza, con un mensaje explícito, un objetivo con valores faltantes, un conjunto sin
columnas de características o con menos de dos clases.

### 4.2 Buscar

```python
from sklearn.neighbors import KNeighborsClassifier

from autopipe import GridSearchEngine, build_tabular_search_space

motor = GridSearchEngine(
    build_tabular_search_space(), val_size=0.2, random_state=42, task="classification"
)
mejor, metricas, resultados = motor.run(features, labels, KNeighborsClassifier())

print(mejor)                    # el pipeline ganador
print(metricas["macro_f1"])     # su métrica primaria
```

**Use el espacio canónico** (`build_tabular_search_space()`) en lugar de armarlo a mano. Tiene
cuatro pasos en orden fijo —imputación, escalado, codificación y selección de
características— y **144 candidatos** contando la omisión de cada paso. Es el mismo para
cualquier conjunto de datos y lleva una versión (`TABULAR_SPACE_VERSION`, hoy `tabular-v1`),
que es lo que permite transferir configuraciones entre conjuntos de datos en el flujo 2.

### 4.3 Leer los resultados

`run(...)` devuelve tres cosas:

| Valor | Contenido |
|---|---|
| `mejor` | El `Pipeline` ganador |
| `metricas` | Todas las métricas de la tarea para el ganador |
| `resultados` | Una fila por candidato, con `pipeline_index`, `params`, `metrics` y `objective` |

Es normal que parte de los candidatos salga con `objective` en `NaN`. Con una sola columna
categórica, por ejemplo, los 48 candidatos que omiten la codificación son inevaluables: el
modelo no puede entrenar con texto. Si **ningún** candidato se puede evaluar, el motor lo
avisa en el registro («Ningún pipeline pudo evaluarse») y el ganador que devuelve no
significa nada: revise los datos, el espacio y el modelo base.

### 4.4 Ver los candidatos antes de buscar

```python
candidatos = motor.build_pipelines()
print(len(candidatos))          # 144
```

No evalúa nada: devuelve exactamente los candidatos que `run` recorrería, en el mismo orden.

### 4.5 Búsqueda bayesiana con Optuna

La búsqueda exhaustiva evalúa los 144 candidatos. `OptunaSearchEngine` explora el mismo
espacio con un presupuesto de ensayos, con la misma interfaz:

```python
from autopipe import OptunaSearchEngine

motor = OptunaSearchEngine(
    build_tabular_search_space(), n_trials=30, random_state=42, task="classification"
)
mejor, metricas, resultados = motor.run(features, labels, KNeighborsClassifier())
print(motor.best_configuration_)   # {'imputation': 2, 'scaling': 3, 'encoding': 0, ...}
```

`best_configuration_` guarda el ganador como **un índice por paso**. Es la forma que sirve
para reconstruirlo más tarde (sección 8) y la que consume el arranque en caliente.

## 5. Flujo 2: arranque en caliente meta-aprendido

### 5.1 El flujo completo

```python
from sklearn.neighbors import KNeighborsClassifier

from autopipe import (
    MetaFeatureExtractor,
    MetaKnowledgeBase,
    MetaRecommender,
    OptunaSearchEngine,
    TabularDataLoader,
    build_tabular_search_space,
)

features, labels = TabularDataLoader("datos.csv").load()
vector = MetaFeatureExtractor(random_state=42).extract(features, labels)

base = MetaKnowledgeBase.load("docs/experimentos/BASE-E2.json")
recomendador = MetaRecommender(strategy="frequent").fit(base)
configuraciones = recomendador.recommend(vector)

motor = OptunaSearchEngine(
    build_tabular_search_space(),
    n_trials=30,
    random_state=42,
    task="classification",
    warm_start=configuraciones,
)
mejor, metricas, resultados = motor.run(features, labels, KNeighborsClassifier())
```

Paso a paso:

1. **`MetaFeatureExtractor`** describe el conjunto de datos con 107 meta-características
   (estadísticas, teórico-informativas y de complejidad). Las de complejidad se calculan
   sobre una submuestra fija de 500 filas, porque su costo crece de forma cuadrática. Use
   `random_state=42`, que es con el que se construyeron las bases incluidas.
2. **`MetaKnowledgeBase.load`** lee una base. El repositorio incluye dos, ambas sobre
   `tabular-v1`:

   | Archivo | Conjuntos de datos |
   |---|---:|
   | `docs/experimentos/BASE-meta-conocimiento.json` | 25 |
   | `docs/experimentos/BASE-E2.json` | 42 |

   Al cargarla se verifica que la versión del espacio sea la vigente: una base de otra
   versión se rechaza, porque sus índices designarían otras transformaciones.
3. **`MetaRecommender`** elige cinco configuraciones **distintas** entre las ganadoras de la
   base. Después de `recommend`, `recomendador.neighbours_` dice de qué conjuntos de datos
   salió cada una.
4. **`warm_start=`** inyecta esas configuraciones como los primeros ensayos. **Consumen
   presupuesto:** con cinco inyectadas y `n_trials=30`, la búsqueda ejecuta 30 ensayos y no
   35.

### 5.2 Qué estrategia usar

| `strategy` | De dónde toma las configuraciones |
|---|---|
| `nearest` | De los conjuntos de datos más parecidos según sus meta-características. Es el valor por defecto |
| `frequent` | Las que más veces ganan en la base, **sin mirar** las meta-características |
| `random` | De conjuntos de datos elegidos al azar (`random_state` fija cuáles) |
| `hybrid` | Dos de `frequent` y el resto de `nearest` |
| `diverse` | Por cercanía, eligiendo las más separadas entre sí. No se evaluó experimentalmente |

**Lo que midieron los experimentos del proyecto** sobre 42 conjuntos de datos de OpenML: el
arranque en caliente redujo el costo de la búsqueda frente al arranque en frío —35,8 % menos
*regret* mediano con `nearest`, significativo tras la corrección de Holm—, pero **elegir por
similitud fue la peor de las cuatro estrategias evaluadas**. El orden por *regret* mediano,
de mejor a peor, fue `random`, `frequent`, `hybrid`, `nearest` y, al final, el arranque en
frío. Por eso el ejemplo de arriba usa `strategy="frequent"`. El detalle está en
[`experimentos/LODO-E2-discusion.md`](experimentos/LODO-E2-discusion.md).

### 5.3 Si va a evaluar el recomendador

Si el conjunto de datos que consulta **está en la base**, el recomendador lo encontrará a sí
mismo y la comparación no valdrá nada. Para medir de forma honesta hay que excluirlo:

```python
recomendador = MetaRecommender().fit(base.without(identificador_del_conjunto))
```

Es lo que hace la validación *leave-one-dataset-out* del proyecto.

## 6. Flujo 3: detección de anomalías en imágenes

### 6.1 Preparar las imágenes

```
mi_dataset/
├── normal/      imágenes sin anomalía (etiqueta 0)
└── anomaly/     imágenes con anomalía (etiqueta 1)
```

Se admiten `.png`, `.jpg` y `.bmp`.

### 6.2 Buscar

```python
from sklearn.ensemble import IsolationForest

from autopipe import GaussianBlur, GridSearchEngine, ImageDataLoader, Normalize, Resize

images, labels = ImageDataLoader(dataset_path="mi_dataset").load()

search_space = {
    "resize": [Resize(width=64, height=64)],
    "blur": [GaussianBlur(kernel_size=3), GaussianBlur(kernel_size=5)],
    "normalize": [Normalize(mean=0.5, std=0.5)],
}
engine = GridSearchEngine(search_space, val_size=0.2, random_state=42)
best_pipeline, metrics, results = engine.run(
    images, labels, base_model=IsolationForest(random_state=0), metric="auc_roc"
)
```

La tarea por defecto es `anomaly_detection`, así que no hace falta indicarla. Hay siete
transformaciones de imagen: `Resize`, `GaussianBlur`, `MedianFilter`, `Normalize`,
`CLAHETransform`, `GammaCorrection` y `GrayscaleConversion`.

**Si las imágenes tienen tamaños distintos**, el candidato que omite `Resize` no se puede
evaluar y queda en `NaN`; el resto de la búsqueda sigue. Con imágenes grandes conviene
redimensionarlas **antes** de buscar, para que ningún candidato tenga que aplanarlas a
resolución completa.

## 7. Evaluación en paralelo

`ParallelRunner` evalúa una lista de pipelines en varios procesos y puede medir la
aceleración:

```python
from autopipe import ParallelRunner

runner = ParallelRunner(n_jobs=-1, val_size=0.3, random_state=42, task="classification")
tabla = runner.run(motor.build_pipelines(), features, labels, KNeighborsClassifier())
informe = runner.benchmark(motor.build_pipelines(), features, labels, KNeighborsClassifier())
print(informe["speedup"], informe["consistent"])
```

`run` devuelve la misma tabla que los motores. Con `n_jobs=1` es secuencial, útil en
Jupyter.

**En Windows**, si lanza trabajos en paralelo desde un archivo `.py`, proteja el punto de
entrada con `if __name__ == "__main__":`; de lo contrario los procesos hijos vuelven a
ejecutar el script.

## 8. Reportes y reutilización del ganador

### 8.1 Generar el reporte

```python
from autopipe import ReportGenerator

rutas = ReportGenerator(output_dir="reportes").generate(
    resultados, mejor, metricas, metric="macro_f1", pdf=False
)
print(rutas["html"], rutas["config"])
```

- Escribe siempre `report.html` y la configuración del ganador, `winning_pipeline.json`
  (o YAML con `config_format="yaml"`).
- Con `pdf=True` intenta además `report.pdf`; si `wkhtmltopdf` no está instalado, registra un
  aviso y `rutas["pdf"]` queda en `None`.
- Pase en `metric` la métrica que optimizó: por defecto es `auc_roc`, la de anomalías.
- Sin `output_dir`, el reporte va a una carpeta `reports/` junto al script, o al directorio
  de trabajo en Jupyter.
- En anomalías de imagen, pasar `images=` y `scores=` añade una galería con las imágenes más
  anómalas.

### 8.2 Reconstruir el ganador

La configuración que guarda el reporte reconstruye el ganador, sea de imagen o tabular:

```python
config = ReportGenerator.load_pipeline_config("reportes/winning_pipeline.json")
pipeline = ReportGenerator.rebuild_pipeline(config)
```

Solo se reconstruyen las transformaciones predefinidas de la librería; una transformación
propia se rechaza con «Transformación no registrada».

Tras una búsqueda con Optuna hay otra forma que no necesita el reporte: guardar los índices
que registró el motor y reconstruir desde ellos con el espacio canónico.

```python
import json

from autopipe import pipeline_from_configuration

with open("ganador.json", "w", encoding="utf-8") as archivo:
    json.dump(motor.best_configuration_, archivo)          # tras OptunaSearchEngine.run

with open("ganador.json", encoding="utf-8") as archivo:
    pipeline = pipeline_from_configuration(json.load(archivo))
```

**Un pipeline reconstruido viene sin ajustar**, en los dos casos: el ajuste depende de los
datos y no de la configuración. Ajústelo con los datos de entrenamiento antes de transformar:
`pipeline.fit(features_entrenamiento, labels_entrenamiento)`.

## 9. Reproducibilidad

- **La misma semilla produce el mismo ganador.** `random_state` fija la partición de
  entrenamiento y validación y el muestreador de Optuna.
- **Semillas distintas evalúan sobre particiones distintas**, así que sus métricas no son
  comparables entre sí. Para una conclusión robusta, repita con varias semillas y tome la
  mediana.
- **Con imágenes**, OpenCV en varios hilos puede variar los últimos bits de un resultado.
  Para una comparación exacta, fije un solo hilo antes de buscar: `cv2.setNumThreads(1)`.
- Guarde junto a una configuración la versión del espacio (`TABULAR_SPACE_VERSION`): unos
  índices solo significan algo respecto de la versión con que se obtuvieron.

## 10. Métricas por tarea

| Tarea | Métricas | Se minimiza |
|---|---|---|
| `anomaly_detection` | `auc_roc` (primaria), `f1`, `precision`, `recall`, `false_positive_rate` | `false_positive_rate` |
| `classification` | `macro_f1` (primaria), `accuracy`, `balanced_accuracy`, `macro_precision`, `macro_recall` | ninguna |

Si omite `metric` en `run(...)`, se optimiza la primaria de la tarea. Las métricas macro pesan
igual a todas las clases, de modo que un modelo que predice siempre la mayoritaria puntúa bajo
aunque las clases estén desbalanceadas.

## 11. Limitaciones conocidas

- **Un único espacio tabular.** El canónico `tabular-v1` tiene 144 candidatos. Puede pasar
  un espacio propio a los motores, pero las bases de meta-conocimiento incluidas solo sirven
  para el canónico.
- **Bases de meta-conocimiento pequeñas.** 25 y 42 conjuntos de datos de la suite
  OpenML-CC18. Un conjunto muy distinto de todos ellos recibirá recomendaciones de vecinos
  lejanos; `recomendador.clipped_features_` indica en qué meta-características se sale del
  rango de la base.
- **La similitud no fue la mejor estrategia** en los experimentos del proyecto (sección 5.2).
- **Métricas binarias de anomalías con modelos que predicen todo «normal».** Si un modelo con
  la convención de scikit-learn predice `+1` para **todas** las muestras, esas predicciones
  se interpretan como «todo anómalo». Afecta a `f1`, `precision`, `recall` y
  `false_positive_rate`, no al `auc_roc`, que usa puntajes continuos. Optimice y reporte
  `auc_roc` en esta tarea.
- ***Isolation Forest* supone que las anomalías son escasas.** Con particiones donde las
  anómalas son mayoría —como la de `test/` de MVTec AD—, su AUC-ROC puede quedar cerca del
  azar.
- Fuera de alcance: agrupamiento, texto, series temporales, grafos, segmentación, interfaz
  gráfica y API REST.

## 12. Solución de problemas

| Mensaje o síntoma | Causa | Qué hacer |
|---|---|---|
| «Ningún pipeline pudo evaluarse» o «Ningún ensayo pudo evaluarse» | Ningún candidato corrió sobre esos datos | Revise que el modelo base entrene con una matriz numérica y que el espacio incluya los pasos que los datos necesitan |
| «la tabla transformada conserva columnas no numéricas» | El candidato omite la codificación | Normal en parte de los candidatos; si pasa con todos, el espacio no tiene paso de codificación |
| «Métrica no soportada para la tarea» | La métrica es de la otra tarea | Revise `task=` y `metric=` (sección 10) |
| «Transformación no registrada» | La configuración guardada nombra una transformación propia, no predefinida | `rebuild_pipeline` solo reconstruye las predefinidas; arme ese pipeline con su propia clase |
| «No se pudo generar el PDF (¿wkhtmltopdf instalado?)» | Falta el binario `wkhtmltopdf` | Instálelo o use `pdf=False`; el HTML se generó igual |
| «no ha sido ajustada; invoque fit()» | Se transformó con un pipeline reconstruido sin ajustar | Llame a `fit` con los datos de entrenamiento |
| «La configuración proviene del espacio de búsqueda X» | Base o configuración de otra versión del espacio | Use una base construida sobre la versión vigente |
| «El recomendador no está ajustado» | Se llamó a `recommend()` antes de `fit(base)` | Ajuste primero el recomendador |
| «no nombra los pasos del espacio de búsqueda» | Una configuración inyectada no encaja en el espacio | Las configuraciones deben tener exactamente los pasos del espacio |
| «Inyectarlo abortaría el estudio completo» | Un índice inyectado está fuera de rango | Revise el origen de la configuración |
| «no llegarían a ejecutarse» | Más configuraciones inyectadas que ensayos | Aumente `n_trials` o pida menos configuraciones |
| `MetaFeatureError` | `pymfe` no pudo extraer algún grupo de meta-características | El mensaje indica qué grupos fallaron, el tamaño de la tabla y la causa original |
| El script en paralelo se queda colgado o se relanza en Windows | Falta la guarda del punto de entrada | Proteja el código con `if __name__ == "__main__":` |

Más mensajes, con su causa técnica, en la sección «Errores frecuentes» de la
[guía técnica](GUIA-tecnica.md).
