# Auditoría de acoplamiento a imágenes y detección de anomalías

- **Fecha:** 2026-07-23
- **Etapa:** E1 (análisis y diseño), insumo de RE1
- **Alcance:** los seis módulos migrados de `autopipe/`, en el commit `f34e176`
- **Propósito:** localizar exactamente qué depende de imágenes y de detección de
  anomalías, para dimensionar el refactor del núcleo agnóstico.

## Resumen

| Módulo | Acoplamiento | Acción |
|---|---|---|
| `data_loader/loader.py` | **Total** | No se refactoriza: se añade un cargador hermano |
| `evaluator/metrics.py` | **Total** | Pasa a ser una implementación dentro de un registro por tarea |
| `evaluator/splitter.py` | **Ninguno** | Se reutiliza sin cambios |
| `pipeline/base.py`, `pipeline/pipeline.py` | Ninguno (tras ADR-001) | Sin cambios |
| `pipeline/transforms.py` | Total, **por diseño** | Sin cambios: son transformaciones de imagen |
| `search_engine/grid_search.py` | **Alto, y parcialmente silencioso** | Es el centro del refactor |
| `search_engine/optuna_search.py` | Bajo | Un parámetro y un valor por defecto |
| `parallel_runner/runner.py` | Bajo | Igual que los motores |
| `reporter/report.py` | Bajo | Menos de lo esperado |

El anteproyecto (§2) afirma que «el acoplamiento a imágenes y a la detección de
anomalías se concentra únicamente en la carga de datos y en el cálculo de métricas».
**La auditoría lo confirma en lo esencial, pero encuentra dos puntos más**, ambos
dentro del motor de búsqueda y ambos de fallo silencioso. Se detallan abajo porque
cambian el alcance del refactor.

## Hallazgo principal: dos fallos silenciosos ante multiclase

Son los dos puntos que el anteproyecto no anticipaba, y son los más peligrosos
precisamente porque **no levantan ninguna excepción**: producen números plausibles
pero incorrectos.

### `_normalize_predictions` colapsa las clases

`grid_search.py:40-59` normaliza las predicciones asumiendo dos clases y la
convención de scikit-learn para detección de anomalías (`-1` anómalo, `+1` normal).
Ante predicciones multiclase cae en la rama de binarización por umbral:

```
_normalize_predictions([0, 1, 2, 3, 4])  ->  [0, 1, 1, 1, 1]
```

Cinco clases quedan reducidas a dos. Las métricas se calculan sobre esas etiquetas
falsas y salen números perfectamente creíbles.

### `_predict_scores` descarta todas las clases menos una

`grid_search.py:62-82` obtiene los puntajes para el AUC-ROC priorizando
`predict_proba(...)[:, 1]`. Sobre un modelo de cinco clases:

```
predict_proba devuelve  (5, 5)
_predict_scores devuelve  (5,)   <- solo la columna de la clase 1
```

Cuatro de las cinco columnas se descartan sin aviso.

**Consecuencia para el diseño:** la normalización de predicciones y la obtención de
puntajes no son detalles internos, sino **parte de la definición de la tarea**. Tienen
que viajar junto con las métricas en cualquier abstracción de tarea que se introduzca,
o el refactor dejará el mismo agujero.

## Detalle por módulo

### `data_loader/loader.py` — acoplamiento total

- Estructura de directorios `normal/` y `anomaly/` fijada en constantes de clase
  (`loader.py:35-38`).
- Extensiones de imagen (`loader.py:39`) y decodificación con PIL (`loader.py:143`).
- Etiquetas 0 y 1 asignadas por subdirectorio.

**No se refactoriza.** La clase ya se llama `ImageDataLoader` y su contrato es
correcto para su dominio; RE2 añade un `TabularDataLoader` hermano. El acoplamiento
aquí no es deuda, es especialización.

### `evaluator/metrics.py` — acoplamiento total

- Conjunto de claves fijo: `auc_roc`, `f1`, `precision`, `recall`,
  `false_positive_rate` (`metrics.py:67-73`).
- `f1_score`, `precision_score` y `recall_score` se invocan en modo binario, con la
  clase positiva implícita en 1.
- `_safe_auc` devuelve `NaN` si hay menos de dos clases (`metrics.py:103`).
- La tasa de falsos positivos se calcula desde una matriz de confusión con
  `labels=[0, 1]` (`metrics.py:63`), que **descarta** cualquier clase adicional.

Es el punto que el anteproyecto identifica correctamente. Las claves están fijadas por
los 134 tests y por el criterio de éxito de ingeniería, así que **no pueden cambiar**;
la salida es convertir esta clase en una de varias implementaciones seleccionables.

### `evaluator/splitter.py` — sin acoplamiento

`DataSplitter` opera únicamente sobre un arreglo de etiquetas y delega en
`train_test_split` y `StratifiedKFold`, que ya soportan multiclase. **Se reutiliza tal
cual, sin una sola línea de cambio.** Es la mejor noticia de la auditoría.

### `search_engine/grid_search.py` — el centro del refactor

Además de los dos fallos silenciosos ya descritos:

- `VALID_METRICS` y `MINIMIZE_METRICS` (`grid_search.py:22-23`) son constantes de
  módulo con las métricas de anomalías. `validate_metric` (`grid_search.py:190`) las
  usa como conjunto único y global.
- `evaluate_pipeline` (`grid_search.py:115-121`) recibe un `MetricCalculator` concreto.
- `_transform_to_matrix` (`grid_search.py:36`) aplana cada muestra con `ravel`. Para
  datos tabulares una fila se aplana en sí misma, de modo que **funciona sin cambios**;
  lo que queda mal es el nombre del parámetro `images`.

### `search_engine/optuna_search.py` y `parallel_runner/runner.py` — acoplamiento bajo

Ambos declaran `metric: str = "auc_roc"` por defecto, nombran `images` al conjunto de
entrada y delegan la lógica acoplada en las funciones de `grid_search`. `ParallelRunner`
además construye su propio `MetricCalculator` (`runner.py:59`) y llama a
`validate_metric` (`runner.py:85`), por lo que es un tercer consumidor del registro de
métricas y debe entrar en el refactor junto con los dos motores.

### `reporter/report.py` — menos acoplado de lo esperado

- **Ya es agnóstico en lo importante:** las columnas de métricas del reporte se derivan
  dinámicamente de los resultados (`report.py:312`), no de una lista fija. Un cambio de
  tarea no requiere tocar la plantilla.
- La galería de imágenes anómalas (`report.py:360-376`, atributo `top_anomalies`) sí es
  específica de imágenes y anomalías, pero la plantilla ya la envuelve en un
  condicional (`report.py:103`), de modo que desaparece sola cuando no hay imágenes ni
  puntajes.
- Queda el valor por defecto `metric="auc_roc"` (`report.py:175`).

## Conclusiones para el refactor

1. El refactor del núcleo es **más acotado de lo que sugiere el anteproyecto en un
   sentido** —`DataSplitter` y el reporter necesitan poco o nada— y **más amplio en
   otro**: hay que cubrir la normalización de predicciones y la obtención de puntajes,
   que no estaban en el inventario original.
2. La abstracción de tarea debe agrupar **cuatro** responsabilidades, no dos: métricas,
   dirección de optimización, normalización de predicciones y obtención de puntajes.
   Separarlas permitiría que se desincronicen, que es exactamente el modo de fallo
   documentado arriba.
3. Hay **tres** consumidores del registro de métricas, no dos: los dos motores de
   búsqueda y `ParallelRunner`.
4. Las claves de `MetricCalculator` y la firma `run(...)` no se tocan; la selección de
   tarea debe entrar por otra vía.

El diseño que responde a estas conclusiones es el ADR-002.
