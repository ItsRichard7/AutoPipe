# ADR-002 — Interfaz de tarea y registro de métricas

- **Estado:** aceptado
- **Fecha:** 2026-07-23
- **Aprobado por:** Ricardo Antonio Borbón Mena, 2026-07-23
- **Etapa:** E1 (análisis y diseño de la generalización), entregable RE1
- **Afecta a:** `autopipe/evaluator/`, `autopipe/search_engine/`,
  `autopipe/parallel_runner/`
- **Insumo:** `docs/decisiones/AUDITORIA-acoplamiento.md`

## Contexto

El framework debe pasar de servir a una sola tarea (detección de anomalías binaria en
imágenes) a soportar además clasificación tabular, y —si el control de alcance de la
semana 7 lo permite— agrupamiento.

La auditoría de acoplamiento localizó dónde vive la dependencia de la tarea. El
resultado corrige el inventario del anteproyecto en un punto importante: además del
cálculo de métricas, el motor de búsqueda contiene **dos funciones específicas de
anomalías que fallan en silencio** ante datos multiclase.

```
_normalize_predictions([0, 1, 2, 3, 4])  ->  [0, 1, 1, 1, 1]
_predict_scores(modelo_de_5_clases, X)   ->  solo la columna de la clase 1
```

Ninguna levanta excepción: producen números plausibles y equivocados.

## Problema

Se necesita seleccionar el comportamiento dependiente de la tarea sin romper las
interfaces públicas ya en uso, y sin permitir que las piezas se desincronicen entre sí.

Restricciones:

1. Las claves de `MetricCalculator` (`auc_roc`, `f1`, `precision`, `recall`,
   `false_positive_rate`) **no pueden cambiar**: las fijan los 134 tests de regresión y
   el criterio de éxito de ingeniería del anteproyecto (§4.3.5).
2. La firma `run(...)` de los motores de búsqueda **no puede cambiar**.
3. `ParallelRunner` es un tercer consumidor del registro de métricas y debe quedar
   cubierto.
4. `CLAUDE.md` pide no introducir herencia sin justificarla: la lógica compartida del
   proyecto son funciones de módulo.

## Decisión

### 1. Una tarea es un objeto de configuración, no una jerarquía de clases

Se introduce `autopipe/tasks/` con una estructura de datos inmutable que agrupa **las
cuatro responsabilidades que deben viajar juntas**:

```python
@dataclass(frozen=True)
class Task:
    name: str
    valid_metrics: tuple[str, ...]
    default_metric: str
    minimize_metrics: tuple[str, ...]
    build_metric_calculator: Callable[[float], MetricCalculatorLike]
    normalize_predictions: Callable[[np.ndarray], np.ndarray]
    predict_scores: Callable[[object, np.ndarray], np.ndarray | None]
```

No se usa herencia. Una tarea no tiene comportamiento propio que extender: es una
colección de decisiones. Un `dataclass` congelado las agrupa, impide que se modifiquen
en caliente y permite compararlas e imprimirlas sin escribir nada.

**Las cuatro responsabilidades van juntas por diseño.** Un registro suelto —un
diccionario de métricas por un lado y funciones de normalización por otro— es
exactamente lo que permite que una tarea nueva declare sus métricas y herede en
silencio la normalización binaria. Ese es el fallo documentado en la auditoría, y esta
agrupación es la que lo impide.

### 2. Un registro de módulo con dos tareas iniciales

```python
ANOMALY_DETECTION = Task(name="anomaly_detection", ...)
CLASSIFICATION    = Task(name="classification", ...)

TASKS: dict[str, Task] = {t.name: t for t in (ANOMALY_DETECTION, CLASSIFICATION)}

def get_task(task: str | Task) -> Task: ...
```

`get_task` acepta el nombre o la instancia, de modo que un usuario pueda registrar una
tarea propia sin tocar la librería.

| | `anomaly_detection` | `classification` |
|---|---|---|
| Métricas | `auc_roc`, `f1`, `precision`, `recall`, `false_positive_rate` | `macro_f1`, `accuracy`, `balanced_accuracy`, `macro_precision`, `macro_recall` |
| Métrica por defecto | `auc_roc` | `macro_f1` |
| Se minimizan | `false_positive_rate` | ninguna |
| Normalización de predicciones | la actual (binaria, convención −1/+1) | **identidad**: las etiquetas de clase se usan tal cual |
| Puntajes | `predict_proba[:, 1]` → `score_samples` → `decision_function` | `None`: macro-F1 no necesita puntajes |

Las métricas primarias son las que fija el anteproyecto (§4.3.2): AUC-ROC para
anomalías, macro-F1 para clasificación.

### 3. La tarea se selecciona en el constructor, nunca en `run()`

```python
GridSearchEngine(search_space, ..., task="classification")
OptunaSearchEngine(search_space, ..., task="classification")
ParallelRunner(..., task="classification")
```

El parámetro es opcional y su valor por defecto es `"anomaly_detection"`, de modo que
**todo el código existente conserva su comportamiento exacto**. La firma de `run(...)`
queda intacta, que es la restricción 2.

### 4. `validate_metric` pasa a ser relativa a la tarea

```python
def validate_metric(metric: str, task: Task | None = None) -> None
```

Con `task=None` valida contra las métricas de detección de anomalías, que es el
comportamiento actual. El parámetro es posicional-opcional, así que ninguna llamada
existente se rompe.

### 5. `MetricCalculator` no se toca

Sigue siendo la implementación de las métricas de detección de anomalías, con sus
mismas claves y su mismo comportamiento. La clasificación estrena un
`ClassificationMetricCalculator` con la misma forma de llamada
(`calculate(y_true, y_pred, y_score=None) -> dict`), de modo que el motor no distingue
entre ambos.

## Alternativas consideradas

**Una jerarquía de clases `BaseTask` con métodos abstractos.** Rechazada: una tarea no
tiene estado ni comportamiento que justifique herencia, y `CLAUDE.md` pide evitarla sin
motivo. El `dataclass` congelado da lo mismo con menos ceremonia.

**Un diccionario global `TASK_METRICS: dict[str, tuple[str, ...]]`.** Rechazada por la
razón central de este ADR: separa las métricas de la normalización de predicciones, que
es precisamente cómo se produce el fallo silencioso documentado en la auditoría.

**Añadir `task` como parámetro de `run(...)`.** Rechazada: cambia una firma pública
usada por los 134 tests y por los notebooks. El constructor logra lo mismo sin romper
nada.

**Inferir la tarea del número de clases de `labels`.** Rechazada: un problema binario de
clasificación y uno de detección de anomalías son indistinguibles por sus etiquetas, y
adivinar mal cambia silenciosamente las métricas reportadas. La tarea la declara quien
usa la librería.

## Consecuencias

**Positivas**

- Los tres consumidores (dos motores y `ParallelRunner`) comparten un único registro.
- Añadir agrupamiento en la semana 7, si entra, es declarar una tercera `Task`.
- Las cuatro decisiones dependientes de la tarea no pueden desincronizarse.
- Ninguna interfaz pública cambia: `run(...)`, las claves de `MetricCalculator` y
  `get_params()` quedan idénticas.

**Negativas y riesgos aceptados**

- Aparece un módulo nuevo (`autopipe/tasks/`) para algo que hoy son dos constantes. Se
  acepta porque es lo que evita el fallo silencioso.
- Guardar funciones dentro de un `dataclass` obliga a definirlas como funciones de
  módulo para que las tareas sigan siendo picklables, requisito del backend `loky` de
  `ParallelRunner`. Queda como restricción explícita de implementación.
- `_normalize_predictions` conserva su defecto conocido para el caso de anomalías (las
  predicciones todas `+1` se interpretan como «todo anómalo»). Corregirlo cambia
  métricas ya reportadas y se decide aparte de este ADR.

## Verificación

1. Los 134 tests de regresión pasan sin editarse: con el valor por defecto, el
   comportamiento es idéntico al actual.
2. Una prueba verifica que `metric="accuracy"` sigue siendo inválida en la tarea de
   anomalías y **es válida** en clasificación.
3. Una prueba con etiquetas multiclase `[0, 1, 2, 3, 4]` verifica que la tarea de
   clasificación **no** colapsa las clases, contrastando con el comportamiento
   documentado en la auditoría.
4. Una prueba verifica que un modelo de cinco clases produce métricas macro correctas y
   que no se descarta información de `predict_proba`.
5. Una prueba verifica que una `Task` es picklable, para que `ParallelRunner` la pueda
   enviar a los procesos de `loky`.

## Trabajo que habilita

- RE1: núcleo agnóstico completo, junto con el refactor de `evaluate_pipeline` de
  ADR-001.
- RE2: la clasificación tabular necesita `CLASSIFICATION` para poder evaluarse.
- El experimento del proyecto, cuyo modelo base congelado es un clasificador de
  $k$ vecinos y cuya métrica primaria es macro-F1.
