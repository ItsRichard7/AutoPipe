# ADR-001 — Extensión del contrato de transformación con semántica `fit`/`transform`

- **Estado:** aceptado
- **Fecha:** 2026-08-05
- **Etapa:** E1 (análisis y diseño de la generalización), entregable RE1
- **Decide:** Ricardo Antonio Borbón Mena
- **Afecta a:** `autopipe/pipeline/base.py`, `autopipe/pipeline/pipeline.py`,
  `autopipe/search_engine/grid_search.py`

## Contexto

`AutoPipeAD` fue construido para detección de anomalías en imágenes. Su contrato de
transformación, definido en `BaseTransform`, expone una sola operación:

```python
@abstractmethod
def apply(self, image: np.ndarray) -> np.ndarray: ...
```

Esa operación es **por muestra y sin estado**: transforma una imagen a la vez y su
resultado no depende de ninguna otra muestra del dataset. Coherente con ello, el
protocolo de evaluación transforma el dataset completo *antes* de partirlo en
entrenamiento y validación (`grid_search.evaluate_pipeline`):

```python
features = _transform_to_matrix(pipeline, images)   # transforma todo
model.fit(features[train_idx], labels[train_idx])   # y solo después parte
```

El TFG añade soporte de datos tabulares (RE2), cuyas transformaciones de
preprocesamiento son **con estado**: el escalado necesita la media y la desviación
de las columnas, la imputación necesita la media o la moda, la codificación necesita
el catálogo de categorías, y la selección de características necesita una medida de
relevancia. Todas esas cantidades deben calcularse **únicamente con el subconjunto
de entrenamiento**.

## Problema

Implementar las transformaciones tabulares como simples subclases del contrato
actual produciría **fuga de datos**: al ajustarse sobre el dataset completo, las
estadísticas del subconjunto de validación entrarían en la transformación aplicada
al entrenamiento. El desempeño medido quedaría optimistamente sesgado y los
experimentos de validación *leave-one-dataset-out* del proyecto —cuya conclusión es
la hipótesis central del TFG— serían inválidos.

El contrato actual no ofrece ningún punto donde una transformación pueda aprender de
un subconjunto y aplicar lo aprendido a otro.

## Restricciones

1. Los 134 tests de regresión de la librería base deben permanecer en verde, sin
   editarse (criterio de éxito de ingeniería del anteproyecto, §4.3.5; mitigación
   del riesgo RI3).
2. Las interfaces públicas ya en uso no pueden cambiar de firma: `BaseTransform.apply`,
   la firma `run(...)` de los motores de búsqueda, y las claves de `get_params()`.
3. Los notebooks de `examples/` usan `Pipeline.apply` directamente.
4. Las siete transformaciones de imagen existentes no deben requerir modificación.

## Decisión

Se extiende el contrato con **tres piezas**, de modo que nada de lo existente cambie
de firma ni de comportamiento.

### 1. `BaseTransform` gana `fit` y `transform` como métodos concretos

```python
class BaseTransform(ABC):
    @abstractmethod
    def apply(self, sample): ...        # sin cambios, sigue abstracto
    @abstractmethod
    def get_params(self) -> dict: ...   # sin cambios, sigue abstracto

    def fit(self, X, y=None) -> "BaseTransform":
        """Por defecto no aprende nada: la transformación es sin estado."""
        return self

    def transform(self, X):
        """Por defecto aplica `apply` muestra por muestra."""
        return [self.apply(sample) for sample in X]
```

Los métodos nuevos son **concretos y no abstractos**. Ésa es la decisión que
preserva la compatibilidad: las siete transformaciones de imagen y las clases
auxiliares de los tests (`Identity`, `Zeroer`) heredan un `fit` que no hace nada y
un `transform` que reproduce exactamente el comportamiento anterior.

`transform` opera **por lote**: recibe el conjunto de muestras y devuelve el conjunto
transformado. Es obligatorio que sea así, porque una transformación con estado no
puede calcular una media viendo una sola fila. `apply` conserva su semántica por
muestra.

### 2. Una subclase `StatefulTransform` para las transformaciones con estado

Se implementa como **método plantilla**: la subclase escribe solo su lógica, en los
métodos abstractos `_fit` y `_transform`, mientras que los métodos públicos `fit` y
`transform` son concretos y se encargan de registrar que el ajuste ocurrió y de
rechazar cualquier transformación previa al ajuste con `NotFittedError`.

Se prefiere esta variante a declarar `fit` y `transform` directamente abstractos
porque una subclase no puede olvidarse de marcar el estado de ajuste ni de validarlo:
esas dos responsabilidades quedan fuera de su alcance. El costo es un nivel de
indirección y dos nombres con guion bajo que la subclase debe conocer.

El estado de ajuste vive en un **atributo de clase** (`_fitted: bool = False`) y no se
inicializa en `__init__`. Así, una subclase que defina su propio `__init__` sin
invocar al de la clase base sigue levantando `NotFittedError` en lugar de un
`AttributeError` confuso.

Su `apply` delega en `_transform` sobre una vista de una sola fila, de modo que
satisface el método abstracto heredado de `BaseTransform` sin inventar semántica
nueva.

### 3. `Pipeline` gana `fit`, `transform` y `fit_transform`

El ajuste es **secuencial**: cada paso se ajusta sobre la salida del paso anterior.
`Pipeline.apply` se conserva intacto.

### 4. El protocolo de evaluación ajusta solo con entrenamiento

```python
pipeline = copy.deepcopy(pipeline)
X_train = [images[i] for i in train_idx]
X_val   = [images[i] for i in val_idx]
features_train = _to_matrix(pipeline.fit_transform(X_train, labels[train_idx]))
features_val   = _to_matrix(pipeline.transform(X_val))
```

La firma de `evaluate_pipeline` no cambia. Para pipelines de imagen el resultado
numérico es idéntico al anterior: las mismas transformaciones sobre las mismas
muestras, calculadas en dos lotes en lugar de uno.

### 5. El pipeline se clona antes de ajustarse

Los motores de búsqueda construyen cada candidato tomando instancias del espacio de
búsqueda (`grid_search._build_pipelines`, `optuna_search.objective`), de modo que
**la misma instancia de transformación aparece en muchos candidatos**. Con
transformaciones sin estado eso es inofensivo; en cuanto `fit` guarde estado en el
objeto, un candidato contaminaría a los siguientes, y bajo `ParallelRunner` con el
backend `loky` habría además una condición de carrera entre procesos.

Por lo tanto `evaluate_pipeline` clona el pipeline con `copy.deepcopy` antes de
ajustarlo, por la misma razón por la que ya clona el modelo base. El costo es
despreciable: las transformaciones solo almacenan parámetros de configuración.

## Alternativas consideradas

**Declarar `fit`/`transform` abstractos en `BaseTransform`.** Rechazada: obligaría a
modificar las siete transformaciones de imagen y las clases auxiliares de los tests,
y rompería cualquier transformación personalizada que un usuario ya haya escrito
contra el contrato publicado.

**Reemplazar `apply` por `transform`.** Rechazada: cambia una interfaz pública en uso
por los notebooks y por los 134 tests, sin ganancia funcional; `apply` por muestra
sigue siendo la operación natural para imágenes.

**Un contrato separado para lo tabular, sin relación con `BaseTransform`.**
Rechazada: obligaría a duplicar `Pipeline` y los motores de búsqueda, lo que
contradice el principio de evolución sobre reinvención del anteproyecto y anularía
la reutilización que justifica el proyecto.

**Ajustar cada transformación con el dataset completo y aceptar la fuga.**
Rechazada: invalidaría los experimentos.

## Consecuencias

**Positivas**

- Las transformaciones tabulares con estado pueden implementarse sin fuga de datos.
- Ninguna interfaz pública cambia; los 134 tests siguen siendo válidos como red de
  regresión.
- `isinstance(paso, StatefulTransform)` permite verificar si un pipeline requiere
  ajuste.
- El clonado por candidato elimina una contaminación entre candidatos y una
  condición de carrera que, aunque hoy latentes, se volverían reales con RE2.

**Negativas y riesgos aceptados**

- Una subclase con estado que herede directamente de `BaseTransform` y olvide
  implementar `fit` heredará silenciosamente el no-op. Lo mitiga `StatefulTransform`,
  que declara ambos métodos abstractos, y la documentación del contrato.
- Existen dos caminos para transformar (`apply` por muestra y `transform` por lote),
  lo que exige mantener la equivalencia entre ambos para las transformaciones sin
  estado. Se cubre con una prueba explícita.
- `ReportGenerator.rebuild_pipeline` reconstruye un pipeline **sin ajustar** a partir
  de su configuración. Es correcto —el ajuste depende de los datos, no de la
  configuración— pero para datos tabulares el usuario debe ajustar el pipeline
  reconstruido antes de transformar. Queda documentado en el método.
- Los motores de búsqueda siguen devolviendo el pipeline **sin ajustar**, igual que
  hoy. Devolver el ajustado sería más útil para el usuario final, pero cambia el
  significado del valor de retorno; se revisará al implementar RE2.

## Verificación

La decisión se considera correctamente implementada cuando:

1. Los 134 tests de regresión pasan sin haber sido editados.
2. Una prueba verifica que, para las siete transformaciones de imagen,
   `transform(X)` es igual a aplicar `apply` a cada muestra.
3. Una prueba verifica que transformar sin ajustar una transformación con estado
   levanta un error específico.
4. **Prueba anti-fuga:** con un dataset cuyos subconjuntos de entrenamiento y
   validación tienen escalas deliberadamente distintas, una transformación con estado
   ajustada solo con entrenamiento produce una validación que *no* queda centrada, lo
   que demuestra que se usaron las estadísticas del entrenamiento. La prueba debe
   fallar si se invierte el orden de ajuste y partición.
5. **Prueba de contaminación:** dos candidatos que comparten la misma instancia con
   estado, evaluados en secuencia, dan el mismo resultado que evaluados de forma
   aislada. La prueba debe fallar si se quita el `copy.deepcopy`.

## Referencias

- Anteproyecto TFG, §3.2 (mapa de reutilización y extensión) y §4.3.5 (criterios de
  éxito de ingeniería).
- Pre-registro del protocolo experimental, v0.1 (etiqueta `prereg-v0.1`).
