# Espacio de búsqueda tabular canónico — `tabular-v1`

- **Estado:** congelado el 2026-08-13 (semana 4)
- **Entregable:** RE1, RE5
- **Implementación:** `autopipe/search_space/tabular.py`
- **Prueba que lo vigila:** `tests/test_search_space.py::test_the_index_of_every_option_is_frozen`
- **Pendiente:** se registra en el pre-registro v0.2 (semana 5), como establece su
  Sección 3.2

Este documento fija la composición del espacio de búsqueda tabular. Es único para todos
los datasets del corpus y está versionado; las dos propiedades son requisitos del
arranque en caliente, no comodidades.

## Por qué versionado

Los parámetros que Optuna registra por ensayo son **índices enteros** sobre la lista de
opciones de cada paso, más un índice adicional que representa omitirlo. Un índice no
significa nada por sí mismo: depende del orden de la lista.

Si el espacio cambiara de claves o de orden entre datasets, una configuración inyectada
como arranque en caliente designaría transformaciones distintas de las previstas **sin
producir ningún error visible**: la búsqueda correría, daría un número, y ese número no
mediría lo que se cree. Comparar la versión convierte ese fallo mudo en un rechazo
explícito (`validate_space_version`).

**Cambiar cualquier cosa de la tabla de abajo obliga a incrementar
`TABULAR_SPACE_VERSION`.** Las configuraciones guardadas con la versión anterior dejan de
ser interpretables.

## Composición congelada

Cuatro pasos, en este orden de aplicación:

```
imputation → scaling → encoding → selection
```

| Paso | Índice | Transformación | Parámetros |
|---|---|---|---|
| `imputation` | 0 | `ImputeMissing` | `numeric_strategy='mean'` |
| `imputation` | 1 | `ImputeMissing` | `numeric_strategy='median'` |
| `scaling` | 0 | `ScaleNumeric` | `method='standard'` |
| `scaling` | 1 | `ScaleNumeric` | `method='minmax'` |
| `scaling` | 2 | `ScaleNumeric` | `method='robust'` |
| `encoding` | 0 | `EncodeCategorical` | `method='onehot'` |
| `encoding` | 1 | `EncodeCategorical` | `method='ordinal'` |
| `selection` | 0 | `SelectFeatures` | `fraction=0.25`, `criterion='f_classif'` |
| `selection` | 1 | `SelectFeatures` | `fraction=0.5`, `criterion='f_classif'` |
| `selection` | 2 | `SelectFeatures` | `fraction=0.75`, `criterion='f_classif'` |

A cada paso los motores de búsqueda le añaden **automáticamente** la opción de omitirse,
con el índice `len(opciones)`. El tamaño del espacio es entonces:

```
3 × 4 × 3 × 4 = 144 pipelines
```

Con los 50 ensayos por dataset que fija el anteproyecto, Optuna explora hasta un 35 % del
espacio.

## Justificación de las decisiones

### Orden: el escalado va antes de la codificación

`ScaleNumeric` solo toca columnas numéricas, así que colocarlo antes de la codificación
hace que actúe sobre las columnas originales y **deje las indicadoras one-hot en `0`/`1`**.

Si el escalado fuera después, estandarizar una indicadora de una categoría rara (digamos
con frecuencia 0,01) la llevaría a valores cercanos a ±10. Con el modelo base congelado
—vecinos más cercanos con distancia euclídea— esa única columna dominaría la distancia y
el resultado mediría la rareza de una categoría y no la calidad del preprocesamiento. Es
además la convención de `ColumnTransformer` en scikit-learn.

### La selección va al final

Cuando la selección se ejecuta, la tabla ya es enteramente numérica y la fracción se
aplica sobre el ancho ya codificado. `SelectFeatures` deja pasar intactas las columnas no
numéricas, de modo que la transformación no rompe si se la coloca en otra posición, pero
la posición canónica es la última.

### Parametrización relativa

Ningún parámetro es absoluto ni nombra columnas. La selección se expresa como **fracción
de columnas**, nunca como un número de columnas: conservar diez columnas significa algo
distinto en un dataset de doce características que en uno de cien, y el valor podría ser
inválido en el dataset receptor. Es la condición para que una configuración recomendada
sea ejecutable en un dataset que el recomendador nunca vio (pre-registro, Sección 3.1).

Lo vigila `tests/test_tabular_transforms.py::test_parameters_do_not_mention_columns`, que
exige que todo parámetro numérico esté en `(0, 1]`.

### Criterio de selección supervisado

`f_classif` (F de un ANOVA de una vía) es el único criterio del espacio canónico. La
clase `SelectFeatures` implementa además `variance`, que queda disponible para quien use
la librería pero **no** forma parte de `tabular-v1`.

### Claves en inglés

Los nombres de los pasos son identificadores públicos de la librería y siguen la
convención del resto del código (módulos y clases en inglés, documentación en español).

## Fracción del espacio no evaluable

Los motores generan candidatos con pasos omitidos, y algunos de esos candidatos son
inevaluables: omitir la codificación en un dataset con columnas categóricas deja una
tabla que el modelo no puede consumir. `evaluate_pipeline_safe` los registra con métricas
`NaN` y la búsqueda continúa, de modo que nunca ganan.

Medición sobre datasets sintéticos de 90 filas y 3 columnas (2026-08-13):

| Dataset | Candidatos `NaN` | Fracción |
|---|---|---|
| Solo numérico, sin faltantes | 0 / 144 | 0 % |
| Solo numérico, con faltantes | 48 / 144 | 33 % |
| Con categóricas, sin faltantes | 48 / 144 | 33 % |
| Con categóricas y faltantes | 80 / 144 | 56 % |

Reproducir con `scripts/` o con el fragmento registrado en la bitácora del 2026-08-13.

**Esto tiene consecuencias sobre el protocolo experimental y no debe darse por resuelto:**

1. **Sobre la variable dependiente del LODO.** «Ensayos hasta el objetivo» cuenta
   ensayos, incluidos los que caen en la región inevaluable. El arranque en frío gasta
   parte de sus ensayos aleatorios ahí; el arranque en caliente inyecta cinco
   configuraciones que son pipelines ganadores y por construcción no son `NaN`. Parte de
   la ventaja medida podría provenir de evitar la región inevaluable y no de acertar con
   el preprocesamiento. La ablación con recomendador aleatorio de la semana 11 controla
   esto en parte, porque el recomendador aleatorio sí puede caer en la región.
2. **Sobre el cribado de sensibilidad de la semana 5.** El pre-registro (Sección 4, paso
   2) extrae doce pipelines al azar del espacio canónico por dataset candidato y mide el
   IQR de macro-F1. En un dataset con categóricas y faltantes, más de la mitad de esos
   doce podrían ser `NaN`, y el IQR se calcularía sobre una muestra efectiva mucho menor
   que doce.

El punto 1 sigue abierto y anotado en las decisiones abiertas de `docs/Agenda.tex`, con
fecha límite en la semana 9 (pre-registro v1.0).

### Resolución del punto 2 (2026-08-14, antes de ejecutar el cribado)

**Se remuestrea hasta reunir doce pipelines evaluables.** Se sigue extrayendo al azar del
espacio canónico con la semilla fija `4242`, descartando los inevaluables, hasta juntar
doce válidos. El número de descartes se registra junto al IQR.

**Razón.** El cribado no solo mide: **ordena** los datasets candidatos por IQR y selecciona
los primeros. Con doce extracciones fijas, el tamaño efectivo de muestra variaría entre
doce (dataset numérico sin faltantes) y unos cinco (categóricas más faltantes), de modo que
el orden compararía estadísticas de precisión muy distinta, y un IQR sobre cinco valores es
ruidoso. Remuestrear mantiene `n = 12` en todos los datasets y hace comparables los IQR.

Es además coherente con la justificación que el propio pre-registro ya da para elegir el
IQR sobre el rango total: mide si **elegir entre** pipelines cambia el resultado. Un
pipeline que no se puede ejecutar no es una de las opciones entre las que se elige.

**Alternativas descartadas.** Contar los `NaN` como macro-F1 igual a cero inflaría el IQR
justo en los datasets con categóricas y faltantes, confundiendo «sensible al
preprocesamiento» con «tiene columnas categóricas» y sesgando la selección del corpus.
Restringir el muestreo a pipelines ejecutables por construcción —excluyendo la omisión de
imputación y codificación— haría que el cribado midiera un espacio distinto del que después
usa la búsqueda.

Esta decisión se toma **antes** de ejecutar el cribado y se registra en la versión v0.2 del
pre-registro, como exige su regla de oro.

## API

```python
from autopipe import (
    TABULAR_SPACE_VERSION,      # 'tabular-v1'
    TABULAR_SPACE_STEPS,        # ('imputation', 'scaling', 'encoding', 'selection')
    build_tabular_search_space, # instancias nuevas en cada invocación
    validate_space_version,     # rechaza configuraciones de otra versión
    describe_tabular_search_space,  # enumeración paso/índice, para docs y reportes
)
```

`build_tabular_search_space()` devuelve **instancias nuevas** cada vez. Las
transformaciones tabulares tienen estado, así que compartir instancias entre dos búsquedas
arrastraría lo aprendido de una hacia la otra.
