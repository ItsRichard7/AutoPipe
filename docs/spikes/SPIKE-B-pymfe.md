# Spike B — Viabilidad de `pymfe` como extractor de meta-características

- **Fecha:** 2026-08-06
- **Etapa:** E1 (análisis y diseño), habilita RE3 y RE4
- **Entorno:** pymfe 0.4.4, openml 0.15.1, pandas 3.0.5, NumPy 2.5.1, scikit-learn 1.9.0
- **Script:** `spikes/spike_b_pymfe.py` (determinista, sin datos externos ni red)
- **Reproducir:** `.\.venv\Scripts\python.exe spikes\spike_b_pymfe.py`

## Por qué se hizo

La semana 5 construye el extractor de meta-características (RE3) y la semana 6 lo
aplica al corpus (RE4). Antes de comprometer ese trabajo hacía falta saber si `pymfe`
funciona con el stack actual, cuánto cuesta, y cómo se comporta ante los datos
heterogéneos que anticipa el riesgo RI4 del anteproyecto.

## Compatibilidad

`pymfe 0.4.4` funciona sin incidencias con NumPy 2.5.1 y scikit-learn 1.9.0. La suite
de regresión del proyecto sigue en 164 pruebas tras la instalación.

Observación: la instalación arrastra un árbol de dependencias considerable
(`statsmodels`, `igraph`, `gower`, `pyarrow`, `minio`, `pycryptodome`, entre otras) y
actualiza pandas a la serie 3.0. No es un stack ligero.

## 1–2. Costo de extracción y escalado

Cuatro grupos, 20 características, tamaños crecientes. Segundos por extracción:

| Grupo | 500 | 1 000 | 2 000 | 4 000 | 8 000 | Exponente | Extrapolado a 50 000 |
|---|---|---|---|---|---|---|---|
| `general` | 0,01 | 0,01 | 0,01 | 0,03 | 0,05 | 0,60 | 0,1 s |
| `statistical` | 0,08 | 0,05 | 0,07 | 0,10 | 0,12 | 0,21 | 0,2 s |
| `info-theory` | 1,07 | 1,15 | 1,31 | 1,45 | 1,88 | 0,20 | 2,5 s |
| `complexity` | 0,20 | 0,71 | 2,42 | 11,01 | 38,65 | **1,91** | **21,6 min** |

El exponente 1,91 confirma el comportamiento cuadrático esperado de las medidas
basadas en distancias. **El grupo `complexity` concentra el 99 % del costo**: extraer
las meta-características de un solo dataset en el techo de 50 000 instancias que
permite el anteproyecto costaría unos 21,6 minutos, frente a menos de tres segundos
para todos los demás grupos juntos.

## 3–4. Estabilidad del vector y valores no finitos

| Caso | Meta-características | No finitas |
|---|---|---|
| Binario equilibrado | 107 | 14 (13,1 %) |
| Binario desbalanceado | 107 | 14 (13,1 %) |
| Multiclase, 5 clases | 107 | 7 (6,5 %) |
| Pocas características (4) | 107 | 14 (13,1 %) |

**El conjunto de nombres es idéntico en los cuatro casos**, que es la condición
necesaria para que el recomendador por vecinos compare vectores alineados. La
proporción de valores no finitos ronda el 13 % y **varía según el dataset**: hay
meta-características que existen para unos datasets y no para otros.

## 5. Determinismo

Dos extracciones sobre el mismo dataset difieren en **2 de 93** meta-características
finitas, `hubs.mean` y `hubs.sd`, con una diferencia relativa máxima de
**2,8 × 10⁻¹⁵**. Es ruido de coma flotante por orden de sumación, no aleatoriedad: los
nombres coinciden, los valores no finitos caen en las mismas posiciones y todo el
vector coincide dentro de una tolerancia de 10⁻⁹.

No compromete el criterio de reproducibilidad del anteproyecto. Persistir el vector
redondeado a una precisión fija lo vuelve estable bit a bit.

## 6. Datos heterogéneos — el riesgo RI4, materializado

Probando grupo por grupo sobre datos con un 10 % de valores faltantes:

| Grupo | Resultado |
|---|---|
| `general` | OK |
| `statistical` | OK |
| `info-theory` | OK |
| `complexity` | **`Exception` con mensaje vacío**, desde `complexity.py:443`, en `precompute_adjacency_graph` |

Las columnas categóricas (arreglo de tipo `object` con cadenas) **sí** se procesan sin
problema: 107 meta-características, misma proporción de no finitas.

**Imputar por la mediana antes de extraer resuelve el fallo por completo**: los cuatro
grupos se extraen sin excepción.

Dos consecuencias: el extractor debe imputar antes de invocar a pymfe, y debe envolver
la excepción, porque un `Exception` con mensaje vacío es indepurable sobre un corpus de
veinte datasets.

## 7. Fidelidad del submuestreo

Referencia de 8 000 instancias. Se excluyen las meta-características dependientes del
tamaño por definición (`attr_to_inst`, `inst_to_attr`, `nr_inst`, `t2`, `t3`) y las de
magnitud menor que 10⁻³, donde la desviación relativa no es informativa. Quedan 77
comparables:

| Muestra | Segundos | Correlación | Desviación mediana | p90 |
|---|---|---|---|---|
| 1 000 | 1,21 | 0,9889 | 11,8 % | 227,6 % |
| 2 000 | 2,48 | 0,9973 | 8,0 % | 90,6 % |
| 4 000 | 7,69 | 0,9992 | 5,3 % | 52,4 % |

El submuestreo preserva el ordenamiento del espacio de meta-características —la
correlación no baja de 0,99— pero **desplaza los valores de forma apreciable**. Dos
datasets solo son comparables entre sí si se les extrajeron las meta-características
con el mismo tamaño de muestra.

## Decisiones de diseño que se derivan

### Tope de submuestreo: 500 instancias, solo para `complexity`

El diseño queda determinado por dos hechos combinados: los criterios de inclusión del
anteproyecto (§4.3.1) admiten datasets desde **500** instancias, y el experimento 7
muestra que los valores no son comparables entre tamaños de muestra distintos. Un
tamaño **fijo e igual para todos los datasets** solo puede ser, por tanto, 500: es el
único valor que todo dataset elegible puede alcanzar.

La propuesta es entonces:

- **`general`, `statistical` e `info-theory`: sobre el dataset completo.** Son
  estimadores de cantidades poblacionales (medias, entropías, asimetrías) que
  *convergen* al crecer la muestra, de modo que más datos dan una mejor estimación de
  la misma cantidad y los valores siguen siendo comparables. Además cuestan menos de
  tres segundos incluso en el techo de 50 000 instancias, así que no hay nada que
  ahorrar.
- **`complexity`: sobre una submuestra aleatoria fija de 500 instancias,
  estratificada por clase y con semilla declarada.** Sus medidas son propiedades
  geométricas de la muestra concreta, no estimadores de una cantidad poblacional, y
  por eso cambian con el tamaño. Fijar el tamaño convierte un sesgo sistemático
  (confundido con el tamaño del dataset) en un ruido homogéneo, que es el intercambio
  correcto para una métrica de similitud.

El costo pasa de 21,6 minutos por dataset grande a **0,2 segundos**, o unos cuatro
segundos para un corpus de veinte datasets.

Contrapartida asumida: con 500 instancias las medidas de complejidad son más ruidosas
que con el dataset completo. Se acepta porque la alternativa —tamaño variable— cambia
ruido por sesgo, y un sesgo correlacionado con el tamaño haría que el recomendador
juzgue similares a datasets que solo comparten su número de filas.

### Meta-características no finitas

Alrededor del 13 % del vector no es finito, y qué posiciones lo son depende del
dataset. La propuesta es **descartar las columnas que resulten no finitas en algún
dataset del corpus**, decidido una sola vez cuando el corpus quede fijado y registrado
en el pre-registro v0.2, en lugar de imputar valores al nivel meta. El vector quedaría
en torno a 93 meta-características.

### Qué debe recoger el pre-registro v0.2

1. El tope de submuestreo de 500 instancias para `complexity`, su estratificación y su
   semilla.
2. Los grupos exactos de pymfe y la versión de la librería.
3. La regla de descarte de meta-características no finitas y la lista resultante.
4. La imputación por mediana previa a la extracción.

Ninguna de estas decisiones modifica la Sección 4.3 del anteproyecto.

## Limitaciones de este spike

- Los datasets son sintéticos. No se ejercitó la descarga desde OpenML, que queda para
  la semana 5 y puede traer sus propios problemas de formato y licencia.
- El escalado se midió hasta 8 000 instancias y el valor a 50 000 es una
  **extrapolación** sobre un ajuste log-log de cinco puntos, no una medición.
- Se midió con 20 características. El techo del anteproyecto es 100, y las medidas de
  complejidad también dependen de la dimensionalidad; el costo real podría ser mayor.
