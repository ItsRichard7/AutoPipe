# Diversidad de los pipelines ganadores (RE4)

> **Archivo generado.** No editar a mano: se regenera con
> `.\.venv\Scripts\python.exe scripts\analyze_winner_diversity.py`.

- **Generado:** 2026-09-03
- **Datasets:** 25
- **Versión del espacio:** `tabular-v1` (144 pipelines)
- **Semilla:** 42 — la misma de la base, de modo que la partición coincide
- **Valores completos:** `DIVERSIDAD-valores.json`

**Análisis exploratorio.** El pre-registro lo declara como tal: informa el diseño
del recomendador y **no puede sostener la hipótesis** del proyecto.

## 1. Forma de la diversidad

**17 configuraciones distintas entre 25 datasets.**
El conteo dice que hay diversidad; esta sección dice de qué tipo.

### Distribución del índice ganador por paso

La entropía normalizada vale 1 cuando todas las opciones del paso ganan por igual
y 0 cuando siempre gana la misma. Un paso de entropía alta discrimina entre
datasets; uno de entropía baja aporta poco a la identidad de una recomendación.

| Paso | Índice ganador: veces | Entropía normalizada |
|---|---|---:|
| `imputation` | `0`: 19, `1`: 5, `2`(omitir): 1 | 0.600 |
| `scaling` | `0`: 7, `1`: 7, `2`: 9, `3`(omitir): 2 | 0.925 |
| `encoding` | `0`: 16, `1`: 9, `2`(omitir): 0 | 0.595 |
| `selection` | `0`: 6, `1`: 3, `2`: 7, `3`(omitir): 9 | 0.953 |

### Distancia entre ganadores

Distancia de Hamming sobre los cuatro pasos, en los 300 pares
de datasets. Media **2.3633** de un máximo de 4.

Pares por distancia — 0: 10, 1: 35, 2: 112, 3: 122, 4: 21

Importa porque el modo de fallo que midió el spike A es inyectar cinco
configuraciones parecidas entre sí y todas equivocadas: TPE las lee como una
región buena y deja de explorar. Cuanto más se parezcan entre sí los ganadores de
datasets vecinos, más probable es ese modo de fallo y más razón hay para exigir
diversidad entre las cinco configuraciones recomendadas.

## 2. Anchura de la meseta

Los 144 pipelines del espacio evaluados de forma exhaustiva sobre cada
dataset, con la misma semilla y la misma partición que construyeron la base. La
columna «meseta» cuenta cuántos alcanzan el 95 % del
óptimo, que es **la definición del objetivo del protocolo**: cuantos más haya,
más fácil le resulta al arranque en frío tropezarse con uno al azar y menos
margen le queda al arranque en caliente.

- Mediana de configuraciones en meseta: **36 de 144**
- Rango: de 3 a 108
- La configuración que la base guarda como ganadora alcanza el óptimo exacto del
  espacio en **23 de 25** datasets

### El óptimo casi nunca es único

La columna «empatan» cuenta cuántas configuraciones del espacio alcanzan
**exactamente** el óptimo. Mediana **9**, máximo 72,
y solo **1 de 25** datasets tienen un óptimo único.

Es el hallazgo que más condiciona el recomendador: cuando varias configuraciones
empatan, la que la base guarda es **un miembro cualquiera del empate**, elegido
por el orden en que el muestreador las visitó. El recomendador estaría aprendiendo
a predecir esa elección arbitraria y no una propiedad del dataset, y dos datasets
gemelos podrían aparecer con ganadores distintos sin que nada los distinga.

La columna «pares» dice de qué naturaleza es el empate. Cuenta cuántos pares
distintos *(escalado, selección)* aparecen entre las configuraciones empatadas:

* **Un solo par** significa que el empate es **degeneración estructural del
  espacio**, no un empate de la métrica. Sobre un dataset numérico y sin
  faltantes, la imputación y la codificación son la identidad, de modo que las
  3 × 3 = 9 combinaciones de esos dos pasos son literalmente el mismo pipeline.
  El dato no dice que dé igual preprocesar: dice que en ese dataset esos dos
  pasos no tienen nada que hacer.
* **Varios pares** significa que pipelines de verdad distintos alcanzan el mismo
  valor.

Ocurre en **18 de 25** datasets que todo
el empate se explica por un solo par.

| ID | Nombre | Eval. | Óptimo | Empatan | Pares | Base ópt. | Meseta | % esp. | E |
|---:|---|---:|---:|---:|---:|:---:|---:|---:|---:|
| 40983 | `wilt` | 144 | 0.8859 | 9 | 1 | sí | 18 | 12.5 % | 8.0 |
| 38 | `sick` | 64 | 0.9383 | 2 | 1 | **no** | 14 | 9.7 % | 10.3 |
| 18 | `mfeat-morphological` | 144 | 0.7415 | 9 | 1 | **no** | 54 | 37.5 % | 2.7 |
| 40975 | `car` | 96 | 0.9108 | 12 | 4 | sí | 12 | 8.3 % | 12.0 |
| 50 | `tic-tac-toe` | 96 | 0.9648 | 12 | 4 | sí | 24 | 16.7 % | 6.0 |
| 40982 | `steel-plates-fault` | 144 | 0.7725 | 9 | 1 | sí | 27 | 18.8 % | 5.3 |
| 188 | `eucalyptus` | 64 | 0.6195 | 2 | 1 | sí | 11 | 7.6 % | 13.1 |
| 29 | `credit-approval` | 64 | 0.8536 | 2 | 1 | sí | 26 | 18.1 % | 5.5 |
| 11 | `balance-scale` | 144 | 0.5906 | 27 | 3 | sí | 36 | 25.0 % | 4.0 |
| 1068 | `pc1` | 144 | 0.7669 | 9 | 1 | sí | 27 | 18.8 % | 5.3 |
| 41027 | `jungle_chess_2pcs_raw_endgame_complete` | 144 | 0.6995 | 9 | 1 | sí | 72 | 50.0 % | 2.0 |
| 1464 | `blood-transfusion-service-center` | 144 | 0.6637 | 9 | 1 | sí | 54 | 37.5 % | 2.7 |
| 40701 | `churn` | 144 | 0.8023 | 3 | 1 | sí | 30 | 20.8 % | 4.8 |
| 1462 | `banknote-authentication` | 144 | 1.0000 | 72 | 8 | sí | 72 | 50.0 % | 2.0 |
| 54 | `vehicle` | 144 | 0.7305 | 9 | 1 | sí | 36 | 25.0 % | 4.0 |
| 46 | `splice` | 96 | 0.8126 | 12 | 4 | sí | 48 | 33.3 % | 3.0 |
| 40994 | `climate-model-simulation-crashes` | 144 | 0.7353 | 18 | 2 | sí | 45 | 31.2 % | 3.2 |
| 1489 | `phoneme` | 144 | 0.8307 | 9 | 1 | sí | 108 | 75.0 % | 1.3 |
| 6332 | `cylinder-bands` | 64 | 0.8466 | 1 | 1 | sí | 3 | 2.1 % | 48.0 |
| 31 | `credit-g` | 96 | 0.6754 | 3 | 1 | sí | 30 | 20.8 % | 4.8 |
| 23381 | `dresses-sales` | 96 | 0.6458 | 2 | 1 | sí | 4 | 2.8 % | 36.0 |
| 4538 | `GesturePhaseSegmentationProcessed` | 144 | 0.5799 | 9 | 1 | sí | 63 | 43.8 % | 2.3 |
| 151 | `electricity` | 144 | 0.8364 | 6 | 1 | sí | 66 | 45.8 % | 2.2 |
| 1494 | `qsar-biodeg` | 144 | 0.8582 | 18 | 2 | sí | 54 | 37.5 % | 2.7 |
| 37 | `diabetes` | 144 | 0.6922 | 9 | 1 | sí | 108 | 75.0 % | 1.3 |

### Efecto piso sobre «ensayos hasta el objetivo»

La columna **E** es el número esperado de ensayos **aleatorios** hasta caer en la
meseta, es decir 144 dividido entre las configuraciones que la alcanzan.

- Mediana de E: **4.0**
- Datasets con E ≤ 2: **4** — el arranque en frío
  acierta casi de inmediato
- Datasets con E ≥ 10: **5**

Con `n_startup_trials` en 10, el arranque en frío suele cruzar el objetivo
**dentro de su propia fase aleatoria**, antes de que TPE empiece a modelar. Y una
reducción del 30 % sobre una mediana de 4.0 ensayos exige bajar a
2.8, con los cinco ensayos inyectados contando dentro.

**E es una cota conservadora, y lo es en la dirección que perjudica al arranque en
caliente.** El objetivo del protocolo no es el 95 % del óptimo del espacio sino el
95 % del mejor valor que el propio arranque en frío alcanza en treinta ensayos,
que es menor o igual. El conjunto de configuraciones que califican es por tanto
mayor que la meseta medida aquí, y el frío acierta aún antes.

Es un análisis exploratorio y **no es un resultado del experimento**: describe el
espacio, no mide las dos condiciones. Su lugar es la decisión abierta sobre la
región inevaluable y la variable «ensayos hasta el objetivo», que la v1.0 del
pre-registro debe cerrar antes de ejecutar nada.

### Desempate entre configuraciones que empatan

La base guarda, entre las que empatan en el mejor valor, **la que omite menos
pasos y, a igualdad, la de índices menores**, y no la que el muestreador visitó
primero. Sin esa regla, la configuración guardada sería una propiedad de la
corrida y no del dataset.

El desempate se calcula **entre los empates observados en los 50 ensayos**,
no sobre la rejilla exhaustiva. Elegir el ganador mirando los 144 pipelines
equivaldría a construir la base por búsqueda exhaustiva, y el anteproyecto fija
50 ensayos por dataset. Mediana de configuraciones empatadas dentro del
presupuesto: **4**.

**La base ya es canónica**: ninguna de sus entradas cambiaría al
aplicar la regla. La comprobación no es una formalidad, porque el
desempate lo aplica el arnés de construcción y esta tabla se calcula por
separado a partir del artefacto guardado: si los dos se separaran, la
base mezclaría dos criterios sin que nada fallara.

| | En la base | Canónica |
|---|---:|---:|
| Entradas que difieren | — | 0 de 25 |
| Configuraciones distintas | 17 | 17 |
| Omiten la imputación | 1 | 1 |
| Omiten la codificación | 0 | 0 |

Las dos últimas filas son las que importan para el arranque en caliente: un
ganador que omite la imputación es **inevaluable** en un dataset receptor con
valores faltantes, y uno que omite la codificación lo es en uno con columnas
categóricas. Cada recomendación así gastaría un ensayo del presupuesto sin
producir una métrica.

Como referencia, desempatar sobre la rejilla exhaustiva —que **no** es
implementable dentro del presupuesto— dejaría 13 configuraciones distintas,
con 0 omisiones de imputación y 0 de codificación. Es la
cota de lo que el desempate puede llegar a corregir.

La contrapartida hay que decirla: la regla reduce las recomendaciones que caen en
la región inevaluable, y eso **refuerza** la asimetría que el pre-registro dejó
abierta, porque el arranque en frío sí gasta ensayos ahí. Debe declararse en la
v1.0 y controlarse con la ablación de recomendador aleatorio.

## 3. Estabilidad de las meta-características supervivientes

El pre-registro v0.2 dice que el descarte de no finitas se calcula **una sola vez
sobre el corpus definitivo**; el código de `autopipe/meta_knowledge/` lo recalcula
**por pliegue**. Las dos reglas no pueden gobernar a la vez el
*leave-one-dataset-out*, y la diferencia entre ellas es medible.

- Supervivientes sobre el corpus completo: **88**
- Pliegues cuyo conjunto difiere: **1 de 25**

**Las dos reglas no coinciden.** 1 de 25 pliegue conserva
meta-características que el corpus completo descarta:

- Pliegue que excluye `blood-transfusion-service-center`: conserva 2 de más
  (`n2.mean`, `n2.sd`), de las cuales 2 son no finitas
  en el propio dataset excluido y caerían al intersecar.

La regla por pliegue es la que no mira el dataset excluido, pero exige
la intersección con las finitas de la consulta; la del corpus completo
usa el patrón de finitud del excluido para decidir qué columnas existen.
