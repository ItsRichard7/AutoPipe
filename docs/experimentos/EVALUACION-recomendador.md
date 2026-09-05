# Evaluación preliminar del recomendador (RE5)

> **Archivo generado.** No editar a mano: se regenera con
> `.\.venv\Scripts\python.exe scripts\evaluate_recommender.py`.

- **Generado:** 2026-09-05
- **Datasets evaluados:** 27, todos **fuera del corpus**
- **Configuraciones por recomendación:** 5
- **Versión del espacio:** `tabular-v1`
- **Valores completos:** `EVALUACION-recomendador.json`

## Dónde se evalúa, y por qué ahí

El corpus son 25 de los 52 candidatos que pasaron el cribado. Los otros
27 **no entran nunca al *leave-one-dataset-out***, de modo que
medir sobre ellos no contamina el experimento. Medir sobre los 25 sí lo haría:
son los pliegues, es decir, el conjunto de prueba.

El recomendador se ajusta con las 25 entradas completas y no con un pliegue,
porque los datasets consultados están fuera: no hay nada que excluir.

## Qué se mide

De cada dataset retenido se evalúan de forma exhaustiva los 144 pipelines del
espacio, lo que da su **óptimo exacto**. Después se comprueba dónde caen las
cinco configuraciones que cada estrategia recomienda, leyéndolas de esa misma
rejilla: la comparación entre estrategias es exacta y no cuesta cómputo extra.

- **Distancia al óptimo:** cuánto pierde la mejor de las cinco recomendaciones
  frente al mejor pipeline posible. Cero significa que la recomendación contiene
  el óptimo exacto.
- **Alcanza el objetivo:** si alguna de las cinco llega al
  95 % del óptimo, que es el umbral del protocolo.

**No se ejecuta ninguna búsqueda de Optuna ni se compara arranque frío contra
caliente.** Eso es el experimento, y producirlo antes de cerrar el pre-registro
v1.0 contaminaría el resultado.

## Las cuatro estrategias

- **`nearest`** toma las de los vecinos más cercanos en meta-características. Es
  la primaria, la que el anteproyecto fija como línea base.
- **`diverse`** toma, entre las candidatas cercanas, el subconjunto más separado
  entre sí. Ataca el modo de fallo que midió el spike A.
- **`frequent`** toma las que más veces ganan en la base, **ignorando por
  completo las meta-características**. Aísla la contribución del
  meta-aprendizaje: si la primaria no le gana, las meta-características no
  aportan nada.
- **`random`** toma las ganadoras de datasets elegidos al azar. Es la ablación
  del pre-registro: aísla si lo que aporta es la **similitud**.

Las cuatro devuelven ganadoras de datasets reales, de modo que la comparación no
se contamina con la calidad intrínseca de lo inyectado: lo único que cambia es
**de qué datasets** se toman.

## Resultado sobre los 27 retenidos

| Estrategia | Distancia mediana | Máxima | Óptimo exacto | Alcanza el objetivo | Inevaluables |
|---|---:|---:|---:|---:|---:|
| `nearest` | 0.0018 | 0.0589 | 8 / 27 | 24 / 27 | 1 |
| `diverse` | 0.0018 | 0.0589 | 8 / 27 | 24 / 27 | 3 |
| `frequent` | 0.0025 | 0.0640 | 7 / 27 | 24 / 27 | 0 |
| `random` | 0.0015 | 0.0762 | 10 / 27 | 25 / 27 | 3 |

### Desglosado por motivo de exclusión

Los dos grupos sesgan en direcciones opuestas y por eso van separados.

**Excluidos por orden (17 datasets).** Su IQR está por debajo
del mínimo del corpus, así que son **menos sensibles al preprocesamiento**: aquí
cualquier estrategia parece buena, y las diferencias entre ellas se comprimen.

| Estrategia | Distancia mediana | Máxima | Óptimo exacto | Alcanza el objetivo | Inevaluables |
|---|---:|---:|---:|---:|---:|
| `nearest` | 0.0018 | 0.0324 | 5 / 17 | 16 / 17 | 1 |
| `diverse` | 0.0014 | 0.0324 | 6 / 17 | 15 / 17 | 2 |
| `frequent` | 0.0017 | 0.0486 | 6 / 17 | 15 / 17 | 0 |
| `random` | 0.0009 | 0.0762 | 7 / 17 | 15 / 17 | 2 |

**Excluidos por regla de familia (10 datasets).** Comparten
origen con algún dataset del corpus, que sí está en la base del recomendador. El
sesgo va en la dirección contraria: **sobreestiman** lo que la similitud aporta,
porque el vecino ideal existe casi por construcción.

| Estrategia | Distancia mediana | Máxima | Óptimo exacto | Alcanza el objetivo | Inevaluables |
|---|---:|---:|---:|---:|---:|
| `nearest` | 0.0017 | 0.0589 | 3 / 10 | 8 / 10 | 0 |
| `diverse` | 0.0045 | 0.0589 | 2 / 10 | 9 / 10 | 1 |
| `frequent` | 0.0058 | 0.0640 | 1 / 10 | 9 / 10 | 0 |
| `random` | 0.0068 | 0.0330 | 3 / 10 | 10 / 10 | 1 |

## Por dataset

Distancia al óptimo de la mejor de las cinco recomendaciones. En **negrita** las
que alcanzan el objetivo.

| ID | Nombre | Motivo | IQR | Óptimo | `nearest` | `diverse` | `frequent` | `random` |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 22 | `mfeat-zernike` | familia | 0.0577 | 0.8191 | **0.0026** | **0.0026** | **0.0131** | **0.0026** |
| 32 | `pendigits` | familia | 0.0566 | 0.9906 | **0.0009** | **0.0009** | **0.0009** | **0.0000** |
| 14 | `mfeat-fourier` | familia | 0.0476 | 0.8709 | **0.0000** | **0.0065** | **0.0062** | **0.0000** |
| 23 | `cmc` | orden | 0.0433 | 0.5042 | 0.0324 | 0.0324 | 0.0356 | 0.0324 |
| 1461 | `bank-marketing` | orden | 0.0425 | 0.7107 | **0.0000** | **0.0193** | **0.0000** | **0.0223** |
| 1497 | `wall-robot-navigation` | orden | 0.0384 | 0.9224 | **0.0246** | **0.0000** | **0.0006** | **0.0000** |
| 1067 | `kc1` | familia | 0.0369 | 0.6797 | **0.0000** | **0.0000** | **0.0000** | **0.0145** |
| 1590 | `adult` | orden | 0.0359 | 0.8045 | **0.0056** | **0.0056** | **0.0056** | **0.0122** |
| 1050 | `pc3` | familia | 0.0311 | 0.6333 | 0.0589 | 0.0589 | 0.0640 | **0.0161** |
| 1480 | `ilpd` | orden | 0.0310 | 0.6340 | **0.0000** | **0.0000** | 0.0486 | **0.0000** |
| 1510 | `wdbc` | orden | 0.0301 | 0.9716 | **0.0097** | **0.0097** | **0.0097** | **0.0000** |
| 16 | `mfeat-karhunen` | familia | 0.0282 | 0.9649 | **0.0000** | **0.0000** | **0.0025** | **0.0000** |
| 182 | `satimage` | orden | 0.0271 | 0.8886 | **0.0014** | **0.0014** | **0.0017** | **0.0015** |
| 40966 | `MiceProtein` | orden | 0.0241 | 0.9716 | **0.0042** | **0.0000** | **0.0004** | **0.0009** |
| 1063 | `kc2` | familia | 0.0227 | 0.6883 | 0.0393 | **0.0236** | **0.0096** | **0.0236** |
| 28 | `optdigits` | familia | 0.0225 | 0.9903 | **0.0009** | **0.0009** | **0.0009** | **0.0009** |
| 1049 | `pc4` | familia | 0.0220 | 0.7511 | **0.0277** | **0.0277** | **0.0277** | **0.0330** |
| 469 | `analcatdata_dmft` | orden | 0.0207 | 0.2329 | **0.0000** | 0.0252 | **0.0000** | 0.0762 |
| 1053 | `jm1` | familia | 0.0180 | 0.6076 | **0.0162** | **0.0182** | **0.0055** | **0.0109** |
| 1475 | `first-order-theorem-proving` | orden | 0.0173 | 0.4819 | **0.0010** | **0.0010** | **0.0112** | **0.0000** |
| 458 | `analcatdata_authorship` | orden | 0.0154 | 0.9959 | **0.0000** | **0.0000** | **0.0000** | **0.0000** |
| 44 | `spambase` | orden | 0.0152 | 0.9032 | **0.0048** | **0.0048** | **0.0076** | **0.0000** |
| 1487 | `ozone-level-8hr` | orden | 0.0143 | 0.7040 | **0.0269** | **0.0000** | **0.0269** | **0.0000** |
| 3 | `kr-vs-kp` | orden | 0.0110 | 0.9624 | **0.0000** | **0.0000** | **0.0000** | **0.0031** |
| 40984 | `segment` | orden | 0.0072 | 0.9124 | **0.0115** | **0.0115** | **0.0000** | **0.0115** |
| 15 | `breast-w` | orden | 0.0049 | 0.9606 | **0.0000** | **0.0000** | **0.0000** | **0.0081** |
| 4534 | `PhishingWebsites` | orden | 0.0027 | 0.9600 | **0.0018** | **0.0018** | **0.0018** | **0.0000** |

## Qué dice el resultado

Las cuatro estrategias quedan **prácticamente empatadas**. La distancia
mediana al óptimo va de 0.0015 con `random` a
0.0025 con `frequent`: una separación de 0.0010 de
macro-F1. Y entre 24 y 25 de los
27 datasets alcanzan el objetivo **con cualquiera de las cuatro**.

**Este conjunto no puede distinguir las estrategias, y eso es un resultado sobre
la medición y no sobre las estrategias.** Es el mismo efecto piso que el análisis
de la semana 7 midió sobre el corpus: cuando una fracción grande del espacio
alcanza el 95 % del óptimo, acertar deja de ser difícil y cualquier recomendación
plausible parece buena.

Conviene decir el dato incómodo con todas las letras. Sobre los
17 excluidos por orden, la ablación aleatoria queda
por delante de la estrategia primaria en distancia mediana.
El único grupo donde la primaria queda por delante es el de los
10 excluidos por familia, que es justo el grupo cuyo sesgo la
favorece: el vecino ideal está en la base casi por construcción.

**Lo que sí queda comprobado** es la sensatez mecánica: las cuatro estrategias
producen configuraciones válidas, ejecutables casi siempre y muy cercanas al
óptimo del espacio, con un número marginal de recomendaciones inevaluables. Eso
era lo que este punto del plan tenía que verificar.

**Lo que anticipa del experimento.** Si el corpus se comporta como este conjunto,
el *leave-one-dataset-out* tendrá el mismo problema de discriminación. Refuerza
la decisión abierta sobre la región inevaluable y sobre la variable «ensayos
hasta el objetivo» que la v1.0 del pre-registro tiene que cerrar.

## Cómo leer esto, y cómo no

Es una **comprobación de sensatez**, no una medida del efecto que el experimento
va a medir. Tres limitaciones que conviene tener delante:

1. **Ninguno de los dos grupos es un sustituto del corpus.** Uno sesga hacia
   abajo y el otro hacia arriba, y ninguno de los dos se parece en sensibilidad
   al conjunto sobre el que se ejecutará el *leave-one-dataset-out*.
2. **Mide la calidad de lo inyectado, no el ahorro de la búsqueda.** La
   hipótesis del proyecto es sobre *ensayos hasta el objetivo*, que depende
   además de cómo el muestreador aprovecha lo inyectado; el spike A midió que
   inyectar bien no garantiza buscar mejor.
3. **No se usó para elegir nada.** El valor de *k*, la métrica de distancia y el
   escalado quedaron fijados a priori antes de esta medición, y esta medición no
   los revisa.
