# Resultados del experimento 2 (RE6)

> **Archivo generado.** No editar a mano: se regenera con
> `.\.venv\Scripts\python.exe scripts\report_lodo_e2.py`.

- **Generado:** 2026-09-08
- **Datos:** `docs/experimentos/LODO-E2-resultados.json`
- **Duración de la ejecución:** 133.1 min
- **Conjuntos de datos:** 42 · **Corridas por condición:**
  420

## Veredicto

La hipótesis del trabajo se juega en los **dos contrastes confirmatorios**, y solo en
ellos. El primero pregunta si el arranque en caliente meta-aprendido reduce el costo
de la búsqueda; el segundo, si la **similitud** aporta algo por encima de un buen
valor por defecto.

1. `nearest` contra `frio`: regret mediano 0.0090 contra 0.0058. **Cumple el criterio**: reducción del 35.8 % —se exige 30 %— con un valor p ajustado por Holm de 0.0019, por debajo de 0,05.

2. `nearest` contra `frequent`: regret mediano 0.0039 contra 0.0058. **No cumple el criterio**: la reducción es del -49.1 % y se exige un 30 %; el valor p ajustado por Holm es 0.9690 —crudo 0.9690— y se exige menos de 0,05.

- `nearest` contra `frio`: gana en **32**, empata en **0** y pierde en **10** de 42.
- `nearest` contra `frequent`: gana en **20**, empata en **0** y pierde en **22** de 42.

## Protocolo ejecutado

- **Pre-registro:** experimento 2, v2.0, etiqueta prereg-v2.0
- **Espacio de búsqueda:** tabular-v1 (144 pipelines, sin cambios)
- **Modelo base:** KNeighborsClassifier(n_neighbors=5, metric=minkowski, p=2, brute)
- **Métrica:** macro-F1, partición estratificada con 20 % de validación
- **Corpus:** 42 conjuntos (25 del experimento 1 + 17; los 10 de familia siguen fuera)
- **Semillas:** 101, 202, 303, 404, 505, 606, 707, 808, 909, 1010
- **Presupuesto:** 60 ensayos por condición; censura en 61
- **Objetivo:** 99 % del óptimo exhaustivo por pareja (conjunto, semilla)
- **Variable primaria:** regret normalizado sobre la curva (menor es mejor)
- **Variable secundaria:** ensayos hasta el objetivo, con el criterio del anteproyecto
- **Contrastes confirmatorios:** nearest contra frio y nearest contra frequent
- **Corrección:** Holm-Bonferroni sobre la familia de dos, α = 0,05

## Contrastes confirmatorios

Menor regret es mejor. El valor $p$ que decide es el **ajustado por Holm**: son dos
contrastes declarados y juzgarlos a 0,05 sin corregir inflaría la probabilidad de
declarar favorable alguno por azar. El efecto es la correlación biserial por rangos
para muestras pareadas, negativa cuando la condición reduce el regret.

| Contraste | Regret referencia | Regret condición | Reducción | p (1 cola) | p (Holm) | Efecto | Empates | Cumple |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `nearest` vs `frio` | 0.0090 | 0.0058 | 35.8 % | 0.0010 | 0.0019 | -0.537 | 0 | sí |
| `nearest` vs `frequent` | 0.0039 | 0.0058 | -49.1 % | 0.9690 | 0.9690 | 0.329 | 0 | no |

## Análisis secundarios

Declarados como tales en el pre-registro. **No sustituyen a los confirmatorios** y no
llevan corrección por multiplicidad, porque no se usan para decidir sobre la
hipótesis. Los dos primeros repiten los contrastes confirmatorios sobre la variable
del experimento 1, que es lo que permite comparar los dos experimentos en los mismos
términos.

| Contraste | Variable | Mediana referencia | Mediana condición | Reducción | p (1 cola) | Efecto |
|---|---|---:|---:|---:|---:|---:|
| `nearest` vs `frio` | ensayos | 7.2 | 4.2 | 41.4 % | 0.3902 | -0.050 |
| `nearest` vs `frequent` | ensayos | 4.2 | 4.2 | 0.0 % | 0.9656 | 0.349 |
| `hybrid` vs `frequent` | regret | 0.0039 | 0.0042 | -7.1 % | 0.9641 | 0.331 |
| `hybrid` vs `nearest` | regret | 0.0058 | 0.0042 | 28.2 % | 0.3559 | -0.066 |
| `random` vs `frio` | regret | 0.0090 | 0.0028 | 68.7 % | 0.0000 | -0.783 |
| `frequent` vs `frio` | regret | 0.0090 | 0.0039 | 57.0 % | 0.0000 | -0.741 |

## Robustez al valor de censura

El regret **no se censura**, de modo que el análisis primario es inmune a esta regla.
Estas filas recalculan solo la variable secundaria con los castigos más severos que
el pre-registro fijó antes de ejecutar.

| Contraste | Censura | Mediana referencia | Mediana condición | Reducción | p (1 cola) |
|---|---:|---:|---:|---:|---:|
| `nearest` vs `frio` | 120 | 7.2 | 4.2 | 41.4 % | 0.3852 |
| `nearest` vs `frio` | 600 | 7.2 | 4.2 | 41.4 % | 0.3852 |
| `nearest` vs `frequent` | 120 | 4.2 | 4.2 | 0.0 % | 0.9656 |
| `nearest` vs `frequent` | 600 | 4.2 | 4.2 | 0.0 % | 0.9656 |

## Corridas censuradas

Con vara externa el arranque en frío **sí puede quedarse corto**, a diferencia del
experimento 1, donde era imposible por construcción.

- `frequent`: 29 de 420
- `frio`: 31 de 420
- `hybrid`: 32 de 420
- `nearest`: 45 de 420
- `random`: 32 de 420

## Resultado por conjunto de datos

Regret mediano de las diez semillas, ordenado por la diferencia entre `nearest` y
`frio`. «Meseta» es cuántas de las 144 configuraciones alcanzan el objetivo del 99 %,
recalculada para este experimento: la del experimento 1 estaba medida al 95 % y no
corresponde al objetivo vigente.

| Conjunto de datos | Meseta | frequent | frio | hybrid | nearest | random | Inevaluables (frío) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `car` | 12 | 0.0034 | 0.0564 | 0.0034 | 0.0016 | 0.0162 | 0.135 |
| `cylinder-bands` | 2 | 0.0240 | 0.0613 | 0.0280 | 0.0272 | 0.0209 | 0.222 |
| `tic-tac-toe` | 12 | 0.0083 | 0.0343 | 0.0095 | 0.0032 | 0.0034 | 0.142 |
| `eucalyptus` | 2 | 0.0383 | 0.0715 | 0.0294 | 0.0505 | 0.0450 | 0.227 |
| `ilpd` | 4 | 0.0082 | 0.0371 | 0.0052 | 0.0165 | 0.0073 | 0.135 |
| `credit-approval` | 4 | 0.0070 | 0.0322 | 0.0200 | 0.0116 | 0.0100 | 0.218 |
| `sick` | 4 | 0.0105 | 0.0282 | 0.0084 | 0.0089 | 0.0063 | 0.225 |
| `credit-g` | 3 | 0.0113 | 0.0259 | 0.0193 | 0.0102 | 0.0129 | 0.138 |
| `bank-marketing` | 6 | 0.0020 | 0.0201 | 0.0036 | 0.0071 | 0.0073 | 0.140 |
| `splice` | 12 | 0.0046 | 0.0176 | 0.0106 | 0.0067 | 0.0074 | 0.143 |
| `kr-vs-kp` | 48 | 0.0005 | 0.0108 | 0.0005 | 0.0001 | 0.0004 | 0.142 |
| `mfeat-morphological` | 32 | 0.0015 | 0.0090 | 0.0004 | 0.0002 | 0.0020 | 0.000 |
| `jungle_chess_2pcs_raw_endgame_complete` | 22 | 0.0024 | 0.0091 | 0.0022 | 0.0011 | 0.0028 | 0.000 |
| `steel-plates-fault` | 9 | 0.0093 | 0.0131 | 0.0166 | 0.0068 | 0.0000 | 0.000 |
| `adult` | 15 | 0.0024 | 0.0164 | 0.0060 | 0.0109 | 0.0013 | 0.138 |
| `qsar-biodeg` | 14 | 0.0044 | 0.0038 | 0.0026 | 0.0008 | 0.0025 | 0.000 |
| `analcatdata_dmft` | 12 | 0.0160 | 0.0335 | 0.0225 | 0.0308 | 0.0143 | 0.143 |
| `dresses-sales` | 3 | 0.0103 | 0.0340 | 0.0126 | 0.0312 | 0.0104 | 0.140 |
| `electricity` | 18 | 0.0019 | 0.0059 | 0.0055 | 0.0033 | 0.0029 | 0.000 |
| `wdbc` | 27 | 0.0009 | 0.0032 | 0.0004 | 0.0007 | 0.0005 | 0.000 |
| `banknote-authentication` | 72 | 0.0002 | 0.0023 | 0.0001 | 0.0000 | 0.0001 | 0.000 |
| `phoneme` | 36 | 0.0018 | 0.0024 | 0.0010 | 0.0004 | 0.0001 | 0.000 |
| `MiceProtein` | 44 | 0.0005 | 0.0020 | 0.0011 | 0.0007 | 0.0011 | 0.137 |
| `cmc` | 9 | 0.0103 | 0.0131 | 0.0111 | 0.0118 | 0.0124 | 0.000 |
| `pc1` | 9 | 0.0077 | 0.0044 | 0.0029 | 0.0032 | 0.0087 | 0.000 |
| `diabetes` | 9 | 0.0054 | 0.0049 | 0.0053 | 0.0038 | 0.0107 | 0.000 |
| `PhishingWebsites` | 60 | 0.0005 | 0.0012 | 0.0004 | 0.0004 | 0.0006 | 0.000 |
| `first-order-theorem-proving` | 22 | 0.0015 | 0.0019 | 0.0038 | 0.0014 | 0.0027 | 0.000 |
| `segment` | 27 | 0.0029 | 0.0022 | 0.0031 | 0.0018 | 0.0020 | 0.000 |
| `spambase` | 36 | 0.0012 | 0.0017 | 0.0014 | 0.0014 | 0.0010 | 0.000 |
| `breast-w` | 56 | 0.0012 | 0.0007 | 0.0006 | 0.0006 | 0.0006 | 0.133 |
| `analcatdata_authorship` | 108 | 0.0002 | 0.0003 | 0.0003 | 0.0002 | 0.0003 | 0.000 |
| `GesturePhaseSegmentationProcessed` | 18 | 0.0059 | 0.0055 | 0.0083 | 0.0057 | 0.0057 | 0.000 |
| `satimage` | 63 | 0.0005 | 0.0016 | 0.0004 | 0.0019 | 0.0003 | 0.000 |
| `blood-transfusion-service-center` | 18 | 0.0048 | 0.0079 | 0.0046 | 0.0097 | 0.0001 | 0.000 |
| `ozone-level-8hr` | 9 | 0.0078 | 0.0041 | 0.0067 | 0.0067 | 0.0064 | 0.000 |
| `climate-model-simulation-crashes` | 36 | 0.0062 | 0.0038 | 0.0102 | 0.0075 | 0.0125 | 0.000 |
| `wall-robot-navigation` | 27 | 0.0011 | 0.0007 | 0.0011 | 0.0059 | 0.0019 | 0.000 |
| `vehicle` | 9 | 0.0045 | 0.0108 | 0.0113 | 0.0162 | 0.0023 | 0.000 |
| `balance-scale` | 27 | 0.0018 | 0.0137 | 0.0018 | 0.0246 | 0.0000 | 0.000 |
| `churn` | 4 | 0.0225 | 0.0143 | 0.0228 | 0.0266 | 0.0046 | 0.000 |
| `wilt` | 9 | 0.0000 | 0.0371 | 0.0000 | 0.0719 | 0.0229 | 0.000 |
