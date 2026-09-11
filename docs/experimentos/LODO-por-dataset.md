# Tabla por conjunto de datos — experimento 1

> **Archivo generado.** No editar a mano: se regenera con
> `.\.venv\Scripts\python.exe scripts\tabulate_by_dataset.py`.

> **Análisis secundario**, declarado en el pre-registro v1.0. Cruza lo que ya estaba medido
> —el cribado del corpus y las corridas del arnés— y no ejecuta ningún ensayo.

- **Generado:** 2026-09-10
- **Datos:** `docs/experimentos/LODO-resultados.json` y `docs/experimentos/CORPUS-cribado-valores.json`
- **Conjuntos:** 25; 0 por debajo del umbral de sensibilidad y
  9 con el pipeline vacío inevaluable
- **Censura:** 31

## Cómo leer la tabla

- **IQR**: sensibilidad al preprocesamiento, medida en el cribado sobre doce pipelines al
  azar. La tabla va de más a menos sensible y **†** marca los conjuntos por debajo del
  umbral de 0.02, que el pre-registro pide ver en su propia fila y no promediados.
- **Margen sobre vacío**: mejor de esos doce pipelines menos el desempeño sin
  preprocesamiento. **n/e** cuando el pipeline vacío no se puede evaluar; de los
  9 conjuntos en que ocurre, 9 tienen columnas categóricas o
  valores faltantes.
- **Meseta**: cuántas de las 144 configuraciones alcanzan el 95 % del
  óptimo; es una de las dos covariables declaradas.
- **Ensayos por semilla**: ensayos hasta el objetivo, en el orden de las semillas
  101, 202, 303, 404, 505. Un 31 es una corrida censurada.
- **Mediana** y **Censurados**: sobre esas mismas semillas.
- **Mejor a k**: mediana del mejor macro-F1 tras k ensayos (15).
- **Inevaluables**: fracción media de ensayos que no se pudieron evaluar; es la otra
  covariable declarada.

## Tabla

| Conjunto | IQR | Margen sobre vacío | Meseta 95 % | Condición | Ensayos por semilla | Mediana | Censurados | Mejor a 15 | Inevaluables |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|
| `wilt` | 0.3239 | 0.0072 | 18 | `frio` | 1, 4, 7, 3, 8 | 4 | 0 | 0.8905 | 0.000 |
|  |  |  |  | `nearest` | 6, 9, 13, 8, 11 | 9 | 0 | 0.8905 | 0.000 |
|  |  |  |  | `random` | 6, 9, 11, 8, 17 | 9 | 0 | 0.8708 | 0.000 |
|  |  |  |  | `frequent` | 2, 2, 2, 2, 2 | 2 | 0 | 0.8994 | 0.000 |
| `sick` | 0.3064 | n/e | 14 | `frio` | 1, 5, 4, 8, 9 | 5 | 0 | 0.8865 | 0.267 |
|  |  |  |  | `nearest` | 5, 2, 2, 17, 2 | 2 | 0 | 0.8812 | 0.273 |
|  |  |  |  | `random` | 1, 1, 1, 1, 1 | 1 | 0 | 0.8865 | 0.280 |
|  |  |  |  | `frequent` | 2, 1, 1, 2, 1 | 1 | 0 | 0.8812 | 0.287 |
| `mfeat-morphological` | 0.2837 | 0.2621 | 54 | `frio` | 1, 2, 3, 4, 2 | 2 | 0 | 0.6973 | 0.000 |
|  |  |  |  | `nearest` | 1, 1, 1, 1, 1 | 1 | 0 | 0.7011 | 0.000 |
|  |  |  |  | `random` | 2, 2, 2, 2, 2 | 2 | 0 | 0.7011 | 0.000 |
|  |  |  |  | `frequent` | 2, 2, 2, 2, 2 | 2 | 0 | 0.7011 | 0.000 |
| `car` | 0.2276 | n/e | 12 | `frio` | 10, 7, 3, 3, 21 | 7 | 0 | 0.8279 | 0.167 |
|  |  |  |  | `nearest` | 1, 1, 1, 1, 1 | 1 | 0 | 0.8424 | 0.160 |
|  |  |  |  | `random` | 2, 2, 2, 2, 2 | 2 | 0 | 0.8368 | 0.160 |
|  |  |  |  | `frequent` | 2, 2, 2, 2, 2 | 2 | 0 | 0.8424 | 0.160 |
| `tic-tac-toe` | 0.2225 | n/e | 24 | `frio` | 1, 20, 3, 11, 5 | 5 | 0 | 0.9588 | 0.167 |
|  |  |  |  | `nearest` | 1, 1, 1, 1, 1 | 1 | 0 | 0.9827 | 0.167 |
|  |  |  |  | `random` | 3, 20, 3, 3, 3 | 3 | 0 | 0.9317 | 0.167 |
|  |  |  |  | `frequent` | 3, 4, 3, 3, 3 | 3 | 0 | 0.9827 | 0.167 |
| `steel-plates-fault` | 0.1827 | 0.4499 | 27 | `frio` | 1, 2, 16, 4, 4 | 4 | 0 | 0.7471 | 0.000 |
|  |  |  |  | `nearest` | 3, 3, 4, 3, 4 | 3 | 0 | 0.7556 | 0.000 |
|  |  |  |  | `random` | 2, 2, 13, 2, 9 | 2 | 0 | 0.7396 | 0.000 |
|  |  |  |  | `frequent` | 2, 5, 12, 2, 2 | 2 | 0 | 0.7228 | 0.000 |
| `eucalyptus` | 0.1716 | n/e | 11 | `frio` | 15, 10, 2, 7, 1 | 7 | 0 | 0.5836 | 0.280 |
|  |  |  |  | `nearest` | 12, 3, 7, 31, 6 | 7 | 1 | 0.5867 | 0.293 |
|  |  |  |  | `random` | 31, 11, 7, 1, 6 | 7 | 1 | 0.5630 | 0.287 |
|  |  |  |  | `frequent` | 1, 1, 1, 1, 1 | 1 | 0 | 0.5813 | 0.253 |
| `credit-approval` | 0.1490 | n/e | 26 | `frio` | 6, 5, 2, 6, 1 | 5 | 0 | 0.8469 | 0.280 |
|  |  |  |  | `nearest` | 1, 1, 1, 1, 1 | 1 | 0 | 0.8614 | 0.293 |
|  |  |  |  | `random` | 1, 1, 1, 1, 1 | 1 | 0 | 0.8614 | 0.280 |
|  |  |  |  | `frequent` | 2, 2, 2, 2, 2 | 2 | 0 | 0.8678 | 0.253 |
| `balance-scale` | 0.1481 | 0.0000 | 36 | `frio` | 1, 3, 7, 3, 8 | 3 | 0 | 0.6094 | 0.000 |
|  |  |  |  | `nearest` | 6, 8, 12, 8, 11 | 8 | 0 | 0.6034 | 0.000 |
|  |  |  |  | `random` | 2, 2, 2, 2, 2 | 2 | 0 | 0.6094 | 0.000 |
|  |  |  |  | `frequent` | 2, 2, 2, 2, 2 | 2 | 0 | 0.6034 | 0.000 |
| `pc1` | 0.1475 | 0.1426 | 27 | `frio` | 3, 2, 9, 2, 7 | 3 | 0 | 0.7020 | 0.000 |
|  |  |  |  | `nearest` | 8, 7, 5, 1, 11 | 7 | 0 | 0.6917 | 0.000 |
|  |  |  |  | `random` | 4, 7, 3, 2, 4 | 4 | 0 | 0.6858 | 0.000 |
|  |  |  |  | `frequent` | 8, 7, 4, 1, 11 | 7 | 0 | 0.6858 | 0.000 |
| `jungle_chess_2pcs_raw_endgame_complete` | 0.1430 | 0.0356 | 72 | `frio` | 1, 2, 3, 3, 2 | 2 | 0 | 0.7027 | 0.000 |
|  |  |  |  | `nearest` | 1, 2, 2, 1, 1 | 1 | 0 | 0.7038 | 0.000 |
|  |  |  |  | `random` | 2, 2, 2, 2, 2 | 2 | 0 | 0.7038 | 0.000 |
|  |  |  |  | `frequent` | 2, 3, 3, 2, 2 | 2 | 0 | 0.7043 | 0.000 |
| `blood-transfusion-service-center` | 0.1180 | 0.0491 | 54 | `frio` | 1, 3, 3, 3, 1 | 3 | 0 | 0.6519 | 0.000 |
|  |  |  |  | `nearest` | 2, 2, 2, 2, 3 | 2 | 0 | 0.6487 | 0.000 |
|  |  |  |  | `random` | 2, 2, 2, 2, 2 | 2 | 0 | 0.6487 | 0.000 |
|  |  |  |  | `frequent` | 2, 2, 2, 2, 3 | 2 | 0 | 0.6487 | 0.000 |
| `churn` | 0.1019 | 0.2043 | 30 | `frio` | 2, 1, 1, 1, 1 | 1 | 0 | 0.7853 | 0.000 |
|  |  |  |  | `nearest` | 7, 1, 2, 6, 2 | 2 | 0 | 0.7853 | 0.000 |
|  |  |  |  | `random` | 7, 1, 6, 4, 6 | 6 | 0 | 0.7853 | 0.000 |
|  |  |  |  | `frequent` | 7, 1, 4, 6, 4 | 4 | 0 | 0.7853 | 0.000 |
| `banknote-authentication` | 0.1008 | -0.0000 | 72 | `frio` | 1, 2, 3, 3, 2 | 2 | 0 | 1.0000 | 0.000 |
|  |  |  |  | `nearest` | 1, 1, 1, 1, 1 | 1 | 0 | 1.0000 | 0.000 |
|  |  |  |  | `random` | 3, 3, 3, 3, 3 | 3 | 0 | 1.0000 | 0.000 |
|  |  |  |  | `frequent` | 2, 2, 2, 2, 2 | 2 | 0 | 1.0000 | 0.000 |
| `vehicle` | 0.0937 | 0.0750 | 36 | `frio` | 1, 2, 9, 15, 1 | 2 | 0 | 0.6967 | 0.000 |
|  |  |  |  | `nearest` | 5, 5, 11, 2, 6 | 5 | 0 | 0.6927 | 0.000 |
|  |  |  |  | `random` | 3, 2, 5, 5, 2 | 3 | 0 | 0.6927 | 0.000 |
|  |  |  |  | `frequent` | 5, 3, 11, 2, 3 | 3 | 0 | 0.6927 | 0.000 |
| `splice` | 0.0897 | n/e | 48 | `frio` | 2, 8, 2, 2, 10 | 2 | 0 | 0.8461 | 0.167 |
|  |  |  |  | `nearest` | 1, 1, 1, 1, 11 | 1 | 0 | 0.8418 | 0.167 |
|  |  |  |  | `random` | 1, 1, 1, 1, 1 | 1 | 0 | 0.8461 | 0.167 |
|  |  |  |  | `frequent` | 2, 2, 2, 2, 2 | 2 | 0 | 0.8461 | 0.167 |
| `climate-model-simulation-crashes` | 0.0808 | 0.1238 | 45 | `frio` | 2, 1, 9, 1, 1 | 1 | 0 | 0.7709 | 0.000 |
|  |  |  |  | `nearest` | 7, 6, 11, 6, 6 | 6 | 0 | 0.7709 | 0.000 |
|  |  |  |  | `random` | 1, 1, 2, 1, 1 | 1 | 0 | 0.7709 | 0.000 |
|  |  |  |  | `frequent` | 2, 2, 4, 2, 2 | 2 | 0 | 0.7709 | 0.000 |
| `phoneme` | 0.0660 | 0.0004 | 108 | `frio` | 1, 2, 3, 3, 2 | 2 | 0 | 0.8547 | 0.000 |
|  |  |  |  | `nearest` | 1, 1, 1, 1, 1 | 1 | 0 | 0.8605 | 0.000 |
|  |  |  |  | `random` | 3, 3, 3, 3, 3 | 3 | 0 | 0.8584 | 0.000 |
|  |  |  |  | `frequent` | 2, 2, 2, 2, 2 | 2 | 0 | 0.8534 | 0.000 |
| `cylinder-bands` | 0.0654 | n/e | 3 | `frio` | 4, 8, 3, 13, 10 | 8 | 0 | 0.7375 | 0.267 |
|  |  |  |  | `nearest` | 2, 11, 8, 13, 1 | 8 | 0 | 0.7363 | 0.233 |
|  |  |  |  | `random` | 1, 3, 8, 4, 1 | 3 | 0 | 0.7468 | 0.267 |
|  |  |  |  | `frequent` | 3, 3, 3, 3, 1 | 3 | 0 | 0.7402 | 0.267 |
| `credit-g` | 0.0616 | n/e | 30 | `frio` | 1, 1, 4, 2, 11 | 2 | 0 | 0.6666 | 0.167 |
|  |  |  |  | `nearest` | 4, 6, 2, 1, 14 | 4 | 0 | 0.6666 | 0.160 |
|  |  |  |  | `random` | 3, 4, 4, 2, 21 | 4 | 0 | 0.6882 | 0.167 |
|  |  |  |  | `frequent` | 1, 1, 9, 1, 12 | 1 | 0 | 0.6707 | 0.160 |
| `dresses-sales` | 0.0589 | n/e | 4 | `frio` | 1, 1, 7, 8, 5 | 5 | 0 | 0.5806 | 0.167 |
|  |  |  |  | `nearest` | 2, 1, 15, 2, 2 | 2 | 0 | 0.5954 | 0.167 |
|  |  |  |  | `random` | 2, 3, 1, 1, 2 | 2 | 0 | 0.5840 | 0.167 |
|  |  |  |  | `frequent` | 1, 6, 31, 4, 2 | 4 | 1 | 0.5806 | 0.167 |
| `GesturePhaseSegmentationProcessed` | 0.0575 | 0.0012 | 63 | `frio` | 4, 2, 3, 3, 3 | 3 | 0 | 0.5672 | 0.000 |
|  |  |  |  | `nearest` | 1, 1, 1, 1, 1 | 1 | 0 | 0.5655 | 0.000 |
|  |  |  |  | `random` | 9, 7, 3, 3, 8 | 7 | 0 | 0.5672 | 0.000 |
|  |  |  |  | `frequent` | 3, 3, 3, 3, 3 | 3 | 0 | 0.5672 | 0.000 |
| `electricity` | 0.0563 | 0.0417 | 66 | `frio` | 1, 2, 3, 3, 2 | 2 | 0 | 0.8308 | 0.000 |
|  |  |  |  | `nearest` | 2, 2, 2, 2, 2 | 2 | 0 | 0.8308 | 0.000 |
|  |  |  |  | `random` | 1, 1, 1, 1, 1 | 1 | 0 | 0.8362 | 0.000 |
|  |  |  |  | `frequent` | 2, 2, 2, 2, 2 | 2 | 0 | 0.8308 | 0.000 |
| `qsar-biodeg` | 0.0502 | 0.0793 | 54 | `frio` | 1, 3, 2, 4, 1 | 2 | 0 | 0.8493 | 0.000 |
|  |  |  |  | `nearest` | 1, 1, 1, 1, 1 | 1 | 0 | 0.8493 | 0.000 |
|  |  |  |  | `random` | 1, 1, 1, 1, 1 | 1 | 0 | 0.8534 | 0.000 |
|  |  |  |  | `frequent` | 2, 1, 2, 3, 2 | 2 | 0 | 0.8493 | 0.000 |
| `diabetes` | 0.0448 | 0.0529 | 108 | `frio` | 3, 2, 2, 1, 1 | 2 | 0 | 0.7306 | 0.000 |
|  |  |  |  | `nearest` | 8, 1, 2, 5, 1 | 2 | 0 | 0.7253 | 0.000 |
|  |  |  |  | `random` | 8, 5, 2, 4, 1 | 4 | 0 | 0.7253 | 0.000 |
|  |  |  |  | `frequent` | 8, 2, 2, 1, 3 | 2 | 0 | 0.7253 | 0.000 |
