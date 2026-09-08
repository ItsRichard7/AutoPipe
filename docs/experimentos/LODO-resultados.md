# Resultados del experimento LODO (RE6)

> **Archivo generado.** No editar a mano: se regenera con
> `.\.venv\Scripts\python.exe scripts\report_lodo.py`.

- **Generado:** 2026-09-08
- **Datos:** `docs/experimentos/LODO-resultados.json`
- **Reporte con gráficas:** `docs/experimentos/reportes/lodo.html`
- **Duración de la ejecución:** 13.4 min

## Veredicto

Mediana de 3.0 ensayos en frío contra 2.0 en caliente. La hipótesis no se confirma: el valor p de una cola es 0.3115 y el criterio exige menos de 0,05. El resultado se reporta tal como salió, que es lo que el pre-registro obliga.

De los 25 conjuntos de datos, el arranque en caliente alcanzó el
objetivo antes en **14**, empató en **4** y llegó después en
**7**.

## Protocolo ejecutado

- **Pre-registro:** v1.0, etiqueta prereg-v1.0
- **Espacio de búsqueda:** tabular-v1 (144 pipelines con omisión)
- **Modelo base:** KNeighborsClassifier(n_neighbors=5, metric=minkowski, p=2, brute)
- **Métrica:** macro-F1, partición estratificada con 20 % de validación
- **Semillas:** 101, 202, 303, 404, 505 (idénticas en ambas condiciones)
- **Presupuesto:** 30 ensayos por condición; censura en 31
- **Arranque en caliente:** 5 configuraciones inyectadas con enqueue_trial, que cuentan
- **Objetivo:** 95 % del mejor valor del arranque en frío, por pareja (dataset, semilla)
- **Prueba:** Wilcoxon pareada de una cola, α = 0,05

## Contraste primario y controles secundarios

Las estrategias de control comparten todo con la primaria salvo cómo eligen las cinco
configuraciones: `random` inyecta las ganadoras de conjuntos al azar y `frequent` las
más frecuentes del pliegue. Si `nearest` no se separa de ellas, la similitud entre
conjuntos de datos no está aportando nada.

**La hipótesis del trabajo se juega únicamente en la fila primaria.** Que un control
cumpla el criterio no la confirma: es un hallazgo sobre ese control, y sobre lo que su
ventaja implica para el meta-aprendizaje.

| Estrategia | Papel | Mediana frío | Mediana caliente | Reducción | p (1 cola) | p (2 colas) | Censurados | Cumple el criterio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `nearest` | **primario** | 3.0 | 2.0 | 33.3 % | 0.3115 | 0.6230 | 1 | no |
| `random` | control | 3.0 | 2.0 | 33.3 % | 0.2693 | 0.5386 | 1 | no |
| `frequent` | control | 3.0 | 2.0 | 33.3 % | 0.0239 | 0.0478 | 1 | sí |

## Robustez al valor de censura

El análisis primario castiga con 31 a las corridas que no alcanzan
el objetivo. Estos recálculos usan los castigos más severos que el pre-registro fijó
antes de ejecutar.

| Estrategia | Censura | Mediana frío | Mediana caliente | Reducción | p (1 cola) | p (2 colas) | Censurados | Cumple el criterio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `nearest` | 60 | 3.0 | 2.0 | 33.3 % | 0.3115 | 0.6230 | 1 | no |
| `nearest` | 300 | 3.0 | 2.0 | 33.3 % | 0.3115 | 0.6230 | 1 | no |

## Resultado por conjunto de datos

Cada celda es la mediana de las cinco semillas, ordenada por la diferencia entre
`nearest` y `frio`. «Meseta» es el número de configuraciones del espacio que alcanzan
el 95 % del óptimo, y «inevaluables» la fracción de ensayos que no se pudieron
evaluar: las dos covariables declaradas en el pre-registro, medidas antes de ejecutar.

| Conjunto de datos | Meseta | frio | nearest | frequent | random | Inevaluables (frío) |
|---|---:|---:|---:|---:|---:|---:|
| `car` | 12 | 7.0 | 1.0 | 2.0 | 2.0 | 0.167 |
| `tic-tac-toe` | 24 | 5.0 | 1.0 | 3.0 | 3.0 | 0.167 |
| `credit-approval` | 26 | 5.0 | 1.0 | 2.0 | 1.0 | 0.280 |
| `sick` | 14 | 5.0 | 2.0 | 1.0 | 1.0 | 0.267 |
| `dresses-sales` | 4 | 5.0 | 2.0 | 4.0 | 2.0 | 0.167 |
| `GesturePhaseSegmentationProcessed` | 63 | 3.0 | 1.0 | 3.0 | 7.0 | 0.000 |
| `mfeat-morphological` | 54 | 2.0 | 1.0 | 2.0 | 2.0 | 0.000 |
| `steel-plates-fault` | 27 | 4.0 | 3.0 | 2.0 | 2.0 | 0.000 |
| `jungle_chess_2pcs_raw_endgame_complete` | 72 | 2.0 | 1.0 | 2.0 | 2.0 | 0.000 |
| `blood-transfusion-service-center` | 54 | 3.0 | 2.0 | 2.0 | 2.0 | 0.000 |
| `banknote-authentication` | 72 | 2.0 | 1.0 | 2.0 | 3.0 | 0.000 |
| `splice` | 48 | 2.0 | 1.0 | 2.0 | 1.0 | 0.167 |
| `phoneme` | 108 | 2.0 | 1.0 | 2.0 | 3.0 | 0.000 |
| `qsar-biodeg` | 54 | 2.0 | 1.0 | 2.0 | 1.0 | 0.000 |
| `eucalyptus` | 11 | 7.0 | 7.0 | 1.0 | 7.0 | 0.280 |
| `cylinder-bands` | 3 | 8.0 | 8.0 | 3.0 | 3.0 | 0.267 |
| `electricity` | 66 | 2.0 | 2.0 | 2.0 | 1.0 | 0.000 |
| `diabetes` | 108 | 2.0 | 2.0 | 2.0 | 4.0 | 0.000 |
| `churn` | 30 | 1.0 | 2.0 | 4.0 | 6.0 | 0.000 |
| `credit-g` | 30 | 2.0 | 4.0 | 1.0 | 4.0 | 0.167 |
| `vehicle` | 36 | 2.0 | 5.0 | 3.0 | 3.0 | 0.000 |
| `pc1` | 27 | 3.0 | 7.0 | 7.0 | 4.0 | 0.000 |
| `wilt` | 18 | 4.0 | 9.0 | 2.0 | 9.0 | 0.000 |
| `balance-scale` | 36 | 3.0 | 8.0 | 2.0 | 2.0 | 0.000 |
| `climate-model-simulation-crashes` | 45 | 1.0 | 6.0 | 2.0 | 1.0 | 0.000 |
