# Sensibilidad del experimento 2 a `n_startup_trials`

> **Archivo generado.** No editar a mano: se regenera con
> `.\.venv\Scripts\python.exe scripts\run_sensitivity_startup.py --solo-reporte`.

> **Análisis de sensibilidad declarado** en el spike A (2026-07-23), cuando
> `n_startup_trials` se fijó en 10 como configuración primaria. **No
> sustituye** a los contrastes confirmatorios del experimento 2: dice si su conclusión
> depende de ese valor.

- **Generado:** 2026-09-10
- **Datos:** `docs/experimentos/SENSIBILIDAD-startup.json` (con 5) y
  `docs/experimentos/LODO-E2-resultados.json` (con 10)
- **Duración de la ejecución:** 90.0 min
- **Conjuntos:** 42 · **Corridas por condición:**
  420 · **Condiciones:** `frio`, `nearest`, `frequent`
- **Los contrastes con 10, recalculados sobre las corridas pareadas,
  coinciden con los del experimento 2:** **sí**, cifra por cifra

## Qué cambia y qué no

Cambia solo el número de ensayos al azar antes de que TPE empiece a modelar. Las semillas,
las particiones, el presupuesto, el óptimo y el objetivo de cada pareja y **las
configuraciones inyectadas** son las del experimento 2, leídas de su documento de
resultados en lugar de recalcularse. Las corridas con 10 no se repiten.

## Los contrastes confirmatorios con 5

Con la misma corrección de Holm y el mismo criterio que el experimento 2: reducción de al
menos el 30 % de la mediana del regret y valor p ajustado por debajo de 0,05.

1. `nearest` contra `frio`: regret mediano 0.0094 contra 0.0050. **Cumple el criterio**: reducción del 46.8 % —se exige 30 %— con un valor p ajustado por Holm de 0.0002, por debajo de 0,05.

2. `nearest` contra `frequent`: regret mediano 0.0045 contra 0.0050. **No cumple el criterio**: la reducción es del -10.2 % y se exige un 30 %; el valor p ajustado por Holm es 0.8761 —crudo 0.8761— y se exige menos de 0,05.

| Contraste | `n_startup_trials` | Regret referencia | Regret condición | Reducción | p (1 cola) | p (Holm) | Cumple |
|---|---:|---:|---:|---:|---:|---:|---:|
| `nearest` vs `frio` | 10 | 0.0090 | 0.0058 | 35.8 % | 0.0010 | 0.0019 | sí |
| `nearest` vs `frio` | 5 | 0.0094 | 0.0050 | 46.8 % | 0.0001 | 0.0002 | sí |
| `nearest` vs `frequent` | 10 | 0.0039 | 0.0058 | -49.1 % | 0.9690 | 0.9690 | no |
| `nearest` vs `frequent` | 5 | 0.0045 | 0.0050 | -10.2 % | 0.8761 | 0.8761 | no |

## La variable secundaria con 5

Los mismos dos pares sobre los ensayos hasta el objetivo, sin corrección.

| Contraste | Mediana referencia | Mediana condición | Reducción | p (1 cola) |
|---|---:|---:|---:|---:|
| `nearest` vs `frio` | 7.0 | 4.8 | 32.1 % | 0.0272 |
| `nearest` vs `frequent` | 3.8 | 4.8 | -26.7 % | 0.7010 |

## El efecto del parámetro en cada brazo

El spike A midió el efecto solo en el brazo caliente y advirtió que había que medirlo
**también en el frío**: si el frío mejora tanto como el caliente, lo que se observa es un
efecto general del muestreador y no uno del arranque en caliente. Menor regret es mejor.
No se declaró dirección, así que la prueba es de dos colas.

| Condición | Regret con 10 | Regret con 5 | Reducción | Mejora con 5 | Igual | Empeora con 5 | p (2 colas) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `frio` | 0.0090 | 0.0094 | -3.4 % | 18 | 3 | 21 | 0.3869 |
| `nearest` | 0.0058 | 0.0050 | 14.2 % | 18 | 9 | 15 | 0.8233 |
| `frequent` | 0.0039 | 0.0045 | -16.1 % | 12 | 9 | 21 | 0.0286 |

## El beneficio del arranque en caliente con 10 y con 5

El beneficio es el regret mediano del frío menos el de la condición, cada uno con su propio
frío: positivo cuando la condición mejora. Si el beneficio no cambia, la conclusión del
experimento 2 no depende del parámetro aunque ambos brazos se muevan.

| Condición | Beneficio mediano con 10 | Beneficio mediano con 5 | Crece con 5 | Igual | Decrece con 5 | p (2 colas) |
|---|---:|---:|---:|---:|---:|---:|
| `nearest` | +0.0021 | +0.0025 | 19 | 3 | 20 | 0.4680 |
| `frequent` | +0.0034 | +0.0030 | 14 | 3 | 25 | 0.1849 |

## Corridas censuradas

| Condición | Con 10 | Con 5 |
|---|---:|---:|
| `frio` | 31 | 47 |
| `nearest` | 45 | 45 |
| `frequent` | 29 | 36 |
