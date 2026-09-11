# Desempeño a presupuesto reducido

> **Archivo generado.** No editar a mano: se regenera con
> `.\.venv\Scripts\python.exe scripts\analyze_reduced_budget.py`.

> **Segunda variable del anteproyecto** (Sección 4.3.3): el mejor macro-F1 alcanzado
> con un presupuesto menor que el completo. Los dos pre-registros la declararon antes de
> ejecutar —descriptiva en el experimento 1, secundaria en el 2— y los dos arneses la
> midieron en cada corrida. **No decide sobre la hipótesis**: eso lo hacen los ensayos
> hasta el objetivo en el experimento 1 y el regret en el 2.

- **Generado:** 2026-09-10
- **Datos:** `docs/experimentos/LODO-resultados.json` y `docs/experimentos/LODO-E2-resultados.json`
- **Mayor es mejor.** Cada observación es la mediana de las semillas de un conjunto;
  «gana» cuenta los conjuntos en que la condición queda por encima de la referencia.

## Experimento 1: 15 ensayos

Medida **descriptiva**, tal como la declaró el pre-registro v1.0: no se somete a
contraste. 25 conjuntos de datos.

| Condición | Mejor a 15 (mediana) | Diferencia mediana con `frio` | Gana | Empata | Pierde |
|---|---:|---:|---:|---:|---:|
| `frio` | 0.7471 | — | — | — | — |
| `nearest` | 0.7556 | +0.0000 | 9 | 7 | 9 |
| `random` | 0.7468 | +0.0000 | 11 | 6 | 8 |
| `frequent` | 0.7402 | +0.0000 | 8 | 8 | 9 |

## Experimento 2: 5, 15, 30 ensayos

Análisis **secundario** del pre-registro v2.0. 42 conjuntos de datos.

### Medianas por condición

| Presupuesto | `frio` | `nearest` | `frequent` | `random` | `hybrid` |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.7692 | 0.7568 | 0.7843 | 0.7946 | 0.7756 |
| 15 | 0.7908 | 0.7826 | 0.7920 | 0.7989 | 0.7866 |
| 30 | 0.7932 | 0.7918 | 0.7937 | 0.7990 | 0.7930 |

### Contrastes

Los mismos dos pares que los contrastes confirmatorios, a cada presupuesto. Wilcoxon
pareada de una cola en la dirección favorable a la condición —ahora, un valor
**mayor**—, sin corrección por multiplicidad. El efecto es la correlación biserial por
rangos, **positiva** cuando la condición mejora.

| Presupuesto | Contraste | Mediana referencia | Mediana condición | Diferencia mediana | Gana | Empata | Pierde | p (1 cola) | Efecto |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | `nearest` vs `frio` | 0.7692 | 0.7568 | +0.0022 | 26 | 2 | 14 | 0.1106 | +0.222 |
| 5 | `nearest` vs `frequent` | 0.7843 | 0.7568 | -0.0002 | 13 | 6 | 23 | 0.9849 | -0.414 |
| 15 | `nearest` vs `frio` | 0.7908 | 0.7826 | +0.0000 | 19 | 9 | 14 | 0.0945 | +0.262 |
| 15 | `nearest` vs `frequent` | 0.7920 | 0.7826 | +0.0002 | 21 | 9 | 12 | 0.2133 | +0.159 |
| 30 | `nearest` vs `frio` | 0.7932 | 0.7918 | +0.0000 | 13 | 16 | 13 | 0.3563 | +0.083 |
| 30 | `nearest` vs `frequent` | 0.7937 | 0.7918 | +0.0000 | 13 | 14 | 15 | 0.5272 | -0.015 |

## Los dos experimentos a igual presupuesto

Mejor valor a 30 ensayos sobre los 25
conjuntos y las semillas que los dos experimentos comparten.

El arranque en frío coincide en **125 de 125** parejas (conjunto, semilla).

Es la comprobación de que la comparación es legítima: con la misma semilla, la misma
partición y el mismo muestreador, el frío no tiene por qué cambiar.

Lo que sí cambia en las condiciones calientes es la base con la que se ajusta el
recomendador: 24 conjuntos en cada pliegue del experimento 1 y
41 en el 2.

| Condición | Experimento 1 | Experimento 2 | Sube en el 2 | Igual | Baja en el 2 |
|---|---:|---:|---:|---:|---:|
| `frio` | 0.7556 | 0.7556 | 0 | 25 | 0 |
| `nearest` | 0.7556 | 0.7556 | 3 | 16 | 6 |
| `random` | 0.7468 | 0.7594 | 10 | 11 | 4 |
| `frequent` | 0.7451 | 0.7451 | 4 | 15 | 6 |
