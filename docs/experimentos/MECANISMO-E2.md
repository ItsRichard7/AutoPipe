# Mecanismo del experimento 2 — análisis exploratorio

> **Archivo generado.** No editar a mano: se regenera con
> `.\.venv\Scripts\python.exe scripts\analyze_e2_mechanism.py`.

> **Exploratorio, no confirmatorio** (pre-registro v2.0 §8.6). Describe el patrón; no
> decide nada sobre la hipótesis. No ejecuta ensayos: lee las configuraciones
> inyectadas y las rejillas exhaustivas ya cacheadas.

- **Generado:** 2026-09-09
- **Pliegues:** 42

## Diversidad de lo inyectado

Pone a prueba la explicación que el spike A había propuesto, según la cual la similitud
fallaría por reducir la dispersión de lo inyectado. La distancia de Hamming va de 0
—las cinco idénticas— a 4 —difieren en los cuatro pasos—.

| Condición | Hamming medio | Configuraciones distintas en todo el corpus | Regret logrado |
|---|---:|---:|---:|
| `nearest` | 2.58 | 28 | 0.0058 |
| `hybrid` | 2.54 | 26 | 0.0042 |
| `random` | 2.34 | 10 | 0.0028 |
| `frequent` | 2.10 | 6 | 0.0039 |

## Calidad de lo inyectado sobre el conjunto excluido

Regret de las configuraciones inyectadas respecto del óptimo de la pareja, leído de la
rejilla exhaustiva. La columna que reproduce el orden del resultado final es la de la
**mejor de las cinco**.

| Condición | Regret medio de las 5 | Regret de la mejor de las 5 | Inevaluables | Regret logrado |
|---|---:|---:|---:|---:|
| `random` | 0.0819 | 0.0209 | 0.0% | 0.0028 |
| `frequent` | 0.0885 | 0.0150 | 0.0% | 0.0039 |
| `hybrid` | 0.0966 | 0.0231 | 2.4% | 0.0042 |
| `nearest` | 0.0928 | 0.0352 | 2.9% | 0.0058 |

## Lectura

La interpretación de estas dos tablas está en `LODO-E2-discusion.md`, Sección 3.
