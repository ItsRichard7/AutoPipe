# Reproducibilidad de punta a punta

> **Archivo generado.** No editar a mano: se regenera con
> `.\.venv\Scripts\python.exe scripts\verify_reproducibility.py --datos <carpeta>`.

- **Generado:** 2026-09-15
- **Verificaciones:** 19 · **Omitidas:** 0

## Veredicto

**Todas las verificaciones ejecutadas se reproducen exactamente.**

La comparación de corridas es **exacta, sin tolerancia**: cada ensayo recién ejecutado
tiene que valer lo mismo que el guardado, bit a bit. Los documentos se comparan línea
por línea ignorando únicamente su fecha de generación.

| Verificación | Qué compara | Estado | Detalle |
|---|---|---|---|
| Resumen del experimento 1 | docs/experimentos/LODO-resultados.md | **idéntico** | 0 líneas distintas |
| Resumen del experimento 2 | docs/experimentos/LODO-E2-resultados.md | **idéntico** | 0 líneas distintas |
| Mecanismo del experimento 2 | docs/experimentos/MECANISMO-E2.md | **idéntico** | 0 líneas distintas |
| Análisis estratificado | docs/experimentos/ESTRATIFICADO-E2.md | **idéntico** | 0 líneas distintas |
| Desempeño a presupuesto reducido | docs/experimentos/PRESUPUESTO-REDUCIDO.md | **idéntico** | 0 líneas distintas |
| Tabla por conjunto, experimento 1 | docs/experimentos/LODO-por-dataset.md | **idéntico** | 0 líneas distintas |
| Tabla por conjunto, experimento 2 | docs/experimentos/LODO-E2-por-dataset.md | **idéntico** | 0 líneas distintas |
| Sensibilidad a n_startup_trials | docs/experimentos/SENSIBILIDAD-startup.md | **idéntico** | 0 líneas distintas |
| Resumen del caso de imágenes | docs/experimentos/IMAGENES-resultados.md | **idéntico** | 0 líneas distintas |
| Recomendaciones E1, balance-scale | nearest, random, frequent | **idéntico** | todas coinciden |
| Corridas E1, balance-scale, semilla 101 | 4 condiciones, 120 ensayos | **idéntico** | 0 ensayos distintos |
| Recomendaciones E1, diabetes | nearest, random, frequent | **idéntico** | todas coinciden |
| Corridas E1, diabetes, semilla 303 | 4 condiciones, 120 ensayos | **idéntico** | 0 ensayos distintos |
| Recomendaciones E2, balance-scale | nearest, frequent, random, hybrid | **idéntico** | todas coinciden |
| Corridas E2, balance-scale, semilla 101 | 5 condiciones, 300 ensayos | **idéntico** | 0 ensayos distintos |
| Recomendaciones E2, blood-transfusion-service-center | nearest, frequent, random, hybrid | **idéntico** | todas coinciden |
| Corridas E2, blood-transfusion-service-center, semilla 606 | 5 condiciones, 300 ensayos | **idéntico** | 0 ensayos distintos |
| Corridas sensibilidad (n_startup_trials=5), balance-scale, semilla 101 | 3 condiciones, 180 ensayos | **idéntico** | 0 ensayos distintos |
| Caso de imágenes, bottle, semilla 101 | óptimo, ganador, métricas y convergencia de Optuna | **idéntico** | todo coincide |

## Qué cubre y qué no

Reejecuta **parejas** de cada experimento y no los experimentos enteros: si una pareja se
reproduce bit a bit, el procedimiento es determinista, y reejecutarlo todo costaría más
de dos horas sin probar nada adicional. Las parejas se eligieron entre los conjuntos más
pequeños, por costo, antes de ejecutar la verificación.

Una verificación **omitida** no es un fallo, pero tampoco una evidencia: indica que los
datos que necesitaba no estaban disponibles en la máquina donde se ejecutó.
