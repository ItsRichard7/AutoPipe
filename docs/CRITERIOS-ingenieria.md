# Criterios de éxito de ingeniería: verificación con evidencia

> **Archivo generado.** No editar a mano: se regenera con
> `.\.venv\Scripts\python.exe scripts\verify_engineering_criteria.py`.

> Los cuatro criterios de ingeniería del anteproyecto (Sección 4.3.5) son independientes
> del resultado de la hipótesis. Aquí cada uno se **mide** con un procedimiento repetible.

- **Generado:** 2026-09-11
- **Commit de migración de la librería base:** `35f1db5`
- **Suite completa:** 884 pasan, 0 fallan, 0 errores
- **Veredicto:** **4 de 4 criterios cumplidos**

| Criterio | Exigido | Medido | Cumple |
|---|---|---|:--:|
| (a) Regresión de la librería base | Las 134 pruebas en verde tras la refactorización | 134 de 134 pasan, ejecutadas tal como estaban en `35f1db5` | **sí** |
| (b) Cobertura de los módulos nuevos | Al menos 70 % | Mínimo 99.4 % (`autopipe/reporter/comparison.py`), 16 módulos | **sí** |
| (c) Reproducibilidad | Misma semilla, mismo resultado | 19 de 19 verificaciones idénticas, 0 omitidas | **sí** |
| (d) Flujo del usuario final | No más de 20 instrucciones | 14 y 19 | **sí** |

## (a) La suite de regresión de la librería base

Se extrajo la carpeta `tests/` tal como estaba en el commit de migración y se ejecutó,
**sin modificarla**, contra el código actual. Es la lectura literal del criterio: si las
pruebas de la librería base pasan sin tocarlas, la refactorización no rompió el caso de uso
original. `autopipe` se importó desde el repositorio: **sí**.

- **Recolectadas:** 134 · **Pasan:** 134 ·
  **Fallan:** 0 · **Errores:** 0
- **Funciones de prueba:** 111; la parametrización las expande en
  las pruebas recolectadas. Comparadas por su árbol sintáctico con las de hoy:
  **111 idénticas**, 0 modificadas y
  0 ausentes.

## (b) Cobertura de los módulos nuevos

«Módulo nuevo» es todo archivo de `autopipe/` que no existía en el commit de migración, según
`git`; los 17 que sí existían quedan fuera del criterio. La cobertura
total del paquete es de 99.3 % sobre 1647
instrucciones.

| Módulo nuevo | Instrucciones | Cobertura |
|---|---:|---:|
| `autopipe/reporter/comparison.py` | 157 | 99.4 % |
| `autopipe/data_loader/tabular.py` | 56 | 100.0 % |
| `autopipe/meta_features/__init__.py` | 2 | 100.0 % |
| `autopipe/meta_features/extractor.py` | 89 | 100.0 % |
| `autopipe/meta_knowledge/__init__.py` | 2 | 100.0 % |
| `autopipe/meta_knowledge/base.py` | 74 | 100.0 % |
| `autopipe/meta_recommender/__init__.py` | 2 | 100.0 % |
| `autopipe/meta_recommender/recommender.py` | 171 | 100.0 % |
| `autopipe/pipeline/tabular_transforms.py` | 193 | 100.0 % |
| `autopipe/protocols.py` | 12 | 100.0 % |
| `autopipe/search_space/__init__.py` | 2 | 100.0 % |
| `autopipe/search_space/tabular.py` | 39 | 100.0 % |
| `autopipe/tasks/__init__.py` | 17 | 100.0 % |
| `autopipe/tasks/anomaly.py` | 25 | 100.0 % |
| `autopipe/tasks/base.py` | 25 | 100.0 % |
| `autopipe/tasks/classification.py` | 15 | 100.0 % |

## (c) Reproducibilidad

Se lee `docs/experimentos/REPRODUCIBILIDAD.md`, que genera
`scripts/verify_reproducibility.py`: regenera los documentos de resultados desde sus datos,
reajusta el recomendador por pliegue y reejecuta corridas ensayo por ensayo, sin tolerancia.
Una verificación omitida no cuenta como evidencia.

- **Idénticas:** 19 · **Difieren:** 0 ·
  **Omitidas:** 0

## (d) Flujo del usuario final

Se cuentan instrucciones de nivel superior y no líneas físicas, con el mismo criterio que
`tests/test_examples.py`, que además ejecuta los dos ejemplos en cada corrida de la suite.

| Ejemplo | Instrucciones |
|---|---:|
| `examples/tabular_quickstart.py` | 14 |
| `examples/warm_start_quickstart.py` | 19 |
