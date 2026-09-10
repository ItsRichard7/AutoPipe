# Caso de anomalías en imágenes (RE6)

> **Archivo generado.** No editar a mano: se regenera con
> `.\.venv\Scripts\python.exe scripts\run_image_case.py --datos <carpeta>`.

- **Generado:** 2026-09-10
- **Duración:** 1.6 min

## Qué verifica y qué no

Verifica la **generalización del framework por tarea**: el mismo cargador, el mismo
registro de tareas, los mismos dos motores, el mismo evaluador y el mismo generador de
reportes que sostienen los experimentos tabulares resuelven la detección de anomalías
en imágenes con su métrica propia, AUC-ROC.

**No es un contraste de la hipótesis**, y **no hay arranque en caliente sobre
imágenes**: exigiría un corpus de datasets de imágenes, su base de meta-conocimiento y su
propio pre-registro, fuera del alcance comprometido.

## Protocolo

- **Datos:** partición `test/` de MVTec AD, categorías `bottle`, `cable`, `capsule`.
- **Resolución de trabajo:** 64 x 64, fijada en la preparación y no buscada.
- **Espacio:** el del proyecto base sin redimensionado —CLAHE (2,0 / 4,0), suavizado
  gaussiano (3 / 5) y normalización, cada uno omitible—: 18 candidatos.
- **Modelo:** `IsolationForest(random_state=0)`, por continuidad con la librería base.
- **Validación:** 30 %, cinco semillas de partición:
  101, 202, 303, 404, 505.
- **Motores:** rejilla exhaustiva (óptimo exacto) y Optuna con 20 ensayos,
  sobre la misma partición.

## Resultado por categoría

El AUC-ROC es el óptimo exacto del espacio sobre cada partición. Se reporta la
**dispersión entre semillas** y no un valor único: con pocas imágenes normales, la
partición de validación deja muy pocas y el AUC-ROC varía de una semilla a otra.

| Categoría | Normales | Anómalas | AUC mediano | Rango | Ganadores | Óptimo en Optuna |
|---|---:|---:|---:|---|---:|---:|
| `bottle` | 20 | 63 | 0.8596 | 0.7719 – 0.9035 | 2 de 5 | 5 de 5 |
| `cable` | 58 | 92 | 0.6765 | 0.6282 – 0.7752 | 3 de 5 | 4 de 5 |
| `capsule` | 23 | 109 | 0.5281 | 0.3680 – 0.6320 | 1 de 5 | 5 de 5 |

## Ganador y convergencia por semilla

«Ensayo» es la posición en que Optuna alcanzó por primera vez el óptimo de la rejilla;
un guion indica que no lo alcanzó dentro del presupuesto.

| Categoría | Semilla | AUC-ROC | Pipeline ganador | Ensayo |
|---|---:|---:|---|---:|
| `bottle` | 101 | 0.7719 | CLAHETransform → GaussianBlur → Normalize | 4 |
| `bottle` | 202 | 0.8947 | CLAHETransform → GaussianBlur → Normalize | 6 |
| `bottle` | 303 | 0.8596 | GaussianBlur → Normalize | 7 |
| `bottle` | 404 | 0.9035 | CLAHETransform → GaussianBlur → Normalize | 4 |
| `bottle` | 505 | 0.8509 | CLAHETransform → GaussianBlur → Normalize | 1 |
| `cable` | 101 | 0.7752 | GaussianBlur → Normalize | 6 |
| `cable` | 202 | 0.6282 | CLAHETransform → GaussianBlur → Normalize | 6 |
| `cable` | 303 | 0.7374 | GaussianBlur → Normalize | — |
| `cable` | 404 | 0.6387 | CLAHETransform → GaussianBlur → Normalize | 4 |
| `cable` | 505 | 0.6765 | CLAHETransform → GaussianBlur → Normalize | 2 |
| `capsule` | 101 | 0.5887 | CLAHETransform → Normalize | 1 |
| `capsule` | 202 | 0.4416 | CLAHETransform → Normalize | 5 |
| `capsule` | 303 | 0.5281 | CLAHETransform → Normalize | 4 |
| `capsule` | 404 | 0.6320 | CLAHETransform → Normalize | 2 |
| `capsule` | 505 | 0.3680 | CLAHETransform → Normalize | 5 |

## Invariante comprobado

En ninguna corrida Optuna superó el óptimo de la rejilla exhaustiva. Como ambos motores
evalúan el mismo espacio sobre la misma partición, superarlo solo podría indicar
particiones distintas; el guion aborta si ocurre.

## Reportes HTML

Cada categoría genera en local un reporte con la galería de las imágenes más anómalas.
**No se versionan**: embeben miniaturas de MVTec AD, cuya licencia prohíbe redistribuir
las imágenes. Antes de generarlos, el guion comprueba que los puntajes reproducidos dan
el mismo AUC-ROC que informó el motor.
