# examples/tabular_quickstart.py
"""Flujo completo de AutoPipe sobre datos tabulares, de carga a pipeline ganador.

Este archivo sirve además como verificación del criterio de éxito de ingeniería
del anteproyecto (Sección 4.3.5): el flujo completo para la persona que usa la
librería debe caber en no más de 20 líneas de código. La prueba
``tests/test_examples.py`` cuenta las **instrucciones** del archivo y lo ejecuta,
de modo que el criterio queda comprobado de forma continua y no de palabra.

Se cuentan instrucciones y no líneas físicas porque el límite de 100 caracteres
del proyecto obliga a repartir una sola instrucción en varias líneas; contar
líneas físicas mediría el estilo de formato y no la complejidad del flujo.

El espacio de búsqueda no se escribe a mano: se pide el **canónico versionado**,
que es el mismo para todos los datasets y el que hace comparables entre sí las
configuraciones halladas (véase ``docs/experimentos/ESPACIO-tabular-v1.md``).

Ejecutar con:
    .\\.venv\\Scripts\\python.exe examples\\tabular_quickstart.py
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from autopipe import (
    TABULAR_SPACE_VERSION,
    GridSearchEngine,
    TabularDataLoader,
    build_tabular_search_space,
)

rng = np.random.default_rng(0)
clase = np.tile([0, 1, 2], 30)
tabla = pd.DataFrame(
    {
        "medida": clase * 2.0 + rng.normal(0, 0.5, size=90),
        "ruido": rng.normal(0, 1.0, size=90),
        "categoria": np.where(rng.random(90) < 0.8, clase, rng.integers(0, 3, 90)),
        "clase": clase,
    }
)
tabla["categoria"] = tabla["categoria"].map({0: "norte", 1: "sur", 2: "este"})
features, labels = TabularDataLoader(tabla).load()
motor = GridSearchEngine(
    build_tabular_search_space(), val_size=0.3, random_state=42, task="classification"
)
mejor, metricas, resultados = motor.run(features, labels, KNeighborsClassifier())
print(f"Espacio de búsqueda: {TABULAR_SPACE_VERSION}")
print(f"Pipeline ganador: {mejor}")
print(f"macro-F1: {metricas['macro_f1']:.3f} sobre {len(resultados)} candidatos")
