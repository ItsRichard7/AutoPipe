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

Ejecutar con:
    .\\.venv\\Scripts\\python.exe examples\\tabular_quickstart.py
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from autopipe import (
    EncodeCategorical,
    GridSearchEngine,
    ImputeMissing,
    ScaleNumeric,
    TabularDataLoader,
)

rng = np.random.default_rng(0)
clase = np.tile([0, 1, 2], 30)
tabla = pd.DataFrame(
    {
        "medida": clase * 2.0 + rng.normal(0, 0.5, size=90),
        "categoria": np.where(clase == 0, "norte", np.where(clase == 1, "sur", "este")),
        "clase": clase,
    }
)
features, labels = TabularDataLoader(tabla).load()
espacio = {
    "imputacion": [ImputeMissing("mean"), ImputeMissing("median")],
    "codificacion": [EncodeCategorical("onehot"), EncodeCategorical("ordinal")],
    "escalado": [ScaleNumeric("standard"), ScaleNumeric("minmax")],
}
motor = GridSearchEngine(espacio, val_size=0.3, random_state=42, task="classification")
mejor, metricas, resultados = motor.run(features, labels, KNeighborsClassifier())
print(f"Pipeline ganador: {mejor}")
print(f"macro-F1: {metricas['macro_f1']:.3f} sobre {len(resultados)} candidatos")
