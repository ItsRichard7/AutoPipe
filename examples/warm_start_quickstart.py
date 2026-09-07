# examples/warm_start_quickstart.py
"""Arranque en caliente meta-aprendido, de un dataset nuevo al pipeline ganador.

Es el flujo que da nombre al proyecto: en lugar de empezar la búsqueda a ciegas,
se describe el dataset con sus meta-características, se buscan en la base de
meta-conocimiento los datasets más parecidos y se inyectan sus configuraciones
ganadoras como los primeros ensayos.

Igual que ``tabular_quickstart.py``, este archivo verifica el criterio de éxito de
ingeniería del anteproyecto (Sección 4.3.5): el flujo completo para quien usa la
librería cabe en no más de 20 instrucciones, y ``tests/test_examples.py`` las
cuenta y ejecuta el archivo.

**Los cinco ensayos inyectados consumen presupuesto.** Con ``n_trials=30`` la
búsqueda ejecuta treinta ensayos en total y no treinta y cinco, de modo que
quedan veinticinco de exploración propia. Es lo que el pre-registro cuenta dentro
de la variable «ensayos hasta el objetivo».

**Esto no es el experimento.** Aquí el recomendador se ajusta con la base
completa porque el dataset consultado es sintético y no pertenece al corpus. En
la validación *leave-one-dataset-out* hay que pasarle ``base.without(d)``, para
que ni la normalización ni los vecinos vean nunca el dataset excluido.

Ejecutar con:
    .\\.venv\\Scripts\\python.exe examples\\warm_start_quickstart.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from autopipe import (
    MetaFeatureExtractor,
    MetaKnowledgeBase,
    MetaRecommender,
    OptunaSearchEngine,
    TabularDataLoader,
    build_tabular_search_space,
)

BASE = Path(__file__).resolve().parents[1] / "docs/experimentos/BASE-meta-conocimiento.json"

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
vector = MetaFeatureExtractor(random_state=42).extract(features, labels)
recomendador = MetaRecommender().fit(MetaKnowledgeBase.load(BASE))
configuraciones = recomendador.recommend(vector)
motor = OptunaSearchEngine(
    build_tabular_search_space(),
    n_trials=30,
    val_size=0.3,
    random_state=42,
    task="classification",
    warm_start=configuraciones,
)
mejor, metricas, resultados = motor.run(features, labels, KNeighborsClassifier())
print(f"Inyectadas {len(configuraciones)} configuraciones de: "
      f"{', '.join(v['dataset_name'] for v in recomendador.neighbours_)}")
print(f"Pipeline ganador: {mejor}")
print(f"macro-F1: {metricas['macro_f1']:.3f} en {len(resultados)} ensayos")
