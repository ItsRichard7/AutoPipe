"""
meta_recommender -- warm-start recommender over the meta-feature space.

Public API:
    MetaRecommender     nearest-neighbour recommender, fitted per LODO fold
    ESCALADOS           supported scalings of the meta-feature space
    METRICAS            supported distance metrics
    ESTRATEGIAS         supported recommendation strategies, with their controls
    FACTOR_DIVERSIDAD   candidates considered per pick by the "diverse" strategy
"""

from autopipe.meta_recommender.recommender import (
    ESCALADOS,
    ESTRATEGIAS,
    FACTOR_DIVERSIDAD,
    METRICAS,
    MetaRecommender,
)

__all__ = [
    "MetaRecommender",
    "ESCALADOS",
    "METRICAS",
    "ESTRATEGIAS",
    "FACTOR_DIVERSIDAD",
]
