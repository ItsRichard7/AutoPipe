"""
meta_recommender -- warm-start recommender over the meta-feature space.

Public API:
    MetaRecommender  nearest-neighbour recommender, fitted per LODO fold
    ESCALADOS        supported scalings of the meta-feature space
    METRICAS         supported distance metrics
"""

from autopipe.meta_recommender.recommender import ESCALADOS, METRICAS, MetaRecommender

__all__ = [
    "MetaRecommender",
    "ESCALADOS",
    "METRICAS",
]
