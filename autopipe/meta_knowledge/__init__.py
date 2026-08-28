"""
meta_knowledge -- the meta-knowledge base the warm-start recommender queries.

Public API:
    MetaKnowledgeEntry  what is known about one corpus dataset
    MetaKnowledgeBase   the collection, with its load/save and fold helpers
"""

from autopipe.meta_knowledge.base import MetaKnowledgeBase, MetaKnowledgeEntry

__all__ = [
    "MetaKnowledgeBase",
    "MetaKnowledgeEntry",
]
