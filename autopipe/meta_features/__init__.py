"""
meta_features -- dataset meta-feature extraction, on top of pymfe.

Public API:
    MetaFeatureExtractor  extracts the meta-feature vector of a tabular dataset
    MetaFeatureError      raised when pymfe cannot extract a group
    columnas_no_finitas   names whose value is not finite, for corpus-level rules
"""

from autopipe.meta_features.extractor import (
    MetaFeatureError,
    MetaFeatureExtractor,
    columnas_no_finitas,
)

__all__ = [
    "MetaFeatureExtractor",
    "MetaFeatureError",
    "columnas_no_finitas",
]
