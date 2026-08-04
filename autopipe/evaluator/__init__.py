"""
evaluator -- dataset splitting and metric calculation.

Public API:
    DataSplitter        splits a dataset into train/validation subsets (k-fold capable)
    MetricCalculator    computes AUC-ROC, F1, precision, recall, FPR
"""

from autopipe.evaluator.metrics import MetricCalculator
from autopipe.evaluator.splitter import DataSplitter

__all__ = ["DataSplitter", "MetricCalculator"]