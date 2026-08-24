"""
evaluator -- dataset splitting and metric calculation.

Public API:
    DataSplitter        splits a dataset into train/validation subsets (k-fold capable)
    MetricCalculator    computes AUC-ROC, F1, precision, recall, FPR (anomaly detection)
    ClassificationMetricCalculator  computes macro-F1, accuracy, balanced accuracy,
                        macro precision and macro recall (classification)
"""

from autopipe.evaluator.metrics import ClassificationMetricCalculator, MetricCalculator
from autopipe.evaluator.splitter import DataSplitter

__all__ = ["DataSplitter", "MetricCalculator", "ClassificationMetricCalculator"]
