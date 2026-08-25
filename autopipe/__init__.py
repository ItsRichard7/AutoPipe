"""
AutoPipe - Automatic Pipeline Selection for Anomaly Detection in Image Datasets.

Main modules:
    data_loader     -- image dataset loading and validation
    pipeline        -- preprocessing transform definition and composition
    tasks           -- learning task definitions and metric registry
    search_engine   -- automatic pipeline search (grid search / Optuna)
    evaluator       -- dataset splitting and metric calculation
    parallel_runner -- parallel pipeline evaluation
    reporter        -- report generation (PDF / HTML)
"""

__version__ = "0.1.0"
__author__ = "Ricardo Antonio Borbón Mena"

from autopipe.data_loader import ImageDataLoader, TabularDataLoader
from autopipe.evaluator import (
    ClassificationMetricCalculator,
    DataSplitter,
    MetricCalculator,
)
from autopipe.parallel_runner import ParallelRunner
from autopipe.pipeline import (
    BaseTransform,
    CLAHETransform,
    GammaCorrection,
    GaussianBlur,
    GrayscaleConversion,
    MedianFilter,
    Normalize,
    NotFittedError,
    Pipeline,
    Resize,
    StatefulTransform,
)
from autopipe.reporter import ReportGenerator
from autopipe.search_engine import GridSearchEngine, OptunaSearchEngine
from autopipe.tasks import (
    ANOMALY_DETECTION,
    CLASSIFICATION,
    TASKS,
    Task,
    get_task,
)

__all__ = [
    "ImageDataLoader",
    "TabularDataLoader",
    "DataSplitter",
    "MetricCalculator",
    "ClassificationMetricCalculator",
    "Task",
    "ANOMALY_DETECTION",
    "CLASSIFICATION",
    "TASKS",
    "get_task",
    "BaseTransform",
    "StatefulTransform",
    "NotFittedError",
    "Pipeline",
    "GaussianBlur",
    "Normalize",
    "Resize",
    "CLAHETransform",
    "MedianFilter",
    "GammaCorrection",
    "GrayscaleConversion",
    "GridSearchEngine",
    "OptunaSearchEngine",
    "ParallelRunner",
    "ReportGenerator",
]