"""
AutoPipe - Task-agnostic automatic pipeline selection with meta-learned warm start.

Main modules:
    data_loader     -- image dataset loading and validation
    pipeline        -- preprocessing transform definition and composition
    tasks           -- learning task definitions and metric registry
    search_engine   -- automatic pipeline search (grid search / Optuna)
    search_space    -- canonical, versioned search spaces shared across datasets
    meta_features   -- dataset meta-feature extraction, on top of pymfe
    meta_knowledge  -- the meta-knowledge base the warm-start recommender queries
    meta_recommender -- nearest-neighbour warm-start recommender over meta-features
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
from autopipe.meta_features import MetaFeatureError, MetaFeatureExtractor
from autopipe.meta_knowledge import MetaKnowledgeBase, MetaKnowledgeEntry
from autopipe.meta_recommender import ESTRATEGIAS, MetaRecommender
from autopipe.parallel_runner import ParallelRunner
from autopipe.pipeline import (
    BaseTransform,
    CLAHETransform,
    EncodeCategorical,
    GammaCorrection,
    GaussianBlur,
    GrayscaleConversion,
    ImputeMissing,
    MedianFilter,
    Normalize,
    NotFittedError,
    Pipeline,
    Resize,
    ScaleNumeric,
    SelectFeatures,
    StatefulTransform,
    TabularTransform,
)
from autopipe.reporter import ComparativeReportGenerator, ReportGenerator
from autopipe.search_engine import GridSearchEngine, OptunaSearchEngine
from autopipe.search_space import (
    TABULAR_SPACE_STEPS,
    TABULAR_SPACE_VERSION,
    build_tabular_search_space,
    canonical_configuration,
    describe_tabular_search_space,
    omitted_steps,
    pipeline_from_configuration,
    validate_space_version,
)
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
    "TabularTransform",
    "ImputeMissing",
    "ScaleNumeric",
    "EncodeCategorical",
    "SelectFeatures",
    "GridSearchEngine",
    "OptunaSearchEngine",
    "TABULAR_SPACE_VERSION",
    "TABULAR_SPACE_STEPS",
    "build_tabular_search_space",
    "validate_space_version",
    "describe_tabular_search_space",
    "pipeline_from_configuration",
    "omitted_steps",
    "canonical_configuration",
    "ParallelRunner",
    "ReportGenerator",
    "ComparativeReportGenerator",
    "MetaFeatureExtractor",
    "MetaFeatureError",
    "MetaKnowledgeBase",
    "MetaKnowledgeEntry",
    "MetaRecommender",
    "ESTRATEGIAS",
]