"""
pipeline -- preprocessing transform definition and pipeline composition.

Public API:
    BaseTransform    abstract base class for all transforms
    StatefulTransform  abstract base class for transforms that require fitting
    NotFittedError   raised when a stateful transform is used before fitting
    Pipeline         ordered sequence of BaseTransform steps
    GaussianBlur     built-in Gaussian blur transform
    Normalize        built-in normalization transform
    Resize           built-in resize transform
    CLAHETransform   built-in CLAHE contrast transform
    MedianFilter     built-in median filter transform
    GammaCorrection  built-in gamma correction transform
    GrayscaleConversion  built-in RGB-to-grayscale transform
    TabularTransform     base class for stateful tabular transforms
    ImputeMissing        fills missing values with a statistic learned on train
    ScaleNumeric         scales numeric columns with statistics learned on train
    EncodeCategorical    encodes categorical columns with categories seen on train
"""

from autopipe.pipeline.base import BaseTransform, NotFittedError, StatefulTransform
from autopipe.pipeline.pipeline import Pipeline
from autopipe.pipeline.tabular_transforms import (
    EncodeCategorical,
    ImputeMissing,
    ScaleNumeric,
    TabularTransform,
)
from autopipe.pipeline.transforms import (
    CLAHETransform,
    GammaCorrection,
    GaussianBlur,
    GrayscaleConversion,
    MedianFilter,
    Normalize,
    Resize,
)

__all__ = [
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
]
