"""
pipeline -- preprocessing transform definition and pipeline composition.

Public API:
    BaseTransform    abstract base class for all transforms
    Pipeline         ordered sequence of BaseTransform steps
    GaussianBlur     built-in Gaussian blur transform
    Normalize        built-in normalization transform
    Resize           built-in resize transform
    CLAHETransform   built-in CLAHE contrast transform
    MedianFilter     built-in median filter transform
    GammaCorrection  built-in gamma correction transform
    GrayscaleConversion  built-in RGB-to-grayscale transform
"""

from autopipe.pipeline.base import BaseTransform
from autopipe.pipeline.pipeline import Pipeline
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
    "Pipeline",
    "GaussianBlur",
    "Normalize",
    "Resize",
    "CLAHETransform",
    "MedianFilter",
    "GammaCorrection",
    "GrayscaleConversion",
]
