"""
data_loader -- image dataset loading and validation.

Public API:
    ImageDataLoader  loads images from a directory structured as normal/ and anomaly/
"""

from autopipe.data_loader.loader import ImageDataLoader

__all__ = ["ImageDataLoader"]