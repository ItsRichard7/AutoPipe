# tests/test_pipeline.py
"""Pruebas unitarias para el módulo autopipe.pipeline."""

from __future__ import annotations

import logging

import numpy as np
import pytest

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


@pytest.fixture
def rgb_image() -> np.ndarray:
    """Imagen RGB uint8 determinista de 16x16."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)


# --------------------------------------------------------------------------- #
# BaseTransform
# --------------------------------------------------------------------------- #
def test_base_transform_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        BaseTransform()  # type: ignore[abstract]


def test_incomplete_subclass_cannot_be_instantiated() -> None:
    class Incomplete(BaseTransform):
        def apply(self, image: np.ndarray) -> np.ndarray:
            return image

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]


def test_repr_includes_params() -> None:
    assert repr(GaussianBlur(kernel_size=3)) == "GaussianBlur(kernel_size=3)"


# --------------------------------------------------------------------------- #
# GaussianBlur
# --------------------------------------------------------------------------- #
def test_gaussian_blur_preserves_shape(rgb_image: np.ndarray) -> None:
    result = GaussianBlur(kernel_size=3).apply(rgb_image)
    assert result.shape == rgb_image.shape


def test_gaussian_blur_get_params() -> None:
    assert GaussianBlur(kernel_size=5).get_params() == {"kernel_size": 5}


@pytest.mark.parametrize("invalid", [0, -1, 4, 2])
def test_gaussian_blur_invalid_kernel_raises(invalid: int) -> None:
    with pytest.raises(ValueError):
        GaussianBlur(kernel_size=invalid)


# --------------------------------------------------------------------------- #
# Normalize
# --------------------------------------------------------------------------- #
def test_normalize_scales_integer_images() -> None:
    image = np.full((4, 4, 3), 128, dtype=np.uint8)
    result = Normalize(mean=0.5, std=0.5).apply(image)
    expected = (128 / 255.0 - 0.5) / 0.5
    assert result.dtype == np.float32
    assert np.allclose(result, expected, atol=1e-5)


def test_normalize_get_params() -> None:
    assert Normalize(mean=0.1, std=0.2).get_params() == {"mean": 0.1, "std": 0.2}


def test_normalize_zero_std_raises() -> None:
    with pytest.raises(ValueError):
        Normalize(mean=0.0, std=0.0)


# --------------------------------------------------------------------------- #
# Resize
# --------------------------------------------------------------------------- #
def test_resize_changes_shape(rgb_image: np.ndarray) -> None:
    result = Resize(width=8, height=4).apply(rgb_image)
    assert result.shape[:2] == (4, 8)


def test_resize_get_params() -> None:
    assert Resize(width=32, height=24).get_params() == {"width": 32, "height": 24}


@pytest.mark.parametrize("width,height", [(0, 4), (4, -1), (4, 0)])
def test_resize_invalid_dims_raise(width: int, height: int) -> None:
    with pytest.raises(ValueError):
        Resize(width=width, height=height)


# --------------------------------------------------------------------------- #
# CLAHETransform
# --------------------------------------------------------------------------- #
def test_clahe_color_image(rgb_image: np.ndarray) -> None:
    result = CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8)).apply(rgb_image)
    assert result.shape == rgb_image.shape
    assert result.dtype == np.uint8


def test_clahe_grayscale_image() -> None:
    rng = np.random.default_rng(1)
    gray = rng.integers(0, 256, size=(16, 16), dtype=np.uint8)
    result = CLAHETransform(clip_limit=2.0, tile_grid_size=(4, 4)).apply(gray)
    assert result.shape == gray.shape
    assert result.dtype == np.uint8


def test_clahe_get_params() -> None:
    params = CLAHETransform(clip_limit=3.0, tile_grid_size=(8, 8)).get_params()
    assert params == {"clip_limit": 3.0, "tile_grid_size": (8, 8)}


def test_clahe_invalid_clip_limit_raises() -> None:
    with pytest.raises(ValueError):
        CLAHETransform(clip_limit=0.0, tile_grid_size=(8, 8))


@pytest.mark.parametrize("grid", [(8,), (8, 8, 8), (0, 8), (8, -1)])
def test_clahe_invalid_grid_raises(grid: tuple) -> None:
    with pytest.raises(ValueError):
        CLAHETransform(clip_limit=2.0, tile_grid_size=grid)


# --------------------------------------------------------------------------- #
# MedianFilter
# --------------------------------------------------------------------------- #
def test_median_filter_preserves_shape(rgb_image: np.ndarray) -> None:
    result = MedianFilter(kernel_size=3).apply(rgb_image)
    assert result.shape == rgb_image.shape


def test_median_filter_get_params() -> None:
    assert MedianFilter(kernel_size=5).get_params() == {"kernel_size": 5}


@pytest.mark.parametrize("invalid", [1, 2, 4, -3])
def test_median_filter_invalid_kernel_raises(invalid: int) -> None:
    with pytest.raises(ValueError):
        MedianFilter(kernel_size=invalid)


# --------------------------------------------------------------------------- #
# GammaCorrection
# --------------------------------------------------------------------------- #
def test_gamma_correction_preserves_shape_and_dtype(rgb_image: np.ndarray) -> None:
    result = GammaCorrection(gamma=2.0).apply(rgb_image)
    assert result.shape == rgb_image.shape
    assert result.dtype == np.uint8


def test_gamma_less_than_one_brightens() -> None:
    image = np.full((4, 4), 100, dtype=np.uint8)
    brightened = GammaCorrection(gamma=0.5).apply(image)
    assert brightened.mean() > image.mean()


def test_gamma_one_is_identity() -> None:
    image = np.arange(256, dtype=np.uint8).reshape(16, 16)
    assert np.array_equal(GammaCorrection(gamma=1.0).apply(image), image)


def test_gamma_on_float_image() -> None:
    image = np.full((4, 4), 0.5, dtype=np.float64)
    result = GammaCorrection(gamma=2.0).apply(image)
    assert result.dtype.kind == "f"
    assert np.allclose(result, 0.25)


def test_gamma_get_params() -> None:
    assert GammaCorrection(gamma=2.2).get_params() == {"gamma": 2.2}


@pytest.mark.parametrize("invalid", [0.0, -1.0])
def test_gamma_invalid_raises(invalid: float) -> None:
    with pytest.raises(ValueError):
        GammaCorrection(gamma=invalid)


# --------------------------------------------------------------------------- #
# GrayscaleConversion
# --------------------------------------------------------------------------- #
def test_grayscale_converts_rgb_to_2d(rgb_image: np.ndarray) -> None:
    result = GrayscaleConversion().apply(rgb_image)
    assert result.ndim == 2
    assert result.shape == rgb_image.shape[:2]


def test_grayscale_is_noop_on_2d() -> None:
    rng = np.random.default_rng(2)
    gray = rng.integers(0, 256, size=(8, 8), dtype=np.uint8)
    assert np.array_equal(GrayscaleConversion().apply(gray), gray)


def test_grayscale_get_params() -> None:
    assert GrayscaleConversion().get_params() == {}


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def test_pipeline_applies_steps_in_order(rgb_image: np.ndarray) -> None:
    pipeline = Pipeline([Resize(width=8, height=8), GaussianBlur(kernel_size=3)])
    result = pipeline.apply(rgb_image)
    assert result.shape[:2] == (8, 8)


def test_pipeline_add_is_chainable(rgb_image: np.ndarray) -> None:
    pipeline = Pipeline([]).add(Resize(width=8, height=8)).add(GaussianBlur(kernel_size=3))
    assert len(pipeline.steps) == 2
    assert pipeline.apply(rgb_image).shape[:2] == (8, 8)


def test_pipeline_add_returns_self() -> None:
    pipeline = Pipeline([])
    assert pipeline.add(Resize(width=8, height=8)) is pipeline


def test_pipeline_add_rejects_non_transform() -> None:
    with pytest.raises(TypeError):
        Pipeline([]).add(object())


def test_pipeline_add_warns_normalize_before_clahe(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="autopipe.pipeline.pipeline"):
        Pipeline([Normalize(mean=0.5, std=0.5)]).add(
            CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8))
        )
    assert any("incompatible" in record.message for record in caplog.records)


def test_pipeline_repr_lists_steps() -> None:
    pipeline = Pipeline([Resize(width=8, height=8), GaussianBlur(kernel_size=3)])
    assert repr(pipeline) == "Pipeline([Resize(width=8, height=8), GaussianBlur(kernel_size=3)])"


def test_empty_pipeline_returns_input(rgb_image: np.ndarray) -> None:
    result = Pipeline([]).apply(rgb_image)
    assert np.array_equal(result, rgb_image)


def test_pipeline_get_params_lists_each_step() -> None:
    pipeline = Pipeline([Resize(width=8, height=8), Normalize(mean=0.5, std=0.5)])
    params = pipeline.get_params()
    assert params == {
        "0_Resize": {"width": 8, "height": 8},
        "1_Normalize": {"mean": 0.5, "std": 0.5},
    }


def test_pipeline_rejects_non_transform_step() -> None:
    with pytest.raises(TypeError):
        Pipeline([GaussianBlur(kernel_size=3), object()])


def test_pipeline_warns_normalize_before_clahe(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="autopipe.pipeline.pipeline"):
        Pipeline(
            [Normalize(mean=0.5, std=0.5), CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8))]
        )
    assert any("incompatible" in record.message for record in caplog.records)


def test_pipeline_no_warning_clahe_before_normalize(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="autopipe.pipeline.pipeline"):
        Pipeline(
            [CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8)), Normalize(mean=0.5, std=0.5)]
        )
    assert not caplog.records
