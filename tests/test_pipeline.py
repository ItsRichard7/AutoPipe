# tests/test_pipeline.py
"""Pruebas unitarias para el módulo autopipe.pipeline."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from autopipe.pipeline.base import (
    BaseTransform,
    Dataset,
    NotFittedError,
    StatefulTransform,
)
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
# BaseTransform: contrato fit/transform (ADR-001)
# --------------------------------------------------------------------------- #
STATELESS_TRANSFORMS = (
    GaussianBlur(kernel_size=3),
    MedianFilter(kernel_size=3),
    Normalize(mean=0.5, std=0.5),
    Resize(width=8, height=8),
    CLAHETransform(clip_limit=2.0, tile_grid_size=(4, 4)),
    GammaCorrection(gamma=1.5),
    GrayscaleConversion(),
)


@pytest.mark.parametrize("transform", STATELESS_TRANSFORMS, ids=lambda t: type(t).__name__)
def test_fit_is_a_noop_for_stateless_transforms(
    transform: BaseTransform, rgb_image: np.ndarray
) -> None:
    """El ajuste por defecto no aprende nada ni altera la configuración."""
    params_before = transform.get_params()
    result = transform.fit([rgb_image, rgb_image])
    assert result is transform
    assert transform.get_params() == params_before


@pytest.mark.parametrize("transform", STATELESS_TRANSFORMS, ids=lambda t: type(t).__name__)
def test_transform_matches_apply_per_sample(
    transform: BaseTransform, rgb_image: np.ndarray
) -> None:
    """``transform`` por lote equivale a aplicar ``apply`` muestra por muestra."""
    dataset = [rgb_image, rgb_image[::-1]]
    expected = [transform.apply(sample) for sample in dataset]
    obtained = transform.transform(dataset)

    assert len(obtained) == len(expected)
    for obtained_sample, expected_sample in zip(obtained, expected, strict=True):
        np.testing.assert_array_equal(obtained_sample, expected_sample)


def test_fit_ignores_labels_by_default(rgb_image: np.ndarray) -> None:
    """Las etiquetas son opcionales y el ajuste por defecto las ignora."""
    transform = GaussianBlur(kernel_size=3)
    assert transform.fit([rgb_image], np.array([0])) is transform


def test_fit_transform_equals_fit_then_transform(rgb_image: np.ndarray) -> None:
    """``fit_transform`` es equivalente a ajustar y transformar por separado."""
    dataset = [rgb_image, rgb_image[::-1]]
    combined = Normalize(mean=0.5, std=0.5).fit_transform(dataset)
    separate = Normalize(mean=0.5, std=0.5).fit(dataset).transform(dataset)

    for combined_sample, separate_sample in zip(combined, separate, strict=True):
        np.testing.assert_array_equal(combined_sample, separate_sample)


def test_transform_preserves_order() -> None:
    """El orden de las muestras transformadas coincide con el de entrada."""
    samples = [np.full((4, 4), value, dtype=np.uint8) for value in (10, 20, 30)]
    obtained = GammaCorrection(gamma=1.0).transform(samples)

    assert [int(sample[0, 0]) for sample in obtained] == [10, 20, 30]


def test_transform_of_empty_dataset_is_empty() -> None:
    """Un conjunto vacío se transforma en un conjunto vacío."""
    assert GaussianBlur(kernel_size=3).transform([]) == []


def test_subclass_inherits_stateless_defaults() -> None:
    """Una subclase que solo implementa el contrato mínimo hereda fit/transform."""

    class Doubler(BaseTransform):
        def apply(self, image: np.ndarray) -> np.ndarray:
            return image * 2

        def get_params(self) -> dict:
            return {}

    transform = Doubler()
    obtained = transform.fit_transform([np.ones((2, 2)), np.zeros((2, 2))])

    np.testing.assert_array_equal(obtained[0], np.full((2, 2), 2.0))
    np.testing.assert_array_equal(obtained[1], np.zeros((2, 2)))


def test_transform_on_tabular_matrix_iterates_rows() -> None:
    """Sobre una matriz 2-D, las muestras son las filas."""
    matrix = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    obtained = GammaCorrection(gamma=1.0).transform(matrix)

    assert len(obtained) == 3
    np.testing.assert_allclose(np.vstack(obtained), matrix)


# --------------------------------------------------------------------------- #
# StatefulTransform (ADR-001)
# --------------------------------------------------------------------------- #
class MeanCenterer(StatefulTransform):
    """Doble de prueba: centra cada columna con la media aprendida en el ajuste."""

    def _fit(self, X: Dataset, y: np.ndarray | None = None) -> None:
        self.mean_ = np.asarray(X, dtype=float).mean(axis=0)

    def _transform(self, X: Dataset) -> Dataset:
        return np.asarray(X, dtype=float) - self.mean_

    def get_params(self) -> dict:
        return {}


def test_stateful_transform_starts_unfitted() -> None:
    assert MeanCenterer().fitted is False


def test_fit_marks_as_fitted_and_returns_self() -> None:
    transform = MeanCenterer()
    result = transform.fit(np.array([[1.0], [3.0]]))

    assert result is transform
    assert transform.fitted is True


def test_transform_before_fit_raises_not_fitted() -> None:
    with pytest.raises(NotFittedError, match="MeanCenterer"):
        MeanCenterer().transform(np.array([[1.0], [2.0]]))


def test_apply_before_fit_raises_not_fitted() -> None:
    with pytest.raises(NotFittedError, match="MeanCenterer"):
        MeanCenterer().apply(np.array([1.0]))


def test_fit_uses_only_the_given_subset() -> None:
    """El estado se aprende del subconjunto de ajuste, no del que se transforma.

    Es la garantía que evita la fuga de datos: la validación se transforma con
    las estadísticas del entrenamiento, de modo que **no** queda centrada.
    """
    train = np.array([[0.0], [10.0]])          # media de entrenamiento = 5.0
    validation = np.array([[100.0], [200.0]])  # media propia = 150.0

    transform = MeanCenterer().fit(train)
    transformed_validation = np.asarray(transform.transform(validation))

    np.testing.assert_allclose(transformed_validation, np.array([[95.0], [195.0]]))
    assert transformed_validation.mean() != pytest.approx(0.0)


def test_refit_replaces_previous_state() -> None:
    transform = MeanCenterer().fit(np.array([[0.0], [10.0]]))
    transform.fit(np.array([[100.0], [100.0]]))

    np.testing.assert_allclose(np.asarray(transform.transform(np.array([[100.0]]))), [[0.0]])


def test_apply_matches_the_corresponding_row_of_transform() -> None:
    dataset = np.array([[1.0, 2.0], [3.0, 6.0]])
    transform = MeanCenterer().fit(dataset)

    by_batch = np.asarray(transform.transform(dataset))
    by_sample = transform.apply(dataset[0])

    np.testing.assert_allclose(by_sample, by_batch[0])


def test_incomplete_stateful_subclass_cannot_be_instantiated() -> None:
    """No implementar el ajuste impide instanciar la subclase."""

    class MissingFit(StatefulTransform):
        def _transform(self, X: Dataset) -> Dataset:
            return X

        def get_params(self) -> dict:
            return {}

    with pytest.raises(TypeError):
        MissingFit()  # type: ignore[abstract]


def test_not_fitted_is_raised_even_without_calling_super_init() -> None:
    """El estado de ajuste vive en un atributo de clase, no en ``__init__``.

    Una subclase que define su propio ``__init__`` sin invocar al de la clase
    base debe seguir levantando ``NotFittedError``, no ``AttributeError``.
    """

    class OwnInit(MeanCenterer):
        def __init__(self) -> None:  # noqa: D107 - doble de prueba
            self.extra = 1

    with pytest.raises(NotFittedError):
        OwnInit().transform(np.array([[1.0]]))


def test_stateful_transform_is_a_base_transform() -> None:
    """El contrato con estado sigue siendo un ``BaseTransform``."""
    assert isinstance(MeanCenterer(), BaseTransform)


# --------------------------------------------------------------------------- #
# Pipeline: fit / transform / fit_transform (ADR-001)
# --------------------------------------------------------------------------- #
class Doubler(BaseTransform):
    """Doble de prueba sin estado: multiplica cada muestra por dos."""

    def apply(self, image: np.ndarray) -> np.ndarray:
        return np.asarray(image, dtype=float) * 2

    def get_params(self) -> dict:
        return {}


class CountingCenterer(MeanCenterer):
    """Doble de prueba que cuenta cuántas veces se le pide transformar."""

    def __init__(self) -> None:
        self.transform_calls = 0

    def _transform(self, X: Dataset) -> Dataset:
        self.transform_calls += 1
        return super()._transform(X)


def test_empty_pipeline_returns_the_input_unchanged() -> None:
    dataset = np.array([[1.0], [2.0]])
    pipeline = Pipeline([])

    assert pipeline.fit(dataset) is pipeline
    np.testing.assert_array_equal(np.asarray(pipeline.transform(dataset)), dataset)


def test_fit_returns_the_pipeline() -> None:
    pipeline = Pipeline([MeanCenterer()])

    assert pipeline.fit(np.array([[1.0], [3.0]])) is pipeline


def test_transform_chains_the_steps_in_order() -> None:
    """Con pasos sin estado, transformar equivale a aplicar paso por paso."""
    dataset = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    pipeline = Pipeline([Doubler(), Doubler()])

    obtenido = pipeline.fit_transform(dataset)

    for muestra, esperada in zip(obtenido, dataset, strict=True):
        np.testing.assert_allclose(np.asarray(muestra), esperada * 4)


def test_steps_are_fitted_on_the_output_of_the_previous_step() -> None:
    """El ajuste es secuencial, no sobre los datos originales.

    El centrador va después de un duplicador, así que debe aprender la media de
    los datos **ya duplicados** (4.0), no la de los originales (2.0). Si se
    ajustara sobre los originales, transformar daría ``[[0], [4]]``.
    """
    train = np.array([[1.0], [3.0]])
    pipeline = Pipeline([Doubler(), MeanCenterer()]).fit(train)

    np.testing.assert_allclose(
        np.asarray(pipeline.transform(train)), np.array([[-2.0], [2.0]])
    )


def test_fit_transform_matches_fit_then_transform() -> None:
    train = np.array([[1.0], [3.0], [5.0]])

    combinado = np.asarray(Pipeline([Doubler(), MeanCenterer()]).fit_transform(train))
    separado = np.asarray(Pipeline([Doubler(), MeanCenterer()]).fit(train).transform(train))

    np.testing.assert_allclose(combinado, separado)


def test_fit_does_not_transform_with_the_last_step() -> None:
    """La salida del último paso se descartaría, así que no se calcula."""
    pipeline = Pipeline([CountingCenterer()])
    pipeline.fit(np.array([[1.0], [3.0]]))

    assert pipeline.steps[0].transform_calls == 0

    pipeline.transform(np.array([[1.0]]))
    assert pipeline.steps[0].transform_calls == 1


def test_transform_before_fit_propagates_not_fitted() -> None:
    with pytest.raises(NotFittedError):
        Pipeline([Doubler(), MeanCenterer()]).transform(np.array([[1.0], [2.0]]))


def test_pipeline_fitted_on_train_uses_train_statistics_on_validation() -> None:
    """Prueba anti-fuga a nivel de pipeline.

    La validación se transforma con las estadísticas del entrenamiento, de modo
    que **no** queda centrada en cero.
    """
    train = np.array([[0.0], [10.0]])          # media de entrenamiento = 5.0
    validation = np.array([[100.0], [200.0]])

    pipeline = Pipeline([MeanCenterer()]).fit(train)
    transformada = np.asarray(pipeline.transform(validation))

    np.testing.assert_allclose(transformada, np.array([[95.0], [195.0]]))
    assert transformada.mean() != pytest.approx(0.0)


def test_stateless_pipeline_transform_matches_apply_per_sample(
    rgb_image: np.ndarray,
) -> None:
    """Un pipeline de imagen produce lo mismo por lote que muestra por muestra."""
    dataset = [rgb_image, rgb_image[::-1]]
    pipeline = Pipeline([Resize(width=8, height=8), GaussianBlur(kernel_size=3)])

    por_lote = pipeline.fit_transform(dataset)
    por_muestra = [pipeline.apply(sample) for sample in dataset]

    for obtenida, esperada in zip(por_lote, por_muestra, strict=True):
        np.testing.assert_array_equal(obtenida, esperada)


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
