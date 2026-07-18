import numpy as np
import pytest
import torch

from hanon.data import (
    PairedImageDataset,
    add_gaussian_noise,
    gaussian_blur,
    normalize_to_uint8_range,
    to_tensor,
)


@pytest.fixture
def image():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(32, 32)).astype(np.float32)


class TestPairedImageDataset:
    def test_length_and_items(self, image):
        data = np.stack([image, image])
        dataset = PairedImageDataset(data, data)
        assert len(dataset) == 2
        x, y = dataset[0]
        np.testing.assert_array_equal(x, image)
        np.testing.assert_array_equal(y, image)

    def test_transforms_are_applied(self, image):
        data = np.stack([image])
        dataset = PairedImageDataset(
            data, data, transform=to_tensor, target_transform=to_tensor
        )
        x, y = dataset[0]
        assert x.shape == (1, 32, 32)
        assert y.shape == (1, 32, 32)

    def test_mismatched_lengths_raise(self, image):
        with pytest.raises(ValueError):
            PairedImageDataset(np.stack([image, image]), np.stack([image]))


def test_to_tensor_shape_dtype_and_range(image):
    tensor = to_tensor(image)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 32, 32)
    assert tensor.dtype == torch.float32
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0


def test_normalize_to_uint8_range(image):
    normalized = normalize_to_uint8_range(image / 50.0)
    assert normalized.max() == pytest.approx(255.0)
    assert normalized.min() >= 0.0


def test_normalize_to_uint8_range_does_not_mutate_input(image):
    original = image.copy()
    normalize_to_uint8_range(image)
    np.testing.assert_array_equal(image, original)


def test_normalize_to_uint8_range_all_zeros():
    normalized = normalize_to_uint8_range(np.zeros((8, 8)))
    assert normalized.max() == 0.0


def test_add_gaussian_noise_stays_in_valid_range(image):
    noisy = add_gaussian_noise(image, noise_std=50.0)
    assert noisy.dtype == np.uint8
    assert noisy.shape == image.shape
    # With a large std, at least some pixels must have changed
    assert not np.array_equal(noisy, image.astype(np.uint8))


def test_gaussian_blur_preserves_shape_and_mean(image):
    blurred = gaussian_blur(image, sigma=2.0)
    assert blurred.shape == image.shape
    # A normalized kernel with reflect padding preserves the average intensity
    assert blurred.mean() == pytest.approx(image.mean(), rel=1e-3)
    # Blurring must reduce the variance
    assert blurred.var() < image.var()
