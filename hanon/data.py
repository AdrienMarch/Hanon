"""Dataset loading, preprocessing and synthetic degradations (noise, blur).

All images are grayscale, handled as 2D numpy arrays in the 0-255 range on the
numpy side, and as (1, H, W) float tensors in [0, 1] on the PyTorch side.
"""

import os
from typing import Callable, Optional

import numpy as np
import torch
from scipy.ndimage import convolve
from torch.utils.data import Dataset


class PairedImageDataset(Dataset):
    """Dataset of (degraded, clean) image pairs for restoration training.

    Args:
        data: degraded input images, shape (N, H, W).
        target: corresponding clean images, shape (N, H, W).
        transform: optional transform applied to each input image.
        target_transform: optional transform applied to each target image.
    """

    def __init__(
        self,
        data: np.ndarray,
        target: np.ndarray,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        if len(data) != len(target):
            raise ValueError(
                f"data and target must have the same length, "
                f"got {len(data)} and {len(target)}"
            )
        self.data = data
        self.target = target
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple:
        x = self.data[index]
        if self.transform:
            x = self.transform(x)

        y = self.target[index]
        if self.target_transform:
            y = self.target_transform(y)

        return x, y


def to_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert a (H, W) image in the 0-255 range to a (1, H, W) float tensor in [0, 1]."""
    return torch.as_tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0


def normalize_to_uint8_range(image: np.ndarray) -> np.ndarray:
    """Rescale an image so its maximum value maps to 255.

    Returns a new float array; the input is left untouched.
    """
    peak = image.max()
    if peak <= 0:
        return np.zeros_like(image, dtype=np.float32)
    return image.astype(np.float32) * (255.0 / peak)


def add_gaussian_noise(image: np.ndarray, noise_std: float = 20.0) -> np.ndarray:
    """Add zero-mean Gaussian noise to a 0-255 image and clip back to [0, 255].

    Args:
        image: clean image in the 0-255 range.
        noise_std: standard deviation of the noise, in pixel-value units.
    """
    noise = np.random.randn(*image.shape) * noise_std
    return np.clip(image + noise, 0, 255).astype(np.uint8)


def gaussian_blur(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Blur an image with a normalized 2D Gaussian kernel.

    The kernel size follows the usual ~6*sigma rule, forced odd, with a
    minimum of 3. Borders use reflect padding.
    """
    kernel_size = max(3, int(6 * sigma))
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Separable Gaussian: build the 1D kernel, then take the outer product.
    ax = np.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=np.float64)
    kernel_1d = np.exp(-0.5 * (ax / sigma) ** 2)
    kernel_1d /= kernel_1d.sum()
    kernel_2d = np.outer(kernel_1d, kernel_1d)

    return convolve(image, kernel_2d, mode="reflect")


def load_lfw_images(cache_dir: str = "Data") -> np.ndarray:
    """Load the LFW face images, downloading and caching them on first use.

    Args:
        cache_dir: directory where the raw images are cached as a .npy file.

    Returns:
        Array of grayscale face images, shape (N, H, W).
    """
    cache_path = os.path.join(cache_dir, "Img.npy")
    if os.path.exists(cache_path):
        print("Image dataset loaded from cache.")
        return np.load(cache_path)

    # First run: fetch from sklearn and cache locally.
    from sklearn.datasets import fetch_lfw_people

    faces = fetch_lfw_people(min_faces_per_person=30)
    print("Image dataset downloaded from sklearn.")

    os.makedirs(cache_dir, exist_ok=True)
    np.save(cache_path, faces.images)
    print("Image dataset cached to disk.")

    return faces.images
