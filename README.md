# Hanon - CNN Image Denoising and Deblurring

This repository contains notebooks and a small Python package to train and run convolutional neural networks for two restoration tasks:

- image denoising
- image deblurring

The project uses residual learning: the network predicts the degradation (noise or blur), then reconstructs the clean image by subtraction.

## Overview

Current workflow:

1. Train a denoising model in [Remove_noise.ipynb](Remove_noise.ipynb)
2. Train a deblurring model in [Remove_blur.ipynb](Remove_blur.ipynb)
3. Test both trained models on one image (jpg format) in [Examples/Single_image.ipynb](Examples/Single_image.ipynb)

Both training notebooks cache dataset images locally to avoid repeated downloads.

## Repository Notes

- The [Data](Data) folder is not provided on GitHub.
- It is created automatically by the notebooks when needed.
- Cached files such as [Data/Img.npy](Data/Img.npy) are generated locally after first run.

## Requirements

Python 3.10+ and the packages listed in [requirements.txt](requirements.txt):

```bash
pip install -r requirements.txt
```

## How To Run

1. Open Jupyter Notebook from the repository root.
2. Run [Remove_noise.ipynb](Remove_noise.ipynb) to train and save [CNN/dncnn_model.pth](CNN/dncnn_model.pth).
3. Run [Remove_blur.ipynb](Remove_blur.ipynb) to train and save [CNN/Blur_cnn_model.pth](CNN/Blur_cnn_model.pth).
4. Run [Examples/Single_image.ipynb](Examples/Single_image.ipynb) to apply denoising then deblurring on a single image.

## Project Structure

```text
Hanon/
|-- README.md
|-- requirements.txt
|-- Remove_noise.ipynb
|-- Remove_blur.ipynb
|-- CNN/
|   |-- dncnn_model.pth
|   `-- Blur_cnn_model.pth
|-- Data/                 # Created automatically at runtime, not provided on GitHub
|-- Examples/
|   `-- Single_image.ipynb
|-- hanon/                # Shared code used by the notebooks
|   |-- data.py           # Dataset, preprocessing, synthetic degradations
|   `-- training.py       # Training loop helpers
`-- tests/
    |-- test_data.py
    `-- test_training.py
```

## Shared Package

The [hanon](hanon) package contains the code shared across notebooks:

- [hanon/data.py](hanon/data.py): paired image dataset, tensor conversion, grayscale normalization, Gaussian noise and blur generation, LFW loading with local caching
- [hanon/training.py](hanon/training.py): residual-learning training loop

## Tests

Unit tests cover the data pipeline and the training loop. They also run in CI on every push (see [.github/workflows/ci.yml](.github/workflows/ci.yml)).

```bash
pip install pytest
python -m pytest tests
```

## References

- K. Zhang et al., Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising, IEEE TIP, 2017.
- LFW dataset: G. B. Huang et al., Labeled Faces in the Wild, 2007.

**Author**: Adrien Marchandou
**Last Updated**: July 2026
