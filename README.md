# Hanon - CNN Image Denoising and Deblurring

This repository contains notebooks and utility code to train and run convolutional neural networks for two restoration tasks:

- image denoising
- image deblurring

The project uses residual learning: the network predicts degradation (noise or blur), then reconstructs the clean image by subtraction.

## Overview

Current workflow in this folder:

1. Train a denoising model in [Remove_noise.ipynb](Remove_noise.ipynb)
2. Train a deblurring model in [Remove_blur.ipynb](Remove_blur.ipynb)
3. Test both trained models on one image (jpg format) in [Examples/Single_image.ipynb](Examples/Single_image.ipynb)

Both training notebooks cache dataset images locally to avoid repeated downloads.

## Repository Notes

- The [Data](Data) folder is not intended to be provided on GitHub.
- It is created automatically by the notebooks when needed (for example via os.makedirs(..., exist_ok=True)).
- Cached files such as [Data/Img.npy](Data/Img.npy) are generated locally after first run.

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- SciPy
- scikit-learn
- Pillow
- Jupyter

Install packages:

```bash
pip install torch torchvision numpy matplotlib scipy scikit-learn pillow jupyter
```

## How To Run

1. Open Jupyter Notebook.
2. Run [Remove_noise.ipynb](Remove_noise.ipynb) to train and save [CNN/dncnn_model.pth](CNN/dncnn_model.pth).
3. Run [Remove_blur.ipynb](Remove_blur.ipynb) to train and save [CNN/Blur_cnn_model.pth](CNN/Blur_cnn_model.pth).
4. Run [Examples/Single_image.ipynb](Examples/Single_image.ipynb) to apply denoising then deblurring on a single image.

## Project Structure

```text
Hanon/
|-- README.md
|-- Remove_noise.ipynb
|-- Remove_blur.ipynb
|-- CNN/
|   |-- dncnn_model.pth
|   `-- Blur_cnn_model.pth
|-- Data/                 # Created automatically at runtime, not provided on GitHub
|   `-- Img.npy           # Cached dataset file generated locally
|-- Examples/
|   `-- Single_image.ipynb
`-- python/
    |-- functions.py
    `-- __pycache__/
```

## Utility Module

[python/functions.py](python/functions.py) contains shared helpers used across notebooks, including:

- dataset class for image/target pairs
- tensor conversion helper (channel and normalization)
- reusable preprocessing utilities

## Outputs

Training and inference notebooks generate:

- trained model files in [CNN](CNN)
- cached dataset files in [Data](Data)
- restoration visualizations in notebook outputs

## References

- K. Zhang et al., Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising, IEEE TIP, 2017.
- LFW dataset: G. B. Huang et al., Labeled Faces in the Wild, 2007.

**Author**: Adrien Marchandou 
**Last Updated**: March 2026
