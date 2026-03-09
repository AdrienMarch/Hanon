# Hanon - DnCNN Image Denoising

A PyTorch implementation of **DnCNN (Denoising Convolutional Neural Network)** for image denoising using residual learning. This project demonstrates how deep learning can effectively remove noise from images while preserving important details.

## 📋 Overview

This project implements a CNN-based image denoising system that:
- Uses the **DnCNN architecture** with residual learning
- Trains on the LFW (Labeled Faces in the Wild) dataset
- Learns to predict and remove noise from images
- Achieves effective denoising through a 17-layer deep CNN

### Key Features

- **Residual Learning**: The network predicts the noise itself rather than the clean image, which has proven more effective
- **Batch Normalization**: Improves training stability and convergence
- **Deep Architecture**: 17 convolutional layers for hierarchical feature learning
- **Automatic Caching**: Downloaded datasets are cached locally for faster subsequent runs

## 🛠️ Installation

### Prerequisites

- Python 3.7+
- PyTorch
- NumPy
- Matplotlib
- scikit-learn
- Pillow (PIL) - for loading JPG/PNG images in examples

### Setup

1. **Clone or download this repository**

2. **Create and activate a virtual environment** (recommended):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1  # PowerShell
   ```

3. **Install dependencies**:
   ```bash
   pip install torch torchvision numpy matplotlib scikit-learn pillow
   ```

   Or if you have a `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Running the Notebook

1. **Launch Jupyter**:
   ```bash
   jupyter notebook
   ```

2. **Open** `Hanon.ipynb`

3. **Run all cells** to:
   - Download/load the LFW dataset
   - Preprocess and add noise to images
   - Build and train the DnCNN model
   - Visualize denoising results

### Workflow

The notebook follows this pipeline:

1. **Data Loading**: Downloads LFW face dataset (cached after first run)
2. **Preprocessing**: Normalizes images to grayscale values (0-255)
3. **Data Augmentation**: Adds Gaussian noise (noise_factor=10)
4. **Model Architecture**: Builds 17-layer DnCNN
5. **Training**: Trains for 50 epochs using MSE loss
6. **Evaluation**: Visualizes denoising performance on test images

## 📦 Python Module (`python/functions.py`)

The `python/` directory contains reusable components for image denoising:

### `MyDataset` Class
Custom PyTorch Dataset for handling image denoising pairs (noisy → clean).

```python
from python.functions import MyDataset

dataset = MyDataset(noisy_images, clean_images, 
                   transform=add_channel,
                   target_transform=add_channel)
```

### `add_channel()` Function
Converts numpy images to PyTorch tensors with proper normalization.

```python
from python.functions import add_channel

# Convert image to tensor: (H, W) → (1, H, W), normalized to [0, 1]
tensor_image = add_channel(numpy_image)
```

## 💾 Saving and Loading Models

### Save a Trained Model

After training, save your model:

```python
# Save model parameters (recommended)
torch.save(cnn.state_dict(), 'CNN/dncnn_model.pth')

# Or save complete model
torch.save(cnn, 'CNN/dncnn_model.pth')

# Save checkpoint with training info
torch.save({
    'epoch': epochs,
    'model_state_dict': cnn.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': train_losses[-1],
}, 'CNN/dncnn_checkpoint.pth')
```

### Load a Saved Model

```python
# Load state dict (requires model architecture)
cnn = nn.Sequential(*layers).to(device)
cnn.load_state_dict(torch.load('CNN/dncnn_model.pth'))
cnn.eval()

# Or load complete model
cnn = torch.load('CNN/dncnn_model.pth')
cnn.eval()
```

## 📸 Examples

### Single Image Denoising (`Examples/Single_image.ipynb`)

This example demonstrates how to use a pre-trained model to denoise a single image:

1. **Navigate to Examples directory**:
   ```bash
   cd Examples
   jupyter notebook
   ```

2. **Open** `Single_image.ipynb`

3. **What it does**:
   - Loads a custom JPG image (`notebook.jpg`)
   - Converts it to grayscale numpy array (0-255)
   - Loads the pre-trained DnCNN model from `CNN/dncnn_model.pth`
   - Denoises the image
   - Displays before/after comparison

4. **Usage**:
   - Replace `Notebook.jpg` with your own image
   - Ensure the model path points to your trained model
   - Run all cells to see denoising results

**Note**: The example imports functions from `python/functions.py`, which is added to the Python path automatically within the notebook.

## 🏗️ Architecture

### DnCNN Structure

```
Input (Noisy Image: 1×H×W)
    ↓
Conv2d(1→64, 3×3) + ReLU
    ↓
[Conv2d(64→64, 3×3) + BatchNorm + ReLU] × 15
    ↓
Conv2d(64→1, 3×3)
    ↓
Predicted Noise (1×H×W)
    ↓
Clean Image = Noisy Image - Predicted Noise
```

**Key Parameters**:
- **Input**: Grayscale images (1 channel)
- **Filters**: 64 feature maps in hidden layers
- **Kernel Size**: 3×3 convolutions
- **Depth**: 17 layers total
- **Batch Normalization**: Applied after each hidden layer
- **Activation**: ReLU (Rectified Linear Unit)

### Training Configuration

- **Optimizer**: Adam (learning rate = 1e-3)
- **Loss Function**: MSE (Mean Squared Error)
- **Batch Size**: 32
- **Epochs**: 50
- **Training Samples**: 1500 images
- **Test Samples**: Remaining images from dataset

## 📊 Results

The model successfully removes Gaussian noise while preserving facial features and image details. Training loss typically decreases consistently over 50 epochs, indicating good convergence.

**Visualization** includes:
- Original clean image
- Noisy image (with Gaussian noise)
- Denoised image (model output)

## 📁 Project Structure

```
Hanon/
├── Hanon.ipynb          # Main training notebook (full pipeline)
├── README.md            # This file
├── Data/
│   ├── Img.npy         # Cached LFW dataset (created on first run)
│   └── Notebook.jpg    # Sample test image
├── CNN/
│   └── dncnn_model.pth # Saved trained model
├── python/
│   └── functions.py    # Reusable dataset and transform functions
├── Examples/
│   └── Single_image.ipynb  # Example: Denoise a single image
└── .venv/              # Virtual environment (optional)
```

## 🔧 Customization

### Adjusting Noise Levels

Modify the `noise_factor` parameter in the noise generation:
```python
img_noisy[i] = add_noise(img_grayscale[i], noise_factor=10)  # Default: 10
```

### Training Parameters

Adjust hyperparameters in the training section:
```python
epochs = 50                # Number of training epochs
batch_size = 32            # Batch size for training
learning_rate = 1e-3       # Adam optimizer learning rate
```

### Loss Function

Switch between MSE and L1 loss:
```python
loss_fn = nn.MSELoss()     # Mean Squared Error (default)
# loss_fn = nn.L1Loss()    # Mean Absolute Error (alternative)
```

## 📚 Background & References

### DnCNN (Denoising CNN)

DnCNN uses **residual learning** to predict the noise component rather than the clean image directly. This approach:
- Simplifies the learning task
- Improves training stability
- Achieves better denoising performance

**Mathematical formulation**:
```
Noisy Image = Clean Image + Noise
Predicted Clean Image = Noisy Image - Network(Noisy Image)
```

### References

1. **Ilesanmi, A. E., & Ilesanmi, T. O. (2021)**  
   *Methods for image denoising using convolutional neural network: a review*  
   Complex & Intelligent Systems, 7(5), 2179-2198.  
   https://link.springer.com/article/10.1007/S40747-021-00428-4

2. **Zhang, K., Zuo, W., Chen, Y., Meng, D., & Zhang, L. (2017)**  
   *Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising*  
   IEEE Transactions on Image Processing, 26(7), 3142-3155.

3. **LFW Dataset**  
   Huang, G. B., Ramesh, M., Berg, T., & Learned-Miller, E. (2007)  
   *Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments*

## 🤝 Contributing

Feel free to:
- Report issues or bugs
- Suggest improvements or new features
- Submit pull requests

## 📄 License

This project is provided for educational purposes. Please refer to the original DnCNN paper and LFW dataset licenses for academic or commercial use.

## 🙏 Acknowledgments

- DnCNN architecture by Zhang et al.

---

**Author**: Adrien Marchandou 
**Last Updated**: March 2026
