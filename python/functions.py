import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    """
    Custom PyTorch Dataset for image denoising.
    
    Args:
        data: Input images (noisy images)
        target: Target images (clean images)
        transform: Optional transform to apply to input images
        target_transform: Optional transform to apply to target images
    """
    def __init__(self, data, target, transform=None, target_transform=None):
        self.data = data
        self.target = target
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, index):
        """
        Retrieves a single sample from the dataset.
        
        Args:
            index: Index of the sample to retrieve
            
        Returns:
            Tuple of (input_image, target_image) with transforms applied
        """
        x = self.data[index]
        if self.transform:
            x = self.transform(x)

        y = self.target[index]
        if self.target_transform:
            y = self.target_transform(y)

        return x, y

# Transform function to add channel dimension and normalize to [0, 1]
def add_channel(image):
    """
    Convert numpy image to PyTorch tensor with channel dimension.
    
    Args:
        image: Input image as numpy array (H x W)
        
    Returns:
        PyTorch tensor of shape (1, H, W) normalized to [0, 1]
    """
    return torch.FloatTensor(image).unsqueeze(0) / 255.0