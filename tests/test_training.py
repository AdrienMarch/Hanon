import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from hanon.data import PairedImageDataset, to_tensor
from hanon.training import train_one_epoch


def test_train_one_epoch_runs_and_updates_the_model():
    rng = np.random.default_rng(0)
    clean = rng.integers(0, 256, size=(8, 16, 16)).astype(np.float32)
    noisy = np.clip(clean + rng.normal(0, 20, size=clean.shape), 0, 255)

    dataset = PairedImageDataset(
        noisy, clean, transform=to_tensor, target_transform=to_tensor
    )
    loader = DataLoader(dataset, batch_size=4)

    device = torch.device("cpu")
    model = nn.Conv2d(1, 1, kernel_size=3, padding=1).to(device)
    weights_before = model.weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    loss = train_one_epoch(loader, model, nn.MSELoss(), optimizer, device)

    assert np.isfinite(loss)
    assert not torch.equal(model.weight.detach(), weights_before)
