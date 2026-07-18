"""Training helpers for the residual restoration models."""

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_one_epoch(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch with residual learning.

    The model predicts the degradation (noise or blur) rather than the clean
    image; the restored image is obtained by subtracting that prediction from
    the input. The loss compares the restored image to the clean target.

    Returns:
        Average training loss over the epoch.
    """
    model.train()
    total_loss = 0.0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        residual = model(x)
        restored = x - residual
        loss = loss_fn(restored, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
