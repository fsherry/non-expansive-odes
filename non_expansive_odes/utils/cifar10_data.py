import os
from math import ceil
from typing import Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms.functional import to_tensor


def get_cifar10_splits(
    frac=0.95,
    batch_size=512,
    device=torch.device(0),
    download=True,
    data_path=None,
    half=False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    batch_size_val = 4 * batch_size
    if half:

        def transform(pic):
            return to_tensor(pic).half()

    else:
        transform = to_tensor

    if data_path is None:
        data_path = os.getenv("DATA_PATH")
        if data_path is None:
            data_path = "."

    train_val_data = CIFAR10(
        data_path,
        train=True,
        transform=transform,
        target_transform=torch.tensor,
        download=download,
    )

    N_train = ceil(frac * len(train_val_data))
    train_data, val_data = random_split(
        train_val_data,
        [N_train, len(train_val_data) - N_train],
    )
    train_ims, train_ys = tuple(
        map(lambda xs: torch.stack(xs, dim=0).to(device), zip(*train_data))
    )
    train_data = TensorDataset(train_ims, train_ys)
    val_ims, val_ys = tuple(
        map(lambda xs: torch.stack(xs, dim=0).to(device), zip(*val_data))
    )
    val_data = TensorDataset(val_ims, val_ys)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size_val)

    test_data = CIFAR10(
        data_path,
        train=False,
        transform=transform,
        target_transform=torch.tensor,
    )
    test_ims, test_ys = tuple(
        map(lambda xs: torch.stack(xs, dim=0).to(device), zip(*test_data))
    )
    test_data = TensorDataset(test_ims, test_ys)
    test_loader = DataLoader(test_data, batch_size=batch_size_val)
    return (train_loader, val_loader, test_loader)
