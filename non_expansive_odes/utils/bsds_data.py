import os
import shutil
import tarfile
from typing import Optional, Tuple

import numpy as np
import requests
import torch
from matplotlib.pyplot import imread
from torch.utils.data import DataLoader
from tqdm import tqdm

_BSDS_URL = (
    "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_full.tgz"
)


def _get_data_path(data_path: Optional[str] = None):
    if data_path is None:
        path_var = os.getenv("DATA_PATH")
        data_path = path_var if path_var is not None else ""
    return data_path


def _download_and_extract_dataset(data_path: Optional[str] = None):
    data_path = _get_data_path(data_path)
    with requests.get(_BSDS_URL, stream=True) as r:
        r.raise_for_status()
        file_size = int(r.headers["content-length"])
        print(f"Downloading dataset to {data_path}:")
        with open(os.path.join(data_path, "BSDS500.tgz"), "wb") as f:
            for chunk in tqdm(r.iter_content(chunk_size=8192), total=file_size // 8192):
                f.write(chunk)
    print("Extracting archive")
    with tarfile.open(os.path.join(data_path, "BSDS500.tgz")) as f:
        f.extractall(data_path)
    os.remove(os.path.join(data_path, "BSDS500.tgz"))


def _process_dataset(data_path: Optional[str] = None):
    data_path = _get_data_path(data_path)
    bsds_path = os.path.join(data_path, "BSDS500")
    if not os.path.isdir(bsds_path):
        os.mkdir(bsds_path)
    train = np.memmap(
        os.path.join(bsds_path, "train.dat"),
        dtype=np.float16,
        mode="w+",
        shape=(200, 3, 321, 481),
    )
    val = np.memmap(
        os.path.join(bsds_path, "val.dat"),
        dtype=np.float16,
        mode="w+",
        shape=(100, 3, 321, 481),
    )

    test = np.memmap(
        os.path.join(bsds_path, "test.dat"),
        dtype=np.float16,
        mode="w+",
        shape=(200, 3, 321, 481),
    )
    print("Extracting training set:")
    train_path = os.path.join(data_path, "BSR/BSDS500/data/images/train/")
    i = 0
    for f in tqdm(os.listdir(train_path)):
        if ".jpg" not in f:
            continue
        im = imread(os.path.join(train_path, f)).astype(np.float16) / 255.0
        if im.shape[:2] == (481, 321):
            im = im.transpose(2, 1, 0)
        else:
            im = im.transpose(2, 0, 1)
        train[i] = im
        i += 1

    print("Extracting validation set:")
    val_path = os.path.join(data_path, "BSR/BSDS500/data/images/val/")
    i = 0
    for f in tqdm(os.listdir(val_path)):
        if ".jpg" not in f:
            continue
        im = imread(os.path.join(val_path, f)).astype(np.float16) / 255.0
        if im.shape[:2] == (481, 321):
            im = im.transpose(2, 1, 0)
        else:
            im = im.transpose(2, 0, 1)
        val[i] = im
        i += 1

    print("Extracting test set:")
    test_path = os.path.join(data_path, "BSR/BSDS500/data/images/test/")
    i = 0
    for f in tqdm(os.listdir(test_path)):
        if ".jpg" not in f:
            continue
        im = imread(os.path.join(test_path, f)).astype(np.float16) / 255.0
        if im.shape[:2] == (481, 321):
            im = im.transpose(2, 1, 0)
        else:
            im = im.transpose(2, 0, 1)
        test[i] = im
        i += 1

    shutil.rmtree(os.path.join(data_path, "BSR"))


class BSDSDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: Optional[str] = None,
        split: str = "train",
        download: bool = True,
        noise_level=0.15,
        device=torch.device(0),
        half=False,
    ):
        self._noise_level = noise_level
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Split '{split}' of dataset does not exist")
        data_path = _get_data_path(data_path)
        if download and not os.path.isdir(os.path.join(data_path, "BSDS500")):
            _download_and_extract_dataset(data_path)
            _process_dataset(data_path)
        filename = os.path.join(data_path, "BSDS500", f"{split}.dat")
        self.split = split
        self._N = 200 if split in ["train", "test"] else 100
        self._buffer = torch.tensor(
            np.memmap(
                filename, mode="r+", dtype=np.float16, shape=(self._N, 3, 321, 481)
            ),
            device=device,
        )
        if not half:
            self._buffer = self._buffer.float()

    def __getitem__(self, i):
        x = self._buffer[i]
        noisy = x + self._noise_level * torch.randn_like(x)
        return x, noisy

    def __len__(self):
        return self._N


def get_bsds_splits(
    batch_size=5,
    device=torch.device(0),
    download=True,
    noise_level=0.15,
    data_path=None,
    half=False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    batch_size_val = 4 * batch_size

    if data_path is None:
        data_path = os.getenv("DATA_PATH")
        if data_path is None:
            data_path = "."

    train_data = BSDSDataset(
        data_path,
        "train",
        download=download,
        noise_level=noise_level,
        device=device,
        half=half,
    )
    val_data = BSDSDataset(
        data_path,
        "val",
        download=download,
        noise_level=noise_level,
        device=device,
        half=half,
    )
    test_data = BSDSDataset(
        data_path,
        "test",
        download=download,
        noise_level=noise_level,
        device=device,
        half=half,
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size_val)
    test_loader = DataLoader(test_data, batch_size=batch_size_val)
    return (train_loader, val_loader, test_loader)
