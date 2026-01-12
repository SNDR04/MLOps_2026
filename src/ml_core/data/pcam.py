from pathlib import Path
from typing import Callable, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class PCAMDataset(Dataset):
    """
    PatchCamelyon (PCAM) Dataset reader for H5 format.
    """

    def __init__(self, x_path: str, y_path: str, transform: Optional[Callable] = None, filter_data: bool = False):
        self.x_path = Path(x_path)
        self.y_path = Path(y_path)
        self.transform = transform
        self.filter_data = filter_data
        self.indices = None


        # TODO: Initialize dataset
        # 1. Check if files exist
        # 2. Open h5 files in read mode
        if not self.x_path.exists():
            raise FileNotFoundError(f"File X has not been found: {self.x_path}")
        if not self.y_path.exists():
            raise FileNotFoundError(f"File Y has not been found: {self.y_path}")
        with h5py.File(self.x_path, "r") as file:
            self.length = len(file["x"])
        if self.filter_data:
            valid_lst = []  
            with h5py.File(self.x_path, "r") as file:
                for i in range(self.length):
                    img = file["x"][i]
                    m = img.mean()
                    if 5 < m < 250:
                        valid_lst.append(i)
            self.indices = valid_lst
            self.length = len(self.indices)

    def __len__(self) -> int:
        # TODO: Return length of dataset
        # The dataloader will know hence how many batches to create
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: Implement data retrieval
        # 1. Read data at idx
        # 2. Convert to uint8 (for PIL compatibility if using transforms)
        # 3. Apply transforms if they exist
        # 4. Return tensor image and label (as long)
        if self.indices is not None:
            idx = self.indices[idx]
        with h5py.File(self.x_path, "r") as filex:
            image = filex["x"][idx]
        with h5py.File(self.y_path, "r") as filey:
            label = filey["y"][idx]

        image = torch.from_numpy(image).float()
        image = torch.clamp(image, 0, 255)
        
        image = image.permute(2, 0, 1) / 255.0

        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(label, dtype=torch.long).squeeze()
        return image, label

