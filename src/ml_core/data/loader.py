from pathlib import Path
from typing import Dict, Tuple

from torch.utils.data import DataLoader
from torchvision import transforms

from .pcam import PCAMDataset
import torch

def get_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """
    Factory function to create Train and Validation DataLoaders
    using pre-split H5 files.
    """
    data_cfg = config["data"]
    base_path = Path(data_cfg["data_path"])

    # TODO: Define Transforms
    train_transform = None
    val_transform = None

    # TODO: Define Paths for X and Y (train and val)
    x_train = base_path / "camelyonpatch_level_2_split_train_x.h5"
    y_train = base_path / "camelyonpatch_level_2_split_train_y.h5"
    x_val = base_path / "camelyonpatch_level_2_split_valid_x.h5"
    y_val = base_path / "camelyonpatch_level_2_split_valid_y.h5"

    # TODO: Instantiate PCAMDataset for train and val
    train_dataset = PCAMDataset(x_train, y_train, transform=train_transform, filter_data=False)
    val_dataset = PCAMDataset(x_val, y_val, transform=val_transform, filter_data=False)

    # TODO: Create DataLoaders
    batch = data_cfg["batch_size"]
    totalworkers = data_cfg.get("num_workers", 0)
    labellst = []
    for i in range(len(train_dataset)):
        _, y = train_dataset[i]
        labellst.append(int(y))
    labels = torch.tensor(labellst)
    totalclasses = torch.bincount(labels)
    weightclass = 1.0 / totalclasses.float()
    weightsamples = weightclass[labels]
    samplers = torch.utils.data.WeightedRandomSampler(weights=weightsamples, num_samples=len(weightsamples), 
                                                      replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=batch,sampler=samplers,
                                             num_workers=totalworkers,)
    val_loader = DataLoader(val_dataset, batch_size=batch, sampler=samplers,
                                            num_workers=totalworkers)
    
    return train_loader, val_loader