import time
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils import ExperimentTracker, setup_logger


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Dict[str, Any],
        device: str,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.config = config
        self.device = device
        
        # TODO: Define Loss Function (Criterion)
        self.criterion = nn.CrossEntropyLoss()

        # TODO: Initialize ExperimentTracker
        self.tracker = ExperimentTracker()
        
        # TODO: Initialize metric calculation (like accuracy/f1-score) if needed
        self.logger = setup_logger()
    def train_epoch(self, dataloader: DataLoader, epoch_idx: int) -> Tuple[float, float, float]:
        self.model.train()
        loss = 0
        # TODO: Implement Training Loop
        # 1. Iterate over dataloader
        # 2. Move data to device
        # 3. Forward pass, Calculate Loss
        # 4. Backward pass, Optimizer step
        # 5. Track metrics (Loss, Accuracy, F1)
        
        for i, j in tqdm(dataloader):
            i = i.to(self.device)
            j  = j.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(i)
            losss = self.criterion(outputs, j)
            losss.backward()
            self.optimizer.step()
            loss += losss.item()
        averageloss = loss /len(dataloader)
        self.tracker.log_metric("train_loss", averageloss, step=epoch_idx)
        return averageloss

    def validate(self, dataloader: DataLoader, epoch_idx: int) -> Tuple[float, float, float]:
        self.model.eval()
        
        # TODO: Implement Validation Loop
        # Remember: No gradients needed here
        
        losstotal = 0

        with torch.no_grad():
            for i,j in dataloader:
                i = i.to(self.device)
                j  = j.to(self.device)
                outputs = self.model(i)
                loss = self.criterion(outputs, j)
                losstotal += loss.item()
        lossaverage = losstotal / len(dataloader)
        self.tracker.log_metric("val_loss", lossaverage, step=epoch_idx)

        return lossaverage
    def save_checkpoint(self, epoch: int, val_loss: float) -> None:
        # TODO: Save model state, optimizer state, and config
        checkpoint = {"epoch": epoch,"model_state_dict": self.model.state_dict(),"optimizer_state_dict": self.optimizer.state_dict(),
        "val_loss": val_loss,"config": self.config,}

        torch.save(checkpoint, f"checkpoint_epoch_{epoch}.pt")

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        epochs = self.config["training"]["epochs"]
        lsttrainlosses = []
        lst_val_losses = []

        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # TODO: Call train_epoch and validate
            # TODO: Log metrics to tracker
            # TODO: Save checkpoints
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.validate(val_loader, epoch)
            lsttrainlosses.append(train_loss)
            lst_val_losses.append(val_loss)
            print(f"Epoch {epoch+1}/{epochs}"
                  f"Train Loss: {train_loss}"
                  f"Val Loss: {val_loss}")
        return lsttrainlosses, lst_val_losses
	# Remember to handle the trackers properly
