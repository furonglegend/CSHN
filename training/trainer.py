"""
Trainer class.

Implements:
- Training loop
- Validation loop
- Early stopping
- Callback integration
"""

import torch
from typing import List, Optional


class Trainer:
    """
    Generic Trainer for PyTorch models.
    """

    def __init__(
        self,
        model,
        loss_fn,
        optimizer,
        scheduler=None,
        device="cuda",
        callbacks: Optional[List] = None,
        logger=None,
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.callbacks = callbacks or []
        self.logger = logger

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0

        for batch in dataloader:
            inputs = batch["features"].to(self.device)
            targets = batch["label"].to(self.device)

            outputs = self.model(inputs)
            logits = outputs["logits"]

            loss = self.loss_fn(logits, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    @torch.no_grad()
    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0.0

        for batch in dataloader:
            inputs = batch["features"].to(self.device)
            targets = batch["label"].to(self.device)

            outputs = self.model(inputs)
            logits = outputs["logits"]

            loss = self.loss_fn(logits, targets)
            total_loss += loss.item()

        return total_loss / len(dataloader)

    def fit(self, train_loader, val_loader, epochs: int):
        for epoch in range(epochs):

            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            if self.logger:
                self.logger.log(epoch, train_loss, val_loss)

            for callback in self.callbacks:
                callback(epoch, self.model, val_loss)

            print(f"Epoch {epoch}: Train {train_loss:.4f} | Val {val_loss:.4f}")