"""
Training callbacks.

Includes:
- Checkpoint saving
- Early stopping
- LR scheduler stepping
"""

import torch


class CheckpointCallback:
    """
    Saves best model based on validation loss.
    """

    def __init__(self, path: str):
        self.path = path
        self.best_loss = float("inf")

    def __call__(self, epoch, model, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            torch.save(model.state_dict(), self.path)


class EarlyStoppingCallback:
    """
    Stops training if validation loss does not improve.
    """

    def __init__(self, patience=10):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0

    def __call__(self, epoch, model, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            raise StopIteration("Early stopping triggered.")


class LRSchedulerCallback:
    """
    Steps scheduler manually.
    """

    def __init__(self, scheduler):
        self.scheduler = scheduler

    def __call__(self, epoch, model, val_loss):
        self.scheduler.step(val_loss)