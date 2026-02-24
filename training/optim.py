"""
Optimizer and scheduler factory.
"""

import torch


def build_optimizer(model, config):
    """
    Create optimizer from config dictionary.
    """
    if config["type"] == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.get("lr", 1e-3),
            weight_decay=config.get("weight_decay", 1e-4),
        )
    elif config["type"] == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config.get("lr", 1e-3),
        )
    else:
        raise ValueError("Unsupported optimizer type.")


def build_scheduler(optimizer, config):
    """
    Create scheduler from config.
    """
    if config["type"] == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=config.get("patience", 5),
            factor=config.get("factor", 0.5),
        )
    elif config["type"] == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.get("T_max", 50),
        )
    else:
        return None