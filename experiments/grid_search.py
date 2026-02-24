"""
Simple grid search for hyperparameter tuning.
"""

import itertools
import torch
from models.sphunc_model import SphUncModel
from training.trainer import Trainer
from training.optim import build_optimizer


def grid_search(param_grid, dataloader):

    keys = param_grid.keys()
    combinations = list(itertools.product(*param_grid.values()))

    best_score = float("inf")
    best_params = None

    for values in combinations:
        config = dict(zip(keys, values))

        model = SphUncModel(input_dim=128, latent_dim=config["latent_dim"])
        optimizer = build_optimizer(model, {"type": "adamw", "lr": config["lr"]})

        loss_fn = torch.nn.CrossEntropyLoss()

        trainer = Trainer(model, loss_fn, optimizer, device="cpu")

        trainer.fit(dataloader, dataloader, epochs=5)

        val_loss = trainer.validate(dataloader)

        if val_loss < best_score:
            best_score = val_loss
            best_params = config

    return best_params