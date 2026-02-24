"""
Full PHEME experiment pipeline.
"""

from data.pheme_dataset import PHEMEDataset
from data.dataset import TorchGraphDataset
from torch.utils.data import DataLoader
from models.sphunc_model import SphUncModel
from training.trainer import Trainer
from training.optim import build_optimizer
import torch


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = PHEMEDataset("data/pheme.csv")
    torch_dataset = TorchGraphDataset(dataset)

    loader = DataLoader(torch_dataset, batch_size=32, shuffle=True)

    model = SphUncModel(input_dim=128, latent_dim=64)

    optimizer = build_optimizer(model, {"type": "adamw", "lr": 1e-3})
    loss_fn = torch.nn.CrossEntropyLoss()

    trainer = Trainer(model, loss_fn, optimizer, device=device)

    trainer.fit(loader, loader, epochs=10)


if __name__ == "__main__":
    main()