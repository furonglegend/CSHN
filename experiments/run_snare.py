"""
Full SNARE experiment pipeline.
"""

from data.snare_dataset import SNAREDataset
from data.dataset import TorchGraphDataset
from torch.utils.data import DataLoader
from models.sphunc_model import SphUncModel
from training.trainer import Trainer
from training.optim import build_optimizer
from robustness import evaluate_under_dropout, plot_dropout_curve
import torch


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = SNAREDataset("data/snare.csv")
    torch_dataset = TorchGraphDataset(dataset)

    loader = DataLoader(torch_dataset, batch_size=32, shuffle=True)

    model = SphUncModel(input_dim=128, latent_dim=64)

    optimizer = build_optimizer(model, {"type": "adamw", "lr": 1e-3})

    loss_fn = torch.nn.CrossEntropyLoss()

    trainer = Trainer(model, loss_fn, optimizer, device=device)

    trainer.fit(loader, loader, epochs=10)

    # Robustness evaluation
    rates = [0.0, 0.1, 0.2, 0.3, 0.4]
    accs = evaluate_under_dropout(model, loader, device, rates)
    plot_dropout_curve(rates, accs, "snare_dropout.png")


if __name__ == "__main__":
    main()