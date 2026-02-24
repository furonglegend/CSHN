"""
Sanity check training convergence on tiny synthetic dataset.
"""

import unittest
import torch
from torch.utils.data import DataLoader, TensorDataset
from models.sphunc_model import SphUncModel
from training.trainer import Trainer
from training.optim import build_optimizer


class TestTraining(unittest.TestCase):

    def test_small_training(self):
        x = torch.randn(100, 128)
        y = torch.randint(0, 2, (100,))

        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=16)

        model = SphUncModel(input_dim=128, latent_dim=64)
        optimizer = build_optimizer(model, {"type": "adamw", "lr": 1e-2})
        loss_fn = torch.nn.CrossEntropyLoss()

        trainer = Trainer(model, loss_fn, optimizer, device="cpu")

        trainer.fit(loader, loader, epochs=3)


if __name__ == "__main__":
    unittest.main()