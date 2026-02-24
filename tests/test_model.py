"""
Unit tests for model forward/backward.
"""

import unittest
import torch
from models.sphunc_model import SphUncModel


class TestModel(unittest.TestCase):

    def test_forward_shape(self):
        model = SphUncModel(input_dim=128, latent_dim=64)
        x = torch.randn(8, 128)
        out = model(x)

        self.assertIn("z", out)
        self.assertEqual(out["z"].shape[0], 8)

    def test_backward(self):
        model = SphUncModel(input_dim=128, latent_dim=64)
        x = torch.randn(8, 128)
        y = torch.randint(0, 2, (8,))

        out = model(x)
        logits = torch.randn(8, 2, requires_grad=True)

        loss = torch.nn.CrossEntropyLoss()(logits, y)
        loss.backward()

        self.assertIsNotNone(logits.grad)


if __name__ == "__main__":
    unittest.main()