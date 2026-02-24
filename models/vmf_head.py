"""
von Mises-Fisher (vMF) head.

Predicts concentration parameter kappa and provides
numerically stable routines for entropy approximation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class VMFHead(nn.Module):
    """
    vMF concentration prediction head.

    Predicts kappa (concentration parameter).
    """

    def __init__(self, latent_dim: int):
        super().__init__()

        self.kappa_layer = nn.Linear(latent_dim, 1)

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: (B, D) unit hypersphere embeddings

        Returns:
            kappa: (B, 1) positive concentration parameter
        """
        raw_kappa = self.kappa_layer(z)
        kappa = F.softplus(raw_kappa) + 1e-6  # ensure positivity
        return kappa

    @staticmethod
    def approximate_entropy(kappa: torch.Tensor, dim: int):
        """
        Approximate entropy of vMF distribution.

        Uses asymptotic expansion for numerical stability.

        Args:
            kappa: (B, 1)
            dim: embedding dimension

        Returns:
            entropy estimate (B,)
        """
        # H ≈ (dim/2 - 1) log(kappa) - kappa + const
        d = dim
        entropy = (d / 2 - 1) * torch.log(kappa + 1e-8) - kappa
        return -entropy.squeeze(-1)  # negative sign for proper entropy interpretation