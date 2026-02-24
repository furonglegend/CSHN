"""
Aleatoric uncertainty head.

Predicts observation-level variance sigma^2_phi(x).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AleatoricHead(nn.Module):
    """
    Predicts aleatoric variance.
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        self.var_layer = nn.Linear(latent_dim, 1)

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: (B, D)

        Returns:
            sigma2: (B, 1) positive variance
        """
        raw = self.var_layer(z)
        sigma2 = F.softplus(raw) + 1e-6  # ensure strictly positive
        return sigma2