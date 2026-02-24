"""
Spherical Encoder.

Projects input features into a latent space and normalizes
them onto the unit hypersphere.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SphericalEncoder(nn.Module):
    """
    Projection + normalization to unit hypersphere.
    """

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()

        self.proj = nn.Linear(input_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim)

        Returns:
            z: (B, latent_dim) normalized to unit sphere
        """
        z = self.proj(x)
        z = F.normalize(z, p=2, dim=-1)  # L2 normalization
        return z