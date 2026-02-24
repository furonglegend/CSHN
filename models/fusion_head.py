"""
Fusion head for combining epistemic and aleatoric uncertainty.

Implements g_omega(U_epi, U_alea).
"""

import torch
import torch.nn as nn


class FusionHead(nn.Module):
    """
    Learnable fusion network.
    """

    def __init__(self, hidden_dim: int = 32):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, u_epi: torch.Tensor, u_alea: torch.Tensor):
        """
        Args:
            u_epi: (B, 1)
            u_alea: (B, 1)

        Returns:
            fused uncertainty score (B, 1)
        """
        x = torch.cat([u_epi, u_alea], dim=-1)
        return self.net(x)