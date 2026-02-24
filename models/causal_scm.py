"""
Structural Causal Module (SCM).

Learns directed parent selection and structural updates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StructuralCausalModel(nn.Module):
    """
    Linear SCM with learnable adjacency matrix.
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.A = nn.Parameter(torch.zeros(latent_dim, latent_dim))
        self.structural = nn.Linear(latent_dim, latent_dim)

    def forward(self, z: torch.Tensor):
        """
        Structural equation:

            z_new = f(A z + noise)

        """
        Az = torch.matmul(z, self.A)
        z_new = self.structural(Az)
        return z_new

    def adjacency_matrix(self):
        """
        Returns learned adjacency weights.
        """
        return self.A