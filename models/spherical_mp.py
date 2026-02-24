"""
Spherical Hypergraph Message Passing.

Implements angular attention using cosine similarity
on the unit hypersphere.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SphericalMessagePassing(nn.Module):
    """
    Angular attention-based message passing.

    Operates on unit vectors.
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        self.linear = nn.Linear(latent_dim, latent_dim)

    def forward(self, z: torch.Tensor, adj: torch.Tensor):
        """
        Args:
            z: (N, D) unit embeddings
            adj: (N, N) adjacency matrix (0/1)

        Returns:
            updated z
        """
        # Angular similarity
        sim = torch.matmul(z, z.T)  # cosine since normalized
        attn = F.softmax(sim * adj, dim=-1)

        message = torch.matmul(attn, z)
        out = self.linear(message)

        out = F.normalize(out, p=2, dim=-1)
        return out