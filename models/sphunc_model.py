"""
Full SphUnc Model.

Combines:
- Spherical encoder
- vMF epistemic head
- Aleatoric head
- Fusion network
- Optional structural causal module placeholder
"""

import torch
import torch.nn as nn
from .spherical_encoder import SphericalEncoder
from .vmf_head import VMFHead
from .aleatoric_head import AleatoricHead
from .fusion_head import FusionHead


class DummySCM(nn.Module):
    """
    Placeholder structural causal model (SCM).

    Replace with full structural equations if needed.
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        self.structural_layer = nn.Linear(latent_dim, latent_dim)

    def forward(self, z: torch.Tensor):
        """
        Applies structural transformation.

        Args:
            z: latent embeddings

        Returns:
            transformed latent representation
        """
        return self.structural_layer(z)


class SphUncModel(nn.Module):
    """
    High-level SphUnc architecture.
    """

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()

        self.encoder = SphericalEncoder(input_dim, latent_dim)
        self.vmf_head = VMFHead(latent_dim)
        self.aleatoric_head = AleatoricHead(latent_dim)
        self.fusion_head = FusionHead()
        self.scm = DummySCM(latent_dim)

    def forward(self, x: torch.Tensor):
        """
        Forward pass.

        Args:
            x: (B, input_dim)

        Returns:
            dict containing:
                - z: spherical embedding
                - kappa: epistemic concentration
                - sigma2: aleatoric variance
                - fused_uncertainty
        """
        z = self.encoder(x)
        z_struct = self.scm(z)

        kappa = self.vmf_head(z_struct)
        sigma2 = self.aleatoric_head(z_struct)

        # Epistemic uncertainty can be inverse concentration
        u_epi = 1.0 / (kappa + 1e-6)
        u_alea = sigma2

        fused = self.fusion_head(u_epi, u_alea)

        return {
            "z": z_struct,
            "kappa": kappa,
            "sigma2": sigma2,
            "u_epistemic": u_epi,
            "u_aleatoric": u_alea,
            "u_fused": fused
        }