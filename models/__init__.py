"""
Model package for SphUnc.

This package provides:
- Hyperspherical encoder
- vMF epistemic head
- Aleatoric variance head
- Fusion module
- Full SphUnc composite model
"""

from .spherical_encoder import SphericalEncoder
from .vmf_head import VMFHead
from .aleatoric_head import AleatoricHead
from .fusion_head import FusionHead
from .sphunc_model import SphUncModel

__all__ = [
    "SphericalEncoder",
    "VMFHead",
    "AleatoricHead",
    "FusionHead",
    "SphUncModel"
]