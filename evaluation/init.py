"""
Evaluation package.

Provides:
- Calibration utilities
- Causal recovery metrics
- Intervention simulation
"""

from .calibration import compute_ece
from .causal_recovery import precision_at_k, rank_correlation
from .intervention import do_intervention

__all__ = [
    "compute_ece",
    "precision_at_k",
    "rank_correlation",
    "do_intervention",
]