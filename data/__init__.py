"""
Data package for SphUnc.

Provides:
- Base dataset abstraction
- Temporal hypergraph container
- Dataset implementations for SNARE, PHEME, and AMIGOS
"""

from .dataset import BaseDataset, TorchGraphDataset
from .temporal_hypergraph import TemporalHypergraph
from .snare_dataset import SNAREDataset
from .pheme_dataset import PHEMEDataset
from .amigos_dataset import AMIGOSDataset

__all__ = [
    "BaseDataset",
    "TorchGraphDataset",
    "TemporalHypergraph",
    "SNAREDataset",
    "PHEMEDataset",
    "AMIGOSDataset"
]