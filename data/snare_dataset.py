"""
SNARE dataset processing.

Includes optional temporal sampling and hypergraph construction.
"""

import pandas as pd
import numpy as np
from .dataset import BaseDataset
from .temporal_hypergraph import TemporalHypergraph


class SNAREDataset(BaseDataset):
    """
    SNARE dataset implementation.
    """

    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.hypergraph = self._build_hypergraph()

    def _build_hypergraph(self):
        edge_df = self.df[["hyperedge_id", "node_id", "timestamp"]]
        return TemporalHypergraph(edge_df)

    def __len__(self):
        return len(self.df)

    def get_item(self, idx):
        row = self.df.iloc[idx]
        return {
            "features": np.array(row["features"]),
            "label": int(row["label"]),
            "timestamp": float(row["timestamp"])
        }