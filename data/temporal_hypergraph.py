"""
Temporal Hypergraph representation.

This class supports:
- Loading hypergraph data
- Time indexing
- Efficient temporal slicing
"""

from typing import List, Dict, Any
import numpy as np
import pandas as pd


class TemporalHypergraph:
    """
    Temporal hypergraph container.

    Nodes interact via hyperedges over time.
    """

    def __init__(self, edge_df: pd.DataFrame):
        """
        Args:
            edge_df: DataFrame with columns:
                ["hyperedge_id", "node_id", "timestamp"]
        """
        self.edge_df = edge_df
        self._build_index()

    def _build_index(self):
        """Build time-based index."""
        self.timestamps = np.sort(self.edge_df["timestamp"].unique())
        self.time_to_edges = {
            t: self.edge_df[self.edge_df["timestamp"] == t]["hyperedge_id"].unique()
            for t in self.timestamps
        }

    def get_hyperedges_at_time(self, t: float) -> List[int]:
        """
        Retrieve hyperedges active at time t.
        """
        return list(self.time_to_edges.get(t, []))

    def get_nodes_in_hyperedge(self, hyperedge_id: int) -> List[int]:
        """
        Retrieve nodes participating in a hyperedge.
        """
        subset = self.edge_df[self.edge_df["hyperedge_id"] == hyperedge_id]
        return subset["node_id"].tolist()

    def time_slice(self, t_start: float, t_end: float) -> pd.DataFrame:
        """
        Extract hyperedges between two timestamps.
        """
        mask = (self.edge_df["timestamp"] >= t_start) & \
               (self.edge_df["timestamp"] <= t_end)
        return self.edge_df[mask].copy()