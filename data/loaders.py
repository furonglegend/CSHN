"""
Common dataset loader utilities.

Provides TemporalHypergraphDataset which reads the CSVs produced by the prepare_*.py scripts.

Returned sample is a dict:
{
    "node_id": int,
    "time": int,
    "features": np.array(float32),
    "neighbors": list of neighbor node ids (assembled from hyperedges for that time),
    "label": int
}
"""
import os
import json
from pathlib import Path
from typing import Optional, List, Dict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class TemporalHypergraphDataset(Dataset):
    """
    Minimal temporal hypergraph dataset.

    Required files (in data_dir):
      <prefix>_nodes.csv        columns: node_id, time, feat_0, feat_1, ...
      <prefix>_hyperedges.csv   columns: time, hyperedge_id, members (JSON list)
      <prefix>_labels.csv       columns: node_id, time, label

    prefix examples: snare, pheme, amigos
    """
    def __init__(self, data_dir: str, prefix: str = "snare", max_time: Optional[int] = None):
        self.data_dir = Path(data_dir)
        nodes_path = self.data_dir / f"{prefix}_nodes.csv"
        hyper_path = self.data_dir / f"{prefix}_hyperedges.csv"
        labels_path = self.data_dir / f"{prefix}_labels.csv"

        if not nodes_path.exists():
            raise FileNotFoundError(f"{nodes_path} not found. Run prepare_{prefix}.py first or provide data.")
        self.nodes_df = pd.read_csv(nodes_path)
        self.hyper_df = pd.read_csv(hyper_path) if hyper_path.exists() else pd.DataFrame(columns=["time","hyperedge_id","members"])
        self.labels_df = pd.read_csv(labels_path) if labels_path.exists() else pd.DataFrame(columns=["node_id","time","label"])

        if max_time is not None:
            self.nodes_df = self.nodes_df[self.nodes_df["time"] < int(max_time)].reset_index(drop=True)

        # build an index mapping (node_id,time) -> row index
        self.nodes_df["key"] = self.nodes_df["node_id"].astype(str) + "_" + self.nodes_df["time"].astype(str)
        self.key_to_idx = {k: idx for idx, k in enumerate(self.nodes_df["key"].tolist())}

        # precompute feature vectors
        feat_cols = [c for c in self.nodes_df.columns if c.startswith("feat_")]
        self.features = self.nodes_df[feat_cols].values.astype("float32")
        # create label array (default -1 for unlabeled)
        self.labels = -1 * np.ones(len(self.nodes_df), dtype=np.int64)
        for _, r in self.labels_df.iterrows():
            k = f"{int(r['node_id'])}_{int(r['time'])}"
            if k in self.key_to_idx:
                self.labels[self.key_to_idx[k]] = int(r["label"])

        # build neighbor lists from hyperedges (flatten hyperedges to pairwise adjacency for simplicity)
        self.neighbors = [[] for _ in range(len(self.nodes_df))]
        for _, r in self.hyper_df.iterrows():
            try:
                members = json.loads(r["members"])
            except Exception:
                # members may already be a list
                members = r["members"]
            for i in range(len(members)):
                for j in range(len(members)):
                    if i == j:
                        continue
                    key_i = f"{int(members[i])}_{int(r['time'])}"
                    if key_i in self.key_to_idx:
                        self.neighbors[self.key_to_idx[key_i]].append(int(members[j]))

    def __len__(self):
        return len(self.nodes_df)

    def __getitem__(self, idx):
        row = self.nodes_df.iloc[idx]
        features = self.features[idx]
        label = int(self.labels[idx]) if self.labels[idx] >= 0 else -1
        neighbor_list = list(set(self.neighbors[idx]))  # unique neighbors
        sample = {
            "node_id": int(row["node_id"]),
            "time": int(row["time"]),
            "features": torch.from_numpy(features),
            "neighbors": neighbor_list,
            "label": label
        }
        return sample

def collate_batch(batch: List[Dict]):
    """
    Collate for DataLoader: stack features, labels; neighbors remain list-of-lists.
    """
    features = torch.stack([b["features"] for b in batch], dim=0)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    node_ids = [b["node_id"] for b in batch]
    times = [b["time"] for b in batch]
    neighbors = [b["neighbors"] for b in batch]
    return {"features": features, "labels": labels, "node_id": node_ids, "time": times, "neighbors": neighbors}