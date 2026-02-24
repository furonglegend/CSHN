#!/usr/bin/env python3
"""
Prepare SNARE dataset (toy or from raw files).

This script creates three CSVs in the current directory (or writes to output_dir):
- snare_nodes.csv         : node features (node_id, time, feature_0, feature_1, ...)
- snare_hyperedges.csv    : hyperedges per time (time, hyperedge_id, member_list as JSON)
- snare_labels.csv        : labels per (node_id, time) for supervised tasks

If real raw files exist, adapt the loader section below to read them.
"""
import os
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def synth_snare(output_dir: str, n_nodes=540, n_times=5, feat_dim=24, seed=0):
    rng = np.random.RandomState(seed)
    nodes = []
    hyperedges = []
    labels = []

    # create node features for each time step
    for t in range(n_times):
        for i in range(n_nodes):
            feats = rng.normal(loc=0.0, scale=1.0, size=feat_dim)
            row = {"node_id": int(i), "time": int(t)}
            for d, v in enumerate(feats):
                row[f"feat_{d}"] = float(v)
            nodes.append(row)
            # synthetic label: three classes (0,1,2) influenced by node id and time
            labels.append({"node_id": int(i), "time": int(t), "label": int((i + t) % 3)})

    # create synthetic hyperedges per time (communities of varying sizes)
    for t in range(n_times):
        n_hyper = n_nodes // 10
        for he in range(n_hyper):
            size = rng.randint(3, 12)
            members = list(map(int, rng.choice(n_nodes, size=size, replace=False)))
            hyperedges.append({
                "time": int(t),
                "hyperedge_id": f"t{t}_he{he}",
                "members": json.dumps(members)
            })

    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(nodes).to_csv(Path(output_dir) / "snare_nodes.csv", index=False)
    pd.DataFrame(hyperedges).to_csv(Path(output_dir) / "snare_hyperedges.csv", index=False)
    pd.DataFrame(labels).to_csv(Path(output_dir) / "snare_labels.csv", index=False)
    print(f"SNARE synthetic files written to {output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=".", help="output directory")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    synth_snare(args.out, seed=args.seed)

if __name__ == "__main__":
    main()