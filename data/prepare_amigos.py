#!/usr/bin/env python3
"""
Prepare AMIGOS dataset (toy or from raw files).

Creates:
- amigos_nodes.csv
- amigos_hyperedges.csv
- amigos_labels.csv
"""
import os
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def synth_amigos(output_dir: str, n_groups=10, members_per_group=4, n_times=4, feat_dim=128, seed=2):
    rng = np.random.RandomState(seed)
    nodes = []
    hyperedges = []
    labels = []

    node_counter = 0
    for g in range(n_groups):
        members = list(range(node_counter, node_counter + members_per_group))
        node_counter += members_per_group
        for t in range(n_times):
            for i in members:
                feats = rng.normal(scale=1.0, size=feat_dim)
                row = {"node_id": int(i), "time": int(t)}
                for d, v in enumerate(feats):
                    row[f"feat_{d}"] = float(v)
                nodes.append(row)
                labels.append({"node_id": int(i), "time": int(t), "label": int(rng.randint(0,4))})
            hyperedges.append({
                "time": int(t),
                "hyperedge_id": f"group{g}_t{t}",
                "members": json.dumps(members)
            })

    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(nodes).to_csv(Path(output_dir) / "amigos_nodes.csv", index=False)
    pd.DataFrame(hyperedges).to_csv(Path(output_dir) / "amigos_hyperedges.csv", index=False)
    pd.DataFrame(labels).to_csv(Path(output_dir) / "amigos_labels.csv", index=False)
    print(f"AMIGOS synthetic files written to {output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=".", help="output directory")
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()
    synth_amigos(args.out, seed=args.seed)

if __name__ == "__main__":
    main()