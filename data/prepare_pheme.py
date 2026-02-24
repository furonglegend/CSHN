#!/usr/bin/env python3
"""
Prepare PHEME dataset (toy or from raw files).

Creates:
- pheme_nodes.csv
- pheme_hyperedges.csv
- pheme_labels.csv

The format mirrors SNARE script and is intentionally simple so loaders can reuse code.
"""
import os
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def synth_pheme(output_dir: str, n_nodes=4800, n_times=10, feat_dim=300, seed=1):
    rng = np.random.RandomState(seed)
    nodes = []
    hyperedges = []
    labels = []

    for t in range(n_times):
        # sample a subset of active nodes per time to mimic threads
        active = int(n_nodes * (0.4 + 0.2 * rng.rand()))
        active_ids = rng.choice(n_nodes, active, replace=False)
        for i in active_ids:
            feats = rng.normal(loc=0.0, scale=1.0, size=feat_dim)
            row = {"node_id": int(i), "time": int(t)}
            for d, v in enumerate(feats[:128]):  # to keep CSV size reasonable, store 128 dims
                row[f"feat_{d}"] = float(v)
            nodes.append(row)
            labels.append({"node_id": int(i), "time": int(t), "label": int(rng.randint(0,4))})

    for t in range(n_times):
        n_hyper = max(10, int(0.002 * n_nodes))
        for he in range(n_hyper):
            size = rng.randint(2, 30)
            members = list(map(int, rng.choice(n_nodes, size=size, replace=False)))
            hyperedges.append({
                "time": int(t),
                "hyperedge_id": f"t{t}_he{he}",
                "members": json.dumps(members)
            })

    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(nodes).to_csv(Path(output_dir) / "pheme_nodes.csv", index=False)
    pd.DataFrame(hyperedges).to_csv(Path(output_dir) / "pheme_hyperedges.csv", index=False)
    pd.DataFrame(labels).to_csv(Path(output_dir) / "pheme_labels.csv", index=False)
    print(f"PHEME synthetic files written to {output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=".", help="output directory")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    synth_pheme(args.out, seed=args.seed)

if __name__ == "__main__":
    main()