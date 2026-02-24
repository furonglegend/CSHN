"""
Inference utility. Loads a saved checkpoint and runs the model on an input CSV or single sample.

Input can be:
 - path to a CSV with columns feat_0, feat_1, ... (one or more rows)
 - or a JSON file with "features": [ ... ] for a single instance

Outputs predictions to STDOUT and optionally saves to CSV.
"""
import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT))

from data.loaders import TemporalHypergraphDataset, collate_batch

def load_checkpoint(path):
    import torch
    ck = torch.load(path, map_location="cpu")
    return ck

def run_inference(config: dict, input_path: str):
    ckpt = config.get("ckpt", "outputs/model_checkpoint.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = load_checkpoint(ckpt)
    # construct model
    from train import ToySphUnc
    # we need input dim; infer from config or try to load example
    if "input_dim" in config:
        input_dim = config["input_dim"]
    else:
        # attempt to derive from a small dataset if provided
        data_dir = config.get("data_dir", "data")
        prefix = config.get("prefix", "snare")
        try:
            ds = TemporalHypergraphDataset(data_dir, prefix=prefix)
            input_dim = ds[0]["features"].shape[0]
        except Exception:
            raise RuntimeError("Cannot infer input_dim. Provide 'input_dim' in config or prepare dataset.")

    model = ToySphUnc(input_dim=input_dim, latent_dim=config.get("latent_dim",128), n_classes=config.get("n_classes",3))
    model.load_state_dict(ck["model_state"])
    model.to(device)
    model.eval()

    # load input
    inp = Path(input_path)
    if inp.suffix.lower() in [".json"]:
        with open(inp, "r") as f:
            data = json.load(f)
        feats = np.array(data["features"], dtype=np.float32)[None, :]
    else:
        # treat as CSV; extract columns starting with feat_
        df = pd.read_csv(inp)
        feat_cols = [c for c in df.columns if c.startswith("feat_")]
        if len(feat_cols) == 0:
            # fallback: take all numeric columns
            feat_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feats = df[feat_cols].values.astype("float32")

    feats_t = torch.from_numpy(feats).to(device)
    with torch.no_grad():
        logits, sigma2, h = model(feats_t, mc_dropout=False)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        pred_conf = probs.max(axis=1)
        pred_class = probs.argmax(axis=1)
        # compute epistemic with MC dropout
        T = 8
        epistemic_probs = []
        for t in range(T):
            logits_t, _, _ = model(feats_t, mc_dropout=True)
            epistemic_probs.append(F.softmax(logits_t, dim=-1).cpu().numpy())
        epistemic_probs = np.stack(epistemic_probs, axis=0)
        mean_probs = epistemic_probs.mean(axis=0)
        entropy = -(mean_probs * np.log(mean_probs + 1e-12)).sum(axis=-1)

    rows = []
    for i in range(len(pred_class)):
        rows.append({
            "pred_class": int(pred_class[i]),
            "pred_conf": float(pred_conf[i]),
            "epistemic": float(entropy[i]),
            "aleatoric": float(sigma2.cpu().numpy()[i])
        })
    out_df = pd.DataFrame(rows)
    out_csv = Path(config.get("out_dir", "outputs")) / "inference_results.csv"
    Path(config.get("out_dir", "outputs")).mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"Inference results saved to {out_csv}")
    print(out_df.to_string(index=False))