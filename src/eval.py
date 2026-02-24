"""
Evaluation script for the toy SphUnc model.

Computes:
- classification accuracy
- simple ECE approximation (by binning predicted confidences and comparing with accuracy per bin)
Saves predictions as CSV: predictions.csv with columns node_id, time, pred_class, pred_conf, epistemic, aleatoric
"""
import os
import sys
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

def compute_ece_from_preds(confidences, correct, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.digitize(confidences, bins) - 1
    ece = 0.0
    n = len(confidences)
    for i in range(n_bins):
        sel = (bin_idx == i)
        if sel.sum() == 0:
            continue
        avg_conf = confidences[sel].mean()
        acc = correct[sel].mean()
        ece += (sel.sum() / n) * abs(avg_conf - acc)
    return float(ece)

def run_evaluation(config):
    data_dir = config.get("data_dir", "data")
    prefix = config.get("prefix", "snare")
    out_dir = Path(config.get("out_dir", "outputs"))
    ckpt_path = config.get("ckpt", str(out_dir / "model_checkpoint.pth"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = TemporalHypergraphDataset(data_dir, prefix=prefix)
    loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False, collate_fn=collate_batch)

    ck = load_checkpoint(ckpt_path)
    # construct model same way as training (toy)
    from train import ToySphUnc
    sample = ds[0]
    input_dim = sample["features"].shape[0]
    model = ToySphUnc(input_dim=input_dim, latent_dim=config.get("latent_dim",128), n_classes=config.get("n_classes",3))
    model.load_state_dict(ck["model_state"])
    model.to(device)
    model.eval()

    all_rows = []
    all_conf = []
    all_correct = []
    with torch.no_grad():
        for batch in loader:
            feats = batch["features"].to(device)
            labels = batch["labels"].cpu().numpy()
            logits, sigma2, h = model(feats, mc_dropout=False)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            pred_conf = probs.max(axis=1)
            pred_class = probs.argmax(axis=1)
            # simple epistemic proxy: run T stochastic forward passes with dropout on
            T = 8
            epistemic_ents = []
            for t in range(T):
                logits_t, _, _ = model(feats, mc_dropout=True)
                probs_t = F.softmax(logits_t, dim=-1).cpu().numpy()
                epistemic_ents.append(probs_t)
            epistemic_ents = np.stack(epistemic_ents, axis=0)  # T x B x C
            # entropy over mean predictive distribution as a simple epistemic proxy
            mean_probs = epistemic_ents.mean(axis=0)  # B x C
            entropy = -(mean_probs * np.log(mean_probs + 1e-12)).sum(axis=-1)  # B
            sigma2_np = sigma2.cpu().numpy()

            for i in range(len(pred_class)):
                node_id = batch["node_id"][i]
                time = batch["time"][i]
                lbl = int(labels[i])
                row = {
                    "node_id": int(node_id),
                    "time": int(time),
                    "true": int(lbl),
                    "pred_class": int(pred_class[i]),
                    "pred_conf": float(pred_conf[i]),
                    "epistemic": float(entropy[i]),
                    "aleatoric": float(sigma2_np[i])
                }
                all_rows.append(row)
                all_conf.append(float(pred_conf[i]))
                all_correct.append(float(int(pred_class[i] == lbl)))

    pred_df = pd.DataFrame(all_rows)
    out_preds = out_dir / f"predictions_{prefix}.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_preds, index=False)
    ece = compute_ece_from_preds(np.array(all_conf), np.array(all_correct), n_bins=15)
    acc = (np.array(all_correct).mean())

    print(f"Saved predictions to {out_preds}")
    print(f"Accuracy: {acc:.4f}, ECE: {ece:.4f}")
    # also save summary
    pd.DataFrame([{"accuracy": acc, "ece": ece}]).to_csv(out_dir / "eval_summary.csv", index=False)