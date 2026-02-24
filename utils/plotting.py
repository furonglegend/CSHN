"""
Matplotlib plotting utilities.

- plot_reliability_diagram(df, model_col, conf_col, true_col, outpath, n_bins=15)
- plot_uncertainty_decomposition(df, epi_col, alea_col, err_col, outpath)
All plots use matplotlib only, no seaborn, and disable background gridlines.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Sequence, Optional

# style defaults
plt.rcParams.update({
    "figure.dpi": 300,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "font.family": "serif"
})

def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def plot_reliability_diagram(
    df: pd.DataFrame,
    conf_col: str,
    pred_col: str,
    true_col: str,
    outpath: str,
    n_bins: int = 10,
    title: Optional[str] = None
) -> None:
    """
    Plot a reliability diagram for one set of predictions.

    df must contain:
      conf_col: confidence for predicted class (float in [0,1])
      pred_col: predicted label (int)
      true_col: true label (int)

    The saved image will be written to outpath (PNG).
    """
    p = Path(outpath)
    ensure_parent(p)

    confidences = np.clip(df[conf_col].values.astype(float), 0.0, 1.0)
    preds = df[pred_col].values
    trues = df[true_col].values
    correct = (preds == trues).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5*(bins[:-1] + bins[1:])
    bin_idx = np.digitize(confidences, bins) - 1

    accs = np.full(n_bins, np.nan)
    conf_means = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        sel = bin_idx == i
        counts[i] = int(sel.sum())
        if counts[i] > 0:
            accs[i] = float(correct[sel].mean())
            conf_means[i] = float(confidences[sel].mean())
        else:
            accs[i] = np.nan
            conf_means[i] = bin_centers[i]

    fig, ax = plt.subplots(figsize=(5,4))
    # perfect calibration line
    ax.plot([0,1], [0,1], linestyle="--", color="0.6", linewidth=1.0, label="Perfect")
    # reliability curve
    ax.plot(conf_means, accs, marker="o", linewidth=1.6, label="Model")
    # optional bars showing accuracy per bin
    for x, y, c in zip(conf_means, accs, counts):
        if not np.isnan(y):
            ax.add_patch(plt.Rectangle((x - 0.025, 0), 0.05, y, alpha=0.06, color="C0"))

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Observed accuracy")
    if title is None:
        title = "Reliability diagram"
    ax.set_title(title)
    ax.legend(frameon=True, fontsize=8, loc="lower right")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(str(p), dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_uncertainty_decomposition(
    df: pd.DataFrame,
    epi_col: str,
    alea_col: str,
    err_col: Optional[str],
    outpath: str,
    title: Optional[str] = None
) -> None:
    """
    Scatter plot of epistemic vs aleatoric uncertainty.

    df should contain numeric columns for epi_col and alea_col. If err_col is supplied (0/1),
    marker sizes or colors will reflect errors.
    """
    p = Path(outpath)
    ensure_parent(p)

    x = df[epi_col].values.astype(float)
    y = df[alea_col].values.astype(float)
    if err_col is not None and err_col in df.columns:
        errs = df[err_col].astype(float).values
        sizes = 18 + 80 * errs
    else:
        sizes = 20

    cmap = plt.get_cmap("viridis")
    colors = cmap(np.clip((x - np.min(x)) / (np.ptp(x) + 1e-12), 0.0, 1.0))

    fig, ax = plt.subplots(figsize=(5,4))
    ax.scatter(x, y, s=sizes, c=colors, alpha=0.75, edgecolors='none')
    ax.set_xlabel("Epistemic uncertainty")
    ax.set_ylabel("Aleatoric uncertainty")
    if title is None:
        title = "Uncertainty decomposition"
    ax.set_title(title)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(str(p), dpi=300, bbox_inches="tight")
    plt.close(fig)