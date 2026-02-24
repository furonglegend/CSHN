"""
Causal recovery metrics.
"""

import numpy as np
from scipy.stats import spearmanr


def precision_at_k(pred_adj, true_adj, k):
    """
    Precision@K for adjacency matrix recovery.
    """
    flat_pred = pred_adj.flatten()
    flat_true = true_adj.flatten()

    idx = np.argsort(-flat_pred)[:k]
    precision = flat_true[idx].sum() / k
    return precision


def rank_correlation(pred_adj, true_adj):
    """
    Spearman rank correlation between predicted and true adjacency.
    """
    return spearmanr(pred_adj.flatten(), true_adj.flatten()).correlation