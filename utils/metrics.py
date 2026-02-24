"""
Evaluation metrics: accuracy, macro-F1, AUC, ECE, Brier score.

Notes:
- Functions accept numpy arrays and return Python floats.
- AUC uses sklearn. For multiclass, it uses 'ovr' by default (requires one-hot or probability matrix).
- ECE is implemented as binning calibration error computed on predicted confidence for predicted class.
"""
from typing import Union
import numpy as np

# sklearn imports (soft dependency)
try:
    from sklearn.metrics import f1_score, roc_auc_score, brier_score_loss
    _SKLEARN_AVAILABLE = True
except Exception:
    f1_score = None
    roc_auc_score = None
    brier_score_loss = None
    _SKLEARN_AVAILABLE = False

def accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute accuracy. preds and labels are 1D arrays of same length.
    """
    preds = np.asarray(preds).ravel()
    labels = np.asarray(labels).ravel()
    if preds.shape != labels.shape:
        raise ValueError("preds and labels must have same shape")
    return float((preds == labels).mean())

def macro_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    """
    Macro-averaged F1 score using sklearn if available, otherwise a simple fallback.
    """
    if not _SKLEARN_AVAILABLE:
        # fallback: approximate using per-class F1 computed manually
        preds = np.asarray(preds).ravel()
        labels = np.asarray(labels).ravel()
        classes = np.unique(np.concatenate([preds, labels]))
        f1s = []
        for c in classes:
            tp = int(((preds == c) & (labels == c)).sum())
            fp = int(((preds == c) & (labels != c)).sum())
            fn = int(((preds != c) & (labels == c)).sum())
            prec = tp / (tp + fp + 1e-12)
            rec = tp / (tp + fn + 1e-12)
            f1s.append(2 * prec * rec / (prec + rec + 1e-12))
        return float(np.mean(f1s))
    else:
        return float(f1_score(labels, preds, average="macro"))

def auc_score(probs: Union[np.ndarray, None], labels: np.ndarray, multi_class: str = "ovr") -> float:
    """
    Compute AUC. For binary classification, 'probs' should be shape (N,) or (N,2).
    For multiclass, 'probs' should be shape (N, C) and multi_class can be 'ovr' or 'ovo'.
    Returns NaN if sklearn is not available.
    """
    if not _SKLEARN_AVAILABLE:
        return float("nan")
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    # if probs is 1D assume positive class probability
    if probs.ndim == 1:
        return float(roc_auc_score(labels, probs))
    return float(roc_auc_score(labels, probs, multi_class=multi_class))

def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Brier score for probabilistic predictions.
    probs: shape (N,) for binary or (N,C) for multiclass (one-hot true labels expected).
    labels: shape (N,) with integer labels.
    """
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    if _SKLEARN_AVAILABLE and probs.ndim == 1:
        return float(brier_score_loss(labels, probs))
    # fallback multiclass: compute mean squared error between one-hot and probs
    if probs.ndim == 2:
        n, c = probs.shape
        onehot = np.zeros_like(probs)
        onehot[np.arange(n), labels] = 1.0
        return float(((probs - onehot) ** 2).sum() / n)
    return float("nan")

def ece_score(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """
    Expected Calibration Error (ECE).
    confidences: predicted probability/confidence for the predicted class (N,)
    correct: binary array (0/1) indicating whether prediction was correct (N,)
    n_bins: number of bins

    ECE = sum_k (|B_k|/N) * |acc(B_k) - conf(B_k)|
    """
    confidences = np.asarray(confidences).ravel()
    correct = np.asarray(correct).ravel().astype(float)
    if confidences.shape[0] != correct.shape[0]:
        raise ValueError("confidences and correct must have same length")
    n = len(confidences)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.digitize(confidences, bins) - 1
    ece = 0.0
    for i in range(n_bins):
        sel = (bin_idx == i)
        count = sel.sum()
        if count == 0:
            continue
        avg_conf = float(confidences[sel].mean())
        acc = float(correct[sel].mean())
        ece += (count / n) * abs(avg_conf - acc)
    return float(ece)