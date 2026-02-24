"""
Calibration utilities.

Includes:
- ECE computation
"""

import numpy as np


def compute_ece(confidences, predictions, labels, n_bins=15):
    """
    Expected Calibration Error.

    Args:
        confidences: predicted probability of predicted class
        predictions: predicted labels
        labels: true labels
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if mask.sum() == 0:
            continue

        acc = (predictions[mask] == labels[mask]).mean()
        conf = confidences[mask].mean()

        ece += (mask.sum() / len(confidences)) * abs(acc - conf)

    return ece