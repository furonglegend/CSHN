"""
Robustness experiments.

Implements:
- Feature dropout experiments
- Performance vs dropout curve generation
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


def apply_feature_dropout(x: torch.Tensor, dropout_rate: float):
    """
    Randomly zero out features.

    Args:
        x: (B, D)
        dropout_rate: fraction of features to remove

    Returns:
        perturbed tensor
    """
    mask = torch.rand_like(x) > dropout_rate
    return x * mask


@torch.no_grad()
def evaluate_under_dropout(model, dataloader, device, rates):
    """
    Evaluate model performance under increasing feature dropout.
    """
    results = []

    model.eval()

    for rate in rates:
        total = 0
        correct = 0

        for batch in dataloader:
            x = batch["features"].to(device)
            y = batch["label"].to(device)

            x = apply_feature_dropout(x, rate)

            outputs = model(x)
            logits = outputs["logits"]
            preds = torch.argmax(logits, dim=-1)

            correct += (preds == y).sum().item()
            total += y.size(0)

        acc = correct / total
        results.append(acc)

    return results


def plot_dropout_curve(rates, accuracies, save_path):
    """
    Plot robustness curve.
    """
    plt.figure(figsize=(5, 4))
    plt.plot(rates, accuracies, marker="o")
    plt.xlabel("Feature Dropout Rate")
    plt.ylabel("Accuracy")
    plt.title("Robustness to Feature Dropout")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()