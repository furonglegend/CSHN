"""
Intervention simulation.

Implements do-operator and causal entropy estimation.
"""

import torch


def do_intervention(model, z, intervene_index, new_value):
    """
    Perform do(Z_i = new_value).

    Args:
        model: structural causal model
        z: latent representation
        intervene_index: index of variable to intervene
        new_value: tensor value to assign

    Returns:
        updated latent representation
    """
    z_new = z.clone()
    z_new[:, intervene_index] = new_value

    z_updated = model.scm(z_new)
    return z_updated


def causal_entropy(logits):
    """
    Compute predictive entropy.

    H = -sum p log p
    """
    probs = torch.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
    return entropy