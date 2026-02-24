"""
Regularization terms.

Includes:
- Entropy calibration loss
- Causal sparsity regularization
"""

import torch


def entropy_calibration_loss(pred_entropy: torch.Tensor,
                             target_entropy: torch.Tensor) -> torch.Tensor:
    """
    Encourage predicted entropy to match target entropy.

    Uses mean squared deviation.
    """
    return torch.mean((pred_entropy - target_entropy) ** 2)


def causal_sparsity_loss(adj_matrix: torch.Tensor) -> torch.Tensor:
    """
    L1 penalty on adjacency matrix to promote sparse structure.
    """
    return torch.mean(torch.abs(adj_matrix))


def acyclicity_constraint(adj_matrix: torch.Tensor) -> torch.Tensor:
    """
    Differentiable acyclicity constraint from NOTEARS.

    h(A) = trace(exp(A ⊙ A)) - d
    """
    d = adj_matrix.shape[0]
    expm = torch.matrix_exp(adj_matrix * adj_matrix)
    h = torch.trace(expm) - d
    return h