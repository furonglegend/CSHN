"""
Joint training loss for SphUnc.

L = L_pred + λ1 L_entropy + λ2 L_causal
"""

import torch
import torch.nn.functional as F
from .regularizers import entropy_calibration_loss, causal_sparsity_loss


class SphUncLoss:
    """
    Composite loss function.
    """

    def __init__(self, lambda_entropy: float = 0.1,
                 lambda_causal: float = 0.01):
        self.lambda_entropy = lambda_entropy
        self.lambda_causal = lambda_causal

    def __call__(self,
                 logits: torch.Tensor,
                 targets: torch.Tensor,
                 pred_entropy: torch.Tensor,
                 target_entropy: torch.Tensor,
                 adj_matrix: torch.Tensor):
        """
        Args:
            logits: model predictions
            targets: ground truth labels
            pred_entropy: predicted epistemic entropy
            target_entropy: supervision signal
            adj_matrix: SCM adjacency

        Returns:
            total loss
        """
        L_pred = F.cross_entropy(logits, targets)

        L_entropy = entropy_calibration_loss(pred_entropy, target_entropy)

        L_causal = causal_sparsity_loss(adj_matrix)

        total = L_pred \
                + self.lambda_entropy * L_entropy \
                + self.lambda_causal * L_causal

        return total