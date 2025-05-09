"""
PyTorch replacement for utils_tf.py
(derive from original TF/Keras implementation)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ------------------------------------------------------------------ #
#                       numpy / pandas helper                         #
# ------------------------------------------------------------------ #
def convert_pd_to_np(*args):
    """Convert pandas objects to numpy, leave others untouched."""
    out = [obj.to_numpy() if hasattr(obj, "to_numpy") else obj for obj in args]
    return out if len(out) > 1 else out[0]


# ------------------------------------------------------------------ #
#                            base losses                              #
# ------------------------------------------------------------------ #
def binary_classification_loss(concat_true: torch.Tensor,
                               concat_pred: torch.Tensor) -> torch.Tensor:
    """
    BCE on treatment assignment.
    concat_true : [N, 2]  -> columns (y_true, t_true)
    concat_pred : [N, 4]  -> columns (y0_pred, y1_pred, t_pred, epsilon)
    """
    t_true = concat_true[:, 1]
    t_pred = concat_pred[:, 2]
    t_pred = (t_pred + 0.001) / 1.002               # smoothing identical to TF
    return F.binary_cross_entropy(t_pred, t_true, reduction="sum")


def regression_loss(concat_true: torch.Tensor,
                    concat_pred: torch.Tensor) -> torch.Tensor:
    """
    ∑ (1-t)(y - y0)^2 + ∑ t (y - y1)^2
    """
    y_true = concat_true[:, 0]
    t_true = concat_true[:, 1]
    y0, y1 = concat_pred[:, 0], concat_pred[:, 1]

    loss0 = ((1.0 - t_true) * (y_true - y0).pow(2)).sum()
    loss1 = (t_true * (y_true - y1).pow(2)).sum()
    return loss0 + loss1


def dragonnet_loss_binarycross(concat_true, concat_pred):
    """regression_loss + binary_classification_loss"""
    return regression_loss(concat_true, concat_pred) + \
           binary_classification_loss(concat_true, concat_pred)


# ------------------------------------------------------------------ #
#                         metrics / trackers                          #
# ------------------------------------------------------------------ #
def treatment_accuracy(concat_true, concat_pred):
    """
    Simple classification accuracy on treatment column.
    """
    t_true = concat_true[:, 1]
    t_pred = (concat_pred[:, 2] >= 0.5).float()
    return (t_true == t_pred).float().mean()


def track_epsilon(concat_true, concat_pred):
    """Mean |epsilon|."""
    eps = concat_pred[:, 3]
    return torch.abs(eps).mean()


# ------------------------------------------------------------------ #
#                targeted-regularisation loss wrapper                #
# ------------------------------------------------------------------ #
def make_tarreg_loss(ratio=1.0, dragonnet_loss=dragonnet_loss_binarycross):
    """
    Factory that wraps a base loss with targeted regularisation term.
    """

    def tarreg_ATE_unbounded_domain_loss(concat_true, concat_pred):
        vanilla = dragonnet_loss(concat_true, concat_pred)

        y_true = concat_true[:, 0]
        t_true = concat_true[:, 1]

        y0_pred, y1_pred, t_pred, eps = \
            concat_pred[:, 0], concat_pred[:, 1], concat_pred[:, 2], concat_pred[:, 3]

        t_pred = (t_pred + 0.01) / 1.02                     # same smoothing
        y_pred = t_true * y1_pred + (1 - t_true) * y0_pred

        h = t_true / t_pred - (1 - t_true) / (1 - t_pred)
        y_pert = y_pred + eps * h

        targ_reg = ((y_true - y_pert) ** 2).sum()
        return vanilla + ratio * targ_reg

    return tarreg_ATE_unbounded_domain_loss


# ------------------------------------------------------------------ #
#                         epsilon “layer”                             #
# ------------------------------------------------------------------ #
class EpsilonLayer(nn.Module):
    """
    Torch analogue of the Keras custom layer that learns a scalar ε
    and expands it to match batch size.
    """

    def __init__(self):
        super().__init__()
        self.epsilon = nn.Parameter(torch.randn(1, 1))   # same init “RandomNormal”

    def forward(self, t_pred):
        """
        t_pred is only used for its batch dimension; output shape [N,1]
        """
        return self.epsilon.expand(t_pred.size(0), 1)
