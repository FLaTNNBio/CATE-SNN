import torch
import torch.nn as nn
import torch.nn.functional as F


class RepresentationNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # Layer 1
            nn.ELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),  # Layer 2
            nn.ELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),  # Layer 3
            nn.ELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),  # Layer 4
            nn.ELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),  # Layer 5
            nn.ELU(),
            nn.BatchNorm1d(hidden_dim),
        )

    def forward(self, x):
        return self.net(x)


class HypothesisNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # Layer 1
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),  # Layer 2
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),  # layer 3
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),  # Layer 4
            nn.ELU(),
            nn.Linear(hidden_dim, 1)  # Predicting a single outcome in Layer 5
        )

    def forward(self, x):
        return self.net(x)


class DualHeadSiameseCATE(nn.Module):
    def __init__(self, input_dim, hidden_dim=100, margin=1.0):
        super().__init__()
        # Moduli Phi distinti
        self.Phi1 = RepresentationNetwork(input_dim, hidden_dim)
        self.Phi2 = RepresentationNetwork(input_dim, hidden_dim)
        # Head per outcome
        self.h0 = HypothesisNetwork(hidden_dim, hidden_dim)
        self.h1 = HypothesisNetwork(hidden_dim, hidden_dim)
        self.margin = margin

    def forward_once(self, x, t):
        # Embedding condizionato: se t=1 usa Phi1, altrimenti Phi2
        e = t * self.Phi1(x) + (1 - t) * self.Phi2(x)
        # Predizione factual
        y_pred = t * self.h1(e) + (1 - t) * self.h0(e)
        return e, y_pred

    def forward(self, x1, x2, t1, t2):
        # Due chiamate forward_once indipendenti
        e1, y1 = self.forward_once(x1, t1)
        e2, y2 = self.forward_once(x2, t2)
        return e1, e2, y1, y2



def contrastive_loss(e1, e2, same_indicator, margin=1.0):
    """
    L_c = s * d^2 + (1-s) * max(0, m - d)^2
    dove d = ||e1 - e2||_2
    """
    distance = torch.norm(e1 - e2, p=2, dim=1)
    # Nel caso di batch, same_indicator deve avere shape (batch,).
    # same_indicator=1 => stesse etichette (z_i=z_j)
    # same_indicator=0 => diverse etichette (z_i!=z_j)
    loss_pos = same_indicator * (distance ** 2)
    loss_neg = (1 - same_indicator) * F.relu(margin - distance) ** 2
    return (loss_pos + loss_neg).mean()


def contrastive_loss_batch(e1, e2, same_indicator, margin=1.0):
    """
    e1: [N, emb_dim]
    e2: [N, emb_dim]
    same_indicator: [N] (0 o 1)
    margin: float

    Ritorna un singolo scalare (media sul batch N).

    L_c(i) = s_i * d_i^2 + (1 - s_i) * max(0, m - d_i)^2
    con d_i = || e1[i] - e2[i] ||_2
    """
    distance = torch.norm(e1 - e2, p=2, dim=1)  # [N]
    s = same_indicator  # [N]

    loss_pos = s * (distance ** 2)  # se s=1
    loss_neg = (1 - s) * F.relu(margin - distance) ** 2  # se s=0

    loss_batch = loss_pos + loss_neg  # [N]
    return loss_batch.mean()
