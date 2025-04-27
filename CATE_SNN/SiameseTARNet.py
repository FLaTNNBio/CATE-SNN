# SiameseTARNet.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# Helper function to get activation layer from name
def get_activation(name):
    name = name.lower()
    if name == 'relu':
        return nn.ReLU()
    elif name == 'elu':
        return nn.ELU()
    else:
        raise ValueError(f"Unknown activation function name: {name}")


class RepresentationNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=100, num_layers=5, activation_fn_name='elu'):
        super().__init__()
        if num_layers < 1:
            raise ValueError("Number of layers must be at least 1")

        layers = []
        current_dim = input_dim
        activation = get_activation(activation_fn_name)

        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(activation)
            # Apply BatchNorm after activation for all layers in representation
            # (Manteniamo la scelta originale, anche se Linear->BN->Act è più comune)
            #layers.append(nn.BatchNorm1d(hidden_dim))
            current_dim = hidden_dim  # Input for the next linear layer is the current hidden_dim

        self.net = nn.Sequential(*layers)
        self.output_dim = current_dim

    def forward(self, x):
        return self.net(x)


class HypothesisNetwork(nn.Module):
    """ Hypothesis network h0 or h1 """

    def __init__(self, input_dim, hidden_dim=100, num_hidden_layers=4, activation_fn_name='elu'):
        # num_hidden_layers = number of hidden Linear layers before the final output layer
        super().__init__()
        if num_hidden_layers < 0:  # Allow 0 hidden layers (direct linear map)
            raise ValueError("Number of hidden layers cannot be negative")

        layers = []
        current_dim = input_dim
        activation = get_activation(activation_fn_name)

        # Hidden layers
        for i in range(num_hidden_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(activation)
            # No BatchNorm in hypothesis network
            current_dim = hidden_dim

        # Final output layer (predicts potential outcome)
        layers.append(nn.Linear(current_dim, 1))
        # Usually no activation after the final regression layer

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# --- MODIFIED DualHeadSiameseCATE ---
class DualHeadSiameseCATE(nn.Module):
    def __init__(self, input_dim, hidden_dim=100, margin=1.0,
                 rep_layers=5, rep_activation='elu',  # Params for Phi
                 hyp_layers=4, hyp_activation='elu'  # Params for h0/h1
                 ):
        super().__init__()

        # *** MODIFICA CHIAVE: Una sola rete Phi condivisa ***
        self.Phi = RepresentationNetwork(input_dim, hidden_dim, rep_layers, rep_activation)
        # Non più self.Phi1 e self.Phi2

        # L'input dimension per h0/h1 è l'output della rete Phi condivisa
        rep_output_dim = self.Phi.output_dim

        # Hypothesis Networks (h0, h1) - rimangono separate
        self.h0 = HypothesisNetwork(rep_output_dim, hidden_dim, hyp_layers, hyp_activation)
        self.h1 = HypothesisNetwork(rep_output_dim, hidden_dim, hyp_layers, hyp_activation)
        self.margin = margin

    def forward_once(self, x, t):
        """ Calcola l'embedding e la predizione fattuale per un singolo input. """
        # Assicura che t sia [B, 1] per il broadcasting
        if t.dim() == 1:
            t = t.unsqueeze(1)

        # *** MODIFICA CHIAVE: Applica sempre la rete Phi condivisa ***
        e = self.Phi(x)  # Calcola l'embedding usando la rete condivisa

        # La selezione basata su 't' avviene SOLO per gli head di ipotesi
        y_pred = t * self.h1(e) + (1 - t) * self.h0(e)
        return e, y_pred

    def forward(self, x1, x2, t1, t2):
        """ Processa una coppia di input (usato nel training loop). """
        # Chiama forward_once per ogni input, usando la logica aggiornata con Phi condiviso
        e1, y1 = self.forward_once(x1, t1)
        e2, y2 = self.forward_once(x2, t2)
        return e1, e2, y1, y2  # y1 e y2 sono le predizioni fattuali


# --- Contrastive Loss Functions (rimangono invariate) ---
def contrastive_loss(e1, e2, same_indicator, margin=1.0):
    """
    L_c = s * d^2 + (1-s) * max(0, m - d)^2
    dove d = ||e1 - e2||_2
    """
    distance = torch.norm(e1 - e2, p=2, dim=1)
    loss_pos = same_indicator * (distance ** 2)
    loss_neg = (1 - same_indicator) * F.relu(margin - distance) ** 2
    return (loss_pos + loss_neg).mean()


def contrastive_loss_batch(e1, e2, same_indicator, margin=1.0):
    """
    e1: [N, emb_dim]
    e2: [N, emb_dim]
    same_indicator: [N] (0 o 1)
    margin: float
    """
    distance = torch.norm(e1 - e2, p=2, dim=1)  # [N]
    s = same_indicator  # [N]

    loss_pos = s * (distance ** 2)  # se s=1
    loss_neg = (1 - s) * F.relu(margin - distance) ** 2  # se s=0

    loss_batch = loss_pos + loss_neg  # [N]
    return loss_batch.mean()

