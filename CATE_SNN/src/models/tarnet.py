import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

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


class TARNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=100):
        super().__init__()

        self.phi = RepresentationNetwork(input_dim, hidden_dim)
        self.h_1 = HypothesisNetwork(hidden_dim, hidden_dim)  # treated hypothesis, t = 1
        self.h_0 = HypothesisNetwork(hidden_dim, hidden_dim)  # control hypothesis, t = 0

    def forward(self, x, t):
        '''
        INPUTS:
            x [batch_size, 25] = 25 covariates for each factual sample in batch
            t [batch_size, 1]  = binary treatment applied for each factual sample in batch
        '''
        # Send x through representation network to learn representation covariates, phi_x
        # Input: x [batch_size, 25] -> Output: phi_x [batch_size, hidden_dim]
        phi_x = self.phi(x)

        # Send phi_x through hypothesis network to learn h1 and h0 estimates
        # Input: phi_x [batch_size, hidden_dim], Output: h_1_phi_x [batch_size, 1]
        h_1_phi_x = self.h_1(phi_x)
        h_0_phi_x = self.h_0(phi_x)

        # Mask the h1 estimates and h0 estimates according to t
        # predictions = [batch_size, 1], the h(\phi(x_i), t_i) for each element, i, in the batch
        predictions = h_1_phi_x * t + h_0_phi_x * (1 - t)  # [batch_size, 1]

        return predictions
