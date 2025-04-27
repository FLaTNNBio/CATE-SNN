import math

import numpy as np
import torch
from torch.utils.data import Dataset


class IHDPLoader(Dataset):
    """
    Adapted from Causal Forge's DataLoader
    https://github.com/anthem-ai/causalforge/blob/main/causalforge/data_loader.py
    """

    def __init__(self, is_train=True):
        super(IHDPLoader, self).__init__()
        self.is_train = is_train
        self.path_train = 'ihdp_npci_1-1000.train.npz'
        self.path_test = 'ihdp_npci_1-1000.test.npz'
        self.loaded = False
        self.load()

    def load(self):
        # carica il .npz
        data = np.load(self.path_train if self.is_train else self.path_test)

        # main tensors
        self.X = torch.tensor(data.f.x.copy(), dtype=torch.float)  # [n_units, d, n_real]
        self.T = torch.tensor(data.f.t.copy(), dtype=torch.float)  # [n_units, n_real]
        self.YF = torch.tensor(data.f.yf.copy(), dtype=torch.float)  # [n_units, n_real]
        self.YCF = torch.tensor(data.f.ycf.copy(), dtype=torch.float)  # [n_units, n_real]
        self.mu_0 = torch.tensor(data.f.mu0.copy(), dtype=torch.float)  # [n_units, n_real]
        self.mu_1 = torch.tensor(data.f.mu1.copy(), dtype=torch.float)  # [n_units, n_real]

        # propensity & weights per realizzazione
        #  u ha shape [1, n_real]
        self.u = self.T.mean(dim=0, keepdim=True)
        #  w ha shape [n_units, n_real]
        self.w = (self.T / (2 * self.u)) + ((1 - self.T) / (2 * (1 - self.u)))

        self.loaded = True

        # il tuo main usa test_dataset.load() per ottenere tutti e 8 i tensori
        return (self.X, self.T, self.YF, self.YCF,
                self.mu_0, self.mu_1, self.u, self.w)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        """
        Qui restituiamo 8 tensori in ordine:
        X, T, YF, YCF, mu0, mu1, u, w
        esattamente come fa load()
        """
        if not self.loaded:
            self.load()

        # estrai per la singola unità idx
        x_i = self.X[idx]  # [d, n_real]
        t_i = self.T[idx]  # [n_real]
        yf_i = self.YF[idx]  # [n_real]
        ycf_i = self.YCF[idx]  # [n_real]
        mu0_i = self.mu_0[idx]  # [n_real]
        mu1_i = self.mu_1[idx]  # [n_real]
        u_i = self.u.squeeze(0)  # [n_real]     (da [1, n_real] -> [n_real])
        w_i = self.w[idx]  # [n_real]

        return x_i, t_i, yf_i, ycf_i, mu0_i, mu1_i, u_i, w_i


from torch.utils.data import Sampler
from collections import defaultdict
import numpy as np


class BalancedBatchSampler(Sampler):
    """
    Ritorna indici tali che in ogni batch ci siano sia trattati che controlli.
    """

    def __init__(self, treatments, batch_size):
        self.batch_size = batch_size
        self.treated_idx = np.where(treatments == 1)[0]
        self.control_idx = np.where(treatments == 0)[0]
        self.num_batches = math.ceil(len(treatments) / batch_size)

    def __iter__(self):
        np.random.shuffle(self.treated_idx)
        np.random.shuffle(self.control_idx)
        t_ptr = c_ptr = 0
        for _ in range(self.num_batches):
            half = self.batch_size // 2
            t_batch = self.treated_idx[t_ptr:t_ptr + half]
            c_batch = self.control_idx[c_ptr:c_ptr + half]
            if len(t_batch) < half:
                t_batch = np.random.choice(self.treated_idx, half, replace=True)
            if len(c_batch) < half:
                c_batch = np.random.choice(self.control_idx, half, replace=True)
            batch_indices = np.concatenate([t_batch, c_batch])
            np.random.shuffle(batch_indices)
            yield batch_indices.tolist()
            t_ptr += half
            c_ptr += half

    def __len__(self):
        return self.num_batches
