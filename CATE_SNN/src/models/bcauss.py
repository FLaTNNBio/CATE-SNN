# --- bcauss_torch.py -------------------------------------------------
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.model import Model  # tua classe base
from src.models.utils import convert_pd_to_np  # helper CausalForge


class EpsilonLayer(nn.Module):
    def forward(self, t_pred):
        return torch.rand_like(t_pred)  # uniforme [0,1) come in Keras


class BCAUSS(Model, nn.Module):
    # -----------------------------------------------------------------
    def __init__(self, **user_params):
        nn.Module.__init__(self)
        self.build(user_params)

    # -----------------------------------------------------------------
    def _init_weights(self):
        """Normal(0,0.05) come in Keras"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.05)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # -----------------------------------------------------------------
    def build(self, user_params):
        # ---------- parametri default ----------
        p = {
            'neurons_per_layer': 200,
            'act_fn': 'relu',
            'reg_l2': 0.01,
            'verbose': True,
            'val_split': 0.22,
            'ratio': 1.0,
            'optim': 'sgd',
            'epochs': 500,
            'learning_rate': 1e-5,
            'momentum': 0.9,
            'use_bce': False,
            'norm_bal_term': True,
            'use_targ_term': False,
            'b_ratio': 1.0,
            'bs_ratio': 1.0,
            'scale_preds': True
        }
        if 'input_dim' not in user_params:
            raise ValueError("input_dim must be specified!")
        p.update(user_params)
        self.params = p

        # ---------- rete ----------
        D, N = p['input_dim'], p['neurons_per_layer']
        act = nn.ReLU if p['act_fn'] == 'relu' else nn.Identity

        # Representation tower
        self.repr_net = nn.Sequential(
            nn.Linear(D, N), act(),
            nn.Linear(N, N), act(),
            nn.Linear(N, N), act()
        )

        # Heads
        self.t_head = nn.Sequential(nn.Linear(N, 1), nn.Sigmoid())
        h = N // 2
        self.y0_net = nn.Sequential(nn.Linear(N, h), act(),
                                    nn.Linear(h, h), act(),
                                    nn.Linear(h, 1))
        self.y1_net = nn.Sequential(nn.Linear(N, h), act(),
                                    nn.Linear(h, h), act(),
                                    nn.Linear(h, 1))
        self.eps_layer = EpsilonLayer()

        self._init_weights()  # ← same init as Keras
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.y_scaler = None

    # -----------------------------------------------------------------
    def forward(self, x):
        z = self.repr_net(x)
        t_pred = self.t_head(z)
        y0 = self.y0_net(z)
        y1 = self.y1_net(z)
        eps = self.eps_layer(t_pred)
        return t_pred, y0, y1, eps

    # -----------------------------------------------------------------
    def compute_loss(self, x, t_true, y_true):
        p = self.params
        bs = x.size(0)

        t_pred, y0, y1, eps = self.forward(x)
        t_smooth = (t_pred + 1e-3) / 1.002

        # -------- (i) factual MSE --------
        loss0 = ((1 - t_true) * (y_true - y0).pow(2)).sum()
        loss1 = (t_true * (y_true - y1).pow(2)).sum()
        reg_loss = loss0 + loss1

        # -------- (ii) BCE sui trattamenti --------
        if p['use_bce']:
            bce = F.binary_cross_entropy(t_smooth, t_true, reduction='sum')
            reg_loss = reg_loss + bce

        # -------- (iii) targeted reg --------
        if p['use_targ_term']:
            y_pred = t_true * y1 + (1 - t_true) * y0
            h_term = t_true / t_smooth - (1 - t_true) / (1 - t_smooth)
            y_pert = y_pred + eps * h_term
            targ = p['ratio'] * ((y_true - y_pert).pow(2)).sum()
            reg_loss = reg_loss + targ

        # -------- (iv) balancing --------
        w1 = t_true / t_smooth
        w0 = (1 - t_true) / (1 - t_smooth)
        if p['norm_bal_term']:
            ones_mean = (w1 * x).sum(0) / w1.sum(0)
            zeros_mean = (w0 * x).sum(0) / w0.sum(0)
        else:
            ones_mean = (w1 * x).sum(0)
            zeros_mean = (w0 * x).sum(0)
        bal = F.mse_loss(zeros_mean, ones_mean, reduction='sum')
        reg_loss = reg_loss + p['b_ratio'] * bal

        # -------- (v) L2 come in Keras --------
        l2_term = 0.0
        for param in self.parameters():
            if param.ndim > 1:  # escludi bias
                l2_term = l2_term + (param ** 2).sum()
        reg_loss = reg_loss + p['reg_l2'] * l2_term

        if torch.isnan(reg_loss):
            pass
            #raise ValueError("NaN in loss")

        return reg_loss

    # -----------------------------------------------------------------
    def fit(self, X, treatment, y):
        X, treatment, y = convert_pd_to_np(X, treatment, y)
        N = X.shape[0]
        p = self.params

        y = y.reshape(-1, 1)
        if p['scale_preds']:
            self.y_scaler = StandardScaler().fit(y)
            y = self.y_scaler.transform(y)

        X_t = torch.tensor(X, dtype=torch.float32)
        t_t = torch.tensor(treatment.reshape(-1, 1), dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)

        # split
        idx = np.random.permutation(N)
        split = int(N * (1 - p['val_split']))
        tr, va = idx[:split], idx[split:]
        bs = int(N * p['bs_ratio'])

        train_loader = DataLoader(TensorDataset(X_t[tr], t_t[tr], y_t[tr]),
                                  batch_size=bs, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_t[va], t_t[va], y_t[va]),
                                batch_size=bs, shuffle=False)

        # optimizer (no weight_decay, già incluso in loss)
        optim_cls = Adam if p['optim'] == 'adam' else SGD
        optim_kwargs = {'lr': p['learning_rate'],
                        'momentum': p['momentum']} if p['optim'] == 'sgd' else \
            {'lr': p['learning_rate']}
        optimizer = optim_cls(self.parameters(), **optim_kwargs)
        scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                      factor=0.5, patience=5,
                                      verbose=p['verbose'])

        best_val, patience = float('inf'), 0
        self.to(self.device)
        X_t, t_t, y_t = None, None, None  # libera RAM host

        for epoch in range(1, p['epochs'] + 1):
            # ---- train ----
            self.train()
            tl = 0.0
            for xb, tb, yb in train_loader:
                xb, tb, yb = xb.to(self.device), tb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = self.compute_loss(xb, tb, yb)
                loss.backward()
                optimizer.step()
                tl += loss.item()
            tl /= len(train_loader)

            # ---- val ----
            self.eval()
            vl = 0.0
            with torch.no_grad():
                for xb, tb, yb in val_loader:
                    xb, tb, yb = xb.to(self.device), tb.to(self.device), yb.to(self.device)
                    vl += self.compute_loss(xb, tb, yb).item()
            vl /= len(val_loader)

            scheduler.step(vl)
            if p['verbose']:
                print(f"Epoch {epoch}/{p['epochs']}  train={tl:.2f}  val={vl:.2f}")

            if vl < best_val - 1e-6:
                best_val = vl
                best_state = self.state_dict()
                patience = 0
            else:
                patience += 1
                if patience >= 40:
                    if p['verbose']:
                        print("Early stopping.")
                    break

        self.load_state_dict(best_state)

    # -----------------------------------------------------------------
    def mu_and_embedding(self, x):
        z = self.repr_net(x)
        y0 = self.y0_net(z)
        y1 = self.y1_net(z)
        mu = torch.cat([y0, y1], dim=1)
        return mu, z

    # -----------------------------------------------------------------
    @property
    def support_ite(self):
        return True

    # -----------------------------------------------------------------
    def predict_ite(self, X):
        X = convert_pd_to_np(X)
        self.eval()
        xt = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            _, y0, y1, _ = self.forward(xt)
        y0_np, y1_np = y0.cpu().numpy(), y1.cpu().numpy()
        if self.params['scale_preds']:
            y0_np = self.y_scaler.inverse_transform(y0_np)
            y1_np = self.y_scaler.inverse_transform(y1_np)
        return (y1_np - y0_np).squeeze()

    def predict_ate(self, X, *args):
        return float(np.mean(self.predict_ite(X)))
# ---------------------------------------------------------------------
