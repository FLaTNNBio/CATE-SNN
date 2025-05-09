"""
Pure-PyTorch rewrite of BCAUS and BCAUS_DR (keeps original public API)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from causalforge.model import Model, Propensity_Estimator, PROBLEM_TYPE


# --------------------------------------------------------------------- #
#                         Helper utilities (no sklearn)                 #
# --------------------------------------------------------------------- #
def check_xy(X, y):
    assert X.ndim == 2, "X must be 2-D"
    assert len(X) == len(y), "X and y length mismatch"
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def check_array(X):
    return np.asarray(X, dtype=np.float32)


# --------------------------------------------------------------------- #
#                           Propensity network                           #
# --------------------------------------------------------------------- #
class _PropensityNet(nn.Module):
    def __init__(self, d, h, p_drop):
        super().__init__()
        self.l1 = nn.Linear(d, h)
        self.l2 = nn.Linear(h, h)
        self.out = nn.Linear(h, 1)
        self.do = nn.Dropout(p_drop)
        nn.init.kaiming_normal_(self.l1.weight, mode="fan_in")
        nn.init.kaiming_normal_(self.l2.weight, mode="fan_in")

    def forward(self, x):
        x = F.relu(self.do(self.l1(x)))
        x = F.relu(self.do(self.l2(x)))
        return torch.sigmoid(self.out(x))


# --------------------------------------------------------------------- #
#                               BCAUS                                    #
# --------------------------------------------------------------------- #
class BCAUS(Propensity_Estimator):
    """
    Propensity estimator con auto-balance loss (PyTorch only).
    """

    def build(self, user_params):
        p = {
            "random_state": 271,
            "hidden_layer_size": None,
            "batch_size": None,
            "shuffle": True,
            "learning_rate_init": 1e-3,
            "nu": 1.0,
            "max_iter": 100,
            "alpha": 0.0,          # weight-decay / L2
            "dropout": 0.0,
            "eps": 1e-5,
            "early_stopping": False,
            "n_iter_no_change": 10,
            "balance_threshold": 0.1,
            "device": "cpu",
            "verbose": False,
        }
        p.update(user_params)
        self.params = p
        self.device = torch.device(p["device"])

    # ------------------------------------------------------------------ #
    #                                fit                                 #
    # ------------------------------------------------------------------ #
    def fit(self, X, treatment):
        rng = np.random.RandomState(self.params["random_state"])
        torch.manual_seed(self.params["random_state"])

        X, y = check_xy(X, treatment)
        X_t = torch.tensor(X, device=self.device)
        y_t = torch.tensor(y, device=self.device)

        d = X.shape[1]
        h = self.params["hidden_layer_size"] or 2 * d
        self.model = _PropensityNet(d, h, self.params["dropout"]).to(self.device)

        criterion_prop = nn.BCELoss()
        criterion_cov = nn.MSELoss()
        optimiser = optim.Adam(
            self.model.parameters(),
            lr=self.params["learning_rate_init"],
            betas=(0.5, 0.999),
            weight_decay=self.params["alpha"],
        )

        if self.params["batch_size"]:
            loader = DataLoader(
                TensorDataset(X_t, y_t),
                batch_size=self.params["batch_size"],
                shuffle=self.params["shuffle"],
            )
        else:
            loader = [(X_t, y_t)]

        best_bal = -1
        stagn = 0

        for epoch in range(self.params["max_iter"]):
            self.model.train()
            for xb, yb in loader:
                zeros = (yb == 0).nonzero(as_tuple=True)[0]
                ones = (yb == 1).nonzero(as_tuple=True)[0]

                score = self.model(xb).squeeze()
                loss_prop = criterion_prop(score, yb)

                w = (yb / (score + self.params["eps"]) + (1 - yb) / (1 - score + self.params["eps"])).unsqueeze(1)
                w = w.repeat(1, xb.size(1))

                z_mean = (w[zeros] * xb[zeros]).sum(0) / w[zeros].sum(0)
                o_mean = (w[ones] * xb[ones]).sum(0) / w[ones].sum(0)

                loss_cov = criterion_cov(z_mean, o_mean)
                loss = loss_prop + self.params["nu"] * (loss_prop / (loss_cov + 1e-8)) * loss_cov

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

            # balance metric sull'intero set
            self.model.eval()
            with torch.no_grad():
                num_bal = self._count_balanced_cov(torch.tensor(X, device=self.device),
                                                   torch.tensor(y, device=self.device))
            if num_bal > best_bal:
                best_bal = num_bal
                stagn = 0
            else:
                stagn += 1

            if self.params["verbose"] and epoch % 50 == 0:
                print(f"[{epoch}] balanced = {num_bal}")

            if self.params["early_stopping"] and stagn >= self.params["n_iter_no_change"]:
                break

        return self

    # ------------------------------------------------------------------ #
    #                helper per numero covariate bilanciate              #
    # ------------------------------------------------------------------ #
    def _count_balanced_cov(self, X, y):
        zeros = (y == 0).nonzero(as_tuple=True)[0]
        ones = (y == 1).nonzero(as_tuple=True)[0]

        score = self.model(X).squeeze()
        w = (y / (score + self.params["eps"]) + (1 - y) / (1 - score + self.params["eps"])).unsqueeze(1)
        w = w.repeat(1, X.size(1))

        z_mean = (w[zeros] * X[zeros]).sum(0) / w[zeros].sum(0)
        o_mean = (w[ones] * X[ones]).sum(0) / w[ones].sum(0)

        z_var = (w[zeros].sum(0) / (w[zeros].sum(0) ** 2 - (w[zeros] ** 2).sum(0))) * (
            (w[zeros] * (X[zeros] - z_mean) ** 2).sum(0)
        )
        o_var = (w[ones].sum(0) / (w[ones].sum(0) ** 2 - (w[ones] ** 2).sum(0))) * (
            (w[ones] * (X[ones] - o_mean) ** 2).sum(0)
        )

        diff = torch.abs(z_mean - o_mean)
        denom = torch.sqrt((z_var + o_var) / 2)
        norm_diff = diff[denom != 0] / denom[denom != 0]
        balanced = (norm_diff <= self.params["balance_threshold"]).sum().item()
        balanced += ((diff[denom == 0]) == 0).sum().item()
        return balanced

    # ------------------------------------------------------------------ #
    #                         predict / predict_proba                    #
    # ------------------------------------------------------------------ #
    def predict(self, X):
        X = torch.tensor(check_array(X), device=self.device)
        with torch.no_grad():
            scores = self.model(X).squeeze().cpu().numpy()
        return (scores >= 0.5).astype(int)

    def predict_proba(self, X):
        X = torch.tensor(check_array(X), device=self.device)
        with torch.no_grad():
            scores = self.model(X).squeeze().cpu().numpy()
        return np.stack([1 - scores, scores], axis=1)


# --------------------------------------------------------------------- #
#                              BCAUS-DR                                  #
# --------------------------------------------------------------------- #
class BCAUS_DR(Model):
    """
    Doubly-Robust ATE con propensity stimata da BCAUS e
    outcome models stimati con Ridge (chiusa) in torch.
    """

    def build(self, params):
        self.props = Model.create_model(
            "bcaus",
            params,
            problem_type=PROBLEM_TYPE.PROPENSITY_ESTIMATION,
            multiple_treatments=False,
        )
        self.alpha = params.get("alpha_ridge", 1e-3)
        self.device = torch.device(params.get("device", "cpu"))

    # -------------------------------------------------------------- #
    def fit(self, X, treatment, y):
        X, t = check_xy(X, treatment)
        y = np.asarray(y, dtype=np.float32)

        # 1) fit propensity
        self.props.fit(X, t)
        e_hat = self.props.predict_proba(X)[:, 1]

        # 2) fit Ridge outcome models (analitico) su torch
        Xt = torch.tensor(X, device=self.device)
        yt = torch.tensor(y, device=self.device).unsqueeze(1)

        idx_t = torch.tensor(np.where(t == 1)[0], device=self.device)
        idx_c = torch.tensor(np.where(t == 0)[0], device=self.device)

        self.w_t = self._ridge_closed(Xt[idx_t], yt[idx_t])
        self.w_c = self._ridge_closed(Xt[idx_c], yt[idx_c])

    # -------------------------------------------------------------- #
    def _ridge_closed(self, X_sub, y_sub):
        d = X_sub.shape[1]
        A = X_sub.T @ X_sub + self.alpha * torch.eye(d, device=self.device)
        b = X_sub.T @ y_sub
        return torch.linalg.solve(A, b)  # (d,1)

    # -------------------------------------------------------------- #
    def support_ite(self):
        return False

    def predict_ite(self, X):
        raise RuntimeError("ITE not supported in BCAUS_DR")

    # -------------------------------------------------------------- #
    def predict_ate(self, X, treatment, y):
        X = check_array(X)
        e_hat = self.props.predict_proba(X)[:, 1]

        Xt = torch.tensor(X, device=self.device)
        yt = torch.tensor(y, device=self.device).unsqueeze(1)
        tt = torch.tensor(treatment, device=self.device).unsqueeze(1)
        et = torch.tensor(e_hat, device=self.device).unsqueeze(1)

        m1 = Xt @ self.w_t
        m0 = Xt @ self.w_c

        dr = tt * (yt - m1) / et - (1 - tt) * (yt - m0) / (1 - et) + (m1 - m0)
        return float(dr.mean().cpu().item())
