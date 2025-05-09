#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Versione solo-PyTorch del training Siamese-BCAUSS con loss contrastiva
e soglia dinamica sul 20° percentile delle differenze di CATE.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader as TorchLoader

from src.data_loader import DataLoader
from src.metrics import eps_ATE_diff, PEHE_with_ite
from src.models.bcauss import BCAUSS

# ------------------------------------------------------------------
# 1) Patch numpy.load per i file picklati (come nell’originale)
_np_load_old = np.load


def np_load_patch(file, *args, **kwargs):
    return _np_load_old(file, *args, allow_pickle=True, **kwargs)


np.load = np_load_patch


# ------------------------------------------------------------------

# ----------------------- Utility functions ------------------------


def compute_tau_threshold(mu0, mu1, percentile=20, sample_size=100_000):
    """
    Soglia epsilon = percentile-esimo delle |tau_i − tau_j|.
    """
    tau = mu1 - mu0
    N = tau.shape[0]

    max_pairs = N * (N - 1) // 2
    M = min(sample_size, max_pairs)

    idx1 = np.random.randint(0, N, size=M)
    idx2 = np.random.randint(0, N, size=M)
    diffs = np.abs(tau[idx1] - tau[idx2])

    return float(np.percentile(diffs, percentile))


def make_pairs(X, T, Y, mu0, mu1, thr_sim, n_pairs):
    """
    Genera n_pairs coppie (ancora, positivo/negativo) + label similarità.
    """
    ite = mu1 - mu0
    N = X.shape[0]
    idx_a = np.random.randint(0, N, size=n_pairs)
    idx_b, labels = [], []
    for i in idx_a:
        if np.random.rand() < .5:  # coppia "simile"
            candidates = np.where(np.abs(ite - ite[i]) < thr_sim)[0]
            labels.append(1)
        else:  # coppia "dissimile"
            candidates = np.where(np.abs(ite - ite[i]) >= thr_sim)[0]
            labels.append(0)
        idx_b.append(np.random.choice(candidates))

    idx_b = np.array(idx_b)
    labels = np.array(labels, dtype=np.int64)

    return (
        X[idx_a], Y[idx_a], T[idx_a],
        X[idx_b], Y[idx_b], T[idx_b],
        labels
    )


# ----------------------- PyTorch Dataset --------------------------


class ContrastiveCausalDataset(Dataset):
    """
    Dataset che, ad ogni __getitem__, restituisce n_pairs=bs coppie.
    In pratica è equivalente a generare batch “on-the-fly”.
    """

    def __init__(self, X, T, Y, mu0, mu1, batch_size=256, percentile=20):
        self.X = X
        self.T = T
        self.Y = Y
        self.mu0 = mu0
        self.mu1 = mu1
        self.bs = batch_size
        self.thr = compute_tau_threshold(mu0, mu1, percentile)

    def __len__(self):
        # numero arbitrario: ogni epoca = 1 000 batch
        return 1_000

    def __getitem__(self, idx):
        (x1, y1, t1,
         x2, y2, t2,
         lab) = make_pairs(
            self.X, self.T, self.Y,
            self.mu0, self.mu1,
            thr_sim=self.thr,
            n_pairs=self.bs
        )

        # to torch.Tensor
        ten = lambda a, dtype=torch.float32: torch.from_numpy(a).type(dtype)
        return (
            ten(x1), ten(y1), ten(t1),
            ten(x2), ten(y2), ten(t2),
            torch.from_numpy(lab)  # int64
        )


# ----------------------- Siamese wrapper --------------------------


class SiameseBCAUSS(torch.nn.Module):
    """
    Wrapper che ingloba il modello BCAUSS base e aggiunge la loss contrastiva.
    """

    def __init__(self, base_model, margin=1.0, lambda_ctr=1.0):
        super().__init__()
        self.base = base_model
        self.margin = margin
        self.lambda_ctr = lambda_ctr

    # ---------- Loss contrastiva ----------
    def contrastive_loss(self, h1, h2, label):
        """
        label = 1 → coppia simile, 0 → dissimile
        """
        d = torch.norm(h1 - h2, dim=1)  # distanza euclidea
        pos = label * d.pow(2)
        neg = (1 - label) * (torch.clamp(self.margin - d, min=0).pow(2))
        return (pos + neg).mean()

    # ---------- Training/eval step ----------
    # ---------- Training / eval step ----------
    def step(self, batch):
        (x1, y1, t1,
         x2, y2, t2,
         lab) = batch

        # ------ forward di entrambe le metà ------
        mu1, h1 = self.base.mu_and_embedding(x1)
        mu2, h2 = self.base.mu_and_embedding(x2)

        # predizione factual per ciascun elemento
        y_pred1 = mu1[:, 0:1]*(1 - t1) + mu1[:, 1:2]*t1
        y_pred2 = mu2[:, 0:1]*(1 - t2) + mu2[:, 1:2]*t2

        # ------ base-loss identica a Keras --------
        base_loss = self.base.compute_loss(
            torch.cat([x1, x2]),        # features
            torch.cat([t1, t2]),        # trattamenti
            torch.cat([y1, y2])         # esiti
        )

        # ------ loss contrastiva ------------------
        ctr_loss  = self.contrastive_loss(h1, h2, lab.float())

        loss = base_loss + self.lambda_ctr * ctr_loss
        return loss, base_loss.detach(), ctr_loss.detach()


    # ---------- API simile a fit() ----------
    def fit(self, loader, epochs=100, lr=1e-4, device='cuda'):
        self.to(device)
        opt = torch.optim.Adam(self.parameters(), lr=lr)

        for ep in range(1, epochs + 1):
            self.train()
            running = []
            for batch in loader:
                batch = [b.to(device) for b in batch]
                opt.zero_grad()
                loss, base_l, ctr_l = self.step(batch)
                loss.backward()
                opt.step()
                running.append([loss.item(), base_l.item(), ctr_l.item()])

            mean_loss = np.mean(running, axis=0)
            print(f"Epoch {ep:3d}/{epochs}: "
                  f"loss={mean_loss[0]:.4f} "
                  f"(base={mean_loss[1]:.4f}, ctr={mean_loss[2]:.4f})")

    # ---------- Predizione ---------------

    @torch.no_grad()
    def predict(self, X, device='cuda'):
        self.eval()
        X = torch.from_numpy(X).float().to(device)
        mu, _ = self.base.mu_and_embedding(X)  # <- usa il metodo corretto
        return mu.cpu().numpy()


# --------------------------- MAIN ---------------------------------


def main():
    # 1. Carica IHDP
    (X_tr, T_tr, YF_tr, YCF_tr, mu0_tr, mu1_tr,
     X_te, T_te, YF_te, YCF_te, mu0_te, mu1_te) = \
        DataLoader.get_loader('IHDP').load()

    n_reps = 10
    bs = 256
    epochs = 5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = []

    for idx in range(n_reps):
        print(f"\n=== Replication {idx + 1}/{n_reps} ===")
        # --- Train split ----
        X_train = X_tr[:, :, idx].astype(np.float32)
        T_train = T_tr[:, idx].reshape(-1, 1).astype(np.float32)
        Y_train = YF_tr[:, idx].reshape(-1, 1).astype(np.float32)
        mu0_train = mu0_tr[:, idx].astype(np.float32)
        mu1_train = mu1_tr[:, idx].astype(np.float32)

        train_set = ContrastiveCausalDataset(
            X_train, T_train, Y_train,
            mu0_train, mu1_train,
            batch_size=bs, percentile=20
        )
        train_loader = TorchLoader(
            train_set,
            batch_size=1,  # lasciato a 1
            shuffle=True,
            collate_fn=lambda x: x[0],  # <-- toglie la dim. extra
            num_workers=0,
            pin_memory=True
        )

        input_dim = X_train.shape[1]  # numero di feature
        base_model = BCAUSS(input_dim=input_dim)

        # --- Siamese training ---
        siam = SiameseBCAUSS(base_model, margin=1.0, lambda_ctr=1.0)
        siam.fit(train_loader, epochs=epochs, lr=1e-4, device=device)

        # --- Valutazione ---
        X_test = X_te[:, :, idx].astype(np.float32)
        T_test = T_te[:, idx].reshape(-1, 1).astype(np.float32)
        Y_test = YF_te[:, idx].reshape(-1, 1).astype(np.float32)
        true_ite = mu1_te[:, idx] - mu0_te[:, idx]

        # Mu0 Mu1 predetti
        mu_pred = siam.predict(X_test, device=device)
        ite_pred = mu_pred[:, 1] - mu_pred[:, 0]

        eps = eps_ATE_diff(ite_pred.mean(), true_ite)
        pehe = PEHE_with_ite(ite_pred, true_ite, sqrt=True)
        print(f"-> eps_ATE: {eps:.4f}, PEHE: {pehe:.4f}")
        results.append((eps, pehe))

    eps_vals, pehe_vals = zip(*results)
    print("\n*** RISULTATI FINALI ***")
    print(f"Average eps_ATE: {np.mean(eps_vals):.4f} ± {np.std(eps_vals):.4f}")
    print(f"Average PEHE:    {np.mean(pehe_vals):.4f} ± {np.std(pehe_vals):.4f}")


if __name__ == "__main__":
    main()
