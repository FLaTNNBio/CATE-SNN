#!/usr/bin/env python3
"""
train_bcauss_reps.py  –  addestra BCAUSS su più repliche IHDP e
salva i pesi in           saved_weights_reps/bcauss_weights_rep_0001.pth  …
                          (zero-padding a 4 cifre, partendo da 1)
oltre a registrare PEHE e ATE_error in saved_weights_reps/metrics.csv
"""

import os
from pathlib import Path
import csv
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from tqdm import trange        # progress bar

from src.data_loader import DataLoader
from src.metrics import PEHE_with_ite, eps_ATE_diff
from src.models.bcauss import BCAUSS

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURAZIONE                                                              │
# ──────────────────────────────────────────────────────────────────────────────
N_REPS        = 1000
OUTPUT_DIR    = Path("../../saved_weights_reps")
WEIGHT_PATTERN = OUTPUT_DIR / "bcauss_weights_rep_{rep:04d}.pth"   # 0001, 0002…

EPOCHS        = 500
LR            = 1e-5
NEURONS       = 200
L2            = 0.01
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED          = 42   # per riproducibilità
# ──────────────────────────────────────────────────────────────────────────────

torch.manual_seed(SEED); np.random.seed(SEED)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
metrics_path = OUTPUT_DIR / "metrics.csv"

# inizializza CSV metriche
with open(metrics_path, "w", newline="") as f:
    csv.writer(f).writerow(["replica", "PEHE", "ATE_error"])


def train_one_rep(rep_idx: int,
                  X_all, W_all, Y_obs_all, mu0_all, mu1_all):
    """Addestra BCAUSS su una singola replica e salva pesi + metriche."""
    # --- estrai slice replica -------------------------------------------------
    X_rep   = X_all[:, :, rep_idx-1].astype(np.float32)  # (n, d)
    W_rep   = W_all[:, rep_idx-1].astype(np.float32)
    Y_rep   = Y_obs_all[:, rep_idx-1].astype(np.float32)
    mu0_rep = mu0_all[:, rep_idx-1].astype(np.float32)
    mu1_rep = mu1_all[:, rep_idx-1].astype(np.float32)

    # --- standardizza Y ------------------------------------------------------
    y_scaler = StandardScaler().fit(Y_rep.reshape(-1, 1))
    Y_scaled = y_scaler.transform(Y_rep.reshape(-1, 1)).astype(np.float32)

    # --- tensori su DEVICE ---------------------------------------------------
    X_t = torch.from_numpy(X_rep).to(DEVICE)
    W_t = torch.from_numpy(W_rep.reshape(-1, 1)).to(DEVICE)
    Y_t = torch.from_numpy(Y_scaled).to(DEVICE)

    # --- modello -------------------------------------------------------------
    model = BCAUSS(
        input_dim=X_rep.shape[1],
        epochs=EPOCHS,
        learning_rate=LR,
        neurons_per_layer=NEURONS,
        act_fn="relu",
        reg_l2=L2,
        verbose=False,
        scale_preds=False,
    ).to(DEVICE)
    model.y_scaler = y_scaler         # per inverse_transform interno
    model.fit(X_t, W_t, Y_t)

    # --- predici ITE ---------------------------------------------------------
    model.eval()
    with torch.no_grad():
        _, y0_raw, y1_raw, _ = model(X_t)
    y0 = y_scaler.inverse_transform(y0_raw.cpu().numpy())
    y1 = y_scaler.inverse_transform(y1_raw.cpu().numpy())
    ite_pred = (y1 - y0).flatten()

    # --- metriche ------------------------------------------------------------
    ite_true = (mu1_rep - mu0_rep)
    pehe     = PEHE_with_ite(ite_true, ite_pred, sqrt=True)
    ate_err  = eps_ATE_diff(ite_true, ite_pred)

    # --- salva pesi ----------------------------------------------------------
    weight_path = WEIGHT_PATTERN.with_name(
        WEIGHT_PATTERN.name.format(rep=rep_idx)
    )
    torch.save(model.state_dict(), weight_path)

    # --- log CSV -------------------------------------------------------------
    with open(metrics_path, "a", newline="") as f:
        csv.writer(f).writerow([rep_idx, pehe, ate_err])

    print(f"rep {rep_idx:04d} | PEHE={pehe:.4f} | ATE_err={ate_err:.4f} | saved → {weight_path}")


def main():
    print(f"Using device: {DEVICE}")
    loader = DataLoader.get_loader("IHDP")
    X_all, W_all, YF_all, _, mu0_all, mu1_all, *_ = loader.load()

    for rep in trange(1, N_REPS + 1, desc="Replicas"):
        train_one_rep(rep, X_all, W_all, YF_all, mu0_all, mu1_all)


if __name__ == "__main__":
    main()
