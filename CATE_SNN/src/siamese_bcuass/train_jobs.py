#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_jobs.py

Pipeline di training “solo per JOBS” basata su BCAUSS + SiameseBCAUSS,
senza parametri da linea di comando: tutte le impostazioni sono definite
in testa al file. Carica due file .npz distinti (train + test), gestisce
automaticamente chiavi alternative e terze dimensioni (repliche).

Come usare:
  1. Impostate i percorsi e gli iperparametri nella sezione “CONFIGURAZIONE” qui sotto.
     - NPZ_TRAIN_PATH: file .npz per il training
     - NPZ_TEST_PATH:  file .npz per il test
  2. Eseguite:
       python train_jobs.py
"""

import os
import random
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from src.models.bcauss import BCAUSS
from src.siamese_bcuass.siamese import SiameseBCAUSS
from src.contrastive import DynamicContrastiveCausalDS

# =============================================================================
# CONFIGURAZIONE: modificare questi valori a piacere prima di eseguire lo script
# =============================================================================

# Percorsi ai file .npz: uno per il train, uno per il test
NPZ_TRAIN_PATH = "../../jobs_DW_bin.new.10.train.npz"
NPZ_TEST_PATH  = "../../jobs_DW_bin.new.10.test.npz"

# Se i file .npz contengono più repliche (terza dimensione), selezionare quale usare:
REP_INDEX = 0  # indice della replica (0-based)

# Iperparametri per il warm‐up di BCAUSS (modificare se necessario)
WARMUP_EPOCHS     = 5
BATCH_SIZE_BASE   = 128
LR_BASE           = 1e-3
WEIGHT_DECAY_BASE = 1e-5

# Iperparametri per l’addestramento di SiameseBCAUSS
MARGIN               = 1.0
LAMBDA_CTR           = 1.0
LR_SIAMESE           = 1e-4
WEIGHT_DECAY_SIAMESE = 1e-5
BATCH_SIZE_SIAMESE   = 128
EPOCHS_SIAMESE       = 50
VAL_SPLIT            = 0.2
CLIP_NORM            = 1.0
USE_AMP              = False        # True per mixed‐precision, False altrimenti
UPDATE_ITE_FREQ      = 1

# Seed per riproducibilità
SEED = 42

# Directory in cui salvare gli output (verrà creata se non esiste)
OUTPUT_DIR = "./outputs"

# =============================================================================
# Fine configurazione
# =============================================================================


def load_from_npz(npz_path, rep_index):
    """
    Ritorna X, T, Y e opzionale mask_RCT da un file .npz.
    Gestisce chiavi alternative ('X','T','Y' o 'x','t','yf') e
    terze dimensioni (repliche). Se i dati hanno dimensione 3, prende
    slice [:, :, rep_index].

    Output:
      X: array float32 di shape (N, d)
      T: array float32 di shape (N, 1)
      Y: array float32 di shape (N, 1)
      mask_RCT: array bool di shape (N,) se presente, altrimenti None
    """
    data = np.load(npz_path)
    keys = set(data.files)

    # Identifico le chiavi di input
    if {"X", "T", "Y"}.issubset(keys):
        X_raw = data["X"]
        T_raw = data["T"]
        Y_raw = data["Y"]
    elif {"x", "t", "yf"}.issubset(keys):
        X_raw = data["x"]
        T_raw = data["t"]
        Y_raw = data["yf"]
    else:
        raise ValueError(
            f"{npz_path}: serve almeno 'X','T','Y' oppure 'x','t','yf'."
        )

    # Estrazione replica se necessario
    # X_raw può essere 2D (N,d) o 3D (N,d,R)
    if X_raw.ndim == 3:
        N, d, R = X_raw.shape
        if not (0 <= rep_index < R):
            raise ValueError(f"REP_INDEX={rep_index} non valido per R={R} repliche.")
        X = X_raw[:, :, rep_index].astype(np.float32)
    else:
        X = X_raw.astype(np.float32)
        N, d = X.shape

    # T_raw può essere 1D (N,) o 2D (N,R) o 2D (N,1)
    if T_raw.ndim == 2 and T_raw.shape[1] > 1:
        # se shape=(N,R), seleziono replica
        T = T_raw[:, rep_index].astype(np.float32).reshape(-1, 1)
    else:
        # se shape=(N,1) o (N,), ridimensiono
        T = T_raw.astype(np.float32).reshape(-1, 1)

    # Y_raw analogamente
    if Y_raw.ndim == 2 and Y_raw.shape[1] > 1:
        Y = Y_raw[:, rep_index].astype(np.float32).reshape(-1, 1)
    else:
        Y = Y_raw.astype(np.float32).reshape(-1, 1)

    # mask_RCT facoltativa
    mask_RCT = None
    if "mask_RCT" in keys:
        m = data["mask_RCT"]
        # se anche mask_RCT è 2D con repliche
        if m.ndim == 2:
            mask_RCT = m[:, rep_index].astype(bool)
        else:
            mask_RCT = m.astype(bool)

    return X, T, Y, mask_RCT


def main():
    # Imposto seed per riproducibilità
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)

    # Preparo cartella di output
    output_root = Path(OUTPUT_DIR)
    output_root.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------------------
    # 1) Caricamento dati da file .npz separati (train + test)
    # --------------------------------------------------------------------------------
    print(f"[INFO] Carico dati di TRAIN da '{NPZ_TRAIN_PATH}' …")
    X_train, T_train, Y_train, mask_RCT_train = load_from_npz(NPZ_TRAIN_PATH, REP_INDEX)
    print(f"[INFO] TRAIN: X_train.shape = {X_train.shape}, T_train.shape = {T_train.shape}, Y_train.shape = {Y_train.shape}")

    print(f"[INFO] Carico dati di TEST da '{NPZ_TEST_PATH}' …")
    X_test, T_test, Y_test, mask_RCT_test = load_from_npz(NPZ_TEST_PATH, REP_INDEX)
    print(f"[INFO]  TEST: X_test.shape  = {X_test.shape},  T_test.shape  = {T_test.shape},  Y_test.shape  = {Y_test.shape}")

    # Calcolo ATT_true combinando eventuali mask_RCT in train e test
    att_true = None
    if mask_RCT_train is not None or mask_RCT_test is not None:
        # Unisco train+test per calcolare ATT_true su tutto l'insieme RCT
        X_all   = np.vstack([X_train,   X_test])
        T_all   = np.vstack([T_train,   T_test])
        Y_all   = np.vstack([Y_train,   Y_test])
        if mask_RCT_train is None:
            mask_RCT_all = mask_RCT_test
        elif mask_RCT_test is None:
            mask_RCT_all = mask_RCT_train
        else:
            mask_RCT_all = np.concatenate([mask_RCT_train, mask_RCT_test])

        treated_rct_all = (T_all[mask_RCT_all].flatten() == 1)
        control_rct_all = (T_all[mask_RCT_all].flatten() == 0)
        Y_rct_all = Y_all[mask_RCT_all]
        att_true = float(
            Y_rct_all[treated_rct_all].mean() - Y_rct_all[control_rct_all].mean()
        )
        print(f"[INFO] ATT_true (RCT complessivo) = {att_true:.4f}")
    else:
        print("[WARNING] Nessuna maschera 'mask_RCT' in TRAIN o TEST: non calcolerò ATT_true.")

    # --------------------------------------------------------------------------------
    # 2) Standardizzazione delle covariate (fit su train, poi apply a test)
    # --------------------------------------------------------------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # --------------------------------------------------------------------------------
    # 3) Pre‐training del modello base (BCAUSS) su TRAIN
    # --------------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device disponibile: {device}")

    print("[INFO] Inizio warm‐up BCAUSS su TRAIN…")
    model_base = BCAUSS(input_dim=X_train.shape[1])
    model_base.to(device)

    if WARMUP_EPOCHS > 0:
        model_base.fit(
            X_train, T_train, Y_train,
            epochs=WARMUP_EPOCHS
        )
    else:
        print("[INFO] Skip warmup BCAUSS (WARMUP_EPOCHS=0).")

    # Calcolo mu0_hat e mu1_hat su TRAIN
    print("[INFO] Calcolo mu0_hat, mu1_hat su TRAIN…")
    with torch.no_grad():
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
        mu_pred_tensor, _ = model_base.mu_and_embedding(X_train_tensor)
        mu_pred = mu_pred_tensor.cpu().numpy()   # (n_train, 2)
        mu0_hat = mu_pred[:, 0].astype(np.float32)
        mu1_hat = mu_pred[:, 1].astype(np.float32)

    # --------------------------------------------------------------------------------
    # 4) Addestramento SiameseBCAUSS su TRAIN
    # --------------------------------------------------------------------------------
    print("[INFO] Inizio training di SiameseBCAUSS…")
    siamese_params = {
        "ds_class": DynamicContrastiveCausalDS,
        "margin": MARGIN,
        "lambda_ctr": LAMBDA_CTR,
        "batch_size": BATCH_SIZE_SIAMESE,
        "lr": LR_SIAMESE,
        "weight_decay": WEIGHT_DECAY_SIAMESE,
        "epochs": EPOCHS_SIAMESE,
        "clip_norm": CLIP_NORM,
        "use_amp": USE_AMP,
        "val_split": VAL_SPLIT,
        "update_ite_freq": UPDATE_ITE_FREQ,
        "warmup_epochs_base": 0,  # già fatto sopra
    }
    model_siamese = SiameseBCAUSS(base_model=model_base, **siamese_params)
    model_siamese.to(device)

    model_siamese.fit(X_train, T_train, Y_train, mu0_hat)

    # --------------------------------------------------------------------------------
    # 5) Valutazione su TEST (e calcolo ATT_pred + R_policy se mask_RCT_test esiste)
    # --------------------------------------------------------------------------------
    print("[INFO] Predizione ITE su TEST…")
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
        mu_test_tensor, _ = model_siamese.base.mu_and_embedding(X_test_tensor)
        mu_test = mu_test_tensor.cpu().numpy()   # (n_test, 2)

    pred_ite_test = mu_test[:, 1] - mu_test[:, 0]  # ITE stimata su TEST
    print(f"[INFO] ITE_pred (TEST): media={pred_ite_test.mean():.4f}, std={pred_ite_test.std():.4f}")

    eps_att, R_policy = float("nan"), float("nan")
    if mask_RCT_test is not None:
        # Identifico gli indici di test corrispondenti a mask_RCT_test
        treated_rct_test = (T_test[mask_RCT_test].flatten() == 1)
        control_rct_test = (T_test[mask_RCT_test].flatten() == 0)
        # ATT_pred: media su trattati RCT in TEST
        if treated_rct_test.sum() > 0:
            pred_ite_rct_test = pred_ite_test[mask_RCT_test]
            att_pred = float(pred_ite_rct_test[treated_rct_test].mean())
            eps_att = abs(att_true - att_pred) if att_true is not None else float("nan")
            print(f"[INFO] ATT_true = {att_true:.4f}, ATT_pred = {att_pred:.4f}, ε_ATT = {eps_att:.4f}")
        else:
            print("[WARNING] Nessun soggetto trattato nella porzione RCT di TEST: ATT_pred non calcolabile.")

        # R_policy
        pi_f = (pred_ite_test[mask_RCT_test] > 0).astype(int)
        mask_pi1 = (pi_f == 1)
        mask_pi0 = (pi_f == 0)

        mask_tr1 = np.logical_and(mask_pi1, treated_rct_test)
        mask_tr0 = np.logical_and(mask_pi0, control_rct_test)

        Y_rct_test = Y_test[mask_RCT_test].flatten()
        Ey1 = float(Y_rct_test[mask_tr1].mean()) if mask_tr1.sum() > 0 else 0.0
        Ey0 = float(Y_rct_test[mask_tr0].mean()) if mask_tr0.sum() > 0 else 0.0
        p1  = float(mask_pi1.mean())
        p0  = float(mask_pi0.mean())

        R_policy = 1.0 - (Ey1 * p1 + Ey0 * p0)
        print(f"[INFO] R_policy (su RCT di TEST) = {R_policy:.4f}")
    else:
        print("[WARNING] Nessuna maschera 'mask_RCT' in TEST: salto ATT_pred e R_policy.")

    # --------------------------------------------------------------------------------
    # 6) Salvataggio metriche su CSV
    # --------------------------------------------------------------------------------
    metrics_csv = output_root / "metrics.csv"
    with open(metrics_csv, "a", newline="") as f:
        writer = csv.writer(f)
        # Riga: [seed, ε_ATT, R_policy, ITE_mean, ITE_std]
        writer.writerow([
            SEED,
            f"{eps_att:.6f}",
            f"{R_policy:.6f}",
            f"{pred_ite_test.mean():.6f}",
            f"{pred_ite_test.std():.6f}"
        ])

    print(f"[INFO] Metriche salvate in: {metrics_csv}")
    print("[INFO] Training completato.")


if __name__ == "__main__":
    main()
