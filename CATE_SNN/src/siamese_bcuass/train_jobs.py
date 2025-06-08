#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_jobs.py

Pipeline di training “solo per JOBS” basata su BCAUSS + SiameseBCAUSS.
Tutto è parametrizzato in testa al file, senza argomenti da linea di comando.
Carica due file .npz distinti (train + test), gestisce automaticamente
chiavi alternative e terze dimensioni (repliche).
"""

import os
import random
import csv
from pathlib import Path
import logging

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from src.models.bcauss import BCAUSS
from src.contrastive import DynamicContrastiveCausalDS
from src.siamese_bcuass.siamese import SiameseBCAUSS

# Configura il logging di base
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ============================= CONFIGURAZIONE ==============================
# Percorsi ai file .npz (train + test). Modifica qui secondo la tua struttura.
NPZ_TRAIN_PATH = "../data/jobs_DW_bin.new.10.train.npz"
NPZ_TEST_PATH  = "../data/jobs_DW_bin.new.10.test.npz"

# Se i .npz contengono più repliche (terza dimensione), seleziona quale utilizzare:
REP_INDEX = 0  # indice della replica (0-based)

# Iperparametri per il warm‐up di BCAUSS (base model)
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
VAL_SPLIT_SIAMESE    = 0.2
CLIP_NORM            = 1.0
USE_AMP              = False
UPDATE_ITE_FREQ      = 1
SIAMESE_PATIENCE     = 20

# Seed per riproducibilità
SEED = 42

# Directory di output (creata automaticamente se non esiste)
OUTPUT_DIR = "./outputs_jobs"
# ===========================================================================


def set_seed(seed: int):
    """Imposta i semi per PyTorch, NumPy, Python random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_from_npz(npz_path: str, rep_index: int = 0):
    """
    Carica X, T, Y e (opzionale) mask_RCT da un file .npz.
    Gestione di chiavi alternative: {"X","T","Y"} o {"x","t","yf"}.
    Gestisce anche eventuale terza dimensione (repliche).
    """
    data = np.load(npz_path)
    keys = set(data.files)

    if {"X", "T", "Y"}.issubset(keys):
        X_raw, T_raw, Y_raw = data["X"], data["T"], data["Y"]
    elif {"x", "t", "yf"}.issubset(keys):
        X_raw, T_raw, Y_raw = data["x"], data["t"], data["yf"]
    else:
        raise ValueError(f"{npz_path}: serve almeno 'X','T','Y' oppure 'x','t','yf'.")

    def extract_rep(val_raw, name):
        if val_raw.ndim == 3 and name == "X":
            # X: (N, d, R)
            _, _, R_val = val_raw.shape
            if not (0 <= rep_index < R_val):
                raise ValueError(f"REP_INDEX={rep_index} non valido per {name} con R={R_val} repliche.")
            return val_raw[:, :, rep_index].astype(np.float32)
        elif val_raw.ndim == 2 and val_raw.shape[1] > 1 and name != "X":
            # T, Y, mask: (N, R)
            _, R_val = val_raw.shape
            if not (0 <= rep_index < R_val):
                raise ValueError(f"REP_INDEX={rep_index} non valido per {name} con R={R_val} repliche.")
            return val_raw[:, rep_index]
        else:
            # Se 2D con una colonna, o 1D, faccio squeeze
            return val_raw.squeeze()

    X = extract_rep(X_raw, "X").astype(np.float32)
    T = extract_rep(T_raw, "T").astype(np.float32)
    Y = extract_rep(Y_raw, "Y").astype(np.float32)

    mask_RCT = None
    if "mask_RCT" in keys:
        mask_raw = data["mask_RCT"]
        mask_RCT = extract_rep(mask_raw, "mask_RCT").astype(bool)

    return X, T, Y, mask_RCT


def compute_true_att(T_train, Y_train, mask_train, T_test, Y_test, mask_test):
    """
    Calcola l’ATT “vero” unendo eventuali RCT da train e test.
    """
    if mask_train is None and mask_test is None:
        return None

    T_list, Y_list = [], []
    if mask_train is not None and np.any(mask_train):
        T_list.append(T_train[mask_train])
        Y_list.append(Y_train[mask_train])
    if mask_test is not None and np.any(mask_test):
        T_list.append(T_test[mask_test])
        Y_list.append(Y_test[mask_test])

    if not T_list:
        return None

    T_rct = np.concatenate(T_list)
    Y_rct = np.concatenate(Y_list)

    if len(T_rct) == 0:
        return None

    Y_treated = Y_rct[T_rct == 1]
    Y_control = Y_rct[T_rct == 0]

    if len(Y_treated) == 0 or len(Y_control) == 0:
        logging.warning("Non ci sono unità trattate o di controllo nel campione RCT per calcolare ATT_true.")
        return None

    return Y_treated.mean() - Y_control.mean()


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Usando device: {device}")

    # Preparo cartella di output
    output_root = Path(OUTPUT_DIR)
    output_root.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------------
    # 1) Caricamento dati da file .npz per JOBS
    # --------------------------------------------------------------------------
    logging.info(f"Carico dati di TRAIN da '{NPZ_TRAIN_PATH}' …")
    X_train, T_train, Y_train, mask_RCT_train = load_from_npz(NPZ_TRAIN_PATH, REP_INDEX)
    logging.info(f"TRAIN: X={X_train.shape}, T={T_train.shape}, Y={Y_train.shape}, mask_RCT={mask_RCT_train.shape if mask_RCT_train is not None else None}")

    logging.info(f"Carico dati di TEST da '{NPZ_TEST_PATH}' …")
    X_test, T_test, Y_test, mask_RCT_test = load_from_npz(NPZ_TEST_PATH, REP_INDEX)
    logging.info(f"TEST:  X={X_test.shape},  T={T_test.shape},  Y={Y_test.shape}, mask_RCT={mask_RCT_test.shape if mask_RCT_test is not None else None}")

    # Calcolo ATT_true (se mask_RCT esiste)
    att_true = compute_true_att(T_train, Y_train, mask_RCT_train, T_test, Y_test, mask_RCT_test)
    if att_true is not None:
        logging.info(f"ATT_true (RCT): {att_true:.6f}")
    else:
        logging.info("ATT_true non disponibile (nessuna mask_RCT valida).")

    # --------------------------------------------------------------------------
    # 2) Standardizzazione delle covariate
    # --------------------------------------------------------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # --------------------------------------------------------------------------
    # 3) Warm‐up: addestramento del modello base BCAUSS per stimare μ₀ e μ₁
    # --------------------------------------------------------------------------
    logging.info("Warm‐up: addestro BCAUSS base...")
    base_model_params_for_warmup = {
        'input_dim': X_train.shape[1],
        'epochs': WARMUP_EPOCHS,
        'learning_rate': LR_BASE,
        'reg_l2': WEIGHT_DECAY_BASE,
        'bs_ratio': 0.05,      # Esempio di bs_ratio; regola se necessario
        'val_split': 0.2,
        'verbose': True,
        'scale_preds': True
    }
    base_model = BCAUSS(**base_model_params_for_warmup)
    base_model.to(device)

    if WARMUP_EPOCHS > 0:
        base_model.fit(X_train, T_train, Y_train.reshape(-1, 1))
        logging.info("Fine warm‐up BCAUSS.")
    else:
        logging.info("Warmup BCAUSS saltato (WARMUP_EPOCHS=0).")
        # Se non facciamo warmup, assicuriamoci di avere lo y_scaler fittato
        if base_model.params.get('scale_preds', False) and base_model.y_scaler is None:
            temp_scaler = StandardScaler()
            base_model.y_scaler = temp_scaler.fit(Y_train.reshape(-1, 1))

    # --------------------------------------------------------------------------
    # 4) Calcolo predizioni μ₀_hat e μ₁_hat su train e test
    # --------------------------------------------------------------------------
    with torch.no_grad():
        # Su TRAIN
        X_tr_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
        mu_tr_tensor, _ = base_model.mu_and_embedding(X_tr_tensor)
        mu_pred_tr = mu_tr_tensor.cpu().numpy().astype(np.float32)  # shape (n_train, 2)
        mu0_hat_train = mu_pred_tr[:, 0]
        mu1_hat_train = mu_pred_tr[:, 1]

        # Su TEST
        X_te_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
        mu_te_tensor, _ = base_model.mu_and_embedding(X_te_tensor)
        mu_pred_te = mu_te_tensor.cpu().numpy().astype(np.float32)  # shape (n_test, 2)
        mu0_hat_test = mu_pred_te[:, 0]
        mu1_hat_test = mu_pred_te[:, 1]

    # --------------------------------------------------------------------------
    # 5) Creazione del DataSet contrastivo per SiameseBCAUSS
    # --------------------------------------------------------------------------
    X_all   = np.vstack([X_train, X_test])                   # (n_train + n_test, d)
    T_all   = np.concatenate([T_train, T_test])               # (n_train + n_test,)
    Y_all   = np.concatenate([Y_train, Y_test])               # (n_train + n_test,)
    mu0_all = np.concatenate([mu0_hat_train, mu0_hat_test])   # (n_tot,)
    mu1_all = np.concatenate([mu1_hat_train, mu1_hat_test])   # (n_tot,)

    ds = DynamicContrastiveCausalDS(
        X_all,
        T_all,
        Y_all.reshape(-1, 1),
        mu0_all,
        mu1_all,
        bs=BATCH_SIZE_SIAMESE
    )

    # --------------------------------------------------------------------------
    # 6) Addestramento di SiameseBCAUSS
    # --------------------------------------------------------------------------
    logging.info("Inizio training di SiameseBCAUSS…")
    siamese_train_params = {
        "ds_class": DynamicContrastiveCausalDS,
        "margin": MARGIN,
        "lambda_ctr": LAMBDA_CTR,
        "batch_size": BATCH_SIZE_SIAMESE,
        "lr": LR_SIAMESE,
        "epochs": EPOCHS_SIAMESE,
        "clip_norm": CLIP_NORM,
        "use_amp": USE_AMP,
        "val_split": VAL_SPLIT_SIAMESE,
        "patience": SIAMESE_PATIENCE,
        "update_ite_freq": UPDATE_ITE_FREQ,
        "warmup_epochs_base": 0,
        "verbose": True
    }
    model_siamese = SiameseBCAUSS(base_model=base_model, **siamese_train_params)

    best_siamese_model_path = output_root / "best_siamese_bcauss_jobs.pth"
    model_siamese.fit(X_train, T_train, Y_train.reshape(-1, 1), best_model_path=best_siamese_model_path)
    logging.info("Fine training SiameseBCAUSS.")

    # --------------------------------------------------------------------------
    # 7) Valutazione su TEST (calcolo ITE_pred, ATT_pred e R_policy se applicabile)
    # --------------------------------------------------------------------------
    logging.info("Predizione ITE su TEST …")
    with torch.no_grad():
        pred_ite_test = model_siamese.predict_ite(X_test)  # forma (n_test,)

    logging.info(f"ITE_pred (TEST): media={pred_ite_test.mean():.4f}, std={pred_ite_test.std():.4f}")

    att_pred_val = float("nan")
    r_policy_val = float("nan")

    if mask_RCT_test is not None and np.any(mask_RCT_test):
        idx_rct_test = np.where(mask_RCT_test)[0]
        if len(idx_rct_test) > 0:
            att_pred_val = pred_ite_test[idx_rct_test].mean()
            n_rct = len(idx_rct_test)
            n_treat_policy = int(n_rct / 2)
            if n_treat_policy > 0 and n_rct > 0:
                # Ordino le RCT per ITE discendente
                sorted_rct_indices_by_ite = idx_rct_test[np.argsort(pred_ite_test[idx_rct_test])[::-1]]
                idx_selected_for_treatment_policy = sorted_rct_indices_by_ite[:n_treat_policy]

                # Payoff policy: outcome medio delle unità trattate tra quelle selezionate
                selected_and_treated_mask = T_test[idx_selected_for_treatment_policy] == 1
                if np.any(selected_and_treated_mask):
                    pay_policy = Y_test[idx_selected_for_treatment_policy][selected_and_treated_mask].mean()
                else:
                    pay_policy = float('nan')

                # Payoff random: outcome medio di n_treat_policy unità trattate scelte casualmente tra le RCT
                rct_treated_indices_test = idx_rct_test[T_test[idx_rct_test] == 1]
                if len(rct_treated_indices_test) >= n_treat_policy:
                    random.seed(SEED)
                    sel_random_treated = np.random.choice(rct_treated_indices_test, size=n_treat_policy, replace=False)
                    pay_random = Y_test[sel_random_treated].mean()
                elif len(rct_treated_indices_test) > 0:
                    pay_random = Y_test[rct_treated_indices_test].mean()
                else:
                    pay_random = float('nan')

                if not np.isnan(pay_policy) and not np.isnan(pay_random):
                    r_policy_val = pay_policy - pay_random
                else:
                    r_policy_val = float('nan')

                logging.info(f"ATT_pred (su RCT test): {att_pred_val:.6f}")
                logging.info(f"R_policy (stimata, su RCT test): {r_policy_val:.6f}")
            else:
                logging.info("Non abbastanza unità RCT o per la policy per calcolare R_policy.")
        else:
            logging.info("Nessuna unità RCT nel test set (mask_RCT_test è tutto False). Salto ATT_pred e R_policy.")
    else:
        logging.info("mask_RCT_test non disponibile o vuota: salto calcolo ATT_pred e R_policy.")

    # --------------------------------------------------------------------------
    # 8) Salvataggio metriche su CSV
    # --------------------------------------------------------------------------
    metrics_csv = output_root / "metrics_jobs.csv"
    header = ["seed", "ATT_true", "ATT_pred_rct", "R_policy_rct", "ITE_mean_test", "ITE_std_test"]
    file_exists = metrics_csv.exists()

    with open(metrics_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([
            SEED,
            f"{att_true:.6f}" if att_true is not None and not np.isnan(att_true) else "nan",
            f"{att_pred_val:.6f}" if not np.isnan(att_pred_val) else "nan",
            f"{r_policy_val:.6f}" if not np.isnan(r_policy_val) else "nan",
            f"{pred_ite_test.mean():.6f}" if pred_ite_test is not None and not np.isnan(pred_ite_test.mean()) else "nan",
            f"{pred_ite_test.std():.6f}" if pred_ite_test is not None and not np.isnan(pred_ite_test.std()) else "nan"
        ])

    logging.info(f"Metriche salvate in: {metrics_csv}")
    logging.info("Training completato.")


if __name__ == "__main__":
    main()
