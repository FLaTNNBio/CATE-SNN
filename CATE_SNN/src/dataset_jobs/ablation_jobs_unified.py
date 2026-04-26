#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified ablation study for HERMES on JOBS
compatible with the real DynamicContrastiveCausalDS implementation.

Supported structural variants:
- BCAUSS_Backbone
- NoContrastive
- RandomPairs
- CovariatePairs
- StaticITEPairs
- DynamicITEPairs

Supported optimization / sensitivity variants:
- HERMES_Full
- NoWarmup
- Warmup5
- RefreshEvery5
- NoRefresh
- LowLambda
- HighLambda
- SmallMargin
- LargeMargin
- Threshold10
- Threshold30
- Pairs16
- Pairs64
"""

import os
import csv
import copy
import random
import logging
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset

from src.dataset_jobs.bcauss import BCAUSS
from src.dataset_jobs.contrastive import DynamicContrastiveCausalDS
from src.metrics import ATT, RPol


# ---------------------------------------------------------------------
# SEED / UTILS
# ---------------------------------------------------------------------
def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_csv_row(csv_path: Path, row) -> None:
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def mean_std_safe(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan, np.nan
    return float(np.nanmean(values)), float(np.nanstd(values))


# ---------------------------------------------------------------------
# LOSSES / TRAINING
# ---------------------------------------------------------------------
def contrastive_loss(h1: torch.Tensor, h2: torch.Tensor, labels: torch.Tensor, margin: float):
    """
    h1, h2: [batch_size, emb_dim]
    labels: [batch_size], 1 = similar, 0 = dissimilar
    """
    d = torch.norm(h1 - h2, p=2, dim=1)
    labels = labels.float().view(-1)

    loss_sim = d.pow(2)
    loss_dis = torch.clamp(margin - d, min=0.0).pow(2)

    return (labels * loss_sim + (1.0 - labels) * loss_dis).mean()


def train_epoch_combined(
    model,
    base_loader,
    contrastive_loader,
    optimizer,
    device,
    lambda_ctr,
    margin,
):
    """
    Alternates one supervised BCAUSS step and one contrastive step.
    Each batch from contrastive_loader is (x1, x2, labels).
    """
    model.train()
    total_loss = 0.0
    total_base_loss = 0.0
    total_ctr_loss = 0.0

    num_batches = max(len(base_loader), len(contrastive_loader))
    base_iter = iter(base_loader)
    ctr_iter = iter(contrastive_loader)

    steps = 0

    for _ in range(num_batches):
        # -------------------------------------------------
        # Step 1: supervised BCAUSS
        # -------------------------------------------------
        try:
            X_batch, T_batch, Y_batch = next(base_iter)
        except StopIteration:
            base_iter = iter(base_loader)
            X_batch, T_batch, Y_batch = next(base_iter)

        X_batch = X_batch.to(device)
        T_batch = T_batch.to(device)
        Y_batch = Y_batch.to(device)

        optimizer.zero_grad()
        base_loss = model.compute_loss(X_batch, T_batch, Y_batch)

        if torch.isfinite(base_loss):
            base_loss.backward()
            optimizer.step()
            total_loss += base_loss.item()
            total_base_loss += base_loss.item()
            steps += 1

        # -------------------------------------------------
        # Step 2: contrastive
        # -------------------------------------------------
        if lambda_ctr > 0.0:
            try:
                x1, x2, labels = next(ctr_iter)
            except StopIteration:
                ctr_iter = iter(contrastive_loader)
                x1, x2, labels = next(ctr_iter)

            x1 = x1.to(device)
            x2 = x2.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            h1 = model.embed(x1)
            h2 = model.embed(x2)

            ctr = contrastive_loss(h1, h2, labels, margin=margin)
            ctr_weighted = lambda_ctr * ctr

            if torch.isfinite(ctr_weighted):
                ctr_weighted.backward()
                optimizer.step()
                total_loss += ctr_weighted.item()
                total_ctr_loss += ctr_weighted.item()
                steps += 1

    if steps == 0:
        return {
            "loss": np.nan,
            "base_loss": np.nan,
            "ctr_loss": np.nan,
        }

    return {
        "loss": total_loss / steps,
        "base_loss": total_base_loss / max(1, len(base_loader)),
        "ctr_loss": total_ctr_loss / max(1, len(contrastive_loader)),
    }


def validate_contrastive_only(model, contrastive_loader, device, margin):
    model.eval()
    total_val = 0.0
    count = 0

    with torch.no_grad():
        for x1, x2, labels in contrastive_loader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            labels = labels.to(device)

            h1 = model.embed(x1)
            h2 = model.embed(x2)

            val_loss = contrastive_loss(h1, h2, labels, margin=margin)
            if torch.isfinite(val_loss):
                total_val += val_loss.item()
                count += 1

    return total_val / max(1, count)


def evaluate_jobs_metrics(model, Xte, Tte, YFte, exp_mask, y_scaler, device):
    """
    Evaluate on the experimental subset only, exactly like the JOBS pipeline.
    """
    model.eval()
    Xte_t = torch.from_numpy(Xte).float().to(device)

    with torch.no_grad():
        y0_pred_scaled, y1_pred_scaled, _ = model.mu_and_embedding(Xte_t)
        hat_y_scaled = np.concatenate(
            [y0_pred_scaled.cpu().numpy(), y1_pred_scaled.cpu().numpy()],
            axis=1
        )

    y0 = y_scaler.inverse_transform(hat_y_scaled[:, 0].reshape(-1, 1)).reshape(-1)
    y1 = y_scaler.inverse_transform(hat_y_scaled[:, 1].reshape(-1, 1)).reshape(-1)
    hat_y = np.stack([y0, y1], axis=1)

    T_exp = Tte[exp_mask].reshape(-1)
    Y_exp = YFte[exp_mask].reshape(-1)
    hat_y_exp = hat_y[exp_mask]

    eps_att = ATT(T_exp, Y_exp, hat_y_exp)
    rpol = RPol(T_exp, Y_exp, hat_y_exp)

    return {
        "eps_att": float(eps_att),
        "rpol": float(rpol),
    }


# ---------------------------------------------------------------------
# GENERIC PAIR DATASETS
# ---------------------------------------------------------------------
class SimpleContrastivePairs(Dataset):
    def __init__(self, X, pairs_idx, labels):
        self.X = X
        self.pairs_idx = pairs_idx
        self.labels = labels

    def __len__(self):
        return len(self.pairs_idx)

    def __getitem__(self, idx):
        i, j = self.pairs_idx[idx]
        return (
            self.X[i],
            self.X[j],
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


def build_pairs_indices(
    X_np,
    model,
    device,
    pair_mode="ite",
    perc=20,
    n_pairs=10000,
    seed=42,
):
    """
    Build pair indices and binary labels:
      - random: random labels
      - covariate: positive if covariate distance is in the lowest perc percentile
      - ite: positive if |tau_i - tau_j| is in the lowest perc percentile
    """
    rng = np.random.default_rng(seed)
    n = X_np.shape[0]

    all_i = rng.integers(0, n, size=n_pairs)
    all_j = rng.integers(0, n, size=n_pairs)

    mask = all_i != all_j
    all_i = all_i[mask]
    all_j = all_j[mask]

    if len(all_i) == 0:
        raise ValueError("No valid pairs generated.")

    pairs_idx = list(zip(all_i.tolist(), all_j.tolist()))

    if pair_mode == "random":
        labels = rng.integers(0, 2, size=len(pairs_idx)).tolist()
        return pairs_idx, labels

    if pair_mode == "covariate":
        dists = np.linalg.norm(X_np[all_i] - X_np[all_j], axis=1)
        thr = np.percentile(dists, perc)
        labels = (dists <= thr).astype(int).tolist()
        return pairs_idx, labels

    if pair_mode == "ite":
        model.eval()
        X_t = torch.from_numpy(X_np).float().to(device)
        with torch.no_grad():
            y0_pred, y1_pred, _ = model.mu_and_embedding(X_t)
            tau_hat = (y1_pred - y0_pred).detach().cpu().numpy().reshape(-1)

        tau_diff = np.abs(tau_hat[all_i] - tau_hat[all_j])
        thr = np.percentile(tau_diff, perc)
        labels = (tau_diff <= thr).astype(int).tolist()
        return pairs_idx, labels

    raise ValueError(f"Unknown pair_mode: {pair_mode}")


# ---------------------------------------------------------------------
# FIT FUNCTIONS
# ---------------------------------------------------------------------
def fit_bcauss_only(
    Xtr,
    Ttr,
    Ytr_scaled,
    input_dim,
    device,
    epochs,
    lr,
    reg_l2=0.01,
    verbose=False,
):
    model = BCAUSS(
        input_dim=input_dim,
        reg_l2=reg_l2,
        learning_rate=lr,
        optim="adam",
        epochs=epochs,
        verbose=verbose,
    ).to(device)

    model.fit(Xtr, Ttr, Ytr_scaled, epochs=epochs)
    return model


def fit_hermes_jobs(
    Xtr,
    Ttr,
    Ytr_scaled,
    mask_rct_train,
    input_dim,
    device,
    epochs,
    warmup_epochs,
    update_ite_freq,
    lr,
    batch_size,
    lambda_ctr,
    margin,
    perc,
    reg_l2=0.01,
    val_split_pairs=0.2,
    min_thr=0.1,
    max_thr=0.5,
    smooth=0.7,
    n_pairs=10000,
    verbose=False,
    pair_mode="ite",
    dynamic_pairs=True,
):
    """
    HERMES-like fit for JOBS:
      1) warm-up BCAUSS on all training samples
      2) contrastive training only on the RCT subset
    """
    model = BCAUSS(
        input_dim=input_dim,
        reg_l2=reg_l2,
        learning_rate=lr,
        optim="adam",
        epochs=epochs,
        verbose=verbose,
    ).to(device)

    # -------------------------
    # Warm-up on all training data
    # -------------------------
    if warmup_epochs > 0:
        if verbose:
            logging.info(f"Warm-up BCAUSS for {warmup_epochs} epochs")
        model.fit(Xtr, Ttr, Ytr_scaled, epochs=warmup_epochs)

    # -------------------------
    # RCT subset for contrastive training
    # -------------------------
    X_rct = Xtr[mask_rct_train]
    T_rct = Ttr[mask_rct_train]
    Y_rct = Ytr_scaled[mask_rct_train]

    if len(X_rct) < 2:
        logging.warning("Not enough RCT samples for contrastive training; returning warm-up model only.")
        return model

    # supervised loader on RCT subset
    base_rct_dataset = TensorDataset(
        torch.from_numpy(X_rct).float(),
        torch.from_numpy(T_rct).float(),
        torch.from_numpy(Y_rct).float(),
    )
    base_loader = DataLoader(base_rct_dataset, batch_size=batch_size, shuffle=True)

    # contrastive dataset
    X_rct_torch = torch.from_numpy(X_rct).float()

    if pair_mode == "ite" and dynamic_pairs:
        contrastive_ds = DynamicContrastiveCausalDS(
            X=X_rct_torch.to(device),
            T=torch.from_numpy(T_rct).float().to(device),
            Y=torch.from_numpy(Y_rct).float().to(device),
            base_model=model,
            n_pairs=n_pairs,
            perc=perc,
            min_thr=min_thr,
            max_thr=max_thr,
            smooth=smooth,
        )
        dynamic_dataset = True
    else:
        pairs_idx, labels = build_pairs_indices(
            X_np=X_rct,
            model=model,
            device=device,
            pair_mode=pair_mode,
            perc=perc,
            n_pairs=n_pairs,
            seed=42,
        )
        contrastive_ds = SimpleContrastivePairs(
            X=X_rct_torch,
            pairs_idx=pairs_idx,
            labels=labels,
        )
        dynamic_dataset = False

    if len(contrastive_ds) < 2:
        logging.warning("Not enough contrastive pairs generated; returning warm-up model only.")
        return model

    n_val = max(1, int(len(contrastive_ds) * val_split_pairs))
    n_train = len(contrastive_ds) - n_val
    if n_train <= 0:
        n_train = len(contrastive_ds) - 1
        n_val = 1

    train_ctr, val_ctr = random_split(contrastive_ds, [n_train, n_val])

    contrastive_loader = DataLoader(train_ctr, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ctr, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")

    for epoch in range(epochs):
        if (
            dynamic_dataset
            and update_ite_freq is not None
            and update_ite_freq > 0
            and epoch > 0
            and epoch % update_ite_freq == 0
        ):
            if verbose:
                logging.info(f"Refreshing contrastive pairs at epoch {epoch}")
            contrastive_ds.update_threshold()

        train_stats = train_epoch_combined(
            model=model,
            base_loader=base_loader,
            contrastive_loader=contrastive_loader,
            optimizer=optimizer,
            device=device,
            lambda_ctr=lambda_ctr,
            margin=margin,
        )

        val_loss = (
            validate_contrastive_only(
                model=model,
                contrastive_loader=val_loader,
                device=device,
                margin=margin,
            )
            if lambda_ctr > 0.0
            else train_stats["loss"]
        )

        if verbose:
            logging.info(
                f"[Epoch {epoch+1}/{epochs}] "
                f"train={train_stats['loss']:.4f} "
                f"base={train_stats['base_loss']:.4f} "
                f"ctr={train_stats['ctr_loss']:.4f} "
                f"val={val_loss:.4f}"
            )

        if np.isfinite(val_loss) and val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------
# ABLATION CONFIGS
# ---------------------------------------------------------------------
STRUCTURAL_ABLATIONS = [
    {
        "group": "structural",
        "name": "BCAUSS_Backbone",
        "use_siamese": False,
    },
    {
        "group": "structural",
        "name": "NoContrastive",
        "use_siamese": True,
        "lambda_ctr": 0.0,
        "pair_mode": "ite",
        "dynamic_pairs": False,
    },
    {
        "group": "structural",
        "name": "RandomPairs",
        "use_siamese": True,
        "pair_mode": "random",
        "dynamic_pairs": False,
    },
    {
        "group": "structural",
        "name": "CovariatePairs",
        "use_siamese": True,
        "pair_mode": "covariate",
        "dynamic_pairs": False,
    },
    {
        "group": "structural",
        "name": "StaticITEPairs",
        "use_siamese": True,
        "pair_mode": "ite",
        "dynamic_pairs": False,
    },
    {
        "group": "structural",
        "name": "DynamicITEPairs",
        "use_siamese": True,
        "pair_mode": "ite",
        "dynamic_pairs": True,
    },
]

OPTIMIZATION_ABLATIONS = [
    {
        "group": "optimization",
        "name": "HERMES_Full",
        "use_siamese": True,
        "pair_mode": "ite",
        "dynamic_pairs": True,
    },
    {
        "group": "optimization",
        "name": "NoWarmup",
        "use_siamese": True,
        "warmup_epochs": 0,
        "pair_mode": "ite",
        "dynamic_pairs": True,
    },
    {
        "group": "optimization",
        "name": "Warmup5",
        "use_siamese": True,
        "warmup_epochs": 5,
        "pair_mode": "ite",
        "dynamic_pairs": True,
    },
    {
        "group": "optimization",
        "name": "RefreshEvery5",
        "use_siamese": True,
        "update_ite_freq": 5,
        "pair_mode": "ite",
        "dynamic_pairs": True,
    },
    {
        "group": "optimization",
        "name": "NoRefresh",
        "use_siamese": True,
        "update_ite_freq": 10**9,
        "pair_mode": "ite",
        "dynamic_pairs": True,
    },
    {
        "group": "optimization",
        "name": "LowLambda",
        "use_siamese": True,
        "lambda_ctr": 0.1,
        "pair_mode": "ite",
        "dynamic_pairs": True,
    },
    {
        "group": "optimization",
        "name": "HighLambda",
        "use_siamese": True,
        "lambda_ctr": 2.0,
        "pair_mode": "ite",
        "dynamic_pairs": True,
    },
    {
        "group": "optimization",
        "name": "SmallMargin",
        "use_siamese": True,
        "margin": 0.2,
        "pair_mode": "ite",
        "dynamic_pairs": True,
    },
    {
        "group": "optimization",
        "name": "LargeMargin",
        "use_siamese": True,
        "margin": 1.0,
        "pair_mode": "ite",
        "dynamic_pairs": True,
    },
    {
        "group": "optimization",
        "name": "Threshold10",
        "use_siamese": True,
        "perc": 10,
        "pair_mode": "ite",
        "dynamic_pairs": True,
    },
    {
        "group": "optimization",
        "name": "Threshold30",
        "use_siamese": True,
        "perc": 30,
        "pair_mode": "ite",
        "dynamic_pairs": True,
    },
    {
        "group": "optimization",
        "name": "Pairs16",
        "use_siamese": True,
        "batch_size": 16,
        "pair_mode": "ite",
        "dynamic_pairs": True,
    },
    {
        "group": "optimization",
        "name": "Pairs64",
        "use_siamese": True,
        "batch_size": 64,
        "pair_mode": "ite",
        "dynamic_pairs": True,
    },
]


# ---------------------------------------------------------------------
# EVALUATION LOOP
# ---------------------------------------------------------------------
def run_single_strategy(
    strategy_cfg,
    args,
    X_tr_all,
    T_tr_all,
    YF_tr_all,
    E_tr_all,
    X_te_all,
    T_te_all,
    YF_te_all,
    E_te_all,
    raw_csv_path,
):
    eps_att_vals = []
    rpol_vals = []

    n_reps_total = X_tr_all.shape[2]
    n_reps = min(args.n_reps, n_reps_total)

    for rep in range(n_reps):
        set_seed(args.seed + rep)
        logging.info(f"[{strategy_cfg['name']}] Replica {rep + 1}/{n_reps}")

        # -----------------------------------------
        # Replica data
        # -----------------------------------------
        Xtr = X_tr_all[:, :, rep].astype(np.float32)
        Ttr = T_tr_all[:, rep].astype(np.float32).reshape(-1, 1)
        Ytr = YF_tr_all[:, rep].astype(np.float32).reshape(-1, 1)

        Xte = X_te_all[:, :, rep].astype(np.float32)
        Tte = T_te_all[:, rep].astype(np.float32).reshape(-1, 1)
        Yte = YF_te_all[:, rep].astype(np.float32).reshape(-1, 1)

        Etr = E_tr_all[:, rep].astype(bool)
        Ete = E_te_all[:, rep].astype(bool)

        # -----------------------------------------
        # Scaling
        # -----------------------------------------
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

        Xtr_scaled = x_scaler.fit_transform(Xtr).astype(np.float32)
        Xte_scaled = x_scaler.transform(Xte).astype(np.float32)
        Ytr_scaled = y_scaler.fit_transform(Ytr).astype(np.float32)

        input_dim = Xtr_scaled.shape[1]

        # -----------------------------------------
        # Resolve strategy hyperparameters
        # -----------------------------------------
        use_siamese = strategy_cfg.get("use_siamese", True)
        warmup_epochs = strategy_cfg.get("warmup_epochs", args.warmup_epochs)
        update_ite_freq = strategy_cfg.get("update_ite_freq", args.update_ite_freq)
        lambda_ctr = strategy_cfg.get("lambda_ctr", args.lambda_ctr)
        margin = strategy_cfg.get("margin", args.margin)
        perc = strategy_cfg.get("perc", args.perc)
        batch_size = strategy_cfg.get("batch_size", args.batch_size)
        pair_mode = strategy_cfg.get("pair_mode", "ite")
        dynamic_pairs = strategy_cfg.get("dynamic_pairs", True)

        # -----------------------------------------
        # Train
        # -----------------------------------------
        if not use_siamese:
            model = fit_bcauss_only(
                Xtr=Xtr_scaled,
                Ttr=Ttr,
                Ytr_scaled=Ytr_scaled,
                input_dim=input_dim,
                device=args.device,
                epochs=args.epochs,
                lr=args.lr,
                reg_l2=args.reg_l2,
                verbose=args.verbose,
            )
            ctr_flag = "No"
            pair_criterion = "--"
            dyn_flag = "No"
            warmup_str = "--"
            refresh_str = "--"
            lambda_str = "0.0"
            margin_str = "--"
            perc_str = "--"
            batch_str = "--"

        else:
            model = fit_hermes_jobs(
                Xtr=Xtr_scaled,
                Ttr=Ttr,
                Ytr_scaled=Ytr_scaled,
                mask_rct_train=Etr,
                input_dim=input_dim,
                device=args.device,
                epochs=args.epochs,
                warmup_epochs=warmup_epochs,
                update_ite_freq=update_ite_freq,
                lr=args.lr,
                batch_size=batch_size,
                lambda_ctr=lambda_ctr,
                margin=margin,
                perc=perc,
                reg_l2=args.reg_l2,
                val_split_pairs=args.val_split_pairs,
                min_thr=args.min_thr,
                max_thr=args.max_thr,
                smooth=args.smooth_thr,
                n_pairs=args.n_pairs,
                verbose=args.verbose,
                pair_mode=pair_mode,
                dynamic_pairs=dynamic_pairs,
            )

            ctr_flag = "Yes" if lambda_ctr > 0 else "No"

            if pair_mode == "random":
                pair_criterion = "Random"
            elif pair_mode == "covariate":
                pair_criterion = "Covariate similarity"
            else:
                pair_criterion = "ITE similarity"

            dyn_flag = "Yes" if dynamic_pairs else "No"
            warmup_str = str(warmup_epochs)
            refresh_str = f"every {update_ite_freq}" if dynamic_pairs else "fixed"
            lambda_str = str(lambda_ctr)
            margin_str = str(margin)
            perc_str = str(perc)
            batch_str = str(batch_size)

        # -----------------------------------------
        # Evaluate
        # -----------------------------------------
        metrics = evaluate_jobs_metrics(
            model=model,
            Xte=Xte_scaled,
            Tte=Tte,
            YFte=Yte,
            exp_mask=Ete,
            y_scaler=y_scaler,
            device=args.device,
        )

        eps_att = metrics["eps_att"]
        rpol = metrics["rpol"]

        eps_att_vals.append(eps_att)
        rpol_vals.append(rpol)

        save_csv_row(
            raw_csv_path,
            [
                strategy_cfg["group"],
                strategy_cfg["name"],
                rep + 1,
                ctr_flag,
                pair_criterion,
                dyn_flag,
                warmup_str,
                refresh_str,
                lambda_str,
                margin_str,
                perc_str,
                batch_str,
                f"{eps_att:.6f}",
                f"{rpol:.6f}",
            ],
        )

    eps_mean, eps_std = mean_std_safe(eps_att_vals)
    rpol_mean, rpol_std = mean_std_safe(rpol_vals)

    return {
        "eps_att_mean": eps_mean,
        "eps_att_std": eps_std,
        "rpol_mean": rpol_mean,
        "rpol_std": rpol_std,
    }


def summarize_group(results_dict, group_name, out_path: Path):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "group",
            "strategy",
            "eps_att_mean",
            "eps_att_std",
            "rpol_mean",
            "rpol_std",
        ])

        for strategy_name, metrics in results_dict.items():
            writer.writerow([
                group_name,
                strategy_name,
                f"{metrics['eps_att_mean']:.6f}",
                f"{metrics['eps_att_std']:.6f}",
                f"{metrics['rpol_mean']:.6f}",
                f"{metrics['rpol_std']:.6f}",
            ])


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Unified JOBS ablation for HERMES")

    parser.add_argument("--train_path", type=str, required=True, help="Path to JOBS train npz")
    parser.add_argument("--test_path", type=str, required=True, help="Path to JOBS test npz")
    parser.add_argument("--out_dir", type=str, default="jobs_ablation_outputs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_reps", type=int, default=10)

    # training defaults
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--warmup_epochs", type=int, default=20)
    parser.add_argument("--update_ite_freq", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1.05e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lambda_ctr", type=float, default=0.145)
    parser.add_argument("--margin", type=float, default=0.561)
    parser.add_argument("--perc", type=int, default=17)
    parser.add_argument("--reg_l2", type=float, default=0.01)

    # pair dataset controls
    parser.add_argument("--val_split_pairs", type=float, default=0.2)
    parser.add_argument("--min_thr", type=float, default=0.1)
    parser.add_argument("--max_thr", type=float, default=0.5)
    parser.add_argument("--smooth_thr", type=float, default=0.7)
    parser.add_argument("--n_pairs", type=int, default=10000)

    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s"
    )

    args.device = torch.device(args.device)
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    raw_csv = out_dir / "jobs_ablation_raw.csv"
    structural_csv = out_dir / "jobs_ablation_structural_summary.csv"
    optimization_csv = out_dir / "jobs_ablation_optimization_summary.csv"

    with open(raw_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "group",
            "strategy",
            "replica",
            "Ctr",
            "PairCriterion",
            "Dynamic",
            "Warmup",
            "Refresh",
            "Lambda",
            "Margin",
            "ThresholdPerc",
            "PairsPerStep",
            "eps_ATT",
            "RPol",
        ])

    # -----------------------------------------
    # Load data
    # -----------------------------------------
    train_data = np.load(args.train_path)
    test_data = np.load(args.test_path)

    X_tr_all = train_data["x"]
    T_tr_all = train_data["t"]
    YF_tr_all = train_data["yf"]
    E_tr_all = train_data["e"]

    X_te_all = test_data["x"]
    T_te_all = test_data["t"]
    YF_te_all = test_data["yf"]
    E_te_all = test_data["e"]

    logging.info(f"Train x shape: {X_tr_all.shape}")
    logging.info(f"Test x shape:  {X_te_all.shape}")

    structural_results = {}
    optimization_results = {}

    # -----------------------------------------
    # Structural ablation
    # -----------------------------------------
    for strategy in STRUCTURAL_ABLATIONS:
        logging.info("=" * 80)
        logging.info(f"Starting structural ablation: {strategy['name']}")
        logging.info("=" * 80)

        metrics = run_single_strategy(
            strategy_cfg=strategy,
            args=args,
            X_tr_all=X_tr_all,
            T_tr_all=T_tr_all,
            YF_tr_all=YF_tr_all,
            E_tr_all=E_tr_all,
            X_te_all=X_te_all,
            T_te_all=T_te_all,
            YF_te_all=YF_te_all,
            E_te_all=E_te_all,
            raw_csv_path=raw_csv,
        )

        structural_results[strategy["name"]] = metrics
        logging.info(
            f"[{strategy['name']}] "
            f"eps_ATT={metrics['eps_att_mean']:.6f} ± {metrics['eps_att_std']:.6f} | "
            f"RPol={metrics['rpol_mean']:.6f} ± {metrics['rpol_std']:.6f}"
        )

    # -----------------------------------------
    # Optimization ablation
    # -----------------------------------------
    for strategy in OPTIMIZATION_ABLATIONS:
        logging.info("=" * 80)
        logging.info(f"Starting optimization ablation: {strategy['name']}")
        logging.info("=" * 80)

        metrics = run_single_strategy(
            strategy_cfg=strategy,
            args=args,
            X_tr_all=X_tr_all,
            T_tr_all=T_tr_all,
            YF_tr_all=YF_tr_all,
            E_tr_all=E_tr_all,
            X_te_all=X_te_all,
            T_te_all=T_te_all,
            YF_te_all=YF_te_all,
            E_te_all=E_te_all,
            raw_csv_path=raw_csv,
        )

        optimization_results[strategy["name"]] = metrics
        logging.info(
            f"[{strategy['name']}] "
            f"eps_ATT={metrics['eps_att_mean']:.6f} ± {metrics['eps_att_std']:.6f} | "
            f"RPol={metrics['rpol_mean']:.6f} ± {metrics['rpol_std']:.6f}"
        )

    summarize_group(structural_results, "structural", structural_csv)
    summarize_group(optimization_results, "optimization", optimization_csv)

    logging.info(f"Saved raw results to: {raw_csv}")
    logging.info(f"Saved structural summary to: {structural_csv}")
    logging.info(f"Saved optimization summary to: {optimization_csv}")


if __name__ == "__main__":
    main()