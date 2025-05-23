#!/usr/bin/env python3
"""Hyper‑parameter search for Siamese BCAUSS on IHDP.

Correzioni:
- Aggiunto `weight_decay` tra gli HP campionati e loggati
- Sistemato `save_metrics()` e le intestazioni CSV
- Ripristinato il training loop nell' `objective()`
- Passato `weight_decay` al modello (o all'optimizer, se lo espone)
"""

import os
import random
import csv
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import yaml
import numpy as np
import torch
import optuna
from codecarbon import EmissionsTracker

from src.data_loader import DataLoader as CFLoader
from src.models.bcauss import BCAUSS
from src.metrics import eps_ATE_diff, PEHE_with_ite
from src.contrastive import DynamicContrastiveCausalDS
from src.siamese_bcuass.siamese import SiameseBCAUSS


# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------

def save_metrics(csv_path: Path, identifier: str, params: dict, eps_ate: float,
                 pehe: float, co2: str | float = "") -> None:
    """Append aggregated metrics and hyperparameters for each trial."""
    with open(csv_path, "a", newline="") as fm:
        writer = csv.writer(fm)
        writer.writerow([
            identifier,
            params["margin"],
            params["lambda_ctr"],
            params["lr"],
            params["batch_size"],
            params["weight_decay"],
            f"{eps_ate:.6f}",
            f"{pehe:.6f}",
            co2,
        ])


# -----------------------------------------------------------------------------
# Optuna objective
# -----------------------------------------------------------------------------

def objective(
        trial: optuna.Trial,
        cfg: DictConfig,
        base: BCAUSS,
        X_tr_all: np.ndarray,
        T_tr_all: np.ndarray,
        YF_tr_all: np.ndarray,
        X_te_all: np.ndarray,
        m0_te_all: np.ndarray,
        m1_te_all: np.ndarray,
        device: str,
        metrics_avg_csv: Path,
        metrics_all_csv: Path,
):
    """Single Optuna trial."""

    # 1️⃣ Hyper‑parameter sampling
    lr = trial.suggest_float("lr", 5e-5, 2e-4, log=True)
    margin = trial.suggest_float("margin", 0.4, 1.2)
    lambda_ctr = trial.suggest_float("lambda_ctr", 0.5, 2.0)
    bs = trial.suggest_categorical("batch_size", [32, 64, 128])
    wd = trial.suggest_float("weight_decay", 1e-5, 5e-3, log=True)

    params = {
        "margin": margin,
        "lambda_ctr": lambda_ctr,
        "lr": lr,
        "batch_size": bs,
        "weight_decay": wd,
    }

    # 2️⃣ Model instantiation
    siamese_params = {
        "ds_class": DynamicContrastiveCausalDS,
        "margin": margin,
        "lambda_ctr": lambda_ctr,
        "batch_size": bs,
        "lr": lr,
        "weight_decay": wd,  # nuova leva
        "epochs": cfg.epochs,
        "clip_norm": cfg.siamese.clip_norm,
        "use_amp": cfg.siamese.use_amp,
        "val_split": cfg.siamese.val_split,
        "update_ite_freq": cfg.siamese.update_ite_freq,
        "warmup_epochs_base": 0,
        "lambda_reg": cfg.siamese.lambda_reg,
    }
    model = SiameseBCAUSS(base_model=base, **siamese_params).to(device)

    # 3️⃣ Training su tutte le repliche IHDP
    eps_vals, pehe_vals = [], []
    for rep in range(cfg.n_reps):
        # Train
        Xtr = X_tr_all[:, :, rep].astype(np.float32)
        Ttr = T_tr_all[:, rep, None].astype(np.float32)
        Ytr = YF_tr_all[:, rep, None].astype(np.float32)
        model.fit(Xtr, Ttr, Ytr)

        # Evaluate
        Xte = X_te_all[:, :, rep].astype(np.float32)
        true_ite = m1_te_all[:, rep] - m0_te_all[:, rep]
        with torch.no_grad():
            pred_ite = model.predict_ite(Xte)

        eps = eps_ATE_diff(pred_ite.mean(), true_ite.mean())
        pehe = PEHE_with_ite(pred_ite, true_ite, sqrt=True)

        # per‑replica logging
        with open(metrics_all_csv, "a", newline="") as fm:
            csv.writer(fm).writerow([
                f"trial_{trial.number}", rep + 1, f"{eps:.6f}", f"{pehe:.6f}",
            ])

        eps_vals.append(eps)
        pehe_vals.append(pehe)

        trial.report(pehe, rep)
        if trial.should_prune():
            raise optuna.TrialPruned()

    # 4️⃣ Aggregated metrics per trial
    avg_eps = float(np.nanmean(eps_vals))
    avg_pehe = float(np.nanmean(pehe_vals))
    save_metrics(metrics_avg_csv, f"trial_{trial.number}", params, avg_eps, avg_pehe)

    return avg_pehe


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

@hydra.main(config_path="../../configs", config_name="default", version_base="1.3")
def run(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print("Config:\n" + OmegaConf.to_yaml(cfg))

    # Reproducibilità
    os.environ["PYTHONHASHSEED"] = str(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = cfg.device if torch.cuda.is_available() else "cpu"

    # Data
    loader = CFLoader.get_loader("IHDP")
    (
        X_tr_all,
        T_tr_all,
        YF_tr_all,
        _,
        _,
        _,
        X_te_all,
        _,
        _,
        _,
        m0_te_all,
        m1_te_all,
    ) = loader.load()
    input_dim = X_tr_all.shape[1]

    # Base BCAUSS warm‑up (solo se richiesto)
    base = BCAUSS(input_dim=input_dim)
    if cfg.siamese.warmup_epochs_base > 0:
        X0 = X_tr_all[:, :, 0].astype(np.float32)
        T0 = T_tr_all[:, 0, None].astype(np.float32)
        Y0 = YF_tr_all[:, 0, None].astype(np.float32)
        base.fit(X0, T0, Y0, epochs=cfg.siamese.warmup_epochs_base)
    base.to(device)

    # Output dir & CSV
    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    metrics_avg_csv = out_dir / "optuna_aggregated_metrics.csv"
    metrics_all_csv = out_dir / "metriche.csv"

    with open(metrics_avg_csv, "w", newline="") as f:
        csv.writer(f).writerow(
            [
                "trial_id",
                "margin",
                "lambda_ctr",
                "lr",
                "batch_size",
                "weight_decay",
                "eps_ate",
                "pehe",
                "co2_kg",
            ]
        )
    with open(metrics_all_csv, "w", newline="") as f:
        csv.writer(f).writerow(["trial_id", "replica", "eps_ate", "pehe"])

    # Emission tracker
    tracker = EmissionsTracker(output_dir=str(out_dir), log_level="error", save_to_file=True)
    tracker.start()

    # Optuna study
    sampler = optuna.samplers.TPESampler(seed=cfg.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

    study.optimize(
        lambda tr: objective(
            tr,
            cfg,
            base,
            X_tr_all,
            T_tr_all,
            YF_tr_all,
            X_te_all,
            m0_te_all,
            m1_te_all,
            device,
            metrics_avg_csv,
            metrics_all_csv,
        ),
        n_trials=cfg.optuna.n_trials,
    )

    # Best params
    best = study.best_params
    print(f"Best parameters found: {best}")
    with open(out_dir / "best_params.yaml", "w") as f:
        yaml.safe_dump({"best_params": best}, f)

    tracker.stop()


if __name__ == "__main__":
    run()
