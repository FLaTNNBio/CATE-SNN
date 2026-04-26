#!/usr/bin/env python3
import os
import random
import csv
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig
import numpy as np
import torch
import yaml

from src.data_loader import DataLoader as CFLoader
from src.models.bcauss import BCAUSS
from src.metrics import eps_ATE_diff, PEHE_with_ite
from src.contrastive import ContrastiveCausalDS
from src.siamese_bcuass.siamese import SiameseBCAUSS


ABLATIONS = [
    {
        "name": "BCAUSS_Base",
        "use_siamese": False,
    },
    {
        "name": "HERMES_NoContrastive",
        "use_siamese": True,
        "pairing_strategy": "dynamic_ite",
        "lambda_ctr": 0.0,
    },
    {
        "name": "HERMES_Random",
        "use_siamese": True,
        "pairing_strategy": "random",
        "lambda_ctr": None,  # filled from best params
    },
    {
        "name": "HERMES_Covariate",
        "use_siamese": True,
        "pairing_strategy": "covariate",
        "lambda_ctr": None,
    },
    {
        "name": "HERMES_Static_ITE",
        "use_siamese": True,
        "pairing_strategy": "static_ite",
        "lambda_ctr": None,
    },
    {
        "name": "HERMES_Dynamic_ITE",
        "use_siamese": True,
        "pairing_strategy": "dynamic_ite",
        "lambda_ctr": None,
    },
]


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_row(csv_path, row):
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def load_best_params(path: str | None):
    if path is not None and Path(path).exists():
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return data["best_params"]

    # fallback: replace with the exact best params from your paper run
    return {
        "margin": 0.432,
        "lambda_ctr": 1.0,
        "lr": 3.19e-4,
        "batch_size": 32,
    }

def build_base_model(input_dim, cfg, X=None, T=None, Y=None):
    """
    Exact same pattern as train_optuna.py:
    instantiate BCAUSS, then optional warm-up through base.fit(...)
    """
    base_model = BCAUSS(input_dim=input_dim)

    if cfg.siamese.warmup_epochs_base > 0 and X is not None:
        base_model.fit(X, T, Y, epochs=cfg.siamese.warmup_epochs_base)

    return base_model


def evaluate_strategy(
    strategy_cfg,
    cfg,
    device,
    input_dim,
    X_tr_all,
    T_tr_all,
    YF_tr_all,
    X_te_all,
    m0_te_all,
    m1_te_all,
    best_params,
    raw_csv_path,
):
    pehe_vals = []
    eps_vals = []

    for rep in range(cfg.n_reps):
        set_seed(cfg.seed + rep)
        print(f"  -> Replication {rep + 1}/{cfg.n_reps}")

        Xtr = X_tr_all[:, :, rep].astype(np.float32)
        Ttr = T_tr_all[:, rep, None].astype(np.float32)
        Ytr = YF_tr_all[:, rep, None].astype(np.float32)

        Xte = X_te_all[:, :, rep].astype(np.float32)
        true_ite = (m1_te_all[:, rep] - m0_te_all[:, rep]).astype(np.float32)

        # build base exactly as in train_optuna.py
        base = build_base_model(
            input_dim=input_dim,
            cfg=cfg,
            X=Xtr,
            T=Ttr,
            Y=Ytr,
        ).to(device)

        if not strategy_cfg["use_siamese"]:
            model = base
            # IMPORTANT:
            # use the SAME training call style you used in the paper for BCAUSS
            model.fit(Xtr, Ttr, Ytr, epochs=cfg.epochs)
        else:
            lambda_ctr = strategy_cfg["lambda_ctr"]
            if lambda_ctr is None:
                lambda_ctr = best_params["lambda_ctr"]

            siamese_params = {
                "ds_class": ContrastiveCausalDS,
                "margin": best_params["margin"],
                "lambda_ctr": lambda_ctr,
                "batch_size": best_params["batch_size"],
                "lr": best_params["lr"],
                "epochs": cfg.epochs,
                "clip_norm": cfg.siamese.clip_norm,
                "use_amp": cfg.siamese.use_amp,
                "val_split": cfg.siamese.val_split,
                "update_ite_freq": cfg.siamese.update_ite_freq,
                "warmup_epochs_base": 0,
                "pairing_strategy": strategy_cfg["pairing_strategy"],
            }

            model = SiameseBCAUSS(base_model=base, **siamese_params).to(device)
            model.fit(Xtr, Ttr, Ytr)

        with torch.no_grad():
            pred_ite = model.predict_ite(Xte)

        eps = eps_ATE_diff(true_ite, pred_ite)
        pehe = PEHE_with_ite(true_ite, pred_ite, sqrt=True)

        eps_vals.append(float(eps))
        pehe_vals.append(float(pehe))

        save_row(
            raw_csv_path,
            [
                strategy_cfg["name"],
                rep + 1,
                f"{float(eps):.6f}",
                f"{float(pehe):.6f}",
            ],
        )

    return (
        float(np.nanmean(eps_vals)),
        float(np.nanstd(eps_vals)),
        float(np.nanmean(pehe_vals)),
        float(np.nanstd(pehe_vals)),
    )


@hydra.main(config_path="../../configs", config_name="default", version_base="1.3")
def run(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    set_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"

    best_params = load_best_params(
        getattr(cfg, "best_params_file", None)
    )

    loader = CFLoader.get_loader("IHDP")
    (
        X_tr_all, T_tr_all, YF_tr_all, _,
        m0_tr_all, m1_tr_all,
        X_te_all, _, _, _,
        m0_te_all, m1_te_all
    ) = loader.load()

    input_dim = X_tr_all.shape[1]

    out_dir = Path("ablation_from_train_optuna_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = out_dir / "ablation_raw.csv"
    summary_csv = out_dir / "ablation_summary.csv"

    with open(raw_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["strategy", "replication", "eps_ate", "pehe"])

    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "strategy",
                "margin",
                "lambda_ctr",
                "lr",
                "batch_size",
                "eps_ate_mean",
                "eps_ate_std",
                "pehe_mean",
                "pehe_std",
            ]
        )

    for strategy in ABLATIONS:
        print("\n" + "=" * 72)
        print(f"Starting ablation: {strategy['name']}")
        print("=" * 72)

        used_lambda = strategy.get("lambda_ctr", None)
        if used_lambda is None and strategy["use_siamese"]:
            used_lambda = best_params["lambda_ctr"]

        eps_mean, eps_std, pehe_mean, pehe_std = evaluate_strategy(
            strategy_cfg=strategy,
            cfg=cfg,
            device=device,
            input_dim=input_dim,
            X_tr_all=X_tr_all,
            T_tr_all=T_tr_all,
            YF_tr_all=YF_tr_all,
            X_te_all=X_te_all,
            m0_te_all=m0_te_all,
            m1_te_all=m1_te_all,
            best_params=best_params,
            raw_csv_path=raw_csv,
        )

        save_row(
            summary_csv,
            [
                strategy["name"],
                best_params["margin"] if strategy["use_siamese"] else "",
                used_lambda if strategy["use_siamese"] else "",
                best_params["lr"] if strategy["use_siamese"] else "",
                best_params["batch_size"] if strategy["use_siamese"] else "",
                f"{eps_mean:.6f}",
                f"{eps_std:.6f}",
                f"{pehe_mean:.6f}",
                f"{pehe_std:.6f}",
            ],
        )

        print(
            f"[{strategy['name']}] "
            f"PEHE: {pehe_mean:.6f} ± {pehe_std:.6f} | "
            f"eps_ATE: {eps_mean:.6f} ± {eps_std:.6f}"
        )

    print(f"\nSaved raw results to: {raw_csv}")
    print(f"Saved summary to:     {summary_csv}")


if __name__ == "__main__":
    run()