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
from sklearn.preprocessing import StandardScaler

from src.data_loader import DataLoader as CFLoader
from src.models.bcauss import BCAUSS
from src.metrics import eps_ATE_diff, PEHE_with_ite
from src.contrastive import ContrastiveCausalDS
from src.siamese_bcuass.siamese import SiameseBCAUSS


# -----------------------------------------------------------------------------
# Experiment families
# -----------------------------------------------------------------------------
STRUCTURAL_ABLATIONS = [
    {
        "group": "structural",
        "name": "BCAUSS_Base",
        "use_siamese": False,
    },
    {
        "group": "structural",
        "name": "HERMES_NoContrastive",
        "use_siamese": True,
        "pairing_strategy": "dynamic_ite",
        "lambda_ctr": 0.0,
    },
    {
        "group": "structural",
        "name": "HERMES_Random",
        "use_siamese": True,
        "pairing_strategy": "random",
    },
    {
        "group": "structural",
        "name": "HERMES_Covariate",
        "use_siamese": True,
        "pairing_strategy": "covariate",
    },
    {
        "group": "structural",
        "name": "HERMES_Static_ITE",
        "use_siamese": True,
        "pairing_strategy": "static_ite",
    },
    {
        "group": "structural",
        "name": "HERMES_Full",
        "use_siamese": True,
        "pairing_strategy": "dynamic_ite",
    },
]

OPTIMIZATION_ABLATIONS = [
    {
        "group": "optimization",
        "name": "HERMES_Full",
        "use_siamese": True,
        "pairing_strategy": "dynamic_ite",
    },
    {
        "group": "optimization",
        "name": "HERMES_NoWarmup",
        "use_siamese": True,
        "pairing_strategy": "dynamic_ite",
        "warmup_epochs_base": 0,
    },
    {
        "group": "optimization",
        "name": "HERMES_Warmup_5",
        "use_siamese": True,
        "pairing_strategy": "dynamic_ite",
        "warmup_epochs_base": 5,
    },
    {
        "group": "optimization",
        "name": "HERMES_Warmup_20",
        "use_siamese": True,
        "pairing_strategy": "dynamic_ite",
        "warmup_epochs_base": 20,
    },
    {
        "group": "optimization",
        "name": "HERMES_NoITERefresh",
        "use_siamese": True,
        "pairing_strategy": "dynamic_ite",
        # initialized once, then practically never refreshed again
        "update_ite_freq": 10**9,
    },
    {
        "group": "optimization",
        "name": "HERMES_RefreshEvery5",
        "use_siamese": True,
        "pairing_strategy": "dynamic_ite",
        "update_ite_freq": 5,
    },
]

SENSITIVITY_EXPERIMENTS = [
    # lambda sensitivity
    {
        "group": "sensitivity",
        "name": "Lambda_0.1",
        "use_siamese": True,
        "pairing_strategy": "dynamic_ite",
        "lambda_ctr": 0.1,
    },
    {
        "group": "sensitivity",
        "name": "Lambda_1.0",
        "use_siamese": True,
        "pairing_strategy": "dynamic_ite",
        "lambda_ctr": 1.0,
    },
    {
        "group": "sensitivity",
        "name": "Lambda_2.0",
        "use_siamese": True,
        "pairing_strategy": "dynamic_ite",
        "lambda_ctr": 2.0,
    },
    # margin sensitivity
    {
        "group": "sensitivity",
        "name": "Margin_0.2",
        "use_siamese": True,
        "pairing_strategy": "dynamic_ite",
        "margin": 0.2,
    },
    {
        "group": "sensitivity",
        "name": "Margin_0.432",
        "use_siamese": True,
        "pairing_strategy": "dynamic_ite",
        "margin": 0.432,
    },
    {
        "group": "sensitivity",
        "name": "Margin_1.0",
        "use_siamese": True,
        "pairing_strategy": "dynamic_ite",
        "margin": 1.0,
    },
    # percentile sensitivity
    {
        "group": "sensitivity",
        "name": "Threshold_10pct",
        "use_siamese": True,
        "pairing_strategy": "dynamic_ite",
        "perc": 10,
    },
    {
        "group": "sensitivity",
        "name": "Threshold_20pct",
        "use_siamese": True,
        "pairing_strategy": "dynamic_ite",
        "perc": 20,
    },
    {
        "group": "sensitivity",
        "name": "Threshold_30pct",
        "use_siamese": True,
        "pairing_strategy": "dynamic_ite",
        "perc": 30,
    },
    # pair budget / pairs per step sensitivity
    {
        "group": "sensitivity",
        "name": "PairsPerStep_16",
        "use_siamese": True,
        "pairing_strategy": "dynamic_ite",
        "batch_size": 16,
    },
    {
        "group": "sensitivity",
        "name": "PairsPerStep_32",
        "use_siamese": True,
        "pairing_strategy": "dynamic_ite",
        "batch_size": 32,
    },
    {
        "group": "sensitivity",
        "name": "PairsPerStep_64",
        "use_siamese": True,
        "pairing_strategy": "dynamic_ite",
        "batch_size": 64,
    },
]

ALL_EXPERIMENTS = STRUCTURAL_ABLATIONS + OPTIMIZATION_ABLATIONS + SENSITIVITY_EXPERIMENTS


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

    return {
        "margin": 0.432,
        "lambda_ctr": 1.0,
        "lr": 3.19e-4,
        "batch_size": 32,
        "perc": 20,
    }


def ensure_target_scaler(base_model, Y):
    if getattr(base_model, "y_scaler", None) is None:
        scaler = StandardScaler()
        scaler.fit(np.asarray(Y).reshape(-1, 1))
        base_model.y_scaler = scaler


def build_base_model(input_dim, cfg, X=None, T=None, Y=None, warmup_epochs=None):
    base_model = BCAUSS(input_dim=input_dim)

    if warmup_epochs is None:
        warmup_epochs = cfg.siamese.warmup_epochs_base

    if warmup_epochs > 0 and X is not None:
        base_model.fit(X, T, Y, epochs=warmup_epochs)

    return base_model


def build_dataset_class(base_perc: int):
    class ContrastiveCausalDSWithPerc(ContrastiveCausalDS):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("perc", base_perc)
            super().__init__(*args, **kwargs)

    return ContrastiveCausalDSWithPerc


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

        warmup_epochs = strategy_cfg.get("warmup_epochs_base", None)
        if warmup_epochs is None:
            warmup_epochs = cfg.siamese.warmup_epochs_base

        base = build_base_model(
            input_dim=input_dim,
            cfg=cfg,
            X=Xtr,
            T=Ttr,
            Y=Ytr,
            warmup_epochs=warmup_epochs,
        ).to(device)

        ensure_target_scaler(base, Ytr)

        if not strategy_cfg["use_siamese"]:
            model = base
            model.fit(Xtr, Ttr, Ytr, epochs=cfg.epochs)
            lambda_ctr = 0.0
            pairing_strategy = "--"
            dynamic_relabeling = "No"
            update_ite_freq = "--"
            margin = "--"
            batch_size = "--"
            perc = "--"
        else:
            lambda_ctr = strategy_cfg.get("lambda_ctr", best_params["lambda_ctr"])
            pairing_strategy = strategy_cfg.get("pairing_strategy", "dynamic_ite")
            update_ite_freq = strategy_cfg.get("update_ite_freq", cfg.siamese.update_ite_freq)
            margin = strategy_cfg.get("margin", best_params["margin"])
            batch_size = strategy_cfg.get("batch_size", best_params["batch_size"])
            perc = strategy_cfg.get("perc", best_params.get("perc", 20))

            dynamic_relabeling = "Yes" if pairing_strategy == "dynamic_ite" and update_ite_freq == 1 else "No"

            siamese_params = {
                "ds_class": build_dataset_class(perc),
                "margin": margin,
                "lambda_ctr": lambda_ctr,
                "batch_size": batch_size,
                "lr": best_params["lr"],
                "epochs": cfg.epochs,
                "clip_norm": cfg.siamese.clip_norm,
                "use_amp": cfg.siamese.use_amp,
                "val_split": cfg.siamese.val_split,
                "update_ite_freq": update_ite_freq,
                "warmup_epochs_base": 0,
                "pairing_strategy": pairing_strategy,
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
                strategy_cfg["group"],
                strategy_cfg["name"],
                rep + 1,
                pairing_strategy,
                dynamic_relabeling,
                warmup_epochs,
                update_ite_freq,
                lambda_ctr,
                margin,
                perc,
                batch_size,
                f"{float(eps):.6f}",
                f"{float(pehe):.6f}",
            ],
        )

    return {
        "eps_mean": float(np.nanmean(eps_vals)),
        "eps_std": float(np.nanstd(eps_vals)),
        "pehe_mean": float(np.nanmean(pehe_vals)),
        "pehe_std": float(np.nanstd(pehe_vals)),
    }


def summarize_group(results_for_group, group_name, out_path):
    full_key = "HERMES_Full" if group_name != "sensitivity" else None
    full_pehe = results_for_group[full_key]["pehe_mean"] if full_key in results_for_group else None

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "group",
                "strategy",
                "pehe_mean",
                "pehe_std",
                "delta_pehe_vs_full_pct",
                "eps_ate_mean",
                "eps_ate_std",
            ]
        )

        for strategy_name, metrics in results_for_group.items():
            delta_pct = ""
            if full_pehe is not None and strategy_name != full_key:
                delta_pct = 100.0 * (metrics["pehe_mean"] - full_pehe) / full_pehe
                delta_pct = f"{delta_pct:.2f}"

            writer.writerow(
                [
                    group_name,
                    strategy_name,
                    f"{metrics['pehe_mean']:.6f}",
                    f"{metrics['pehe_std']:.6f}",
                    delta_pct,
                    f"{metrics['eps_mean']:.6f}",
                    f"{metrics['eps_std']:.6f}",
                ]
            )


@hydra.main(config_path="../../configs", config_name="default", version_base="1.3")
def run(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    set_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"

    best_params = load_best_params(getattr(cfg, "best_params_file", None))

    loader = CFLoader.get_loader("IHDP")
    (
        X_tr_all, T_tr_all, YF_tr_all, _,
        m0_tr_all, m1_tr_all,
        X_te_all, _, _, _,
        m0_te_all, m1_te_all
    ) = loader.load()

    input_dim = X_tr_all.shape[1]

    out_dir = Path("ablation_sensitivity_suite_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = out_dir / "ablation_sensitivity_raw.csv"
    structural_csv = out_dir / "ablation_structural_summary.csv"
    optimization_csv = out_dir / "ablation_optimization_summary.csv"
    sensitivity_csv = out_dir / "ablation_sensitivity_summary.csv"

    with open(raw_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "group",
            "strategy",
            "replication",
            "pairing_strategy",
            "dynamic_relabeling",
            "warmup_epochs_base",
            "update_ite_freq",
            "lambda_ctr",
            "margin",
            "perc",
            "batch_size",
            "eps_ate",
            "pehe",
        ])

    structural_results = {}
    optimization_results = {}
    sensitivity_results = {}

    for strategy in ALL_EXPERIMENTS:
        print("\n" + "=" * 80)
        print(f"Starting {strategy['group']} experiment: {strategy['name']}")
        print("=" * 80)

        metrics = evaluate_strategy(
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

        print(
            f"[{strategy['name']}] "
            f"PEHE: {metrics['pehe_mean']:.6f} ± {metrics['pehe_std']:.6f} | "
            f"eps_ATE: {metrics['eps_mean']:.6f} ± {metrics['eps_std']:.6f}"
        )

        if strategy["group"] == "structural":
            structural_results[strategy["name"]] = metrics
        elif strategy["group"] == "optimization":
            optimization_results[strategy["name"]] = metrics
        else:
            sensitivity_results[strategy["name"]] = metrics

    summarize_group(structural_results, "structural", structural_csv)
    summarize_group(optimization_results, "optimization", optimization_csv)
    summarize_group(sensitivity_results, "sensitivity", sensitivity_csv)

    print(f"\nSaved raw results to:          {raw_csv}")
    print(f"Saved structural summary to:   {structural_csv}")
    print(f"Saved optimization summary to: {optimization_csv}")
    print(f"Saved sensitivity summary to:  {sensitivity_csv}")


if __name__ == "__main__":
    run()
