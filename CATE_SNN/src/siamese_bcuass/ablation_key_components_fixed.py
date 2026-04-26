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
# Two families of ablations:
# 1) structural: what makes HERMES conceptually different
# 2) optimization: what makes HERMES training stable/effective
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
        "warmup_epochs_base": None,
        "update_ite_freq": None,
    },
    {
        "group": "structural",
        "name": "HERMES_Random",
        "use_siamese": True,
        "pairing_strategy": "random",
        "lambda_ctr": None,
        "warmup_epochs_base": None,
        "update_ite_freq": None,
    },
    {
        "group": "structural",
        "name": "HERMES_Covariate",
        "use_siamese": True,
        "pairing_strategy": "covariate",
        "lambda_ctr": None,
        "warmup_epochs_base": None,
        "update_ite_freq": None,
    },
    {
        "group": "structural",
        "name": "HERMES_Static_ITE",
        "use_siamese": True,
        "pairing_strategy": "static_ite",
        "lambda_ctr": None,
        "warmup_epochs_base": None,
        "update_ite_freq": None,
    },
    {
        "group": "structural",
        "name": "HERMES_Full",
        "use_siamese": True,
        "pairing_strategy": "dynamic_ite",
        "lambda_ctr": None,
        "warmup_epochs_base": None,
        "update_ite_freq": None,
    },
]


OPTIMIZATION_ABLATIONS = [
    {
        "group": "optimization",
        "name": "HERMES_Full",
        "use_siamese": True,
        "pairing_strategy": "dynamic_ite",
        "lambda_ctr": None,
        "warmup_epochs_base": None,
        "update_ite_freq": None,
    },
    {
        "group": "optimization",
        "name": "HERMES_NoWarmup",
        "use_siamese": True,
        "pairing_strategy": "dynamic_ite",
        "lambda_ctr": None,
        "warmup_epochs_base": 0,
        "update_ite_freq": None,
    },
    {
        "group": "optimization",
        "name": "HERMES_Warmup_5",
        "use_siamese": True,
        "pairing_strategy": "dynamic_ite",
        "lambda_ctr": None,
        "warmup_epochs_base": 5,
        "update_ite_freq": None,
    },
    {
        "group": "optimization",
        "name": "HERMES_Warmup_20",
        "use_siamese": True,
        "pairing_strategy": "dynamic_ite",
        "lambda_ctr": None,
        "warmup_epochs_base": 20,
        "update_ite_freq": None,
    },
    {
        "group": "optimization",
        "name": "HERMES_NoITERefresh",
        "use_siamese": True,
        "pairing_strategy": "dynamic_ite",
        "lambda_ctr": None,
        "warmup_epochs_base": None,
        # huge value == initialized once, then practically never updated again
        "update_ite_freq": 10**9,
    },
    {
        "group": "optimization",
        "name": "HERMES_RefreshEvery5",
        "use_siamese": True,
        "pairing_strategy": "dynamic_ite",
        "lambda_ctr": None,
        "warmup_epochs_base": None,
        "update_ite_freq": 5,
    },
]


ALL_ABLATIONS = STRUCTURAL_ABLATIONS + OPTIMIZATION_ABLATIONS


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

    # fallback: replace with exact values from your final paper run if needed
    return {
        "margin": 0.432,
        "lambda_ctr": 1.0,
        "lr": 3.19e-4,
        "batch_size": 32,
    }




def ensure_target_scaler(base_model, Y):
    """
    Initialize the target scaler without doing a warm-up optimization step.
    This is needed for true "NoWarmup" runs: we still want the same target
    normalization / inverse-transform behavior used by BCAUSS, but without
    pretraining the backbone before Siamese optimization.
    """
    if getattr(base_model, "y_scaler", None) is None:
        scaler = StandardScaler()
        scaler.fit(np.asarray(Y).reshape(-1, 1))
        base_model.y_scaler = scaler

def build_base_model(input_dim, cfg, X=None, T=None, Y=None, warmup_epochs=None):
    """
    Same logic as your current runner, but with an explicit ablation override
    for the backbone warm-up.
    """
    base_model = BCAUSS(input_dim=input_dim)

    if warmup_epochs is None:
        warmup_epochs = cfg.siamese.warmup_epochs_base

    if warmup_epochs > 0 and X is not None:
        base_model.fit(X, T, Y, epochs=warmup_epochs)

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

        # Important: "NoWarmup" should disable only the optimization warm-up,
        # not the target normalization machinery expected by BCAUSS.predict_ite.
        # So, if the base.fit(...) warm-up was skipped, initialize y_scaler
        # explicitly from the training outcomes.
        ensure_target_scaler(base, Ytr)

        if not strategy_cfg["use_siamese"]:
            model = base
            model.fit(Xtr, Ttr, Ytr, epochs=cfg.epochs)
            lambda_ctr = 0.0
            pairing_strategy = "--"
            dynamic_relabeling = "No"
            update_ite_freq = "--"
        else:
            lambda_ctr = strategy_cfg["lambda_ctr"]
            if lambda_ctr is None:
                lambda_ctr = best_params["lambda_ctr"]

            pairing_strategy = strategy_cfg["pairing_strategy"]
            update_ite_freq = strategy_cfg.get("update_ite_freq", None)
            if update_ite_freq is None:
                update_ite_freq = cfg.siamese.update_ite_freq

            dynamic_relabeling = "Yes" if pairing_strategy == "dynamic_ite" and update_ite_freq == 1 else "No"

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
                "update_ite_freq": update_ite_freq,
                # keep 0 here to avoid doing warm-up twice
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
                f"{float(lambda_ctr):.6f}",
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
    """
    Save a group summary and compute delta PEHE vs the full HERMES variant in that group.
    """
    full_key = "HERMES_Full"
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

    out_dir = Path("ablation_key_components_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = out_dir / "ablation_key_components_raw.csv"
    structural_csv = out_dir / "ablation_structural_summary.csv"
    optimization_csv = out_dir / "ablation_optimization_summary.csv"

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
            "eps_ate",
            "pehe",
        ])

    structural_results = {}
    optimization_results = {}

    for strategy in ALL_ABLATIONS:
        print("\n" + "=" * 80)
        print(f"Starting {strategy['group']} ablation: {strategy['name']}")
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
        else:
            optimization_results[strategy["name"]] = metrics

    summarize_group(structural_results, "structural", structural_csv)
    summarize_group(optimization_results, "optimization", optimization_csv)

    print(f"\nSaved raw results to:        {raw_csv}")
    print(f"Saved structural summary to: {structural_csv}")
    print(f"Saved optimization summary to: {optimization_csv}")


if __name__ == "__main__":
    run()
