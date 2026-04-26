#!/usr/bin/env python3
"""
Ablation script aligned with the protocol reported in the HERMES paper.

Main fixes vs the previous script:
1) same IHDP protocol as in the paper: lr=3.19e-4, batch_size=32,
   total epochs=500, warm-up=20, relabel every 3 epochs;
2) NO double warm-up for HERMES;
3) BCAUSS base trained for 500 epochs (not 520);
4) explicit validation split=0.2, when supported by the implementation;
5) added HERMES_NoContrastive to isolate the contribution of the contrastive term;
6) robust signature filtering: only supported kwargs are passed to constructors / fit methods.

If your local implementation of BCAUSS / SiameseBCAUSS exposes different argument names,
check the alias blocks in `build_base_model` and `build_siamese_model`.
"""

import os
import random
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data_loader import DataLoader as CFLoader
from src.models.bcauss import BCAUSS
from src.metrics import eps_ATE_diff, PEHE_with_ite
from src.contrastive import ContrastiveCausalDS
from src.siamese_bcuass.siamese import SiameseBCAUSS


# -----------------------------------------------------------------------------
# PAPER PROTOCOL (IHDP)
# -----------------------------------------------------------------------------
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_REPS = 10  # use 1000 for the full IHDP protocol

PAPER_CFG = {
    "epochs": 500,
    "warmup_epochs": 20,        # warm-up phase INSIDE HERMES training
    "update_ite_freq": 3,       # relabel every 3 epochs
    "lr": 3.19e-4,
    "batch_size": 32,
    "margin": 0.432,
    "lambda_ctr": 1.0,
    "val_split": 0.20,
    "grad_clip_norm": 2.0,
    "use_amp": False,
    # loss weights reported in the paper
    "lambda_bal": 1.0,
    "lambda_bce": 0.0,
    "lambda_targ": 0.0,
    "lambda_reg": 0.1,
    # optional controls, passed only if supported by your code
    "early_stopping": True,
    "patience": 30,
}

ABLATIONS = [
    {"name": "BCAUSS_Base", "use_siamese": False},
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
        "lambda_ctr": PAPER_CFG["lambda_ctr"],
    },
    {
        "name": "HERMES_Covariate",
        "use_siamese": True,
        "pairing_strategy": "covariate",
        "lambda_ctr": PAPER_CFG["lambda_ctr"],
    },
    {
        "name": "HERMES_Static_ITE",
        "use_siamese": True,
        "pairing_strategy": "static_ite",
        "lambda_ctr": PAPER_CFG["lambda_ctr"],
    },
    {
        "name": "HERMES_Dynamic_ITE",
        "use_siamese": True,
        "pairing_strategy": "dynamic_ite",
        "lambda_ctr": PAPER_CFG["lambda_ctr"],
    },
]

def make_train_val_split_with_cf(X, T, Y, m0, m1, val_ratio=0.2, seed=42):
    n = X.shape[0]
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    n_val = max(1, int(round(n * val_ratio)))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    return (
        X[tr_idx], T[tr_idx], Y[tr_idx], m0[tr_idx], m1[tr_idx],
        X[val_idx], T[val_idx], Y[val_idx], m0[val_idx], m1[val_idx],
    )
# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_signature_info(obj):
    """Return accepted parameter names and whether the callable accepts **kwargs."""
    sig = inspect.signature(obj)
    params = sig.parameters
    accepted = set(params.keys())
    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    return accepted, has_var_keyword


def keep_supported_kwargs(callable_obj, kwargs: dict) -> dict:
    accepted, has_var_keyword = get_signature_info(callable_obj)
    if has_var_keyword:
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in accepted}


def add_first_supported_alias(target_kwargs: dict, callable_obj, aliases_and_value):
    """
    aliases_and_value: list of tuples [(name1, value), (name2, value), ...]
    Adds only the first alias supported by the callable signature.
    If the callable accepts **kwargs, the first alias is used.
    """
    accepted, has_var_keyword = get_signature_info(callable_obj)
    if has_var_keyword:
        name, value = aliases_and_value[0]
        target_kwargs[name] = value
        return
    for name, value in aliases_and_value:
        if name in accepted:
            target_kwargs[name] = value
            return


def make_train_val_split_with_cf(X, T, Y, m0, m1, val_ratio=0.2, seed=42):
    n = X.shape[0]
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    n_val = max(1, int(round(n * val_ratio)))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    return (
        X[tr_idx], T[tr_idx], Y[tr_idx], m0[tr_idx], m1[tr_idx],
        X[val_idx], T[val_idx], Y[val_idx], m0[val_idx], m1[val_idx],
    )
def maybe_to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# -----------------------------------------------------------------------------
# MODEL BUILDERS
# -----------------------------------------------------------------------------
def build_base_model(input_dim: int):
    """
    Build BCAUSS using the paper loss configuration whenever the implementation
    exposes the relevant kwargs.
    """
    kwargs = {}

    # fixed / common
    add_first_supported_alias(kwargs, BCAUSS, [("input_dim", input_dim)])
    add_first_supported_alias(kwargs, BCAUSS, [("device", DEVICE)])

    # loss weights from the paper
    add_first_supported_alias(kwargs, BCAUSS, [
        ("lambda_bal", PAPER_CFG["lambda_bal"]),
        ("bal_weight", PAPER_CFG["lambda_bal"]),
    ])
    add_first_supported_alias(kwargs, BCAUSS, [
        ("lambda_bce", PAPER_CFG["lambda_bce"]),
        ("bce_weight", PAPER_CFG["lambda_bce"]),
    ])
    add_first_supported_alias(kwargs, BCAUSS, [
        ("lambda_targ", PAPER_CFG["lambda_targ"]),
        ("targ_weight", PAPER_CFG["lambda_targ"]),
        ("targeted_reg_weight", PAPER_CFG["lambda_targ"]),
    ])
    add_first_supported_alias(kwargs, BCAUSS, [
        ("lambda_reg", PAPER_CFG["lambda_reg"]),
        ("lambda_l2", PAPER_CFG["lambda_reg"]),
        ("weight_decay", PAPER_CFG["lambda_reg"]),
    ])

    model = BCAUSS(**kwargs).to(DEVICE)
    return model


def build_siamese_model(base_model, pairing_strategy: str, lambda_ctr: float):
    """
    Build HERMES/Siamese wrapper with the protocol used in the paper.
    """
    kwargs = {
        "base_model": base_model,
        "ds_class": ContrastiveCausalDS,
        "margin": PAPER_CFG["margin"],
        "lambda_ctr": lambda_ctr,
        "batch_size": PAPER_CFG["batch_size"],
        "lr": PAPER_CFG["lr"],
        "epochs": PAPER_CFG["epochs"],
        "warmup_epochs_base": PAPER_CFG["warmup_epochs"],
        "pairing_strategy": pairing_strategy,
        "update_ite_freq": PAPER_CFG["update_ite_freq"],
        "val_split": 0.0,   # importantissimo: validation esterna
    }

    add_first_supported_alias(kwargs, SiameseBCAUSS, [
        ("clip_grad_norm", PAPER_CFG["grad_clip_norm"]),
        ("max_grad_norm", PAPER_CFG["grad_clip_norm"]),
    ])
    add_first_supported_alias(kwargs, SiameseBCAUSS, [
        ("use_amp", PAPER_CFG["use_amp"]),
        ("amp", PAPER_CFG["use_amp"]),
    ])
    add_first_supported_alias(kwargs, SiameseBCAUSS, [
        ("early_stopping", PAPER_CFG["early_stopping"]),
    ])
    add_first_supported_alias(kwargs, SiameseBCAUSS, [
        ("patience", PAPER_CFG["patience"]),
        ("early_stopping_patience", PAPER_CFG["patience"]),
    ])

    kwargs = keep_supported_kwargs(SiameseBCAUSS, kwargs)
    model = SiameseBCAUSS(**kwargs).to(DEVICE)
    return model
# -----------------------------------------------------------------------------
# FIT HELPERS
# -----------------------------------------------------------------------------
def fit_base_model(model, Xtr, Ttr, Ytr, Xval, Tval, Yval):
    """
    Train BCAUSS with the SAME total protocol budget as the paper.
    Warm-up is NOT added here; base model gets 500 epochs total.
    """
    fit_kwargs = {
        "epochs": PAPER_CFG["epochs"],
        "lr": PAPER_CFG["lr"],
        "batch_size": PAPER_CFG["batch_size"],
        "validation_split": PAPER_CFG["val_split"],
        "val_split": PAPER_CFG["val_split"],
        "X_val": Xval,
        "T_val": Tval,
        "Y_val": Yval,
        "validation_data": (Xval, Tval, Yval),
        "early_stopping": PAPER_CFG["early_stopping"],
        "patience": PAPER_CFG["patience"],
        "use_amp": PAPER_CFG["use_amp"],
        "amp": PAPER_CFG["use_amp"],
        "clip_grad_norm": PAPER_CFG["grad_clip_norm"],
        "max_grad_norm": PAPER_CFG["grad_clip_norm"],
    }
    fit_kwargs = keep_supported_kwargs(model.fit, fit_kwargs)
    model.fit(Xtr, Ttr, Ytr, **fit_kwargs)



def fit_siamese_model(model, Xtr, Ttr, Ytr, Xval, Tval, Yval, true_ite_val):
    """
    Train HERMES using:
    - pseudo-ITE of the model for dynamic pair construction
    - true validation PEHE for early stopping / checkpoint selection on IHDP
    """
    model.fit(
        Xtr, Ttr, Ytr,
        X_val_np=Xval,
        T_val_np=Tval,
        Y_val_np=Yval,
        true_ite_val=true_ite_val
    )
# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def run_ablation(output_dir="ablation_paper_protocol"):
    set_seed(SEED)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading IHDP data...")
    loader = CFLoader.get_loader("IHDP")
    (
        X_tr_all, T_tr_all, YF_tr_all, _, m0_tr_all, m1_tr_all,
        X_te_all, _, _, _, m0_te_all, m1_te_all,
    ) = loader.load()

    input_dim = X_tr_all.shape[1]
    all_results = []

    for ab in ABLATIONS:
        print("\n" + "=" * 70)
        print(f"Starting ablation: {ab['name']}")
        print("=" * 70)

        pehe_list = []
        eps_list = []

        for rep in range(N_REPS):
            rep_seed = SEED + rep
            set_seed(rep_seed)

            print(f"  -> Replication {rep + 1}/{N_REPS}")

            Xtr = X_tr_all[:, :, rep].astype(np.float32)
            Ttr = T_tr_all[:, rep, None].astype(np.float32)
            Ytr = YF_tr_all[:, rep, None].astype(np.float32)

            Xte = X_te_all[:, :, rep].astype(np.float32)
            true_ite = (m1_te_all[:, rep] - m0_te_all[:, rep]).astype(np.float32)

            true_m0_tr = m0_tr_all[:, rep].astype(np.float32)
            true_m1_tr = m1_tr_all[:, rep].astype(np.float32)

            (
                Xtr_fit, Ttr_fit, Ytr_fit, m0tr_fit, m1tr_fit,
                Xval, Tval, Yval, m0val, m1val
            ) = make_train_val_split_with_cf(
                Xtr, Ttr, Ytr, true_m0_tr, true_m1_tr,
                val_ratio=PAPER_CFG["val_split"], seed=rep_seed
            )

            true_ite_val = (m1val - m0val).reshape(-1)

            base_model = build_base_model(input_dim=input_dim)

            if not ab["use_siamese"]:
                model = base_model
                fit_base_model(model, Xtr_fit, Ttr_fit, Ytr_fit, Xval, Tval, Yval)
            else:
                model = build_siamese_model(
                    base_model=base_model,
                    pairing_strategy=ab["pairing_strategy"],
                    lambda_ctr=ab["lambda_ctr"],
                )
                fit_siamese_model(
                    model,
                    Xtr_fit, Ttr_fit, Ytr_fit,
                    Xval, Tval, Yval,
                    true_ite_val=true_ite_val
                )

            with torch.no_grad():
                pred_ite = maybe_to_numpy(model.predict_ite(Xte)).reshape(-1)

            eps = float(eps_ATE_diff(true_ite, pred_ite))
            pehe = float(PEHE_with_ite(true_ite, pred_ite, sqrt=True))

            pehe_list.append(pehe)
            eps_list.append(eps)

            all_results.append({
                "Strategy": ab["name"],
                "Replication": rep,
                "PEHE": pehe,
                "eps_ATE": eps,
            })

        print(f"[{ab['name']}] PEHE mean ± std: {np.mean(pehe_list):.6f} ± {np.std(pehe_list):.6f}")
        print(f"[{ab['name']}] eps_ATE mean ± std: {np.mean(eps_list):.6f} ± {np.std(eps_list):.6f}")

    df = pd.DataFrame(all_results)
    df.to_csv(output_dir / "ablation_results_raw.csv", index=False)

    summary = (
        df.groupby("Strategy")[["PEHE", "eps_ATE"]]
        .agg(["mean", "std", "median"])
        .round(6)
    )
    summary.to_csv(output_dir / "ablation_results_summary.csv")

    print("\nFinal summary:\n")
    print(summary)
    print(f"\nSaved raw results to: {output_dir / 'ablation_results_raw.csv'}")
    print(f"Saved summary to:     {output_dir / 'ablation_results_summary.csv'}")

if __name__ == "__main__":
    run_ablation()
