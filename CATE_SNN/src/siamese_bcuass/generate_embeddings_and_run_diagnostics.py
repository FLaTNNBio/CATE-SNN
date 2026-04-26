from __future__ import annotations

"""
Unified pipeline for HERMES/BCAUSS post-hoc analysis:
  1) load an evaluation split from the project
  2) load saved PyTorch weights for HERMES and BCAUSS
  3) extract embeddings / predicted potential outcomes / tau_hat
  4) save all arrays as .npy
  5) compute both the paper's existing latent-space metrics and the new reviewer-focused metrics
  6) export CSV/JSON/LaTeX tables

IMPORTANT
---------
You only need to edit the PROJECT ADAPTER block:
  - load_eval_bundle(rep)
  - build_hermes(device)
  - build_bcauss(device)
  - infer_outputs(model, x)

Once those 4 functions are connected to your project, this script can do everything:
  extraction + diagnostics.

Why placeholders remain
-----------------------
The exact class names / dataset loader of your local project are not visible from here.
So the *pipeline* is complete, but the adapter still needs your real imports.
"""

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from scipy.stats import spearmanr


# =========================================================
# PROJECT ADAPTER -- EDIT ONLY THIS BLOCK
# =========================================================

import re
from pathlib import Path


def _project_root() -> Path:
    # .../CATE_SNN/src/siamese_bcuass/script.py -> .../CATE_SNN
    return Path(__file__).resolve().parents[2]


def _score_candidate(p: Path) -> tuple:
    name = p.name.lower()
    # preferisci test > ihdp > npz
    return (
        0 if "test" in name else 1,
        0 if "ihdp" in name else 1,
        0 if p.suffix.lower() == ".npz" else 1,
        len(name),
    )


def _find_npz_candidates(root: Path) -> list[Path]:
    files = [p for p in root.rglob("*.npz") if "ihdp" in p.name.lower() or "npci" in p.name.lower()]
    return sorted(files, key=_score_candidate)


def _select_rep(arr: np.ndarray, rep_idx: int, key: str) -> np.ndarray:
    """
    rep_idx is 0-based.
    Handles common IHDP layouts:
      - x: [n, d]
      - t/y/mu0/mu1: [n, R]
      - x: [n, d, R] or [R, n, d]
      - split keys x_test, t_test, ...
    """
    arr = np.asarray(arr)

    if arr.ndim == 1:
        return arr

    if arr.ndim == 2:
        # x is usually [n, d] with d small (e.g. 25)
        if key.startswith("x") and arr.shape[1] <= 200:
            return arr
        # t/y/mu0/mu1 often [n, R]
        if rep_idx < arr.shape[1]:
            return arr[:, rep_idx]
        # fallback
        return arr

    if arr.ndim == 3:
        # [n, d, R]
        if rep_idx < arr.shape[2] and arr.shape[1] <= 200:
            return arr[:, :, rep_idx]
        # [R, n, d]
        if rep_idx < arr.shape[0] and arr.shape[2] <= 200:
            return arr[rep_idx]
        # [n, R, d]
        if rep_idx < arr.shape[1] and arr.shape[2] <= 200:
            return arr[:, rep_idx, :]
        return arr

    raise RuntimeError(f"Unsupported array shape for key={key}: {arr.shape}")


def _extract_from_npz(npz_path: Path, rep_idx: int) -> Dict[str, np.ndarray] | None:
    data = np.load(npz_path, allow_pickle=True)
    keys = set(data.files)

    aliases = {
        "x": ["x_test", "X_test", "x", "X"],
        "t": ["t_test", "T_test", "t", "T"],
        "y": ["yf_test", "y_test", "Y_test", "yf", "y", "Y"],
        "mu0": ["mu0_test", "mu0"],
        "mu1": ["mu1_test", "mu1"],
    }

    out: Dict[str, np.ndarray] = {}

    for canonical, cands in aliases.items():
        found = None
        for c in cands:
            if c in keys:
                found = c
                break
        if found is not None:
            out[canonical] = _select_rep(data[found], rep_idx, found)

    # need at least x,t,y
    if not {"x", "t", "y"}.issubset(out.keys()):
        return None

    out["x"] = np.asarray(out["x"], dtype=np.float32)
    out["t"] = np.asarray(out["t"], dtype=np.float32).reshape(-1)
    out["y"] = np.asarray(out["y"], dtype=np.float32).reshape(-1)

    if "mu0" in out and "mu1" in out:
        out["mu0"] = np.asarray(out["mu0"], dtype=np.float32).reshape(-1)
        out["mu1"] = np.asarray(out["mu1"], dtype=np.float32).reshape(-1)
        out["tau"] = out["mu1"] - out["mu0"]

    return out




from src.models.bcauss import BCAUSS
from src.siamese_bcuass.siamese import SiameseBCAUSS
from src.data_loader import DataLoader as CFLoader


def load_eval_bundle(rep: int) -> Dict[str, np.ndarray]:
    """
    Loads the IHDP TEST split for replication `rep` (1-based).
    Assumed loader order:
      X_tr, T_tr, YF_tr, YCF_tr, MU0_tr, MU1_tr,
      X_te, T_te, YF_te, YCF_te, MU0_te, MU1_te
    """
    rep_index = rep - 1

    loader = CFLoader.get_loader('IHDP')
    (
        X_tr_all, T_tr_all, YF_tr_all, YCF_tr_all, MU0_tr_all, MU1_tr_all,
        X_te_all, T_te_all, YF_te_all, YCF_te_all, MU0_te_all, MU1_te_all
    ) = loader.load()

    x = X_te_all[:, :, rep_index].astype(np.float32)
    t = T_te_all[:, rep_index].astype(np.float32)
    y = YF_te_all[:, rep_index].astype(np.float32)
    mu0 = MU0_te_all[:, rep_index].astype(np.float32)
    mu1 = MU1_te_all[:, rep_index].astype(np.float32)

    return {
        "x": x,
        "t": t.reshape(-1),
        "y": y.reshape(-1),
        "mu0": mu0.reshape(-1),
        "mu1": mu1.reshape(-1),
        "tau": (mu1 - mu0).reshape(-1),
    }


def build_hermes(device: torch.device) -> torch.nn.Module:
    """
    HERMES/Siamese model constructor.
    Based on your existing evaluation/extraction scripts.
    """
    # input_dim inferred from IHDP
    input_dim = 25
    base_model = BCAUSS(input_dim=input_dim)

    model = SiameseBCAUSS(
        base_model=base_model,
        ds_class=None,
        margin=1.0,
        lambda_ctr=1.0,
        lr=1e-4,
        batch_size=128,
        epochs=1,
        val_split=0.2,
        update_ite_freq=1,
        warmup_epochs_base=0
    )
    return model.to(device)


def build_bcauss(device: torch.device) -> torch.nn.Module:
    """
    Plain BCAUSS model constructor.
    """
    input_dim = 25
    model = BCAUSS(input_dim=input_dim)
    return model.to(device)


@torch.no_grad()
def infer_outputs(model: torch.nn.Module, x: torch.Tensor):
    """
    Returns:
      phi, mu0, mu1

    Robust handling for:
      - SiameseBCAUSS
      - plain BCAUSS
      - nested tuples/lists/dicts
    """
    core = model.base if hasattr(model, "base") else model

    def collect_tensors(obj):
        out = []
        if torch.is_tensor(obj):
            out.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                out.extend(collect_tensors(v))
        elif isinstance(obj, (tuple, list)):
            for v in obj:
                out.extend(collect_tensors(v))
        return out

    # 1) Preferred path: mu_and_embedding
    if hasattr(core, "mu_and_embedding"):
        raw = core.mu_and_embedding(x)
        tensors = collect_tensors(raw)

        if len(tensors) >= 3:
            # Heuristic:
            # first two small outputs = mu0, mu1
            # last tensor = embedding
            mu0 = tensors[0]
            mu1 = tensors[1]
            phi = tensors[-1]
            return phi, mu0, mu1

        if len(tensors) == 1:
            phi = tensors[0]
        else:
            phi = None
    else:
        phi = None

    # 2) Fallback to forward
    raw2 = core(x)
    tensors2 = collect_tensors(raw2)

    if len(tensors2) >= 2:
        mu0 = tensors2[0]
        mu1 = tensors2[1]
        if phi is None:
            phi = tensors2[-1] if len(tensors2) >= 3 else torch.cat([mu0, mu1], dim=1)
        return phi, mu0, mu1

    raise RuntimeError(
        "infer_outputs() could not recover phi/mu0/mu1. "
        "Try printing type(core.mu_and_embedding(x)) once for debugging."
    )
# =========================================================
# UTILITIES
# =========================================================


@dataclass
class ModelEval:
    label: str
    phi: np.ndarray
    mu0: np.ndarray
    mu1: np.ndarray
    tau_hat: np.ndarray
    metrics: Dict[str, float]


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)
def ensure_1d(x: Any) -> np.ndarray:
    return to_numpy(x).reshape(-1)


def ensure_2d(x: Any) -> np.ndarray:
    arr = to_numpy(x)
    if arr.ndim == 1:
        return arr[:, None]
    return arr


def save_npy(path: Path, arr: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, to_numpy(arr))


def load_state_dict_any(weights_path: str) -> Dict[str, torch.Tensor]:
    blob = torch.load(weights_path, map_location="cpu")
    if isinstance(blob, dict):
        for k in ["state_dict", "model_state_dict", "net", "model"]:
            if k in blob and isinstance(blob[k], dict):
                return blob[k]
        if all(isinstance(v, torch.Tensor) for v in blob.values()):
            return blob
    raise RuntimeError(f"Could not extract a state_dict from: {weights_path}")


def pairwise_sq_dists(a: np.ndarray, b: Optional[np.ndarray] = None) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = a if b is None else np.asarray(b, dtype=np.float64)
    aa = np.sum(a * a, axis=1, keepdims=True)
    bb = np.sum(b * b, axis=1, keepdims=True).T
    d2 = aa + bb - 2.0 * (a @ b.T)
    np.maximum(d2, 0.0, out=d2)
    return d2


def factual_predictions(mu0: np.ndarray, mu1: np.ndarray, t: np.ndarray) -> np.ndarray:
    return t * mu1 + (1.0 - t) * mu0


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def ate_error(tau_hat: np.ndarray, tau_true: np.ndarray) -> float:
    return float(abs(np.mean(tau_hat) - np.mean(tau_true)))


def linear_mmd(phi: np.ndarray, t: np.ndarray) -> float:
    t = ensure_1d(t).astype(int)
    phi = ensure_2d(phi).astype(np.float64)
    if not np.any(t == 0) or not np.any(t == 1):
        return np.nan
    return float(np.linalg.norm(np.mean(phi[t == 1], axis=0) - np.mean(phi[t == 0], axis=0), ord=2))


def _median_heuristic_bandwidth(z: np.ndarray) -> float:
    d2 = pairwise_sq_dists(z)
    vals = d2[np.triu_indices(len(z), k=1)]
    vals = vals[np.isfinite(vals)]
    vals = vals[vals > 0]
    if len(vals) == 0:
        return 1.0
    med = np.median(vals)
    return float(max(np.sqrt(0.5 * med), 1e-6))


def rbf_mmd(phi: np.ndarray, t: np.ndarray, sigma: Optional[float] = None) -> float:
    phi = ensure_2d(phi).astype(np.float64)
    t = ensure_1d(t).astype(int)
    x = phi[t == 1]
    y = phi[t == 0]
    if len(x) < 2 or len(y) < 2:
        return np.nan
    z = np.vstack([x, y])
    sigma = _median_heuristic_bandwidth(z) if sigma is None else float(max(sigma, 1e-6))
    gamma = 1.0 / (2.0 * sigma * sigma)
    kxx = np.exp(-gamma * pairwise_sq_dists(x))
    kyy = np.exp(-gamma * pairwise_sq_dists(y))
    kxy = np.exp(-gamma * pairwise_sq_dists(x, y))
    m = len(x)
    n = len(y)
    term_xx = (np.sum(kxx) - np.trace(kxx)) / (m * (m - 1))
    term_yy = (np.sum(kyy) - np.trace(kyy)) / (n * (n - 1))
    term_xy = np.sum(kxy) / (m * n)
    return float(max(term_xx + term_yy - 2.0 * term_xy, 0.0))


def tail_ipm(phi: np.ndarray, t: np.ndarray, tail_q: float = 0.10) -> float:
    phi = ensure_2d(phi).astype(np.float64)
    t = ensure_1d(t).astype(int)
    center = np.mean(phi, axis=0, keepdims=True)
    radii = np.sqrt(np.sum((phi - center) ** 2, axis=1))
    thr = np.quantile(radii, 1.0 - tail_q)
    mask = radii >= thr
    if np.sum(mask & (t == 0)) < 2 or np.sum(mask & (t == 1)) < 2:
        return np.nan
    return linear_mmd(phi[mask], t[mask])


def knn_support_scores(phi: np.ndarray, t: np.ndarray, k: int = 10) -> np.ndarray:
    n = len(phi)
    d2 = pairwise_sq_dists(phi)
    np.fill_diagonal(d2, np.inf)
    nn_idx = np.argpartition(d2, kth=min(k, n - 1) - 1, axis=1)[:, :k]
    t = ensure_1d(t).astype(int)
    scores = np.mean(t[nn_idx] != t[:, None], axis=1)
    return scores.astype(np.float64)


def nearest_opposite_distances(phi: np.ndarray, t: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=np.float64)
    t = ensure_1d(t).astype(int)
    d2 = pairwise_sq_dists(phi)
    out = np.full(len(phi), np.nan, dtype=np.float64)
    for i in range(len(phi)):
        mask = (t != t[i])
        if np.any(mask):
            out[i] = float(np.sqrt(np.min(d2[i, mask])))
    return out


def support_tail_mask(support_scores: np.ndarray, tail_q: float = 0.25) -> np.ndarray:
    thr = float(np.quantile(support_scores, tail_q))
    return support_scores <= thr


def hard_region_mask(nn_opp_dist: np.ndarray, support_scores: np.ndarray, dist_q: float = 0.75, support_q: float = 0.25) -> np.ndarray:
    d_thr = float(np.quantile(nn_opp_dist[~np.isnan(nn_opp_dist)], dist_q))
    s_thr = float(np.quantile(support_scores, support_q))
    return (nn_opp_dist >= d_thr) | (support_scores <= s_thr)


def worst_smd_latent(phi: np.ndarray, t: np.ndarray) -> float:
    phi = ensure_2d(phi).astype(np.float64)
    t = ensure_1d(t).astype(int)
    if not np.any(t == 0) or not np.any(t == 1):
        return np.nan
    x1 = phi[t == 1]
    x0 = phi[t == 0]
    smds = []
    for j in range(phi.shape[1]):
        m1, m0 = np.mean(x1[:, j]), np.mean(x0[:, j])
        v1, v0 = np.var(x1[:, j], ddof=1), np.var(x0[:, j], ddof=1)
        sd_pooled = math.sqrt(max((v1 + v0) / 2.0, 1e-12))
        smds.append(abs(m1 - m0) / sd_pooled)
    return float(np.max(smds))


def cross_treatment_effect_geometry(phi: np.ndarray, t: np.ndarray, tau_ref: np.ndarray, pos_q: float = 0.20, neg_q: float = 0.80) -> Dict[str, float]:
    t = ensure_1d(t).astype(int)
    tau_ref = ensure_1d(tau_ref).astype(np.float64)
    d = np.sqrt(pairwise_sq_dists(phi))
    tau_gap = np.abs(tau_ref[:, None] - tau_ref[None, :])
    cross = t[:, None] != t[None, :]
    iu = np.triu_indices(len(phi), k=1)
    cross_mask = cross[iu]
    if not np.any(cross_mask):
        return {
            "cross_treat_effect_distance_spearman": np.nan,
            "pos_pair_mean_latent_dist": np.nan,
            "neg_pair_mean_latent_dist": np.nan,
            "separation_ratio": np.nan,
        }
    d_vec = d[iu][cross_mask]
    tau_vec = tau_gap[iu][cross_mask]
    pos_thr = float(np.quantile(tau_vec, pos_q))
    neg_thr = float(np.quantile(tau_vec, neg_q))
    pos_mask = tau_vec <= pos_thr
    neg_mask = tau_vec >= neg_thr
    rho = spearmanr(d_vec, tau_vec).correlation
    pos_mean = float(np.mean(d_vec[pos_mask])) if np.any(pos_mask) else np.nan
    neg_mean = float(np.mean(d_vec[neg_mask])) if np.any(neg_mask) else np.nan
    sep = float(neg_mean / max(pos_mean, 1e-8)) if np.isfinite(pos_mean) and np.isfinite(neg_mean) else np.nan
    return {
        "cross_treat_effect_distance_spearman": float(rho) if rho is not None else np.nan,
        "pos_pair_mean_latent_dist": pos_mean,
        "neg_pair_mean_latent_dist": neg_mean,
        "separation_ratio": sep,
    }


def summarize_model(label: str, x: np.ndarray, t: np.ndarray, y: np.ndarray, phi: np.ndarray, mu0: np.ndarray, mu1: np.ndarray, tau_true: Optional[np.ndarray], k: int) -> ModelEval:
    t = ensure_1d(t).astype(np.float64)
    y = ensure_1d(y).astype(np.float64)
    phi = ensure_2d(phi).astype(np.float64)
    mu0 = ensure_1d(mu0).astype(np.float64)
    mu1 = ensure_1d(mu1).astype(np.float64)
    tau_hat = mu1 - mu0

    yhat_fact = factual_predictions(mu0, mu1, t)
    support = knn_support_scores(phi, t, k=k)
    nn_opp = nearest_opposite_distances(phi, t)
    tail_mask = support_tail_mask(support, tail_q=0.25)
    hard_mask = hard_region_mask(nn_opp, support, dist_q=0.75, support_q=0.25)

    metrics: Dict[str, float] = {
        "n": float(len(x)),
        "factual_mse": float(np.mean((yhat_fact - y) ** 2)),
        "linear_mmd": linear_mmd(phi, t),
        "rbf_mmd": rbf_mmd(phi, t),
        "tail_ipm": tail_ipm(phi, t),
        "knn_overlap": float(np.mean(support)),
        "worst_smd_latent": worst_smd_latent(phi, t),
        "support_mean": float(np.mean(support)),
        "support_q25": float(np.quantile(support, 0.25)),
        "support_q50": float(np.quantile(support, 0.50)),
        "support_q75": float(np.quantile(support, 0.75)),
        "nn_opp_mean": float(np.nanmean(nn_opp)),
        "nn_opp_median": float(np.nanmedian(nn_opp)),
        "hard_region_frac": float(np.mean(hard_mask)),
        "low_support_frac": float(np.mean(tail_mask)),
    }

    tau_ref_for_geometry = tau_hat
    if tau_true is not None:
        tau_true = ensure_1d(tau_true).astype(np.float64)
        metrics.update({
            "rmse_tau_overall": rmse(tau_hat, tau_true),
            "ate_error_overall": ate_error(tau_hat, tau_true),
            "rmse_tau_low_support": rmse(tau_hat[tail_mask], tau_true[tail_mask]),
            "rmse_tau_easy_region": rmse(tau_hat[~tail_mask], tau_true[~tail_mask]),
            "rmse_tau_hard_region": rmse(tau_hat[hard_mask], tau_true[hard_mask]),
            "rmse_tau_nonhard_region": rmse(tau_hat[~hard_mask], tau_true[~hard_mask]),
            "ate_error_low_support": ate_error(tau_hat[tail_mask], tau_true[tail_mask]),
            "ate_error_hard_region": ate_error(tau_hat[hard_mask], tau_true[hard_mask]),
            "spearman_rho": float(spearmanr(tau_hat, tau_true).correlation),
        })
        metrics["support_gap_ratio"] = float(metrics["rmse_tau_low_support"] / max(metrics["rmse_tau_easy_region"], 1e-8))
        metrics["hard_gap_ratio"] = float(metrics["rmse_tau_hard_region"] / max(metrics["rmse_tau_nonhard_region"], 1e-8))
        tau_ref_for_geometry = tau_true
    else:
        metrics["spearman_rho"] = np.nan

    metrics.update(cross_treatment_effect_geometry(phi, t, tau_ref_for_geometry))

    return ModelEval(label=label, phi=phi, mu0=mu0, mu1=mu1, tau_hat=tau_hat, metrics=metrics)


def compare_two_models(a: ModelEval, b: ModelEval) -> Dict[str, float]:
    keys = sorted(set(a.metrics.keys()) & set(b.metrics.keys()))
    out = {}
    for k in keys:
        va, vb = a.metrics[k], b.metrics[k]
        if np.isfinite(va) and np.isfinite(vb):
            out[f"delta__{k}"] = float(vb - va)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_latex_table_existing(path: Path, rows: list[dict], labels: Tuple[str, str]) -> None:
    a_label, b_label = labels
    lookup = {row["label"]: row for row in rows}
    a = lookup[a_label]
    b = lookup[b_label]
    metrics = [
        ("factual_mse", r"Factual MSE $\downarrow$"),
        ("linear_mmd", r"Linear MMD $\downarrow$"),
        ("rbf_mmd", r"RBF MMD $\downarrow$"),
        ("tail_ipm", r"Tail IPM $\downarrow$"),
        ("spearman_rho", r"Spearman's $\rho$ $\uparrow$"),
        ("knn_overlap", r"k-NN Overlap $\uparrow$"),
        ("worst_smd_latent", r"worst-SMD (latent) $\downarrow$"),
    ]
    lines = [r"\begin{tabular}{lcc}", r"\toprule", rf"Metric & {a_label} & {b_label} \\", r"\midrule"]
    for key, title in metrics:
        va, vb = a.get(key, np.nan), b.get(key, np.nan)
        fa = "--" if not np.isfinite(va) else f"{va:.4f}"
        fb = "--" if not np.isfinite(vb) else f"{vb:.4f}"
        lines.append(f"{title} & {fa} & {fb} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_latex_table_reviewer(path: Path, rows: list[dict], labels: Tuple[str, str]) -> None:
    a_label, b_label = labels
    lookup = {row["label"]: row for row in rows}
    a = lookup[a_label]
    b = lookup[b_label]
    metrics = [
        ("rmse_tau_overall", r"RMSE overall $\downarrow$"),
        ("rmse_tau_low_support", r"RMSE low-support $\downarrow$"),
        ("rmse_tau_hard_region", r"RMSE hard-region $\downarrow$"),
        ("support_mean", r"Local support $\uparrow$"),
        ("nn_opp_mean", r"NN opposite distance $\downarrow$"),
        ("cross_treat_effect_distance_spearman", r"Cross-treatment $\rho$ $\uparrow$"),
        ("pos_pair_mean_latent_dist", r"Same-effect x-treat dist $\downarrow$"),
        ("neg_pair_mean_latent_dist", r"Diff-effect x-treat dist $\uparrow$"),
        ("separation_ratio", r"Separation ratio $\uparrow$"),
        ("support_gap_ratio", r"Support gap ratio $\downarrow$"),
        ("hard_gap_ratio", r"Hard-region gap ratio $\downarrow$"),
    ]
    lines = [r"\begin{tabular}{lcc}", r"\toprule", rf"Metric & {a_label} & {b_label} \\", r"\midrule"]
    for key, title in metrics:
        va, vb = a.get(key, np.nan), b.get(key, np.nan)
        fa = "--" if not np.isfinite(va) else f"{va:.4f}"
        fb = "--" if not np.isfinite(vb) else f"{vb:.4f}"
        lines.append(f"{title} & {fa} & {fb} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines), encoding="utf-8")


def extract_and_save_one(kind: str, label: str, weights_path: str, bundle: Dict[str, np.ndarray], device: torch.device, out_dir: Path) -> Dict[str, np.ndarray]:
    x = torch.as_tensor(bundle["x"], dtype=torch.float32, device=device)
    model = build_hermes(device) if kind.lower() == "hermes" else build_bcauss(device)
    state_dict = load_state_dict_any(weights_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[{label}] missing keys: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
    if unexpected:
        print(f"[{label}] unexpected keys: {unexpected[:10]}{' ...' if len(unexpected) > 10 else ''}")
    model.to(device)
    model.eval()
    phi_t, mu0_t, mu1_t = infer_outputs(model, x)

    payload = {
        "embeddings": ensure_2d(phi_t.detach().cpu().numpy()).astype(np.float32),
        "mu0_hat": ensure_1d(mu0_t).astype(np.float32),
        "mu1_hat": ensure_1d(mu1_t).astype(np.float32),
        "tau_hat": (ensure_1d(mu1_t) - ensure_1d(mu0_t)).astype(np.float32),
    }
    save_npy(out_dir / f"{label.lower()}_embeddings.npy", payload["embeddings"])
    save_npy(out_dir / f"{label.lower()}_mu0_hat.npy", payload["mu0_hat"])
    save_npy(out_dir / f"{label.lower()}_mu1_hat.npy", payload["mu1_hat"])
    save_npy(out_dir / f"{label.lower()}_tau_hat.npy", payload["tau_hat"])
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified extraction + diagnostics pipeline for HERMES vs BCAUSS.")
    parser.add_argument("--rep", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--hermes-label", type=str, default="HERMES")
    parser.add_argument("--bcauss-label", type=str, default="BCAUSS")
    parser.add_argument(
        "--hermes-weights",
        type=str,
        default=r"C:\Users\aless\Desktop\CATE-SNN\CATE_SNN\src\siamese_bcuass\saved_weights\weights_trial_9_rep_8.pth",
    )
    parser.add_argument(
        "--bcauss-weights",
        type=str,
        default=r"C:\Users\aless\Desktop\CATE-SNN\CATE_SNN\saved_weights_reps\bcauss_weights_rep_0008.pth",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/full_diagnostics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.output_dir) / f"rep_{args.rep}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading evaluation bundle for replication {args.rep} ...")
    bundle = load_eval_bundle(args.rep)
    required = {"x", "t", "y"}
    missing = sorted(required - set(bundle.keys()))
    if missing:
        raise RuntimeError(f"Evaluation bundle missing required keys: {missing}")

    # Save shared arrays too
    save_npy(out_dir / "x.npy", bundle["x"])
    save_npy(out_dir / "t.npy", bundle["t"])
    save_npy(out_dir / "y.npy", bundle["y"])
    if "mu0" in bundle:
        save_npy(out_dir / "mu0_true.npy", bundle["mu0"])
    if "mu1" in bundle:
        save_npy(out_dir / "mu1_true.npy", bundle["mu1"])
    if "tau" in bundle:
        save_npy(out_dir / "tau_true.npy", bundle["tau"])

    print(f"Extracting {args.hermes_label} from: {args.hermes_weights}")
    hermes_payload = extract_and_save_one("hermes", args.hermes_label, args.hermes_weights, bundle, device, out_dir)

    print(f"Extracting {args.bcauss_label} from: {args.bcauss_weights}")
    bcauss_payload = extract_and_save_one("bcauss", args.bcauss_label, args.bcauss_weights, bundle, device, out_dir)

    tau_true = bundle.get("tau", None)

    hermes_eval = summarize_model(
        label=args.hermes_label,
        x=ensure_2d(bundle["x"]),
        t=ensure_1d(bundle["t"]),
        y=ensure_1d(bundle["y"]),
        phi=hermes_payload["embeddings"],
        mu0=hermes_payload["mu0_hat"],
        mu1=hermes_payload["mu1_hat"],
        tau_true=None if tau_true is None else ensure_1d(tau_true),
        k=args.k,
    )

    bcauss_eval = summarize_model(
        label=args.bcauss_label,
        x=ensure_2d(bundle["x"]),
        t=ensure_1d(bundle["t"]),
        y=ensure_1d(bundle["y"]),
        phi=bcauss_payload["embeddings"],
        mu0=bcauss_payload["mu0_hat"],
        mu1=bcauss_payload["mu1_hat"],
        tau_true=None if tau_true is None else ensure_1d(tau_true),
        k=args.k,
    )

    rows = [
        {"label": hermes_eval.label, **hermes_eval.metrics},
        {"label": bcauss_eval.label, **bcauss_eval.metrics},
    ]
    deltas = compare_two_models(bcauss_eval, hermes_eval)
    delta_row = {"label": f"delta__{args.hermes_label}_minus_{args.bcauss_label}", **deltas}

    write_csv(out_dir / "raw_metrics.csv", rows)
    write_csv(out_dir / "summary_metrics.csv", [delta_row])
    write_latex_table_existing(out_dir / "table_existing_metrics.tex", rows, labels=(args.bcauss_label, args.hermes_label))
    write_latex_table_reviewer(out_dir / "table_reviewer_metrics.tex", rows, labels=(args.bcauss_label, args.hermes_label))

    payload = {
        "rep": args.rep,
        "hermes_label": args.hermes_label,
        "bcauss_label": args.bcauss_label,
        "rows": rows,
        "delta_row": delta_row,
    }
    (out_dir / "summary_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nSaved arrays:")
    for name in [
        "x.npy", "t.npy", "y.npy", "mu0_true.npy", "mu1_true.npy", "tau_true.npy",
        f"{args.hermes_label.lower()}_embeddings.npy", f"{args.hermes_label.lower()}_mu0_hat.npy", f"{args.hermes_label.lower()}_mu1_hat.npy", f"{args.hermes_label.lower()}_tau_hat.npy",
        f"{args.bcauss_label.lower()}_embeddings.npy", f"{args.bcauss_label.lower()}_mu0_hat.npy", f"{args.bcauss_label.lower()}_mu1_hat.npy", f"{args.bcauss_label.lower()}_tau_hat.npy",
    ]:
        p = out_dir / name
        if p.exists():
            print(f"  - {p}")

    print("\nSaved reports:")
    for name in ["raw_metrics.csv", "summary_metrics.csv", "table_existing_metrics.tex", "table_reviewer_metrics.tex", "summary_metrics.json"]:
        print(f"  - {out_dir / name}")

    print("\nSummary:")
    for row in rows:
        print(f"\n[{row['label']}]")
        for k in sorted(k for k in row.keys() if k != "label"):
            print(f"  {k}: {row[k]}")


if __name__ == "__main__":
    main()
