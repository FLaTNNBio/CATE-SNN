#!/usr/bin/env python3
"""
evaluate_overlap_models_v12.py – Analisi semantica & bilanciamento (no-CLI)
==============================================================================
Script **autonomo** (nessun parametro da linea di comando) per confrontare
BCAUSS e Siamese-BCAUSS sul benchmark IHDP con 999 repliche, includendo
le analisi aggiuntive richieste (MMD locale, overlap, SMD, Δ-test e grafici).

⚠️  Assicùrati di avere:
    • PyTorch, numpy, pandas, scipy, scikit-learn, tqdm
    • matplotlib (opzionale: solo per grafici)

Per cambiare dataset, percorsi pesi, ecc., modifica la dataclass `Config`.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy import stats as st
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

try:
    import torch
except ImportError as err:
    print(f"❌  PyTorch mancante: {err}")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use('TkAgg')  # Imposta un backend compatibile
    import matplotlib.pyplot as plt
    # facoltativo
except ImportError:
    plt = None

# -----------------------------------------------------------------------------
# Moduli di progetto (assumiamo repo root come cwd)
# -----------------------------------------------------------------------------
from src.data_loader import DataLoader  # noqa: E402
from src.models.bcauss import BCAUSS  # noqa: E402
from src.contrastive import DynamicContrastiveCausalDS  # noqa: E402
from src.siamese_bcuass.siamese import SiameseBCAUSS  # noqa: E402

# -----------------------------------------------------------------------------
# Configurazione
# -----------------------------------------------------------------------------

@dataclass
class Config:
    dataset: str = "IHDP"
    replicas: List[int] | None = None        # None → 1-999

    k_neighbors: int = 10                    # per overlap k-NN
    tail_percentile: int = 90                # per tail-IPM

    csv_path: Path = Path("semantic_results_v12.csv")
    write_csv: bool = True

    # Percorsi modelli - usare «{rep}» come placeholder
    weights_bcauss: str = "saved_weights_reps/bcauss_weights_rep_{rep:04d}.pth"
    weights_siamese: str = "src/siamese_bcuass/saved_weights/weights_trial_4_rep_{rep}.pth"

    seed: int = 42
    log_level: str = "INFO"
    make_plots: bool = True                  # disattiva se headless o plt None

    def __post_init__(self):
        if self.replicas is None:
            self.replicas = list(range(1, 1000))

CFG = Config()

# -----------------------------------------------------------------------------
# Metriche
# -----------------------------------------------------------------------------

def calculate_linear_mmd(z0: np.ndarray, z1: np.ndarray) -> float:
    if len(z0) == 0 or len(z1) == 0:
        return np.nan
    return float(np.sum((z0.mean(0) - z1.mean(0)) ** 2))


def _rbf_kernel(A: np.ndarray, B: np.ndarray, gamma: float) -> np.ndarray:
    return np.exp(-gamma * cdist(A, B, "sqeuclidean"))


def calculate_rbf_mmd(z0: np.ndarray, z1: np.ndarray) -> float:
    if len(z0) == 0 or len(z1) == 0:
        return np.nan
    z_all = np.vstack([z0, z1])
    if len(z_all) < 2:
        return np.nan
    dists_sq = cdist(z_all, z_all, "sqeuclidean")
    median_sq = np.median(dists_sq[np.triu_indices_from(dists_sq, k=1)])
    gamma = 1.0 / (median_sq + 1e-9)
    xx = _rbf_kernel(z0, z0, gamma).mean()
    yy = _rbf_kernel(z1, z1, gamma).mean()
    xy = _rbf_kernel(z0, z1, gamma).mean()
    return float(xx + yy - 2 * xy)


def calculate_knn_overlap(z: np.ndarray, t: np.ndarray, k: int, mask: Optional[np.ndarray] = None) -> float:
    if mask is not None:
        z = z[mask]
        t = t[mask]
    if len(z) <= k:
        return np.nan
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(z)
    idx_mat = nbrs.kneighbors(z, return_distance=False)[:, 1:]
    mixed = [(t[idx] != t[i]).any() for i, idx in enumerate(idx_mat)]
    return float(np.mean(mixed))


def calculate_tail_ipm(z0: np.ndarray, z1: np.ndarray, percentile: int, max_pairs: int = 50_000) -> float:
    if len(z0) == 0 or len(z1) == 0:
        return np.nan
    total = len(z0) * len(z1)
    if total > max_pairs:
        rows = np.random.choice(len(z0), size=max_pairs, replace=True)
        cols = np.random.choice(len(z1), size=max_pairs, replace=True)
    else:
        rows, cols = zip(*product(range(len(z0)), range(len(z1))))
    dists = np.linalg.norm(z0[list(rows)] - z1[list(cols)], axis=1)
    thr = np.percentile(dists, percentile)
    tail = dists[dists >= thr]
    return float(np.mean(tail)) if tail.size else np.nan


def distance_ite_correlation(z: np.ndarray, hat_tau: np.ndarray, n_pairs: int = 20_000) -> float:
    if z.shape[0] < 2:
        return np.nan
    idx_i = np.random.randint(0, len(z), n_pairs)
    idx_j = np.random.randint(0, len(z), n_pairs)
    d_z = np.linalg.norm(z[idx_i] - z[idx_j], axis=1)
    d_tau = np.abs(hat_tau[idx_i] - hat_tau[idx_j])
    rho, _ = st.spearmanr(d_z, d_tau)
    return float(rho)


def local_mmd(z: np.ndarray, t: np.ndarray, mask: np.ndarray) -> float:
    z_t0 = z[mask & (t == 0)]
    z_t1 = z[mask & (t == 1)]
    return calculate_rbf_mmd(z_t0, z_t1)


from sklearn.preprocessing import StandardScaler
_scaler = StandardScaler()          # definito una volta, fuori dal loop

def smd(X: np.ndarray, t: np.ndarray, mask: np.ndarray) -> float:
    if X.ndim == 2:                 # (n, d) passato al volo
        Xz = _scaler.fit_transform(X)       # z-score per replica
    else:
        Xz = X
    treated, control = Xz[mask & (t == 1)], Xz[mask & (t == 0)]
    if len(treated) == 0 or len(control) == 0:
        return np.nan
    sd_pool = np.sqrt((treated.var(0) + control.var(0)) / 2)
    return float(np.nanmax(np.abs(treated.mean(0) - control.mean(0)) / (sd_pool + 1e-9)))


# Questa è la formula SMD standard, che ora applichiamo a 'z'.
def smd_latent(z: np.ndarray, t: np.ndarray, mask: np.ndarray) -> float:
    """Calcola la Standardized Mean Difference (SMD) nello spazio latente 'z'."""
    # Filtra i dati in base alla maschera (es. 'simili' o 'dissimili')
    z_masked = z[mask]
    t_masked = t[mask]

    treated = z_masked[t_masked == 1]
    control = z_masked[t_masked == 0]

    if len(treated) == 0 or len(control) == 0:
        return np.nan

    # Calcola la differenza delle medie
    mean_diff = np.abs(treated.mean(0) - control.mean(0))

    # Calcola la deviazione standard "pooled"
    sd_pool = np.sqrt((treated.var(0) + control.var(0)) / 2)

    # Calcola la SMD per ogni dimensione e restituisce la peggiore (massima)
    # Aggiungiamo 1e-9 per evitare divisioni per zero
    smd_values = mean_diff / (sd_pool + 1e-9)

    return float(np.nanmax(smd_values))

# -----------------------------------------------------------------------------
# Modelli e predizioni
# -----------------------------------------------------------------------------

def get_model(model_type: str, weight_path: Path, input_dim: int):
    if not weight_path.exists():
        logging.warning("Pesi %s non trovati – replica saltata.", weight_path)
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_type == "bcauss":
        model = BCAUSS(input_dim=input_dim)
    elif model_type == "siamese":
        base = BCAUSS(input_dim=input_dim)
        model = SiameseBCAUSS(base_model=base, ds_class=DynamicContrastiveCausalDS)
    else:
        raise ValueError("Tipo modello non supportato")
    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model


def predict(model, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    with torch.no_grad():
        x_t = torch.from_numpy(X).float().to(device)
        if isinstance(model, SiameseBCAUSS):
            mu, z = model.base.mu_and_embedding(x_t)
            tau_hat = (mu[:, 1] - mu[:, 0]).cpu().numpy()
            return z.cpu().numpy(), tau_hat
        else:
            tau_hat = model.predict_ite(X)
            _, z = model.mu_and_embedding(x_t)
            return z.cpu().numpy(), tau_hat


def _predict_mu(m,X):
    """Restituisce matrice (n,2) con E[Y|X,T=0], E[Y|X,T=1]."""
    dev=next(m.parameters()).device
    with torch.no_grad():
        xt=torch.from_numpy(X).float().to(dev)
        if isinstance(m,SiameseBCAUSS):
            mu,_=m.base.mu_and_embedding(xt)
        else:
            mu,_=m.mu_and_embedding(xt)
    return mu.cpu().numpy()

# -----------------------------------------------------------------------------
# Replica
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------
# Modelli e predizioni  (tutto identico a prima: get_model(), predict(), _predict_mu() )
# -----------------------------------------------------------------

def get_model(model_type: str, weight_path: Path, input_dim: int):
    if not weight_path.exists():
        logging.warning("Pesi %s non trovati – replica %d saltata.", weight_path, weight_path)
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_type == "bcauss":
        model = BCAUSS(input_dim=input_dim)
    elif model_type == "siamese":
        base = BCAUSS(input_dim=input_dim)
        model = SiameseBCAUSS(base_model=base, ds_class=DynamicContrastiveCausalDS)
    else:
        raise ValueError("Tipo modello non supportato")
    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model


def predict(model, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Restituisce (embedding z, stima ITE)."""
    device = next(model.parameters()).device
    with torch.no_grad():
        xt = torch.from_numpy(X).float().to(device)
        if isinstance(model, SiameseBCAUSS):           # Siamese → mu and embedding sul backbone
            mu, z = model.base.mu_and_embedding(xt)
        else:                                          # BCAUSS “liscio”
            mu, z = model.mu_and_embedding(xt)

        tau_hat = (mu[:, 1] - mu[:, 0]).cpu().numpy()
        return z.cpu().numpy(), tau_hat


# -----------------------------------------------------------------
# Replica (NUOVA versione robusta allo y_scaler mancante)
# -----------------------------------------------------------------

class _IdentityScaler:
    """Fallback: applica f(x)=x ed evita AttributeError se lo scaler manca nei pesi."""
    def transform(self,  x): return x
    def inverse_transform(self, x): return x


@torch.no_grad()
def evaluate_replica(
        rep: int,
        X: np.ndarray,
        t: np.ndarray,
        y_obs: np.ndarray,
        scaler_y: Optional[StandardScaler],        # può essere None
        tau_true: Optional[np.ndarray]
) -> Dict[str, Any]:

    res: Dict[str, Any] = {"rep": rep}

    # ---------------- carica modelli ----------------
    w_b = Path(CFG.weights_bcauss.format(rep=rep))
    w_s = Path(CFG.weights_siamese.format(rep=rep))

    model_b = get_model("bcauss",  w_b, X.shape[1])
    model_s = get_model("siamese", w_s, X.shape[1])

    # ---------------- y_scaler safety ----------------
    for m in (model_b, model_s):
        if m is None:
            continue

        # (1) Inserisce _IdentityScaler dove lo scaler manca
        if getattr(m, "y_scaler", None) is None:
            m.y_scaler = _IdentityScaler()
        if hasattr(m, "base") and getattr(m.base, "y_scaler", None) is None:
            m.base.y_scaler = _IdentityScaler()

        # (2) Se l’utente ha fornito uno scaler reale, lo imposta (sovrascrive l’identità)
        if scaler_y is not None:
            m.y_scaler = scaler_y
            if hasattr(m, "base"):
                m.base.y_scaler = scaler_y

    # ---------------- forward: embedding + ITE ----------------
    z_b, tau_b = (np.nan, np.nan) if model_b is None else predict(model_b, X)
    z_s, tau_s = (np.nan, np.nan) if model_s is None else predict(model_s, X)

    # ---------------- metriche sullo spazio latente ----------------
    z0_b, z1_b = (np.empty((0, X.shape[1])),) * 2 if model_b is None else (z_b[t == 0], z_b[t == 1])
    z0_s, z1_s = (np.empty((0, X.shape[1])),) * 2 if model_s is None else (z_s[t == 0], z_s[t == 1])

    # soglia 20° percentile su |τ̂_B| per simili/dissimili
    if isinstance(tau_b, np.ndarray):
        thresh     = np.percentile(np.abs(tau_b), 20)
        mask_sim   = np.abs(tau_b) <= thresh
    else:
        mask_sim   = np.full(len(t), False)
    mask_dsim = ~mask_sim

    # ---------- BCAUSS ----------
    if model_b is not None:
        res.update({
            "lin_mmd_b":   calculate_linear_mmd(z0_b, z1_b),
            "rbf_mmd_b":   calculate_rbf_mmd(z0_b, z1_b),
            "knn_overlap_b": calculate_knn_overlap(z_b, t, CFG.k_neighbors),
            "tail_ipm_b":  calculate_tail_ipm(z0_b, z1_b, CFG.tail_percentile),
            "rho_b":       distance_ite_correlation(z_b, tau_b),
            "mmd_sim_b":   local_mmd(z_b, t, mask_sim),
            "mmd_dsim_b":  local_mmd(z_b, t, mask_dsim),
            "smd_latent_b_sim": smd_latent(z_b, t, mask_sim),
            "smd_latent_b_dsim": smd_latent(z_b, t, mask_dsim),
        })
    else:
        for k in ["lin_mmd_b", "rbf_mmd_b", "knn_overlap_b", "tail_ipm_b",
                  "rho_b", "mmd_sim_b", "mmd_dsim_b",
                  "smd_latent_b_sim", "smd_latent_b_dsim"]:
            res[k] = np.nan

    # ---------- Siamese ----------
    if model_s is not None:
        res.update({
            "lin_mmd_s":   calculate_linear_mmd(z0_s, z1_s),
            "rbf_mmd_s":   calculate_rbf_mmd(z0_s, z1_s),
            "knn_overlap_s": calculate_knn_overlap(z_s, t, CFG.k_neighbors),
            "tail_ipm_s":  calculate_tail_ipm(z0_s, z1_s, CFG.tail_percentile),
            "rho_s":       distance_ite_correlation(z_s, tau_s),
            "mmd_sim_s":   local_mmd(z_s, t, mask_sim),
            "mmd_dsim_s":  local_mmd(z_s, t, mask_dsim),
            "smd_latent_s_sim": smd_latent(z_s, t, mask_sim),
            "smd_latent_s_dsim": smd_latent(z_s, t, mask_dsim),
        })
    else:
        for k in ["lin_mmd_s", "rbf_mmd_s", "knn_overlap_s", "tail_ipm_s",
                  "rho_s", "mmd_sim_s", "mmd_dsim_s",
                  "smd_latent_s_sim", "smd_latent_s_dsim"]:
            res[k] = np.nan

    # ---------- MSE factual ----------
    if model_b is not None:
        mu_b     = _predict_mu(model_b, X)
        y_hat_b  = mu_b[np.arange(len(t)), t]
        res["mse_factual_b"] = float(np.mean((y_hat_b - y_obs.squeeze()) ** 2))
    else:
        res["mse_factual_b"] = np.nan

    if model_s is not None:
        mu_s     = _predict_mu(model_s, X)
        y_hat_s  = mu_s[np.arange(len(t)), t]
        res["mse_factual_s"] = float(np.mean((y_hat_s - y_obs.squeeze()) ** 2))
    else:
        res["mse_factual_s"] = np.nan

    # ---------- Overlap & SMD su gruppi (embedding BCAUSS) ----------
    res.update({
        "ov_sim":  calculate_knn_overlap(z_b if isinstance(z_b, np.ndarray) else np.empty((0, 1)), t,
                                         CFG.k_neighbors, mask_sim),
        "ov_dsim": calculate_knn_overlap(z_b if isinstance(z_b, np.ndarray) else np.empty((0, 1)), t,
                                         CFG.k_neighbors, mask_dsim),
        "smd_sim": smd(X, t, mask_sim),
        "smd_dsim": smd(X, t, mask_dsim),
    })

    # ---------- PEHE (se ground-truth disponibile) ----------
    if tau_true is not None and isinstance(tau_b, np.ndarray):
        res["pehe_b"] = float(np.mean((tau_true - tau_b) ** 2))
        res["pehe_s"] = float(np.mean((tau_true - tau_s) ** 2))
    else:
        res["pehe_b"] = res["pehe_s"] = np.nan

    return res


# -----------------------------------------------------------------------------
# Logging helper
# -----------------------------------------------------------------------------

def setup_logging():
    logging.basicConfig(level=getattr(logging, CFG.log_level.upper()), format="%(levelname)s | %(message)s")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    setup_logging()
    np.random.seed(CFG.seed)

    logging.info("Config: %s", {k: v for k, v in asdict(CFG).items() if not k.startswith("weights")})

    # Carica dataset IHDP
    loader = DataLoader.get_loader(CFG.dataset)
    X_te, T_te, YF_te, YCF_te, MU0_te, MU1_te, *_ = loader.load()
    # X_tr: (n, d, rep)

    input_dim = X_te.shape[1]
    results: List[Dict[str, Any]] = []

    for rep in tqdm(CFG.replicas, desc="Repliche", unit="rep"):
        X = X_te[:, :, rep].astype(np.float32)
        t = T_te[:, rep].astype(int)
        y = YF_te[:, rep].astype(np.float32).reshape(-1, 1)
        tau_true = (MU1_te[:, rep] - MU0_te[:, rep]).astype(np.float32) if MU0_te is not None else None

        scaler_y = StandardScaler().fit(y)
        y = YF_te[:, rep].astype(np.float32).reshape(-1, 1)
        res = evaluate_replica(rep, X, t, y, scaler_y, tau_true)
        results.append(res)

    df = pd.DataFrame(results)

    # -------------------------------------------------- sintesi quick console
    logging.info("\n==== Sintesi metriche (medie su repliche valide) ====")
    for suffix, name in [("b", "BCAUSS"), ("s", "Siamese")]:
        rho_col = f"rho_{suffix}"
        mmd_col = f"rbf_mmd_{suffix}"
        valid = df[[rho_col, mmd_col]].dropna()
        if valid.empty:
            continue
        logging.info("%s: ρ=%.4f   MMD=%.4f  (n=%d)", name, valid[rho_col].mean(), valid[mmd_col].mean(), len(valid))

    # Δ-test rho e MMD
    paired_rho = df[["rho_b", "rho_s"]].dropna()
    if len(paired_rho) > 1:
        t_stat, p_val = st.ttest_rel(paired_rho["rho_s"], paired_rho["rho_b"])
        logging.info("T-test ρ_s vs ρ_b: t=%.2f, p=% .3g", t_stat, p_val)
    paired_mmd = df[["rbf_mmd_b", "rbf_mmd_s"]].dropna()
    if len(paired_mmd) > 1:
        t_stat, p_val = st.ttest_rel(paired_mmd["rbf_mmd_s"], paired_mmd["rbf_mmd_b"])
        logging.info("T-test MMD_s vs MMD_b: t=%.2f, p=% .3g", t_stat, p_val)

    # -------------------------------------------------- grafici opzionali
    if CFG.make_plots and plt is not None:
        # Box-plot MMD locale
        box_data = [df["mmd_sim_b"].dropna(), df["mmd_sim_s"].dropna(), df["mmd_dsim_b"].dropna(), df["mmd_dsim_s"].dropna()]
        plt.figure(figsize=(6, 4))
        plt.boxplot(box_data, labels=["sim-B", "sim-S", "dsim-B", "dsim-S"])
        plt.title("MMD locale (RBF)")
        plt.ylabel("MMD")
        plt.tight_layout()
        plt.show()

        # Scatter ΔMMD vs ΔPEHE se PEHE presente
        if df["pehe_b"].notna().any():
            df["delta_mmd"] = df["rbf_mmd_s"] - df["rbf_mmd_b"]
            df["delta_pehe"] = df["pehe_s"] - df["pehe_b"]
            rho, p = st.spearmanr(df["delta_mmd"], df["delta_pehe"], nan_policy="omit")
            plt.figure(figsize=(5, 4))
            plt.scatter(df["delta_mmd"], df["delta_pehe"], s=10, alpha=.6)
            plt.axvline(0, ls="--", lw=.8)
            plt.axhline(0, ls="--", lw=.8)
            plt.xlabel("ΔMMD (S-B)")
            plt.ylabel("ΔPEHE (S-B)")
            plt.title(f"ΔMMD vs ΔPEHE  ρ={rho:.2f}, p={p:.3g}")
            plt.tight_layout()
            plt.show()

    # ----------------- CHECK OVERLAP & SMD, salvati su testo -----------------
    for g in ["sim", "dsim"]:
        ov_mean = df[f"ov_{g}"].mean()
        ov_ci = df[f"ov_{g}"].quantile([0.025, 0.975]).values
        smd_mean = df[f"smd_{g}"].mean()
        smd_ci = df[f"smd_{g}"].quantile([0.025, 0.975]).values
        logging.info(
            "Gruppo %-4s | overlap=%.3f [%.3f, %.3f] | worst-SMD=%.3f [%.3f, %.3f]",
            g, ov_mean, *ov_ci, smd_mean, *smd_ci
        )
    # (facoltativo) salva un histogram PNG dei worst-SMD
    if CFG.make_plots and plt is not None:
        plt.figure(figsize=(5, 3))
        plt.hist(df["smd_sim"].dropna(), bins=30, alpha=.6, label="simili")
        plt.hist(df["smd_dsim"].dropna(), bins=30, alpha=.6, label="dissimili")
        plt.axvline(0.1, color="red", ls="--")
        plt.xlabel("worst-SMD per replica")
        plt.legend();
        plt.tight_layout()
        plt.savefig("hist_worstSMD.png", dpi=300);
        plt.close()
        logging.info("Saved hist_worstSMD.png")


    # -------------------------------------------------- salva CSV
    if CFG.write_csv:
        CFG.csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(CFG.csv_path, index=False)
        logging.info("Risultati salvati in %s", CFG.csv_path.resolve())


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.error("Interrotto dall'utente – uscita.")
