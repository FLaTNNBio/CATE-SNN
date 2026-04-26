import os
import csv
import random
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import yaml
import optuna

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.models.bcauss import BCAUSS
from src.metrics import eps_ATE_diff, PEHE_with_ite
from src.contrastive import ContrastiveCausalDS
from src.siamese_bcuass.siamese import SiameseBCAUSS


# =============================================================================
# CONFIG BASE
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOGGER = logging.getLogger("ACIC_SETTING_BASED_ALL_IN_ONE")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
optuna.logging.set_verbosity(optuna.logging.INFO)

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
        "lambda_ctr": None,
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


# =============================================================================
# UTILITIES
# =============================================================================
def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_csv_row(csv_path: Path, row: List):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def save_yaml(path: Path, data: Dict):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def load_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# =============================================================================
# DATA LOADING
# =============================================================================
def read_table_auto(file_path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(file_path)
    except Exception:
        return pd.read_csv(file_path, sep=None, engine="python")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def find_first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_covariates(x_path: Path) -> np.ndarray:
    X_df = read_table_auto(x_path)
    X_df.columns = [str(c).strip() for c in X_df.columns]

    for col in X_df.columns:
        X_df[col] = pd.to_numeric(X_df[col], errors="ignore")

    cat_cols = X_df.select_dtypes(include=["object", "category"]).columns.tolist()

    if len(cat_cols) > 0:
        LOGGER.info(f"Categorical columns found in X: {cat_cols}")
        X_df = pd.get_dummies(X_df, columns=cat_cols, drop_first=False)

    bool_cols = X_df.select_dtypes(include=["bool"]).columns.tolist()
    if len(bool_cols) > 0:
        X_df[bool_cols] = X_df[bool_cols].astype(np.int32)

    X_df = X_df.astype(np.float32)

    LOGGER.info(f"Covariates shape after encoding: {X_df.shape}")
    return X_df.values


def parse_simulation_dataframe(df: pd.DataFrame, sim_file: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Restituisce:
      T
      Y factual
      true_ite

    Formato atteso:
      z, y0, y1, mu0, mu1
    """
    df = normalize_columns(df)

    z_col = find_first_existing_column(df, ["z", "t", "a", "treatment"])
    y0_col = find_first_existing_column(df, ["y0", "y_0"])
    y1_col = find_first_existing_column(df, ["y1", "y_1"])
    mu0_col = find_first_existing_column(df, ["mu0", "mu_0"])
    mu1_col = find_first_existing_column(df, ["mu1", "mu_1"])
    y_col = find_first_existing_column(df, ["y", "yf", "outcome"])

    if z_col is None:
        raise ValueError(
            f"Nel file {sim_file} manca la colonna trattamento. "
            f"Colonne trovate: {list(df.columns)}"
        )

    if y0_col is not None and y1_col is not None:
        T = df[z_col].values.astype(np.float32)
        y0 = df[y0_col].values.astype(np.float32)
        y1 = df[y1_col].values.astype(np.float32)

        Y = T * y1 + (1.0 - T) * y0

        if mu0_col is not None and mu1_col is not None:
            mu0 = df[mu0_col].values.astype(np.float32)
            mu1 = df[mu1_col].values.astype(np.float32)
            true_ite = mu1 - mu0
        else:
            true_ite = y1 - y0

        return T, Y, true_ite

    if y_col is not None:
        raise ValueError(
            f"Il file {sim_file} contiene solo z e y, ma non y0/y1. "
            f"Questo non basta per valutazione CATE con PEHE."
        )

    raise ValueError(
        f"Formato non riconosciuto per {sim_file}. "
        f"Colonne trovate: {list(df.columns)}"
    )


def discover_acic_by_setting(acic_root: str) -> Tuple[Path, Dict[str, List[Path]]]:
    """
    Struttura attesa:
      acic_root/
        x o x.csv
        1/
        2/
        ...
        77/

    Restituisce:
      x_path
      settings_dict: {setting_id: [rep_file1, rep_file2, ...]}
    """
    root = Path(acic_root)
    if not root.exists():
        raise FileNotFoundError(f"Cartella ACIC non trovata: {root}")

    x_candidates = [root / "x", root / "x.csv", root / "X", root / "X.csv"]
    x_path = None
    for cand in x_candidates:
        if cand.exists() and cand.is_file():
            x_path = cand
            break

    if x_path is None:
        raise FileNotFoundError(
            f"Non ho trovato il file covariate nella root {root}. "
            f"Attesi: x, x.csv, X, X.csv"
        )

    setting_dirs = [p for p in root.iterdir() if p.is_dir() and p.name.isdigit()]
    setting_dirs = sorted(setting_dirs, key=lambda p: int(p.name))

    if len(setting_dirs) == 0:
        raise RuntimeError(
            f"Nessuna cartella numerata trovata in {root}. "
            f"Mi aspettavo cartelle tipo 1, 2, 3, ..."
        )

    allowed_suffixes = {"", ".csv", ".txt", ".tsv"}
    settings = {}

    for sdir in setting_dirs:
        rep_files = [
            f for f in sorted(sdir.iterdir(), key=lambda p: p.name)
            if f.is_file() and f.suffix.lower() in allowed_suffixes
        ]
        if len(rep_files) > 0:
            settings[sdir.name] = rep_files

    if len(settings) == 0:
        raise RuntimeError("Nessun file di simulazione trovato nelle cartelle numerate.")

    return x_path, settings


def prepare_acic_grouped_data(
    acic_root: str,
    seed: int,
) -> Tuple[Dict[str, List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]], int]:
    """
    Restituisce:
      grouped_data[setting_id] = lista di repliche, ciascuna come:
         (X_tr, T_tr, Y_tr, X_te, true_ite_te)
      input_dim
    """
    x_path, settings = discover_acic_by_setting(acic_root)

    LOGGER.info(f"ACIC root: {acic_root}")
    LOGGER.info(f"Covariates file: {x_path}")
    LOGGER.info(f"Number of settings found: {len(settings)}")

    X = load_covariates(x_path)
    input_dim = X.shape[1]

    grouped_data = {}

    for setting_id, rep_files in settings.items():
        LOGGER.info(f"[LOAD] setting={setting_id} | n_rep_files={len(rep_files)}")
        grouped_data[setting_id] = []

        for rep_idx, sim_file in enumerate(rep_files):
            df = read_table_auto(sim_file)
            T, Y, true_ite = parse_simulation_dataframe(df, sim_file)

            if len(X) != len(T):
                raise ValueError(
                    f"Mismatch righe tra X ({len(X)}) e {sim_file} ({len(T)})."
                )

            X_tr, X_te, T_tr, T_te, Y_tr, Y_te, ite_tr, ite_te = train_test_split(
                X,
                T,
                Y,
                true_ite,
                test_size=0.2,
                random_state=seed + rep_idx,
            )

            scaler_x = StandardScaler()
            X_tr = scaler_x.fit_transform(X_tr).astype(np.float32)
            X_te = scaler_x.transform(X_te).astype(np.float32)

            T_tr = T_tr.reshape(-1, 1)
            Y_tr = Y_tr.reshape(-1, 1)

            grouped_data[setting_id].append((X_tr, T_tr, Y_tr, X_te, ite_te))

    return grouped_data, input_dim


# =============================================================================
# SPLIT SETTINGS
# =============================================================================
def split_settings(
    grouped_data: Dict[str, List],
    seed: int,
    n_tuning_settings: int,
) -> Tuple[List[str], List[str]]:
    all_setting_ids = sorted([str(k) for k in grouped_data.keys()], key=lambda x: int(x))

    rng = np.random.default_rng(seed)
    shuffled = all_setting_ids.copy()
    rng.shuffle(shuffled)

    n_tuning_settings = min(n_tuning_settings, len(shuffled) - 1)

    tuning_settings = [str(x) for x in shuffled[:n_tuning_settings]]
    final_settings = [str(x) for x in shuffled[n_tuning_settings:]]

    return tuning_settings, final_settings

# =============================================================================
# MODEL BUILDING / TRAINING
# =============================================================================
def build_base_model(input_dim: int, warmup_epochs_base: int, X=None, T=None, Y=None):
    base_model = BCAUSS(input_dim=input_dim)

    if warmup_epochs_base > 0 and X is not None:
        base_model.fit(X, T, Y, epochs=warmup_epochs_base)

    return base_model


def build_siamese_params(params: Dict, pairing_strategy: str) -> Dict:
    return {
        "ds_class": ContrastiveCausalDS,
        "margin": params["margin"],
        "lambda_ctr": params["lambda_ctr"],
        "batch_size": params["batch_size"],
        "lr": params["lr"],
        "epochs": params["epochs"],
        "clip_norm": params["clip_norm"],
        "use_amp": params["use_amp"],
        "val_split": params["val_split"],
        "update_ite_freq": params["update_ite_freq"],
        "warmup_epochs_base": 0,
        "pairing_strategy": pairing_strategy,
    }


def train_and_eval_one_rep(
    rep_global_idx: int,
    rep_data,
    input_dim: int,
    params: Dict,
    use_siamese: bool,
    pairing_strategy: str = "dynamic_ite",
):
    set_seed(params["seed"] + rep_global_idx)

    Xtr, Ttr, Ytr, Xte, true_ite_te = rep_data

    base = build_base_model(
        input_dim=input_dim,
        warmup_epochs_base=params["warmup_epochs_base"],
        X=Xtr,
        T=Ttr,
        Y=Ytr,
    ).to(params["device"])

    if not use_siamese:
        model = base
        model.fit(Xtr, Ttr, Ytr, epochs=params["epochs"])
    else:
        siamese_params = build_siamese_params(params, pairing_strategy)
        model = SiameseBCAUSS(base_model=base, **siamese_params).to(params["device"])
        model.fit(Xtr, Ttr, Ytr)

    with torch.no_grad():
        pred_ite = model.predict_ite(Xte).reshape(-1)

    pehe = PEHE_with_ite(true_ite_te, pred_ite, sqrt=True)
    ate_err = eps_ATE_diff(true_ite_te, pred_ite)

    return float(pehe), float(ate_err)


# =============================================================================
# EVALUATION HELPERS
# =============================================================================
def aggregate_metrics(pehes: List[float], ate_errs: List[float]) -> Dict[str, float]:
    return {
        "pehe_mean": float(np.nanmean(pehes)),
        "pehe_std": float(np.nanstd(pehes)),
        "ate_mean": float(np.nanmean(ate_errs)),
        "ate_std": float(np.nanstd(ate_errs)),
    }


def evaluate_on_settings(
    grouped_data: Dict[str, List],
    setting_ids: List[str],
    input_dim: int,
    params: Dict,
    use_siamese: bool,
    pairing_strategy: str,
    max_reps_per_setting: Optional[int] = None,
    raw_csv_path: Optional[Path] = None,
    strategy_name: str = "model",
):
    """
    Aggregazione corretta:
      1) media sulle repliche di ogni setting
      2) media sui settings
    """
    setting_pehes = []
    setting_ates = []

    global_rep_counter = 0

    for setting_id in setting_ids:
        reps = grouped_data[setting_id]
        if max_reps_per_setting is not None:
            reps = reps[:max_reps_per_setting]

        rep_pehes = []
        rep_ates = []

        LOGGER.info(
            f"[{strategy_name}] setting={setting_id} | n_reps_used={len(reps)}"
        )

        for rep_idx, rep_data in enumerate(reps):
            pehe, ate_err = train_and_eval_one_rep(
                rep_global_idx=global_rep_counter,
                rep_data=rep_data,
                input_dim=input_dim,
                params=params,
                use_siamese=use_siamese,
                pairing_strategy=pairing_strategy,
            )
            global_rep_counter += 1

            rep_pehes.append(pehe)
            rep_ates.append(ate_err)

            LOGGER.info(
                f"[{strategy_name}] setting={setting_id} | rep {rep_idx+1}/{len(reps)} | "
                f"PEHE={pehe:.6f} | eps_ATE={ate_err:.6f}"
            )

            if raw_csv_path is not None:
                save_csv_row(
                    raw_csv_path,
                    [
                        strategy_name,
                        setting_id,
                        rep_idx + 1,
                        f"{pehe:.6f}",
                        f"{ate_err:.6f}",
                    ],
                )

        setting_pehe = float(np.mean(rep_pehes))
        setting_ate = float(np.mean(rep_ates))

        setting_pehes.append(setting_pehe)
        setting_ates.append(setting_ate)

        LOGGER.info(
            f"[{strategy_name}] setting={setting_id} | "
            f"SETTING_MEAN_PEHE={setting_pehe:.6f} | "
            f"SETTING_MEAN_eps_ATE={setting_ate:.6f}"
        )

    return aggregate_metrics(setting_pehes, setting_ates)


# =============================================================================
# OPTUNA
# =============================================================================
def suggest_params(trial: optuna.trial.Trial, base_cfg: Dict) -> Dict:
    return {
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "margin": trial.suggest_float("margin", 0.1, 1.5),
        "lambda_ctr": trial.suggest_float("lambda_ctr", 1e-3, 5.0, log=True),
        "epochs": trial.suggest_categorical("epochs", [80, 100, 150, 200]),
        "warmup_epochs_base": trial.suggest_categorical("warmup_epochs_base", [0, 10, 20, 30]),
        "update_ite_freq": trial.suggest_categorical("update_ite_freq", [1, 2, 5, 10]),
        "clip_norm": trial.suggest_categorical("clip_norm", [1.0, 5.0, 10.0]),
        "val_split": base_cfg["val_split"],
        "use_amp": base_cfg["use_amp"],
        "seed": base_cfg["seed"],
        "device": base_cfg["device"],
    }


def run_optuna_tuning(
    grouped_data: Dict[str, List],
    tuning_setting_ids: List[str],
    input_dim: int,
    reps_per_setting_tune: int,
    n_trials: int,
    out_dir: Path,
    base_cfg: Dict,
):
    ensure_dir(out_dir)

    trials_csv = out_dir / "optuna_trials.csv"
    with open(trials_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "trial",
            "value",
            "lr",
            "batch_size",
            "margin",
            "lambda_ctr",
            "epochs",
            "warmup_epochs_base",
            "update_ite_freq",
            "clip_norm",
        ])

    def objective(trial: optuna.trial.Trial):
        params = suggest_params(trial, base_cfg)

        try:
            metrics = evaluate_on_settings(
                grouped_data=grouped_data,
                setting_ids=tuning_setting_ids,
                input_dim=input_dim,
                params=params,
                use_siamese=True,
                pairing_strategy="dynamic_ite",
                max_reps_per_setting=reps_per_setting_tune,
                raw_csv_path=None,
                strategy_name=f"trial_{trial.number}",
            )
            score = metrics["pehe_mean"]

        except Exception as e:
            LOGGER.exception(f"Trial {trial.number} failed: {e}")
            score = float("inf")

        save_csv_row(
            trials_csv,
            [
                trial.number,
                score,
                params["lr"],
                params["batch_size"],
                params["margin"],
                params["lambda_ctr"],
                params["epochs"],
                params["warmup_epochs_base"],
                params["update_ite_freq"],
                params["clip_norm"],
            ],
        )

        return score

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = suggest_params(study.best_trial, base_cfg)
    best_bundle = {
        "best_value": float(study.best_value),
        "best_params": best_params,
        "n_trials": n_trials,
        "tuning_settings": tuning_setting_ids,
        "reps_per_setting_tune": reps_per_setting_tune,
    }

    save_yaml(out_dir / "best_params.yaml", best_bundle)
    study.trials_dataframe().to_csv(out_dir / "optuna_trials_dataframe.csv", index=False)

    LOGGER.info(f"Best score: {study.best_value:.6f}")
    LOGGER.info(f"Best params: {study.best_params}")

    return best_bundle


# =============================================================================
# FINAL / FULL / ABLATION
# =============================================================================
def run_final_evaluation(
    grouped_data: Dict[str, List],
    setting_ids: List[str],
    input_dim: int,
    params: Dict,
    out_dir: Path,
    reps_per_setting_eval: Optional[int],
    tag: str = "final",
):
    ensure_dir(out_dir)

    raw_csv = out_dir / f"{tag}_raw.csv"
    summary_csv = out_dir / f"{tag}_summary.csv"

    with open(raw_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["strategy", "setting", "rep", "pehe", "eps_ate"])

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "strategy",
            "n_settings",
            "reps_per_setting",
            "pehe_mean",
            "pehe_std",
            "ate_mean",
            "ate_std",
        ])

    metrics = evaluate_on_settings(
        grouped_data=grouped_data,
        setting_ids=setting_ids,
        input_dim=input_dim,
        params=params,
        use_siamese=True,
        pairing_strategy="dynamic_ite",
        max_reps_per_setting=reps_per_setting_eval,
        raw_csv_path=raw_csv,
        strategy_name="HERMES_Dynamic_ITE",
    )

    save_csv_row(
        summary_csv,
        [
            "HERMES_Dynamic_ITE",
            len(setting_ids),
            "ALL" if reps_per_setting_eval is None else reps_per_setting_eval,
            f"{metrics['pehe_mean']:.6f}",
            f"{metrics['pehe_std']:.6f}",
            f"{metrics['ate_mean']:.6f}",
            f"{metrics['ate_std']:.6f}",
        ],
    )

    LOGGER.info(
        f"[{tag.upper()}] PEHE={metrics['pehe_mean']:.6f} ± {metrics['pehe_std']:.6f} | "
        f"eps_ATE={metrics['ate_mean']:.6f} ± {metrics['ate_std']:.6f}"
    )

    return metrics


def run_ablation(
    grouped_data: Dict[str, List],
    setting_ids: List[str],
    input_dim: int,
    params: Dict,
    out_dir: Path,
    reps_per_setting_eval: Optional[int],
):
    ensure_dir(out_dir)

    raw_csv = out_dir / "ablation_raw.csv"
    summary_csv = out_dir / "ablation_summary.csv"

    with open(raw_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["strategy", "setting", "rep", "pehe", "eps_ate"])

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "strategy",
            "n_settings",
            "reps_per_setting",
            "margin",
            "lambda_ctr",
            "lr",
            "batch_size",
            "epochs",
            "warmup_epochs_base",
            "update_ite_freq",
            "clip_norm",
            "pehe_mean",
            "pehe_std",
            "ate_mean",
            "ate_std",
        ])

    for strategy in ABLATIONS:
        LOGGER.info("=" * 80)
        LOGGER.info(f"Starting ablation: {strategy['name']}")
        LOGGER.info("=" * 80)

        strategy_params = dict(params)

        if strategy["use_siamese"] and strategy["lambda_ctr"] is not None:
            strategy_params["lambda_ctr"] = strategy["lambda_ctr"]

        pairing_strategy = strategy.get("pairing_strategy", "dynamic_ite")

        metrics = evaluate_on_settings(
            grouped_data=grouped_data,
            setting_ids=setting_ids,
            input_dim=input_dim,
            params=strategy_params,
            use_siamese=strategy["use_siamese"],
            pairing_strategy=pairing_strategy,
            max_reps_per_setting=reps_per_setting_eval,
            raw_csv_path=raw_csv,
            strategy_name=strategy["name"],
        )

        save_csv_row(
            summary_csv,
            [
                strategy["name"],
                len(setting_ids),
                "ALL" if reps_per_setting_eval is None else reps_per_setting_eval,
                strategy_params["margin"] if strategy["use_siamese"] else "",
                strategy_params["lambda_ctr"] if strategy["use_siamese"] else "",
                strategy_params["lr"] if strategy["use_siamese"] else "",
                strategy_params["batch_size"] if strategy["use_siamese"] else "",
                strategy_params["epochs"] if strategy["use_siamese"] else "",
                strategy_params["warmup_epochs_base"] if strategy["use_siamese"] else "",
                strategy_params["update_ite_freq"] if strategy["use_siamese"] else "",
                strategy_params["clip_norm"] if strategy["use_siamese"] else "",
                f"{metrics['pehe_mean']:.6f}",
                f"{metrics['pehe_std']:.6f}",
                f"{metrics['ate_mean']:.6f}",
                f"{metrics['ate_std']:.6f}",
            ],
        )

        LOGGER.info(
            f"[{strategy['name']}] PEHE={metrics['pehe_mean']:.6f} ± {metrics['pehe_std']:.6f} | "
            f"eps_ATE={metrics['ate_mean']:.6f} ± {metrics['ate_std']:.6f}"
        )


# =============================================================================
# CONFIG
# =============================================================================
def default_config() -> Dict:
    return {
        # mode:
        # "tune"       -> Optuna su tuning settings
        # "final"      -> final eval sui final settings
        # "ablation"   -> ablation sui final settings
        # "full"       -> eval del modello migliore su TUTTI i settings
        # "all"        -> esegue tune + final + ablation + full
        "mode": "all",

        # dataset
        "acic_root": r"C:\Users\aless\Desktop\CATE-SNN\CATE_SNN\src\acic\acic",
        "seed": 42,
        "test_size": 0.2,

        # split per settings
        "n_tuning_settings": 20,

        # quante repliche usare
        "reps_per_setting_tune": 20,     # per Optuna
        "reps_per_setting_final": None,  # None = tutte
        "reps_per_setting_ablation": None,  # None = tutte
        "reps_per_setting_full": None,   # None = tutte

        # optuna
        "optuna_trials": 50,

        # output
        "output_root": "acic_outputs_setting_based",

        # training defaults
        "device": DEVICE,
        "use_amp": True if DEVICE == "cuda" else False,
        "val_split": 0.2,

        # best params
        "best_params_file": None,
    }


# =============================================================================
# MAIN
# =============================================================================
def main():
    cfg = default_config()
    set_seed(cfg["seed"])

    output_root = Path(cfg["output_root"])
    ensure_dir(output_root)

    grouped_data, input_dim = prepare_acic_grouped_data(
        acic_root=cfg["acic_root"],
        seed=cfg["seed"],
    )

    all_setting_ids = sorted(grouped_data.keys(), key=lambda x: int(x))
    tuning_settings, final_settings = split_settings(
        grouped_data=grouped_data,
        seed=cfg["seed"],
        n_tuning_settings=cfg["n_tuning_settings"],
    )

    LOGGER.info(f"input_dim = {input_dim}")
    LOGGER.info(f"n_settings_total = {len(all_setting_ids)}")
    LOGGER.info(f"tuning_settings ({len(tuning_settings)}): {tuning_settings}")
    LOGGER.info(f"final_settings ({len(final_settings)}): {final_settings}")

    split_info = {
        "all_settings": [str(x) for x in all_setting_ids],
        "tuning_settings": [str(x) for x in tuning_settings],
        "final_settings": [str(x) for x in final_settings],
    }
    save_yaml(output_root / "setting_split.yaml", split_info)

    base_cfg = {
        "seed": cfg["seed"],
        "device": cfg["device"],
        "use_amp": cfg["use_amp"],
        "val_split": cfg["val_split"],
    }

    best_params_path = output_root / "tuning" / "best_params.yaml"

    if cfg["mode"] in ["tune", "all"]:
        run_optuna_tuning(
            grouped_data=grouped_data,
            tuning_setting_ids=tuning_settings,
            input_dim=input_dim,
            reps_per_setting_tune=cfg["reps_per_setting_tune"],
            n_trials=cfg["optuna_trials"],
            out_dir=output_root / "tuning",
            base_cfg=base_cfg,
        )

    if cfg["mode"] in ["final", "ablation", "full", "all"]:
        if cfg["best_params_file"] is None:
            cfg["best_params_file"] = str(best_params_path)

        bundle = load_yaml(Path(cfg["best_params_file"]))
        params = bundle["best_params"]

    if cfg["mode"] in ["final", "all"]:
        run_final_evaluation(
            grouped_data=grouped_data,
            setting_ids=final_settings,
            input_dim=input_dim,
            params=params,
            out_dir=output_root / "final",
            reps_per_setting_eval=cfg["reps_per_setting_final"],
            tag="final",
        )

    if cfg["mode"] in ["ablation", "all"]:
        run_ablation(
            grouped_data=grouped_data,
            setting_ids=final_settings,
            input_dim=input_dim,
            params=params,
            out_dir=output_root / "ablation",
            reps_per_setting_eval=cfg["reps_per_setting_ablation"],
        )

    if cfg["mode"] in ["full", "all"]:
        run_final_evaluation(
            grouped_data=grouped_data,
            setting_ids=all_setting_ids,
            input_dim=input_dim,
            params=params,
            out_dir=output_root / "full",
            reps_per_setting_eval=cfg["reps_per_setting_full"],
            tag="full",
        )


if __name__ == "__main__":
    main()