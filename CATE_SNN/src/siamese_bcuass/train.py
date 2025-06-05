#!/usr/bin/env python3
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


def save_metrics(csv_path, identifier, params, eps_ate, pehe, co2=""):
    """
    Append aggregated metrics and hyperparameters for each trial (aggregated file).
    """
    with open(csv_path, 'a', newline='') as fm:
        writer = csv.writer(fm)
        writer.writerow([
            identifier,
            params['margin'],
            params['lambda_ctr'],
            params['lr'],
            params['batch_size'],
            f"{eps_ate:.6f}",
            f"{pehe:.6f}",
            co2
        ])


def objective(trial, cfg, base,
              X_tr_all, T_tr_all, YF_tr_all,
              X_te_all, m0_te_all, m1_te_all,
              device, metrics_avg_csv, metrics_all_csv):
    """
    Optuna objective: train su ogni replica, log dei metrici per-replica,
    salvataggio dei pesi alla fine di ciascuna replica, e infine log dei metrici aggregati.
    """
    # Sample dei parametri
    margin = trial.suggest_float('margin', min(cfg.grid.margin), max(cfg.grid.margin))
    lambda_ctr = trial.suggest_float('lambda_ctr', min(cfg.grid.lambda_ctr), max(cfg.grid.lambda_ctr))
    lr = trial.suggest_float('lr', min(cfg.grid.lr), max(cfg.grid.lr), log=True)
    bs = trial.suggest_categorical('batch_size', cfg.grid.batch_size)
    params = {'margin': margin, 'lambda_ctr': lambda_ctr, 'lr': lr, 'batch_size': bs}

    # Inizializzo il modello per questo trial
    siamese_params = {
        'ds_class': DynamicContrastiveCausalDS,
        'margin': margin,
        'lambda_ctr': lambda_ctr,
        'batch_size': bs,
        'lr': lr,
        'epochs': cfg.epochs,
        'clip_norm': cfg.siamese.clip_norm,
        'use_amp': cfg.siamese.use_amp,
        'val_split': cfg.siamese.val_split,
        'update_ite_freq': cfg.siamese.update_ite_freq,
        'warmup_epochs_base': 0,
        'lambda_reg': cfg.siamese.lambda_reg,
    }
    model = SiameseBCAUSS(base_model=base, **siamese_params).to(device)

    eps_vals, pehe_vals = [], []

    # Preparo la cartella di salvataggio fissa (ad esempio "./saved_weights/")
    fixed_dir = Path("saved_weights")
    fixed_dir.mkdir(parents=True, exist_ok=True)

    # Loop su ciascuna replica
    for rep in range(cfg.n_reps):
        # 1) Train sulla replica rep
        Xtr = X_tr_all[:, :, rep].astype(np.float32)
        Ttr = T_tr_all[:, rep, None].astype(np.float32)
        Ytr = YF_tr_all[:, rep, None].astype(np.float32)
        model.fit(Xtr, Ttr, Ytr)

        # 2) Salvare i pesi appena finito di allenare questa replica
        peso_rep_path = fixed_dir / f"weights_trial_{trial.number}_rep_{rep + 1}.pth"
        try:
            torch.save(model.state_dict(), peso_rep_path)
            print(f"[DEBUG] Trial {trial.number}, replica {rep + 1}: pesi salvati in {peso_rep_path.resolve()}")
        except Exception as e:
            print(f"[ERROR] Trial {trial.number}, replica {rep + 1}: impossibile salvare i pesi. Errore: {e}")

        # 3) Evaluate sulla replica rep
        Xte = X_te_all[:, :, rep].astype(np.float32)
        true_ite = m1_te_all[:, rep] - m0_te_all[:, rep]
        with torch.no_grad():
            pred_ite = model.predict_ite(Xte)
        eps = eps_ATE_diff(pred_ite.mean(), true_ite.mean())
        pehe = PEHE_with_ite(pred_ite, true_ite, sqrt=True)

        # 4) Log per-replica metrics
        with open(metrics_all_csv, 'a', newline='') as fm:
            writer = csv.writer(fm)
            writer.writerow([f"trial_{trial.number}", rep + 1, f"{eps:.6f}", f"{pehe:.6f}"])

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


@hydra.main(config_path="../../configs", config_name="default", version_base="1.3")
def run(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    print("Config:\n" + OmegaConf.to_yaml(cfg))

    # Reproducibility
    os.environ['PYTHONHASHSEED'] = str(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = cfg.device if torch.cuda.is_available() else "cpu"

    # Load data
    loader = CFLoader.get_loader('IHDP')
    (X_tr_all, T_tr_all, YF_tr_all,
     _, m0_tr_all, m1_tr_all,
     X_te_all, _, _, _, m0_te_all, m1_te_all) = loader.load()
    input_dim = X_tr_all.shape[1]

    # Base model warmup
    base = BCAUSS(input_dim=input_dim)
    if cfg.siamese.warmup_epochs_base > 0:
        X0 = X_tr_all[:, :, 0].astype(np.float32)
        T0 = T_tr_all[:, 0, None].astype(np.float32)
        Y0 = YF_tr_all[:, 0, None].astype(np.float32)
        base.fit(X0, T0, Y0,
                 epochs=cfg.siamese.warmup_epochs_base)
    base.to(device)

    # Prepare output files
    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    metrics_avg_csv = out_dir / "optuna_aggregated_metrics.csv"
    metrics_all_csv = out_dir / "metriche.csv"

    # Write headers
    with open(metrics_avg_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'trial_id', 'margin', 'lambda_ctr', 'lr',
            'batch_size', 'eps_ate', 'pehe', 'co2_kg'
        ])
    with open(metrics_all_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'trial_id', 'replica', 'eps_ate', 'pehe'
        ])

    # Emissions tracking
    tracker = EmissionsTracker(output_dir=str(out_dir), log_level="error", save_to_file=True)
    tracker.start()

    # Optuna study setup
    sampler = optuna.samplers.TPESampler(seed=cfg.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

    # Run optimization
    study.optimize(
        lambda trial: objective(
            trial, cfg, base,
            X_tr_all, T_tr_all, YF_tr_all,
            X_te_all, m0_te_all, m1_te_all,
            device, metrics_avg_csv, metrics_all_csv
        ),
        n_trials=cfg.optuna.n_trials
    )

    # Save best params
    best = study.best_params
    print(f"Best parameters found: {best}")
    best_file = out_dir / "best_params.yaml"
    with open(best_file, 'w') as f:
        yaml.safe_dump({'best_params': best}, f)
    print(f"Saved best parameters to {best_file}")

    tracker.stop()


if __name__ == "__main__":
    run()
