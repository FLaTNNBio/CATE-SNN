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

from CATE_SNN.src.data_loader import DataLoader as CFLoader
from CATE_SNN.src.models.bcauss import BCAUSS
from CATE_SNN.src.metrics import eps_ATE_diff, PEHE_with_ite
from CATE_SNN.src.contrastive import DynamicContrastiveCausalDS
from CATE_SNN.src.siamese_bcuass.siamese import SiameseBCAUSS

def save_metrics(csv_path, identifier, params, eps_ate, pehe, co2=""):
    with open(csv_path, 'a', newline='') as fm:
        writer = csv.writer(fm)
        writer.writerow([
            identifier,
            params['margin'],
            params['optim'],
            params['activation'],
            params['pair_pct'],
            params['lr'],
            params['batch_size'],
            f"{eps_ate:.6f}",
            f"{pehe:.6f}",
            co2
        ])

def objective(trial, cfg, base, X_tr_all, T_tr_all, YF_tr_all, X_te_all, m0_te_all, m1_te_all, device, metrics_avg_csv, metrics_all_csv):
    # Campionamento parametri
    margin = trial.suggest_float('margin', 0.1, 1.0)
    optim = trial.suggest_categorical('optim', ['adam', 'sgd'])
    activation = trial.suggest_categorical('activation', ['relu', 'elu', 'tanh'])
    pair_pct = trial.suggest_float('pair_pct', 0.1, 1.0)

    # ✅ Tolto riferimento a cfg.grid.lr e batch_size
    lr = cfg.bcauss_params.learning_rate
    bs = cfg.batch  # uso quello globale, se preferisci puoi anche scegliere un valore fisso

    params = {
        'margin': margin,
        'optim': optim,
        'activation': activation,
        'pair_pct': pair_pct,
        'lr': lr,
        'batch_size': bs
    }

    siamese_params = {
        'ds_class': DynamicContrastiveCausalDS,
        'margin': margin,
        'lambda_ctr': cfg.siamese.lambda_ctr,
        'batch_size': bs,
        'lr': lr,
        'epochs': cfg.epochs,
        'clip_norm': cfg.siamese.clip_norm,
        'use_amp': cfg.siamese.use_amp,
        'val_split': cfg.siamese.val_split,
        'update_ite_freq': cfg.siamese.update_ite_freq,
        'warmup_epochs_base': 0,
        'lambda_reg': cfg.siamese.lambda_reg,
        'pair_sampling_fraction': pair_pct,
        'bcauss_params': {
            'act_fn': activation,
            'optim': optim
        }
    }

    model = SiameseBCAUSS(base_model=base, **siamese_params).to(device)

    eps_vals, pehe_vals = [], []
    fixed_dir = Path("saved_weights")
    fixed_dir.mkdir(parents=True, exist_ok=True)

    for rep in range(cfg.n_reps):
        Xtr = X_tr_all[:, :, rep].astype(np.float32)
        Ttr = T_tr_all[:, rep, None].astype(np.float32)
        Ytr = YF_tr_all[:, rep, None].astype(np.float32)
        model.fit(Xtr, Ttr, Ytr)

        peso_rep_path = fixed_dir / f"weights_trial_{trial.number}_rep_{rep + 1}.pth"
        torch.save(model.state_dict(), peso_rep_path)

        Xte = X_te_all[:, :, rep].astype(np.float32)
        true_ite = m1_te_all[:, rep] - m0_te_all[:, rep]
        with torch.no_grad():
            pred_ite = model.predict_ite(Xte)
        eps = eps_ATE_diff(pred_ite.mean(), true_ite.mean())
        pehe = PEHE_with_ite(pred_ite, true_ite, sqrt=True)

        with open(metrics_all_csv, 'a', newline='') as fm:
            writer = csv.writer(fm)
            writer.writerow([f"trial_{trial.number}", rep + 1, f"{eps:.6f}", f"{pehe:.6f}"])

        eps_vals.append(eps)
        pehe_vals.append(pehe)
        trial.report(pehe, rep)
        if trial.should_prune():
            raise optuna.TrialPruned()

    avg_eps = float(np.nanmean(eps_vals))
    avg_pehe = float(np.nanmean(pehe_vals))
    save_metrics(metrics_avg_csv, f"trial_{trial.number}", params, avg_eps, avg_pehe)

    return avg_pehe

@hydra.main(config_path="../../configs", config_name="default", version_base="1.3")
def run(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    print("Config:\n" + OmegaConf.to_yaml(cfg))

    os.environ['PYTHONHASHSEED'] = str(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = cfg.device if torch.cuda.is_available() else "cpu"

    loader = CFLoader.get_loader('IHDP')
    X_tr_all, T_tr_all, YF_tr_all, _, m0_tr_all, m1_tr_all, X_te_all, _, _, _, m0_te_all, m1_te_all = loader.load()
    input_dim = X_tr_all.shape[1]

    base = BCAUSS(input_dim=input_dim)
    if cfg.siamese.warmup_epochs_base > 0:
        X0 = X_tr_all[:, :, 0].astype(np.float32)
        T0 = T_tr_all[:, 0, None].astype(np.float32)
        Y0 = YF_tr_all[:, 0, None].astype(np.float32)
        base.fit(X0, T0, Y0, epochs=cfg.siamese.warmup_epochs_base)
    base.to(device)

    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    metrics_avg_csv = out_dir / "optuna_aggregated_metrics.csv"
    metrics_all_csv = out_dir / "metriche.csv"

    with open(metrics_avg_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'trial_id', 'margin', 'optim', 'activation', 'pair_pct', 'lr', 'batch_size', 'eps_ate', 'pehe', 'co2_kg'
        ])
    with open(metrics_all_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['trial_id', 'replica', 'eps_ate', 'pehe'])

    tracker = EmissionsTracker(output_dir=str(out_dir), log_level="error", save_to_file=True)
    tracker.start()

    sampler = optuna.samplers.TPESampler(seed=cfg.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

    study.optimize(
        lambda trial: objective(
            trial, cfg, base,
            X_tr_all, T_tr_all, YF_tr_all,
            X_te_all, m0_te_all, m1_te_all,
            device, metrics_avg_csv, metrics_all_csv
        ),
        n_trials=cfg.optuna.n_trials
    )

    best = study.best_params
    print(f"Best parameters found: {best}")
    best_file = out_dir / "best_params.yaml"
    with open(best_file, 'w') as f:
        yaml.safe_dump({'best_params': best}, f)
    print(f"Saved best parameters to {best_file}")

    tracker.stop()

if __name__ == "__main__":
    run()
