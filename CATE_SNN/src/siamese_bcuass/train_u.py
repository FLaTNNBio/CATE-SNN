#!/usr/bin/env python3
import os
import random
import csv
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
from codecarbon import EmissionsTracker

from src.data_loader import DataLoader as CFLoader
from src.models.bcauss import BCAUSS
from src.metrics import eps_ATE_diff, PEHE_with_ite
from src.contrastive import DynamicContrastiveCausalDS
from src.siamese_bcuass.siamese import SiameseBCAUSS


def save_metrics(csv_path, step_idx, eps_ate, pehe, co2=""):
    with open(csv_path, "a", newline="") as fm:
        csv.writer(fm).writerow([step_idx, f"{eps_ate:.6f}", f"{pehe:.6f}", co2])


@hydra.main(config_path="../../configs", config_name="default", version_base="1.3")
def run(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    print("Config:\n" + OmegaConf.to_yaml(cfg))

    # Seeds and device
    os.environ['PYTHONHASHSEED'] = str(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = cfg.device if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # Output setup
    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    metrics_csv = out_dir / "metrics.csv"
    metrics_csv.write_text("step,eps_ate,pehe,co2_kg\n")
    models_dir = out_dir / "best_models"
    models_dir.mkdir(exist_ok=True)

    # Track emissions
    tracker = EmissionsTracker(output_dir=str(out_dir), log_level="error", save_to_file=True)
    tracker.start()

    # Load all replicas data
    loader = CFLoader.get_loader('IHDP')
    X_tr_all, T_tr_all, YF_tr_all, _, m0_tr_all, m1_tr_all, X_te_all, _, _, _, m0_te_all, m1_te_all = loader.load()

    # Initialize base model once with warmup
    base = BCAUSS(input_dim=X_tr_all.shape[1])
    if cfg.warmup_epochs_base > 0:
        X0 = X_tr_all[:, :, 0].astype(np.float32)
        T0 = T_tr_all[:, 0, None].astype(np.float32)
        Y0 = YF_tr_all[:, 0, None].astype(np.float32)
        base.fit(X0, T0, Y0, epochs=cfg.warmup_epochs_base)
    base.to(device)

    # Initialize Siamese model once
    siamese_params = {
        'ds_class': DynamicContrastiveCausalDS,
        'margin': cfg.margin,
        'lambda_ctr': cfg.lambda_ctr,
        'batch_size': cfg.batch,
        'lr': cfg.lr,
        'epochs': cfg.epochs,
        'clip_norm': cfg.clip_norm,
        'use_amp': cfg.use_amp,
        'val_split': cfg.val_split,
        'update_ite_freq': cfg.update_ite_freq,
        'warmup_epochs_base': 0,
        'lambda_reg': cfg.get('lambda_reg', 0.0),
    }
    model = SiameseBCAUSS(base_model=base, **siamese_params).to(device)

    # Sequential training and evaluation over replicas
    for idx in range(cfg.n_reps):
        logging.info(f"--- Step {idx + 1}/{cfg.n_reps}: training on replica {idx} ---")
        Xtr = X_tr_all[:, :, idx].astype(np.float32)
        Ttr = T_tr_all[:, idx, None].astype(np.float32)
        Ytr = YF_tr_all[:, idx, None].astype(np.float32)

        # Train on this replica (continual learning)
        best_model_path = models_dir / f"best_siamese_step_{idx + 1}.pt"
        model.fit(Xtr, Ttr, Ytr, best_model_path=str(best_model_path))

        # Evaluate on this replica
        Xte = X_te_all[:, :, idx].astype(np.float32)
        true_ite = m1_te_all[:, idx] - m0_te_all[:, idx]
        if Xte.shape[0] > 0:
            with torch.no_grad():
                pred_ite = model.predict_ite(Xte)
            eps = eps_ATE_diff(pred_ite.mean(), true_ite.mean())
            pehe = PEHE_with_ite(pred_ite, true_ite, sqrt=True)
        else:
            eps, pehe = np.nan, np.nan
        logging.info(f"After step {idx + 1}: eps_ATE={eps:.4f}, PEHE={pehe:.4f}")
        save_metrics(metrics_csv, idx + 1, eps, pehe)

    # Stop emissions
    total_co2 = tracker.stop() or 0.0

    # Compute overall summary
    eps_vals, pehe_vals = [], []
    with open(metrics_csv) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            eps_vals.append(float(row[1]))
            pehe_vals.append(float(row[2]))
    mean_eps = np.nanmean(eps_vals)
    mean_pehe = np.nanmean(pehe_vals)

    # Append final summary row
    with open(metrics_csv, "a", newline="") as fm:
        csv.writer(fm).writerow(["AVERAGE", f"{mean_eps:.6f}", f"{mean_pehe:.6f}", f"{total_co2:.3f}"])

    print(f"Total CO2: {total_co2:.3f} kg | Mean eps_ATE={mean_eps:.4f}, Mean PEHE={mean_pehe:.4f}")


if __name__ == "__main__":
    run()
