#!/usr/bin/env python3
import os
import random
import csv
import logging
from pathlib import Path
import itertools

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


def save_metrics(csv_path, identifier, eps_ate, pehe, co2=""):
    """Append metrics to the specified CSV file."""
    with open(csv_path, "a", newline="") as fm:
        csv.writer(fm).writerow([identifier, f"{eps_ate:.6f}", f"{pehe:.6f}", co2])


@hydra.main(config_path="../../configs", config_name="default", version_base="1.3")
def run(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    print("Config:\n" + OmegaConf.to_yaml(cfg))

    # Seeds and device
    os.environ['PYTHONHASHSEED'] = str(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = cfg.device if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # Output setup
    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    metrics_csv = out_dir / "metrics.csv"
    # Header: identifier can be step or config_idx:step
    metrics_csv.write_text("id,eps_ate,pehe,co2_kg\n")
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
    if cfg.siamese.warmup_epochs_base > 0:
        X0 = X_tr_all[:, :, 0].astype(np.float32)
        T0 = T_tr_all[:, 0, None].astype(np.float32)
        Y0 = YF_tr_all[:, 0, None].astype(np.float32)
        base.fit(X0, T0, Y0, epochs=cfg.siamese.warmup_epochs_base)
    base.to(device)

    # Grid search parameters from config
    param_grid = {
        'margin': cfg.grid.margin,
        'lambda_ctr': cfg.grid.lambda_ctr,
        'lr': cfg.grid.lr,
        'batch_size': cfg.grid.batch_size,
    }
    grid = list(itertools.product(
        param_grid['margin'],
        param_grid['lambda_ctr'],
        param_grid['lr'],
        param_grid['batch_size'],
    ))

    # CSV for grid search summary
    grid_csv = out_dir / 'grid_search_results.csv'
    grid_csv.write_text('margin,lambda_ctr,lr,batch_size,avg_eps_ATE,avg_PEHE\n')

    # Perform grid search
    for cfg_idx, (margin, lam, lr, bs) in enumerate(grid, start=1):
        eps_vals, pehe_vals = [], []
        logging.info(
            f"--- Grid Search {cfg_idx}/{len(grid)}: margin={margin}, lambda_ctr={lam}, lr={lr}, batch_size={bs} ---")

        # Re-initialize siamese model for each config
        siamese_params = {
            'ds_class': DynamicContrastiveCausalDS,
            'margin': margin,
            'lambda_ctr': lam,
            'batch_size': bs,
            'lr': lr,
            'epochs': cfg.epochs,
            'clip_norm': cfg.clip_norm,
            'use_amp': cfg.use_amp,
            'val_split': cfg.val_split,
            'update_ite_freq': cfg.update_ite_freq,
            'warmup_epochs_base': 0,
            'lambda_reg': cfg.get('lambda_reg', 0.0),
        }
        model = SiameseBCAUSS(base_model=base, **siamese_params).to(device)

        # Train and evaluate over replicas
        for rep in range(cfg.n_reps):
            # Training on replica rep
            Xtr = X_tr_all[:, :, rep].astype(np.float32)
            Ttr = T_tr_all[:, rep, None].astype(np.float32)
            Ytr = YF_tr_all[:, rep, None].astype(np.float32)
            model.fit(Xtr, Ttr, Ytr)

            # Evaluation
            Xte = X_te_all[:, :, rep].astype(np.float32)
            true_ite = m1_te_all[:, rep] - m0_te_all[:, rep]
            if Xte.shape[0] > 0:
                with torch.no_grad():
                    pred_ite = model.predict_ite(Xte)
                eps = eps_ATE_diff(pred_ite.mean(), true_ite.mean())
                pehe = PEHE_with_ite(pred_ite, true_ite, sqrt=True)
            else:
                eps, pehe = np.nan, np.nan

            # Save per-experiment metrics
            identifier = f"cfg{cfg_idx}_rep{rep + 1}"
            save_metrics(metrics_csv, identifier, eps, pehe)
            eps_vals.append(eps)
            pehe_vals.append(pehe)
            logging.info(f"[{identifier}] eps_ATE={eps:.4f}, PEHE={pehe:.4f}")

        # Average metrics per config
        avg_eps = np.nanmean(eps_vals)
        avg_pehe = np.nanmean(pehe_vals)
        # Save grid summary
        with open(grid_csv, "a", newline="") as f:
            csv.writer(f).writerow([margin, lam, lr, bs, f"{avg_eps:.6f}", f"{avg_pehe:.6f}"])
        logging.info(f"Config {cfg_idx} summary: avg_eps_ATE={avg_eps:.4f}, avg_PEHE={avg_pehe:.4f}")

    # Finish emissions tracking
    tracker.stop()
    logging.info("Grid search completed. Results saved to grid_search_results.csv and metrics.csv")


if __name__ == "__main__":
    run()
