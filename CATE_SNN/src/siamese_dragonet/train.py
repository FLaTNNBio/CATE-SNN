#!/usr/bin/env python3
# train_siamese_dragonnet_hydra.py
# Script Hydra-based per training del modello SiameseDragonNet sulla replicazione continua di IHDP

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
from src.metrics import eps_ATE_diff, PEHE_with_ite

from src.contrastive import DynamicContrastiveCausalDS
from src.siamese_dragonet.siamese import SiameseDragonNet


def save_metrics(csv_path, step_idx, eps_ate, pehe, co2=""):
    with open(csv_path, "a", newline="") as fm:
        csv.writer(fm).writerow([step_idx, f"{eps_ate:.6f}", f"{pehe:.6f}", co2])


@hydra.main(config_path="../../configs", config_name="siamese_dragonnet_config", version_base="1.3")
def run(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    print("Config:" + OmegaConf.to_yaml(cfg))

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

    # Outputs
    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    metrics_csv = out_dir / "metrics.csv"
    metrics_csv.write_text("step,eps_ate,pehe,co2_kg")
    models_dir = out_dir / "best_models"
    models_dir.mkdir(exist_ok=True)

    # Track emissions
    tracker = EmissionsTracker(output_dir=str(out_dir), log_level="error", save_to_file=True)
    tracker.start()

    # Load IHDP data
    loader = CFLoader.get_loader('IHDP')
    X_tr_all, T_tr_all, YF_tr_all, _, m0_tr_all, m1_tr_all, X_te_all, _, _, _, m0_te_all, m1_te_all = loader.load()

    # Instantiate SiameseDragonNet
    model = SiameseDragonNet(
        input_dim=X_tr_all.shape[1],
        ds_class=DynamicContrastiveCausalDS,
        margin=cfg.margin,
        lambda_ctr=cfg.lambda_ctr,
        device=device,
        # pass-through params
        val_split=cfg.val_split,
        batch_size=cfg.batch,
        optim=cfg.optim,
        lr=cfg.lr,
        momentum=cfg.momentum,
        epochs=cfg.epochs,
        patience=cfg.patience,
        clip_norm=cfg.clip_norm,
        use_amp=cfg.use_amp,
        warmup_epochs_base=cfg.warmup_epochs_base
    ).to(device)

    # Continuous training over replicas
    for idx in range(cfg.n_reps):
        step = idx + 1
        logging.info(f"--- Step {step}/{cfg.n_reps}: replica {idx} ---")
        # Prepare replica data
        Xtr = X_tr_all[:, :, idx].astype(np.float32)
        Ttr = T_tr_all[:, idx, None].astype(np.float32)
        Ytr = YF_tr_all[:, idx, None].astype(np.float32)

        # Train
        best_model_path = models_dir / f"best_siamese_dn_step_{step}.pth"
        model.fit(Xtr, Ttr, Ytr, best_model_path=str(best_model_path))

        # Evaluate
        Xte = X_te_all[:, :, idx].astype(np.float32)
        true_ite = m1_te_all[:, idx] - m0_te_all[:, idx]
        if Xte.shape[0] > 0:
            with torch.no_grad():
                pred_ite = model.predict_ite(Xte)
            eps = eps_ATE_diff(pred_ite.mean(), true_ite.mean())
            pehe = PEHE_with_ite(pred_ite, true_ite, sqrt=True)
        else:
            eps, pehe = np.nan, np.nan
        logging.info(f"After step {step}: eps_ATE={eps:.4f}, PEHE={pehe:.4f}")
        save_metrics(metrics_csv, step, eps, pehe)

    # Stop emissions
    total_co2 = tracker.stop() or 0.0

    # Summary
    eps_vals, pehe_vals = [], []
    with open(metrics_csv) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            eps_vals.append(float(row[1]));
            pehe_vals.append(float(row[2]))
    mean_eps = np.nanmean(eps_vals)
    mean_pehe = np.nanmean(pehe_vals)
    with open(metrics_csv, "a", newline="") as fm:
        csv.writer(fm).writerow(["AVERAGE", f"{mean_eps:.6f}", f"{mean_pehe:.6f}", f"{total_co2:.3f}"])

    print(f"Total CO2: {total_co2:.3f} kg | Mean eps_ATE={mean_eps:.4f}, Mean PEHE={mean_pehe:.4f}")


if __name__ == "__main__":
    run()
