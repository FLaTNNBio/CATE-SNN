#!/usr/bin/env python3
import os, random, csv, logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
from codecarbon import EmissionsTracker

from src.data_loader import DataLoader as CFLoader
from src.models.bcauss import BCAUSS
from src.metrics import eps_ATE_diff, PEHE_with_ite
from src.contrastive import DynamicContrastiveCausalDS, first_item
from src.siamese import SiameseBCAUSS


@hydra.main(config_path="../configs", config_name="default", version_base="1.3")
def run(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    print("Config:\n" + OmegaConf.to_yaml(cfg))

    # Seeds and device
    os.environ['PYTHONHASHSEED'] = str(cfg.seed)
    random.seed(cfg.seed);
    np.random.seed(cfg.seed);
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True;
    torch.backends.cudnn.benchmark = False
    device = cfg.device if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # Output setup
    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    (out_dir / "train_loss.csv").write_text("replica,epoch,loss,base_loss,ctr_loss\n")
    (out_dir / "metrics.csv").write_text("replica,eps_ate,pehe,co2_kg\n")

    tracker = EmissionsTracker(output_dir=str(out_dir), log_level="error", save_to_file=True)
    tracker.start()

    # Load data
    try:
        loader = CFLoader.get_loader('IHDP')
        data = loader.load()
    except Exception as e:
        logging.error(f"Errore caricamento dati IHDP: {e}")
        return

    (X_tr_all, T_tr_all, YF_tr_all, _, m0_tr_all, m1_tr_all,
     X_te_all, _, _, _, m0_te_all, m1_te_all) = data

    results = []
    for idx in range(cfg.n_reps):
        logging.info(f"--- Replica {idx + 1}/{cfg.n_reps} ---")
        # Extract replica
        Xtr = X_tr_all[:, :, idx].astype(np.float32)
        Ttr = T_tr_all[:, idx, None].astype(np.float32)
        Ytr = YF_tr_all[:, idx, None].astype(np.float32)

        # Base model warmup via fit()
        base = BCAUSS(input_dim=Xtr.shape[1])
        if cfg.warmup_epochs_base > 0:
            base.fit(Xtr, Ttr, Ytr, epochs=cfg.warmup_epochs_base)
        base.to(device)

        # Siamese model
        model = SiameseBCAUSS(
            base_model=base,
            ds_class=DynamicContrastiveCausalDS,
            margin=cfg.margin,
            lambda_ctr=cfg.lambda_ctr,
            batch_size=cfg.batch,
            lr=cfg.lr,
            epochs=cfg.epochs,
            clip_norm=cfg.clip_norm,
            use_amp=cfg.use_amp,
            val_split=cfg.val_split,
            update_ite_freq=cfg.update_ite_freq,
            warmup_epochs_base=0
        ).to(device)

        # Train via unified fit()
        model.fit(Xtr, Ttr, Ytr)

        # Evaluate on test
        Xte = X_te_all[:, :, idx].astype(np.float32)
        true_ite_te = m1_te_all[:, idx] - m0_te_all[:, idx]
        if Xte.shape[0] > 0:
            pred_ite_te = model.predict_ite(Xte)
            eps_ate = eps_ATE_diff(pred_ite_te.mean(), true_ite_te.mean())
            pehe = PEHE_with_ite(pred_ite_te, true_ite_te, sqrt=True)
        else:
            eps_ate, pehe = np.nan, np.nan

        results.append((eps_ate, pehe))
        with open(out_dir / "metrics.csv", "a", newline="") as fm:
            csv.writer(fm).writerow([idx, eps_ate, pehe, ""])

    # Save last base
    torch.save(model.base.state_dict(), out_dir / "model_final_base_last_rep.pt")

    # Averages & emissions
    eps_vals, pehe_vals = zip(*results) if results else ([], [])
    mean_eps = np.nanmean(eps_vals) if eps_vals else np.nan
    mean_pehe = np.nanmean(pehe_vals) if pehe_vals else np.nan
    total_co2 = tracker.stop() or 0.0
    with open(out_dir / "metrics.csv", "a", newline="") as fm:
        csv.writer(fm).writerow(["AVERAGE", mean_eps, mean_pehe, total_co2])

    print(f"Emissioni CO2 totali: {total_co2:.3f} kg")
    print(f"Risultati medi: EPS_ATE={mean_eps:.4f}, PEHE={mean_pehe:.4f}")


if __name__ == "__main__":
    run()
