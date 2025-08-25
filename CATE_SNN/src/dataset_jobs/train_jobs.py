# !/usr/bin/env python3
import os
import random
import csv
import logging
from pathlib import Path
import hydra
import optuna
from omegaconf import DictConfig, OmegaConf
import yaml
import numpy as np
import torch
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.dataset_jobs.contrastive import DynamicContrastiveCausalDS
from src.dataset_jobs.bcauss import BCAUSS
from src.metrics import RPol, ATT
from src.dataset_jobs.siamese import SiameseBCAUSS


# ==============================================================================
# SEZIONE 1: FUNZIONI HELPER PER IL TRAINING
# ==============================================================================
def train_epoch_combined(model, base_loader, contrastive_loader, optimizer, device):
    """
    Esegue un'epoca di training che gestisce entrambe le loss.
    Alterna un batch per la loss base e un batch per la loss contrastiva.
    """
    model.train()
    total_loss = 0.0

    # Usiamo il loader più lungo come riferimento per la durata dell'epoca
    num_batches = max(len(base_loader), len(contrastive_loader))
    base_iter = iter(base_loader)
    contrastive_iter = iter(contrastive_loader)

    for _ in range(num_batches):
        # --- Passo di training per la loss supervisionata (BASE) ---
        try:
            X_batch, T_batch, Y_batch = next(base_iter)
            X_batch, T_batch, Y_batch = X_batch.to(device), T_batch.to(device), Y_batch.to(device)

            optimizer.zero_grad()
            mu0_hat, mu1_hat, _ = model.base.mu_and_embedding(X_batch)
            base_loss = model.base.compute_loss(X_batch, T_batch, Y_batch)

            if torch.isfinite(base_loss):
                base_loss.backward()
                optimizer.step()
                total_loss += base_loss.item()
        except StopIteration:
            # Se il loader base finisce, lo resettiamo per il prossimo giro
            base_iter = iter(base_loader)

        # --- Passo di training per la loss contrastiva (CONTRASTIVE) ---
        try:
            x1, x2, labels = next(contrastive_iter)
            x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)

            optimizer.zero_grad()
            h1 = model.base.embed(x1)
            h2 = model.base.embed(x2)
            ctr_loss = model.contrastive_loss(h1, h2, labels.float())

            if not torch.isfinite(ctr_loss):
                logging.warning("Contrastive loss is NaN/Inf, skipping.")
                ctr_loss = torch.tensor(0.0, device=device)

            # Applichiamo il lambda solo alla loss contrastiva
            loss_to_backward = model.lambda_ctr * ctr_loss
            loss_to_backward.backward()
            optimizer.step()
            total_loss += loss_to_backward.item()
        except StopIteration:
            contrastive_iter = iter(contrastive_loader)

    return total_loss / (2 * num_batches) if num_batches > 0 else 0.0


def validate_epoch(model, loader, device):
    """Calcola la validation loss (solo contrastiva, per semplicità)."""
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for x1, x2, labels in loader:
            x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
            h1 = model.base.embed(x1)
            h2 = model.base.embed(x2)
            val_loss = model.contrastive_loss(h1, h2, labels.float())
            if torch.isfinite(val_loss):
                total_val_loss += val_loss.item()
    return total_val_loss / len(loader) if len(loader) > 0 else float('inf')


def save_aggregated_metrics(csv_path, identifier, params, metrics):
    """Salva le metriche aggregate e gli iperparametri per ogni trial."""
    with open(csv_path, 'a', newline='') as fm:
        writer = csv.writer(fm)
        writer.writerow([
            identifier, params['margin'], params['lambda_ctr'], params['lr'],
            params['batch_size'], params['perc'], f"{metrics['eps_att']:.6f}", f"{metrics['rpol']:.6f}",
            f"{metrics['true_att']:.6f}", f"{metrics['pred_att']:.6f}", f"{metrics['rel_err']:.6f}",
            f"{metrics['cohen_d']:.6f}"
        ])


def save_per_replica_metrics(csv_path, trial_id, replica_num, metrics):
    """Aggiunge una riga al file di metriche per-replica."""
    with open(csv_path, 'a', newline='') as fm:
        writer = csv.writer(fm)
        writer.writerow([
            f"trial_{trial_id}",
            replica_num,
            f"{metrics['eps_att']:.6f}",
            f"{metrics['rpol']:.6f}",
            f"{metrics['true_att']:.6f}",
            f"{metrics['pred_att']:.6f}",
            f"{metrics['rel_err']:.6f}",
            f"{metrics['cohen_d']:.6f}"
        ])

# ==============================================================================
# SEZIONE 2: FUNZIONE OBJECTIVE E LOGICA PRINCIPALE
# ==============================================================================

def objective(trial, cfg, base,
              X_tr_all, T_tr_all, YF_tr_all, E_tr_all,
              X_te_all, T_te_all, YF_te_all, E_te_all, I_te_all,
              device, metrics_avg_csv, metrics_all_csv, out_dir):
    """
    Funzione objective di Optuna che gestisce il training e la valutazione
    su più repliche, con report dettagliato per-replica.
    """
    # 1. Campionamento degli iperparametri per questo trial
    params = {
        'margin': trial.suggest_float('margin', 0.2, 0.8),
        'lambda_ctr': trial.suggest_float('lambda_ctr', 0.01, 0.2),
        'lr': trial.suggest_float('lr', 1e-4, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', cfg.grid.batch_size),
        'perc': trial.suggest_int('perc', 10, 50)  # Range ridotto per stabilità
    }

    # 2. Inizializzazione per il trial
    metric_vals = {k: [] for k in ['eps_att', 'rpol', 'true_att', 'pred_att', 'rel_err', 'cohen_d']}
    weights_dir = out_dir / "saved_weights_jobs" / f"trial_{trial.number}"
    weights_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0

    # 3. Ciclo su ogni replica
    for rep in range(cfg.n_reps):
        logging.info(f"--- Trial {trial.number}, Replica {rep + 1}/{cfg.n_reps} ---")

        # Preparazione dati per la replica corrente
        x_scaler, y_scaler = StandardScaler(), StandardScaler()
        Xtr, Ttr, Ytr = X_tr_all[:, :, rep].astype(np.float32), T_tr_all[:, rep, None].astype(np.float32), YF_tr_all[:,
                                                                                                           rep,
                                                                                                           None].astype(
            np.float32)
        Xte = X_te_all[:, :, rep].astype(np.float32)
        Xtr, Xte = x_scaler.fit_transform(Xtr), x_scaler.transform(Xte)
        Ytr_scaled = y_scaler.fit_transform(Ytr)
        mask_rct_train = E_tr_all[:, rep].astype(bool)

        # Fase 1: Warmup del modello base
        logging.info(f"[Replica {rep + 1}] Phase 1: Training base model...")
        base.fit(Xtr, Ttr, Ytr_scaled, epochs=cfg.siamese.warmup_epochs_base)

        # Fase 2: Training contrastivo sui dati RCT
        logging.info(f"[Replica {rep + 1}] Phase 2: Contrastive training...")
        Xtr_rct, Ttr_rct, Ytr_rct_scaled = Xtr[mask_rct_train], Ttr[mask_rct_train], Ytr_scaled[mask_rct_train]

        if len(Xtr_rct) < params['batch_size']:
            logging.warning("Not enough RCT data for a full batch. Skipping replica.")
            continue

        # Istanziazione del modello Siamese per la replica
        siamese_params = {
            'ds_class': DynamicContrastiveCausalDS,
            'margin': params['margin'], 'lambda_ctr': params['lambda_ctr'],
            'clip_norm': cfg.siamese.clip_norm, 'perc': params['perc']
        }
        model = SiameseBCAUSS(base_model=base, **siamese_params).to(device)

        # Preparazione dei DataLoader
        base_rct_dataset = TensorDataset(torch.from_numpy(Xtr_rct), torch.from_numpy(Ttr_rct),
                                         torch.from_numpy(Ytr_rct_scaled))
        base_loader = DataLoader(base_rct_dataset, batch_size=params['batch_size'], shuffle=True)

        contrastive_ds = DynamicContrastiveCausalDS(
            torch.from_numpy(Xtr_rct).to(device), torch.from_numpy(Ttr_rct).to(device),
            torch.from_numpy(Ytr_rct_scaled).to(device),
            base_model=model.base, perc=params['perc'], n_pairs=cfg.siamese.n_pairs,
            min_thr=cfg.siamese.min_thr, max_thr=cfg.siamese.max_thr, smooth=cfg.siamese.smooth
        )

        if len(contrastive_ds) < 2:
            logging.warning("Not enough contrastive pairs generated. Skipping replica.")
            continue

        n_val = int(len(contrastive_ds) * cfg.siamese.val_split)
        n_train = len(contrastive_ds) - n_val
        if n_train == 0 or n_val == 0:
            logging.warning("Not enough pairs for train/val split. Skipping replica.")
            continue

        train_contrastive_ds, val_contrastive_ds = torch.utils.data.random_split(contrastive_ds, [n_train, n_val])
        contrastive_loader = DataLoader(train_contrastive_ds, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_contrastive_ds, batch_size=params['batch_size'])
        optimizer = optim.Adam(model.parameters(), lr=params['lr'])

        # Loop di training
        for epoch in range(cfg.epochs):
            if epoch > 0 and epoch % cfg.siamese.update_ite_freq == 0:
                contrastive_ds.update_threshold()
            tr_loss = train_epoch_combined(model, base_loader, contrastive_loader, optimizer, device)
            val_loss = validate_epoch(model, val_loader, device)
            logging.info(f"Epoch {epoch + 1}/{cfg.epochs} | Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f}")

            trial.report(val_loss, global_step)
            global_step += 1
            if trial.should_prune():
                raise optuna.TrialPruned()

        # 4. Valutazione e salvataggio per la replica
        torch.save(model.state_dict(), weights_dir / f"weights_rep_{rep + 1}.pth")

        with torch.no_grad():
            Xte_t = torch.from_numpy(Xte).float().to(device)
            y0_pred_scaled, y1_pred_scaled, _ = model.base.mu_and_embedding(Xte_t)
            hat_y_scaled = np.concatenate([y0_pred_scaled.cpu().numpy(), y1_pred_scaled.cpu().numpy()], axis=1)

        hat_y = y_scaler.inverse_transform(hat_y_scaled)
        exp_mask = E_te_all[:, rep].astype(bool)
        # Linea Corretta
        Tte_exp, YFte_exp, hat_y_exp = T_te_all[:, rep][exp_mask], YF_te_all[:, rep][exp_mask], hat_y[exp_mask]

        # Calcolo metriche per la replica
        current_metrics = {
            'eps_att': ATT(Tte_exp, YFte_exp, hat_y_exp),
            'rpol': RPol(Tte_exp, YFte_exp, hat_y_exp),
            'true_att': np.mean(YFte_exp[Tte_exp == 1]) - np.mean(YFte_exp[Tte_exp == 0]),
            'pred_att': np.mean(hat_y_exp[Tte_exp == 1, 1] - hat_y_exp[Tte_exp == 1, 0])
        }
        current_metrics['rel_err'] = current_metrics['eps_att'] / abs(current_metrics['true_att']) if abs(
            current_metrics['true_att']) > 1e-9 else np.nan
        sd_train_rct = Ytr[mask_rct_train].std() if np.any(mask_rct_train) else 0
        current_metrics['cohen_d'] = current_metrics['eps_att'] / sd_train_rct if sd_train_rct > 1e-6 else np.nan

        save_per_replica_metrics(metrics_all_csv, trial.number, rep + 1, current_metrics)
        for key, value in current_metrics.items():
            metric_vals[key].append(value)

    # 5. Aggregazione e report finale per il trial
    if not metric_vals['eps_att']:
        logging.warning(f"Trial {trial.number} failed to produce any results.")
        return float('inf')

    avg_metrics = {k: float(np.nanmean(v)) for k, v in metric_vals.items()}
    save_aggregated_metrics(metrics_avg_csv, f"trial_{trial.number}", params, avg_metrics)

    return avg_metrics['eps_att']


def save_metrics(csv_path, identifier, params, eps_att, rpol,
                 avg_true_att, avg_pred_att, avg_rel_err, avg_cohen_d):
    """Salva le metriche aggregate e gli iperparametri per ogni trial."""
    with open(csv_path, 'a', newline='') as fm:
        writer = csv.writer(fm)
        writer.writerow([
            identifier, params['margin'], params['lambda_ctr'], params['lr'],
            params['batch_size'], params['perc'], f"{eps_att:.6f}", f"{rpol:.6f}",
            f"{avg_true_att:.6f}", f"{avg_pred_att:.6f}", f"{avg_rel_err:.6f}",
            f"{avg_cohen_d:.6f}"
        ])


# ==============================================================================
# SEZIONE 2: FUNZIONE OBJECTIVE E LOGICA PRINCIPALE
# ==============================================================================


@hydra.main(config_path="../../configs", config_name="jobs_config", version_base="1.3")
def run(cfg: DictConfig):
    """Funzione principale che orchestra l'esperimento."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', encoding='utf-8')
    print("Config:\n" + OmegaConf.to_yaml(cfg))

    # Riproducibilità
    os.environ['PYTHONHASHSEED'] = str(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = cfg.device if torch.cuda.is_available() else "cpu"

    # Caricamento Dati
    train_data = np.load(cfg.data.train_path)
    test_data = np.load(cfg.data.test_path)
    X_tr_all, T_tr_all, YF_tr_all, E_tr_all = train_data['x'], train_data['t'], train_data['yf'], train_data['e']
    X_te_all, T_te_all, YF_te_all, E_te_all, I_te_all = test_data['x'], test_data['t'], test_data['yf'], test_data['e'], \
        test_data['I']

    input_dim = X_tr_all.shape[1]
    base = BCAUSS(input_dim=input_dim, reg_l2=cfg.bcauss.reg_l2).to(device)

    # Prepara i file di output
    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    exp_name = cfg.data.name

    metrics_avg_csv = out_dir / f"{exp_name}_aggregated_metrics.csv"
    metrics_all_csv = out_dir / f"{exp_name}_per_replica_metrics.csv"

    # Scrivi gli header per entrambi i file
    with open(metrics_avg_csv, 'w', newline='') as f:
        csv.writer(f).writerow([
            'trial_id', 'margin', 'lambda_ctr', 'lr', 'batch_size', 'perc',
            'avg_eps_att', 'avg_rpol', 'avg_true_att', 'avg_pred_att',
            'avg_relative_error', 'avg_cohen_d_error'
        ])

    with open(metrics_all_csv, 'w', newline='') as f:
        csv.writer(f).writerow([
            'trial_id', 'replica', 'eps_att', 'rpol', 'true_att', 'pred_att',
            'relative_error', 'cohen_d_error'
        ])

    # Setup di Optuna
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=cfg.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10 * cfg.n_reps)  # Aumentato warmup_steps
    )

    # Esegui l'ottimizzazione
    # VERSIONE CORRETTA
    study.optimize(
        lambda trial: objective(
            trial, cfg, base,
            X_tr_all, T_tr_all, YF_tr_all, E_tr_all,
            X_te_all, T_te_all, YF_te_all, E_te_all, I_te_all,  # <-- AGGIUNTO DI NUOVO
            device, metrics_avg_csv, metrics_all_csv, out_dir
        ),
        n_trials=cfg.optuna.n_trials
    )

    # Salva i parametri migliori
    best = study.best_params
    print(f"Best parameters found: {best}")
    with open(out_dir / f"{exp_name}_best_params.yaml", 'w') as f:
        yaml.safe_dump({'best_params': best}, f)


if __name__ == "__main__":
    run()
