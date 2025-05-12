#!/usr/bin/env python3
"""
Siamese-BCAUSS con Hydra (PyTorch) + CodeCarbon
----------------------------------------------
• Contrastive loss con soglia dinamica sul 20° percentile delle CATE stimate (ITE_hat)
• ITE_hat aggiornati periodicamente usando il modello BCAUSS base
• Mixed-precision (AMP) opzionale
• Gradient clipping
• Log CSV per ogni epoca (`train_loss.csv`) e per replica (`metrics.csv`)
• Grafico `loss_curve.png` (media su tutte le repliche)
• CodeCarbon: `emissions.csv` nella cartella di output Hydra

Esempio:
```bash
python siamese_bcauss_hydra.py batch=512 lr=5e-5 epochs=30 use_amp=true use_proxy_ite=true update_ite_freq=1 warmup_epochs_base=0
"""

import os, random, csv, logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch

try:
    from torch.amp import GradScaler  # PyTorch ≥ 2.1
except ImportError:
    from torch.cuda.amp import GradScaler  # type: ignore
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from codecarbon import EmissionsTracker

from src.data_loader import DataLoader as CFLoader  # Assumendo che questo sia un tuo modulo locale
from src.metrics import eps_ATE_diff, PEHE_with_ite  # Assumendo che questo sia un tuo modulo locale
from src.models.bcauss import BCAUSS  # Assumendo che questo sia un tuo modulo locale


###############################################################################
# Helper Functions                                                            #
###############################################################################

def compute_tau_threshold(mu0_or_mu0_hat, mu1_or_mu1_hat, perc=20, sample=100_000):
    tau_or_tau_hat = mu1_or_mu1_hat - mu0_or_mu0_hat
    actual_sample_size = min(sample, len(tau_or_tau_hat))

    if actual_sample_size == 0:
        logging.warning("Cannot compute tau threshold with 0 samples for tau_or_tau_hat.")
        return 0.1  # Fallback

    # Questa condizione dovrebbe essere coperta da actual_sample_size == 0 se len(tau_or_tau_hat) == 0
    if len(tau_or_tau_hat) == 0:
        logging.warning("tau_or_tau_hat is empty, cannot compute threshold.")
        return 0.1  # Fallback

    if len(tau_or_tau_hat) == 1 and actual_sample_size > 0:
        # Se c'è un solo campione ITE, la differenza tra campioni casuali (che saranno lo stesso campione) è 0
        # Questo porterebbe a una soglia di 0, il che potrebbe non essere desiderabile.
        # Considerare un comportamento alternativo o accettare che la soglia sia 0.
        # Per ora, si procede come se la differenza fosse 0, percentile di 0 è 0.
        idx1 = np.zeros(actual_sample_size, dtype=int)
        idx2 = np.zeros(actual_sample_size, dtype=int)
    else:
        idx1 = np.random.randint(0, len(tau_or_tau_hat), size=actual_sample_size)
        idx2 = np.random.randint(0, len(tau_or_tau_hat), size=actual_sample_size)

    return float(np.percentile(np.abs(tau_or_tau_hat[idx1] - tau_or_tau_hat[idx2]), perc))


def make_pairs_from_hat(X, T, Y, mu0_hat, mu1_hat, thr, n_pairs, seed=None):
    if seed is not None:
        np.random.seed(seed)

    ite_hat = mu1_hat - mu0_hat
    N = X.shape[0]
    all_indices = np.arange(N)

    idx_a_final, idx_b_final, lab_final = [], [], []

    if N == 0:
        logging.warning("make_pairs_from_hat: Input data (X) is empty.")
        X_shape_tail = X.shape[1:] if X.ndim > 1 else ()
        Y_shape_tail = Y.shape[1:] if Y.ndim > 1 else ()
        T_shape_tail = T.shape[1:] if T.ndim > 1 else ()
        return (np.empty((0,) + X_shape_tail, dtype=X.dtype), np.empty((0,) + Y_shape_tail, dtype=Y.dtype),
                np.empty((0,) + T_shape_tail, dtype=T.dtype), np.empty((0,) + X_shape_tail, dtype=X.dtype),
                np.empty((0,) + Y_shape_tail, dtype=Y.dtype), np.empty((0,) + T_shape_tail, dtype=T.dtype),
                np.array([], dtype=np.int64))

    anchor_indices = np.random.randint(0, N, size=n_pairs)

    for anchor_idx in anchor_indices:
        if N <= 1:
            logging.debug(f"Not enough samples (N={N}) to form a pair for anchor {anchor_idx}.")
            continue

        anchor_ite_hat = ite_hat[anchor_idx]
        other_indices = np.delete(all_indices, anchor_idx)

        if other_indices.size == 0:  # Dovrebbe accadere solo se N=1, già gestito sopra.
            logging.debug(f"No other indices to form a pair for anchor {anchor_idx} after deletion (N={N}).")
            continue

        ite_hat_others = ite_hat[other_indices]
        abs_diff_ite_hat = np.abs(ite_hat_others - anchor_ite_hat)

        is_target_similar = np.random.rand() < 0.5
        partner_idx = -1
        label = -1

        if is_target_similar:
            possible_partner_indices_local = np.where(abs_diff_ite_hat < thr)[0]
            if possible_partner_indices_local.size > 0:
                chosen_local_idx = np.random.choice(possible_partner_indices_local)
                partner_idx = other_indices[chosen_local_idx]
                label = 1
            else:
                logging.debug(
                    f"No similar partner for anchor {anchor_idx} (ITE_hat={anchor_ite_hat:.2f}, thr={thr:.2f}). "
                    f"Trying dissimilar fallback.")
                possible_partner_indices_local_dissim = np.where(abs_diff_ite_hat >= thr)[0]
                if possible_partner_indices_local_dissim.size > 0:
                    hardest_negative_local_idx = np.argmin(abs_diff_ite_hat[possible_partner_indices_local_dissim])
                    partner_idx = other_indices[possible_partner_indices_local_dissim[hardest_negative_local_idx]]
                    label = 0
        else:  # target dissimilar
            possible_partner_indices_local = np.where(abs_diff_ite_hat >= thr)[0]
            if possible_partner_indices_local.size > 0:
                hardest_negative_local_idx = np.argmin(abs_diff_ite_hat[possible_partner_indices_local])
                partner_idx = other_indices[possible_partner_indices_local[hardest_negative_local_idx]]
                label = 0
            else:
                logging.debug(
                    f"No dissimilar partner for anchor {anchor_idx} (ITE_hat={anchor_ite_hat:.2f}, thr={thr:.2f}). "
                    f"Trying similar fallback.")
                possible_partner_indices_local_sim = np.where(abs_diff_ite_hat < thr)[0]
                if possible_partner_indices_local_sim.size > 0:
                    chosen_local_idx = np.random.choice(possible_partner_indices_local_sim)
                    partner_idx = other_indices[chosen_local_idx]
                    label = 1

        if partner_idx != -1:
            idx_a_final.append(anchor_idx)
            idx_b_final.append(partner_idx)
            lab_final.append(label)

    if not idx_a_final:
        logging.warning("make_pairs_from_hat generated 0 pairs for this call.")
        X_shape_tail = X.shape[1:] if X.ndim > 1 else ()
        Y_shape_tail = Y.shape[1:] if Y.ndim > 1 else ()
        T_shape_tail = T.shape[1:] if T.ndim > 1 else ()
        return (np.empty((0,) + X_shape_tail, dtype=X.dtype), np.empty((0,) + Y_shape_tail, dtype=Y.dtype),
                np.empty((0,) + T_shape_tail, dtype=T.dtype), np.empty((0,) + X_shape_tail, dtype=X.dtype),
                np.empty((0,) + Y_shape_tail, dtype=Y.dtype), np.empty((0,) + T_shape_tail, dtype=T.dtype),
                np.array(lab_final, dtype=np.int64))

    idx_a_final = np.array(idx_a_final)
    idx_b_final = np.array(idx_b_final)

    return (X[idx_a_final], Y[idx_a_final], T[idx_a_final],
            X[idx_b_final], Y[idx_b_final], T[idx_b_final],
            np.array(lab_final, np.int64))


def first_item(batch):
    return batch[0]


###############################################################################
# Dataset Class                                                               #
###############################################################################

class DynamicContrastiveCausalDS(Dataset):
    def __init__(self, X_all_replica, T_all_replica, Y_all_replica,
                 initial_mu0_hat, initial_mu1_hat,
                 bs=256, perc=20, sample_for_thr_calc=100_000):

        self.X_all, self.T_all, self.Y_all = X_all_replica, T_all_replica, Y_all_replica
        self.bs = bs
        self.perc = perc
        self.sample_for_thr_calc = sample_for_thr_calc

        if initial_mu0_hat.size == 0 or initial_mu1_hat.size == 0:
            logging.error("Initial mu_hat arrays are empty. Cannot initialize dataset properly.")
            self.current_mu0_hat = np.array([])
            self.current_mu1_hat = np.array([])
            self.thr = 0.1
        else:
            self.current_mu0_hat = initial_mu0_hat
            self.current_mu1_hat = initial_mu1_hat
            self.update_threshold()

    def update_threshold(self):
        self.thr = compute_tau_threshold(self.current_mu0_hat, self.current_mu1_hat,
                                         self.perc, self.sample_for_thr_calc)
        logging.info(f"Updated dynamic threshold for pair generation based on ITE_hat: {self.thr:.4f}")

    def update_ite_estimates(self, new_mu0_hat, new_mu1_hat):
        if new_mu0_hat.size == 0 or new_mu1_hat.size == 0:
            logging.warning("Received empty new_mu_hat arrays during update. Retaining old estimates.")
            return
        self.current_mu0_hat = new_mu0_hat
        self.current_mu1_hat = new_mu1_hat
        self.update_threshold()

    def __len__(self):
        return 32  # batch virtuali per epoca

    def __getitem__(self, _):
        if self.current_mu0_hat.size == 0 or self.X_all.shape[0] == 0:
            logging.warning(f"Dataset has no data or no ITE estimates. Returning empty tensors for __getitem__.")
            X_shape_tail = self.X_all.shape[1:] if self.X_all.ndim > 1 else ()
            Y_shape_tail = self.Y_all.shape[1:] if self.Y_all.ndim > 1 else ()
            T_shape_tail = self.T_all.shape[1:] if self.T_all.ndim > 1 else ()
            empty_X = torch.empty((0,) + X_shape_tail, dtype=torch.float32)
            empty_Y = torch.empty((0,) + Y_shape_tail, dtype=torch.float32)
            empty_T = torch.empty((0,) + T_shape_tail, dtype=torch.float32)
            empty_lab = torch.empty((0,), dtype=torch.long)
            return empty_X, empty_Y, empty_T, empty_X, empty_Y, empty_T, empty_lab

        tpl = make_pairs_from_hat(self.X_all, self.T_all, self.Y_all,
                                  self.current_mu0_hat, self.current_mu1_hat,
                                  self.thr, self.bs)

        tf = lambda a: torch.tensor(a, dtype=torch.float32)
        x1, y1, t1, x2, y2, t2 = map(tf, tpl[:6])
        lab = torch.tensor(tpl[6], dtype=torch.long)

        return x1, y1, t1, x2, y2, t2, lab


###############################################################################
# Siamese Model Wrapper                                                       #
###############################################################################

class SiameseBCAUSS(torch.nn.Module):
    def __init__(self, base, margin, lambda_ctr):
        super().__init__()
        self.base = base
        self.margin = margin
        self.lambda_ctr = lambda_ctr

    def contrastive_loss(self, h1, h2, l):
        d = torch.norm(h1 - h2, p=2, dim=1)
        loss_similar = d.pow(2)
        loss_dissimilar = torch.clamp(self.margin - d, min=0).pow(2)
        final_loss = (l * loss_similar) + ((1 - l) * loss_dissimilar)
        return final_loss.mean()

    def step(self, b, scaler, opt, clip, amp):
        x1, y1, t1, x2, y2, t2, l = b

        if x1.shape[0] == 0:
            logging.warning("Skipping training step due to empty batch.")
            return 0.0, 0.0, 0.0

        with autocast(enabled=amp):
            mu_outcome1, h1 = self.base.mu_and_embedding(x1)
            mu_outcome2, h2 = self.base.mu_and_embedding(x2)

            base_loss_val = self.base.compute_loss(
                torch.cat([x1, x2]),
                torch.cat([t1, t2]),
                torch.cat([y1, y2])
            )
            ctr_loss_val = self.contrastive_loss(h1, h2, l.float())
            loss = base_loss_val + self.lambda_ctr * ctr_loss_val

        if torch.isnan(loss) or torch.isinf(loss):
            logging.error(
                f"NaN/Inf loss detected. base_loss: {base_loss_val.item() if not torch.isnan(base_loss_val) else 'NaN'}, ctr_loss: {ctr_loss_val.item() if not torch.isnan(ctr_loss_val) else 'NaN'}. Skipping update.")
            opt.zero_grad(set_to_none=True)
            bl = base_loss_val.item() if not (torch.isnan(base_loss_val) or torch.isinf(base_loss_val)) else 0.0
            cl = ctr_loss_val.item() if not (torch.isnan(ctr_loss_val) or torch.isinf(ctr_loss_val)) else 0.0
            lo = bl + self.lambda_ctr * cl
            return lo, bl, cl

        scaler.scale(loss).backward()
        if clip > 0:  # Solitamente clip_norm è > 0 per attivarlo
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        return loss.item(), base_loss_val.item(), ctr_loss_val.item()

    @torch.no_grad()
    def predict_ite(self, X_np, device):
        self.eval()
        xt = torch.tensor(X_np, dtype=torch.float32, device=device)
        if xt.shape[0] == 0:
            return np.array([])
        mu_matrix, _ = self.base.mu_and_embedding(xt)
        return (mu_matrix[:, 1] - mu_matrix[:, 0]).cpu().numpy()

    @torch.no_grad()
    def predict_mu_hat(self, X_np, device):
        self.base.eval()
        if X_np.shape[0] == 0:
            logging.warning("predict_mu_hat called with empty X_np.")
            return np.array([]), np.array([])
        x_tensor = torch.tensor(X_np, dtype=torch.float32).to(device)
        mu_matrix, _ = self.base.mu_and_embedding(x_tensor)
        mu0_hat = mu_matrix[:, 0].cpu().numpy()
        mu1_hat = mu_matrix[:, 1].cpu().numpy()
        return mu0_hat, mu1_hat


###############################################################################
# Hydra Main Function                                                         #
###############################################################################

@hydra.main(config_path="../configs", config_name="default", version_base="1.3")
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
    logging.info(f"Using device: {device}")
    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    (out_dir / "train_loss.csv").write_text("replica,epoch,loss,base_loss,ctr_loss\n")
    (out_dir / "metrics.csv").write_text("replica,eps_ate,pehe,co2_kg\n")

    tracker = EmissionsTracker(output_dir=str(out_dir), log_level="error", save_to_file=True)
    tracker.start()

    try:
        data_loader_instance = CFLoader.get_loader('IHDP')  # type: ignore
        data = data_loader_instance.load()
    except Exception as e:
        logging.error(f"Errore durante il caricamento dei dati IHDP: {e}")
        return

    (X_tr_all_reps, T_tr_all_reps, YF_tr_all_reps, _, m0_tr_all_reps, m1_tr_all_reps,
     X_te_all_reps, _, _, _, m0_te_all_reps, m1_te_all_reps) = data

    results = []
    loss_per_epoch_all_reps_df_list = []

    for idx in range(cfg.n_reps):
        logging.info(f"--- Replica {idx + 1}/{cfg.n_reps} ---")
        Xtr_rep, Ttr_rep, YFtr_rep = (X_tr_all_reps[:, :, idx].astype(np.float32),
                                      T_tr_all_reps[:, idx, None].astype(np.float32),
                                      YF_tr_all_reps[:, idx, None].astype(np.float32))

        true_m0_tr_rep, true_m1_tr_rep = m0_tr_all_reps[:, idx], m1_tr_all_reps[:, idx]

        base_model_instance = BCAUSS(input_dim=Xtr_rep.shape[1])
        siamese_model_instance = SiameseBCAUSS(base_model_instance, cfg.margin, cfg.lambda_ctr).to(device)
        optimizer_instance = torch.optim.Adam(siamese_model_instance.parameters(), lr=cfg.lr)
        grad_scaler_instance = GradScaler(enabled=cfg.use_amp)

        current_mu0_hat_tr, current_mu1_hat_tr = np.array([]), np.array([])  # Inizializza come vuoti

        if cfg.use_proxy_ite:
            if cfg.warmup_epochs_base > 0:
                logging.info(f"Warmup del modello BCAUSS base per {cfg.warmup_epochs_base} epoche...")
                base_model_instance.train()
                temp_optimizer_base = torch.optim.Adam(base_model_instance.parameters(), lr=cfg.lr)

                if Xtr_rep.shape[0] > 0:  # Solo se ci sono dati di training
                    x_warmup_t = torch.tensor(Xtr_rep, dtype=torch.float32).to(device)
                    t_warmup_t = torch.tensor(Ttr_rep, dtype=torch.float32).to(device)
                    y_warmup_t = torch.tensor(YFtr_rep, dtype=torch.float32).to(device)

                    for w_ep in range(cfg.warmup_epochs_base):
                        temp_optimizer_base.zero_grad()
                        with autocast(enabled=cfg.use_amp):
                            loss_w = base_model_instance.compute_loss(x_warmup_t, t_warmup_t, y_warmup_t)

                        if cfg.use_amp:
                            grad_scaler_instance.scale(loss_w).backward()
                            if cfg.clip_norm > 0:
                                grad_scaler_instance.unscale_(temp_optimizer_base)
                                torch.nn.utils.clip_grad_norm_(base_model_instance.parameters(), cfg.clip_norm)
                            grad_scaler_instance.step(temp_optimizer_base)
                            grad_scaler_instance.update()
                        else:
                            loss_w.backward()
                            if cfg.clip_norm > 0:
                                torch.nn.utils.clip_grad_norm_(base_model_instance.parameters(), cfg.clip_norm)
                            temp_optimizer_base.step()

                        if (w_ep + 1) % 5 == 0 or w_ep == cfg.warmup_epochs_base - 1:
                            logging.info(f"Replica {idx + 1} Warmup Epoca Base {w_ep + 1}, Loss: {loss_w.item():.4f}")
                    del temp_optimizer_base, x_warmup_t, t_warmup_t, y_warmup_t
                else:
                    logging.warning("Dati di training per warmup vuoti, warmup saltato.")

            current_mu0_hat_tr, current_mu1_hat_tr = siamese_model_instance.predict_mu_hat(Xtr_rep, device)
            if current_mu0_hat_tr.size == 0:  # Se predict_mu_hat restituisce vuoto (es. Xtr_rep vuoto)
                logging.error(
                    f"Replica {idx + 1}: Impossibile ottenere stime ITE iniziali, Xtr_rep potrebbe essere vuoto. "
                    f"Salto questa replica.")
                results.append((np.nan, np.nan))  # Registra NaN per questa replica
                continue  # Passa alla prossima replica

            dataset_instance = DynamicContrastiveCausalDS(Xtr_rep, Ttr_rep, YFtr_rep,
                                                          current_mu0_hat_tr, current_mu1_hat_tr,
                                                          bs=cfg.batch, perc=cfg.percentile)
        else:
            logging.info("use_proxy_ite=False: si usano ITE veri per il pairing (con DynamicContrastiveCausalDS).")
            if true_m0_tr_rep.size == 0:  # Verifica che i dati veri non siano vuoti
                logging.error(f"Replica {idx + 1}: Dati ITE veri per il training vuoti. Salto questa replica.")
                results.append((np.nan, np.nan))
                continue
            dataset_instance = DynamicContrastiveCausalDS(Xtr_rep, Ttr_rep, YFtr_rep,
                                                          true_m0_tr_rep, true_m1_tr_rep,
                                                          bs=cfg.batch, perc=cfg.percentile)

        dataloader_instance = DataLoader(dataset_instance, batch_size=1, shuffle=True,
                                         collate_fn=first_item, num_workers=cfg.num_workers,
                                         pin_memory=True,
                                         persistent_workers=(cfg.num_workers > 0))

        for ep in range(1, cfg.epochs + 1):
            siamese_model_instance.train()

            if cfg.use_proxy_ite and ep > 1 and (ep - 1) % cfg.update_ite_freq == 0:
                logging.info(f"Replica {idx + 1} Epoca {ep}: Aggiornamento stime ITE per il dataset...")
                current_mu0_hat_tr, current_mu1_hat_tr = siamese_model_instance.predict_mu_hat(Xtr_rep, device)
                if current_mu0_hat_tr.size > 0:
                    dataset_instance.update_ite_estimates(current_mu0_hat_tr, current_mu1_hat_tr)
                else:
                    logging.warning(
                        f"Replica {idx + 1} Epoca {ep}: Stime ITE vuote durante aggiornamento, non aggiorno il dataset.")

            epoch_losses_vals, epoch_base_losses_vals, epoch_ctr_losses_vals = [], [], []
            for batch_idx_loop, batch_data_item in enumerate(dataloader_instance):  # Rinominato batch_idx
                batch_data_item = [t.to(device) for t in batch_data_item]
                loss_val, base_loss_val, ctr_loss_val = siamese_model_instance.step(
                    batch_data_item, grad_scaler_instance, optimizer_instance,
                    cfg.clip_norm, cfg.use_amp
                )
                if batch_data_item[0].shape[0] > 0:
                    epoch_losses_vals.append(loss_val)
                    epoch_base_losses_vals.append(base_loss_val)
                    epoch_ctr_losses_vals.append(ctr_loss_val)

            mean_loss_ep = np.mean(epoch_losses_vals) if epoch_losses_vals else 0.0
            mean_base_loss_ep = np.mean(epoch_base_losses_vals) if epoch_base_losses_vals else 0.0
            mean_ctr_loss_ep = np.mean(epoch_ctr_losses_vals) if epoch_ctr_losses_vals else 0.0

            loss_per_epoch_all_reps_df_list.append({'replica': idx, 'epoch': ep, 'loss': mean_loss_ep})

            if ep == 1 or ep % 10 == 0 or ep == cfg.epochs:
                print(
                    f"rep{idx + 1} ep{ep:03d} loss={mean_loss_ep:.3f} base={mean_base_loss_ep:.3f} ctr={mean_ctr_loss_ep:.3f}")

            with open(out_dir / "train_loss.csv", "a", newline="") as f_loss:
                csv_writer = csv.writer(f_loss)
                csv_writer.writerow(
                    [idx, ep, round(mean_loss_ep, 6), round(mean_base_loss_ep, 6), round(mean_ctr_loss_ep, 6)])

        true_ite_te_rep = m1_te_all_reps[:, idx] - m0_te_all_reps[:, idx]
        if X_te_all_reps[:, :, idx].shape[0] > 0:
            pred_ite_te_rep = siamese_model_instance.predict_ite(X_te_all_reps[:, :, idx].astype(np.float32), device)
            if pred_ite_te_rep.size > 0 and true_ite_te_rep.size > 0:
                eps_ate_val = eps_ATE_diff(pred_ite_te_rep.mean(), true_ite_te_rep.mean())
                pehe_val = PEHE_with_ite(pred_ite_te_rep, true_ite_te_rep, sqrt=True)
            else:
                logging.warning(
                    f"Replica {idx + 1}: Predizioni ITE o ITE veri di test vuoti. Metriche impostate a NaN.")
                eps_ate_val, pehe_val = np.nan, np.nan
        else:
            logging.warning(f"Replica {idx + 1}: Dati di test vuoti. Metriche impostate a NaN.")
            eps_ate_val, pehe_val = np.nan, np.nan

        results.append((eps_ate_val, pehe_val))

        with open(out_dir / "metrics.csv", "a", newline="") as f_metrics:
            csv_writer = csv.writer(f_metrics)
            csv_writer.writerow([idx, round(eps_ate_val, 6) if not np.isnan(eps_ate_val) else "NaN",
                                 round(pehe_val, 6) if not np.isnan(pehe_val) else "NaN", ""])

    total_co2_emissions = tracker.stop()
    if total_co2_emissions is None:
        total_co2_emissions = 0.0

    if 'siamese_model_instance' in locals() and siamese_model_instance is not None:  # Salva solo se il modello esiste
        torch.save(siamese_model_instance.base.state_dict(), out_dir / "model_final_base_last_rep.pt")
        print(f"Modello base (ultima replica) salvato in {out_dir / 'model_final_base_last_rep.pt'}")
    else:
        print("Nessun modello da salvare (possibile errore in tutte le repliche).")

    eps_values, pehe_values = zip(*results) if results else ([], [])
    mean_eps_ate = np.nanmean([e for e in eps_values if not np.isnan(e)]) if any(
        not np.isnan(e) for e in eps_values) else np.nan
    mean_pehe = np.nanmean([p for p in pehe_values if not np.isnan(p)]) if any(
        not np.isnan(p) for p in pehe_values) else np.nan

    with open(out_dir / "metrics.csv", "a", newline="") as f_metrics:
        csv_writer = csv.writer(f_metrics)
        csv_writer.writerow(["AVERAGE", round(mean_eps_ate, 6) if not np.isnan(mean_eps_ate) else "NaN",
                             round(mean_pehe, 6) if not np.isnan(mean_pehe) else "NaN",
                             round(total_co2_emissions, 9)])

    if loss_per_epoch_all_reps_df_list:
        df_loss_all = pd.DataFrame(loss_per_epoch_all_reps_df_list)
        if not df_loss_all.empty:
            plt.figure(figsize=(10, 6))
            for rep_idx_plot in df_loss_all['replica'].unique():
                rep_data = df_loss_all[df_loss_all['replica'] == rep_idx_plot]
                plt.plot(rep_data['epoch'], rep_data['loss'], color='gray', alpha=0.2)

            mean_loss_per_epoch_overall = df_loss_all.groupby('epoch')['loss'].mean()
            plt.plot(mean_loss_per_epoch_overall.index, mean_loss_per_epoch_overall.values, color='blue', linewidth=2,
                     label='Loss Media su Repliche')
            plt.xlabel('Epoca')
            plt.ylabel('Loss Totale Media')
            plt.title('Training Loss Media per Epoca (Tutte le Repliche)')
            plt.legend()
            plt.grid(True)
            plt.savefig(out_dir / "loss_curve.png")
            plt.close()
        else:
            logging.warning("DataFrame df_loss_all vuoto, nessun grafico di loss generato.")
    else:
        logging.warning("Nessun dato di loss registrato per il grafico.")

    print(f"Emissioni CO2 totali: {total_co2_emissions:.9f} kg")
    print(f"Risultati medi ({cfg.n_reps} repliche): EPS_ATE={mean_eps_ate:.4f}, PEHE={mean_pehe:.4f}")
    print("Completato.")


if __name__ == "__main__":
    run()
