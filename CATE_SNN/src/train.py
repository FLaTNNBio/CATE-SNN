import os, random, csv, logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch

from CATE_SNN.src.contrastive import first_item, DynamicContrastiveCausalDS
from CATE_SNN.src.siamese import SiameseBCAUSS

try:
    from torch.amp import GradScaler  # PyTorch ≥ 2.1
except ImportError:
    from torch.cuda.amp import GradScaler  # type: ignore
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from codecarbon import EmissionsTracker

from CATE_SNN.src.data_loader import DataLoader as CFLoader  # Assumendo che questo sia un tuo modulo locale
from CATE_SNN.src.metrics import eps_ATE_diff, PEHE_with_ite  # Assumendo che questo sia un tuo modulo locale
from CATE_SNN.src.models.bcauss import BCAUSS  # Assumendo che questo sia un tuo modulo locale

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
