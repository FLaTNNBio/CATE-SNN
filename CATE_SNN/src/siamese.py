#!/usr/bin/env python3
import logging
import numpy as np
import torch
from torch import nn
# torch.cuda.amp è deprecato, usa torch.amp per il futuro
# from torch.cuda.amp import autocast, GradScaler # Deprecato
from torch.amp import autocast  # Per 'cuda' o 'cpu'
from torch.amp.grad_scaler import GradScaler  # Specifico per CUDA, o generico torch.amp.GradScaler
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset  # Aggiunto Dataset per chiarezza

from sklearn.preprocessing import StandardScaler


# Assumendo che convert_pd_to_np e first_item siano definiti correttamente
from src.models.utils import convert_pd_to_np # Se è in un file utils.py dentro models
from src.contrastive import first_item # Se è in contrastive.py





class SiameseBCAUSS(nn.Module):
    """
    Siamese wrapper for BCAUSS with integrated fit logic.
    """

    def __init__(
            self,
            base_model: nn.Module,
            ds_class: type[Dataset],  # La classe del dataset contrastivo (es. PairDatasetFromITE)
            margin: float = 1.0,
            lambda_ctr: float = 1.0,
            **user_params
    ):
        super().__init__()
        self.base = base_model
        # default fit params
        p = {
            'val_split': 0.2,
            'batch_size': 128,  # Questo batch_size è per il dataset contrastivo, se lo usa
            'optim': 'adam',
            'lr': 1e-4,
            'momentum': 0.9,
            'epochs': 100,
            'patience': 20,  # Patience per l'early stopping del training siamese
            'clip_norm': 1.0,
            'use_amp': False,  # Automatic Mixed Precision
            'verbose': True,
            'update_ite_freq': 1,  # Frequenza di aggiornamento delle stime ITE
            'warmup_epochs_base': 0,  # Epoche per il warmup del modello base BCAUSS
        }
        p.update(user_params)
        self.params = p
        self.ds_class = ds_class
        self.margin = margin
        self.lambda_ctr = lambda_ctr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base.to(self.device)  # Sposta il modello base sul dispositivo corretto

    def contrastive_loss(self, h1: torch.Tensor, h2: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calcola la loss contrastiva.
        h1, h2: embeddings dei due elementi della coppia. Shape: (batch_size_pairs, embedding_dim)
        labels: 0 se dissimili (da allontanare), 1 se simili (da avvicinare). Shape: (batch_size_pairs,)
        """
        d = torch.norm(h1 - h2, p=2, dim=1)  # Distanza euclidea, shape: (batch_size_pairs,)
        # Per coppie simili (label=1), la loss è d^2 (minimizzare la distanza)
        loss_sim = d.pow(2)
        # Per coppie dissimili (label=0), la loss è max(0, margin - d)^2 (distanza almeno margin)
        loss_dis = torch.clamp(self.margin - d, min=0).pow(2)

        # labels è 1 per simili, 0 per dissimili.
        loss = labels * loss_sim + (1 - labels) * loss_dis
        return loss.mean()  # Media della loss sulle coppie nel batch

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray):
        # prepare numpy data
        # Assicurati che X, T, Y siano NumPy array qui
        if not all(isinstance(arr, np.ndarray) for arr in [X, T, Y]):
            logging.info("Converting input data to NumPy arrays for SiameseBCAUSS.fit")
            X_np, T_np, Y_np = convert_pd_to_np(X, T, Y)
        else:
            X_np, T_np, Y_np = X, T, Y

        N = X_np.shape[0]
        p = self.params

        # Warmup base model (BCAUSS)
        Y_flat = Y_np.reshape(-1, 1)
        if p['warmup_epochs_base'] > 0:
            logging.info(f"Starting BCAUSS warmup for {p['warmup_epochs_base']} epochs...")
            self.base.fit(X_np, T_np, Y_flat, epochs=p['warmup_epochs_base'])
            logging.info("BCAUSS warmup finished.")

        self.base.to(self.device)

        # Split data into training and validation sets
        idx = np.random.permutation(N)
        split_idx = int(N * (1 - p['val_split']))
        tr_idx, va_idx = idx[:split_idx], idx[split_idx:]

        if len(tr_idx) == 0:
            logging.error("Training set is empty after split. Check val_split and data size.")
            return
        if len(va_idx) == 0:
            logging.warning(
                "Validation set is empty after split. Proceeding without validation for ITE updates if needed by dataset.")
            # Considera come gestire l'assenza di un validation set per val_ds

        # Initial ITE (Individual Treatment Effect) estimates from the base model
        with torch.no_grad():
            mu_tr_tensor, _ = self.base.mu_and_embedding(
                torch.tensor(X_np[tr_idx], dtype=torch.float32, device=self.device)
            )
        mu_tr = mu_tr_tensor.detach()
        mu0_tr = mu_tr[:, 0].cpu().numpy()
        mu1_tr = mu_tr[:, 1].cpu().numpy()

        mu0_va, mu1_va = None, None
        if len(va_idx) > 0:
            with torch.no_grad():
                mu_va_tensor, _ = self.base.mu_and_embedding(
                    torch.tensor(X_np[va_idx], dtype=torch.float32, device=self.device)
                )
            mu_va = mu_va_tensor.detach()
            mu0_va = mu_va[:, 0].cpu().numpy()
            mu1_va = mu_va[:, 1].cpu().numpy()

        # Create datasets
        # Il parametro 'bs' qui è passato al costruttore del dataset ds_class.
        # Se ds_class non lo usa, potrebbe essere rimosso o gestito diversamente.
        train_ds = self.ds_class(X_np[tr_idx], T_np[tr_idx], Y_np[tr_idx], mu0_tr, mu1_tr, bs=p['batch_size'])

        val_loader = None
        if len(va_idx) > 0 and mu0_va is not None and mu1_va is not None:
            val_ds = self.ds_class(X_np[va_idx], T_np[va_idx], Y_np[va_idx], mu0_va, mu1_va, bs=p['batch_size'])
            val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=first_item, num_workers=0)
        else:
            # Se non c'è validation set, val_ds non viene creato, val_loader rimane None
            logging.info(
                "Validation set is empty or ITEs for validation could not be computed. Siamese training will proceed without validation loop.")

        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=first_item, num_workers=0)

        # Optimizer
        optim_cls = Adam if p['optim'] == 'adam' else SGD
        optim_kwargs = {'lr': p['lr']}
        if p['optim'] == 'sgd':
            optim_kwargs['momentum'] = p['momentum']
        optimizer = optim_cls(self.parameters(), **optim_kwargs)

        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=max(1, p['patience'] // 3))

        # GradScaler per Automatic Mixed Precision (AMP)
        # CORREZIONE: device_type non è un argomento per GradScaler.__init__
        # L'abilitazione dipende da use_amp e se il device è cuda.
        # autocast gestirà il device_type.
        use_cuda_for_amp = p['use_amp'] and self.device.type == 'cuda'
        scaler = GradScaler(enabled=use_cuda_for_amp)

        best_val_loss, patience_counter, best_state_dict = float('inf'), 0, None

        logging.info(
            f"Starting Siamese training for {p['epochs']} epochs on device {self.device}. AMP enabled: {use_cuda_for_amp}")

        # Training loop
        for epoch in range(1, p['epochs'] + 1):
            self.train()
            total_train_loss = 0.0
            num_train_batches = 0

            for batch_data in train_loader:
                x1, y1_true, t1_true, x2, y2_true, t2_true, labels_ctr = batch_data

                x1, y1_true, t1_true = x1.to(self.device), y1_true.to(self.device), t1_true.to(self.device)
                x2, y2_true, t2_true = x2.to(self.device), y2_true.to(self.device), t2_true.to(self.device)
                labels_ctr = labels_ctr.to(self.device).float()

                with autocast(device_type=self.device.type, dtype=torch.float16 if use_cuda_for_amp else None,
                              enabled=use_cuda_for_amp):
                    mu_preds1, h1 = self.base.mu_and_embedding(x1)
                    mu_preds2, h2 = self.base.mu_and_embedding(x2)

                    X_combined = torch.cat([x1, x2], dim=0)
                    T_combined_true = torch.cat([t1_true, t2_true], dim=0)
                    Y_combined_true = torch.cat([y1_true, y2_true], dim=0)

                    Y_combined_for_base_loss = Y_combined_true
                    if self.base.y_scaler is not None and hasattr(self.base.y_scaler, 'transform'):
                        y_np_original = Y_combined_true.cpu().numpy()
                        original_shape = y_np_original.shape
                        y_np_reshaped = y_np_original.reshape(-1, 1) if len(original_shape) == 1 else y_np_original

                        try:
                            y_np_scaled = self.base.y_scaler.transform(y_np_reshaped)
                            Y_combined_for_base_loss = torch.tensor(y_np_scaled, dtype=torch.float32,
                                                                    device=self.device).reshape(original_shape)
                        except Exception as e:
                            logging.error(f"Error during y_scaler.transform: {e}. Using unscaled Y for base loss.")
                            Y_combined_for_base_loss = Y_combined_true  # Fallback

                    base_loss = self.base.compute_loss(X_combined, T_combined_true, Y_combined_for_base_loss)
                    ctr_loss = self.contrastive_loss(h1, h2, labels_ctr)
                    loss = base_loss + self.lambda_ctr * ctr_loss

                if torch.isnan(loss) or torch.isinf(loss):
                    logging.warning(
                        f"NaN or Inf loss detected during training epoch {epoch}. Loss: {loss.item()}. Skipping batch.")
                    if torch.isnan(base_loss) or torch.isinf(base_loss):
                        logging.warning(f"Base loss component was NaN/Inf: {base_loss.item()}")
                    if torch.isnan(ctr_loss) or torch.isinf(ctr_loss):
                        logging.warning(f"Contrastive loss component was NaN/Inf: {ctr_loss.item()}")
                    # Considera di fermare il training o di investigare ulteriormente
                    continue

                optimizer.zero_grad(set_to_none=True)  # set_to_none=True per potenziale miglioramento performance
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.parameters(), p['clip_norm'])
                scaler.step(optimizer)
                scaler.update()

                total_train_loss += loss.item()
                num_train_batches += 1

            avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else 0.0

            if epoch % p['update_ite_freq'] == 0:
                logging.debug(f"Updating ITE estimates at epoch {epoch}...")
                with torch.no_grad():
                    mu_all_tensor, _ = self.base.mu_and_embedding(
                        torch.tensor(X_np, dtype=torch.float32, device=self.device)
                    )
                mu_all_detached = mu_all_tensor.detach()
                mu0_all_updated = mu_all_detached[:, 0].cpu().numpy()
                mu1_all_updated = mu_all_detached[:, 1].cpu().numpy()

                mu0_tr_slice = mu0_all_updated[tr_idx]
                mu1_tr_slice = mu1_all_updated[tr_idx]
                if hasattr(train_ds, 'update_ite_estimates'):
                    train_ds.update_ite_estimates(mu0_tr_slice, mu1_tr_slice)

                if len(va_idx) > 0 and val_loader is not None and hasattr(val_ds,
                                                                          'update_ite_estimates'):  # val_ds esiste
                    mu0_va_slice = mu0_all_updated[va_idx]
                    mu1_va_slice = mu1_all_updated[va_idx]
                    val_ds.update_ite_estimates(mu0_va_slice, mu1_va_slice)
                logging.debug("ITE estimates updated for datasets (if applicable).")

            # Validation step (solo se val_loader è stato creato)
            avg_val_loss = float('inf')  # Default se non c'è validazione
            if val_loader is not None:
                self.eval()
                total_val_loss = 0.0
                num_val_batches = 0
                with torch.no_grad():
                    for batch_data_val in val_loader:
                        x1_val, y1_val_true, t1_val_true, x2_val, y2_val_true, t2_val_true, labels_ctr_val = batch_data_val

                        x1_val, y1_val_true, t1_val_true = x1_val.to(self.device), y1_val_true.to(
                            self.device), t1_val_true.to(self.device)
                        x2_val, y2_val_true, t2_val_true = x2_val.to(self.device), y2_val_true.to(
                            self.device), t2_val_true.to(self.device)
                        labels_ctr_val = labels_ctr_val.to(self.device).float()

                        with autocast(device_type=self.device.type, dtype=torch.float16 if use_cuda_for_amp else None,
                                      enabled=use_cuda_for_amp):
                            mu_preds1_val, h1_val = self.base.mu_and_embedding(x1_val)
                            mu_preds2_val, h2_val = self.base.mu_and_embedding(x2_val)

                            X_combined_val = torch.cat([x1_val, x2_val], dim=0)
                            T_combined_val_true = torch.cat([t1_val_true, t2_val_true], dim=0)
                            Y_combined_val_true = torch.cat([y1_val_true, y2_val_true], dim=0)

                            Y_combined_for_base_loss_val = Y_combined_val_true
                            if self.base.y_scaler is not None and hasattr(self.base.y_scaler, 'transform'):
                                y_np_original_val = Y_combined_val_true.cpu().numpy()
                                original_shape_val = y_np_original_val.shape
                                y_np_reshaped_val = y_np_original_val.reshape(-1, 1) if len(
                                    original_shape_val) == 1 else y_np_original_val

                                try:
                                    y_np_scaled_val = self.base.y_scaler.transform(y_np_reshaped_val)
                                    Y_combined_for_base_loss_val = torch.tensor(y_np_scaled_val, dtype=torch.float32,
                                                                                device=self.device).reshape(
                                        original_shape_val)
                                except Exception as e:
                                    logging.error(
                                        f"Error during y_scaler.transform (validation): {e}. Using unscaled Y.")
                                    Y_combined_for_base_loss_val = Y_combined_val_true

                            base_loss_val = self.base.compute_loss(X_combined_val, T_combined_val_true,
                                                                   Y_combined_for_base_loss_val)
                            ctr_loss_val = self.contrastive_loss(h1_val, h2_val, labels_ctr_val)
                            val_loss_batch = base_loss_val + self.lambda_ctr * ctr_loss_val

                        if not (torch.isnan(val_loss_batch) or torch.isinf(val_loss_batch)):
                            total_val_loss += val_loss_batch.item()
                        else:
                            logging.warning(f"NaN or Inf validation loss detected epoch {epoch}. Skipping batch.")
                        num_val_batches += 1

                avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else float('inf')

            # Se non c'è validazione, lo scheduler e l'early stopping potrebbero basarsi sulla training loss
            # o essere disabilitati/modificati. Per ora, lo scheduler usa avg_val_loss (che sarà inf se non c'è val).
            if val_loader is not None:  # Solo se c'è validazione
                scheduler.step(avg_val_loss)
            else:  # Opzionale: step sulla training loss se non c'è validazione
                scheduler.step(avg_train_loss)

            if p['verbose']:
                val_loss_str = f"{avg_val_loss:.4f}" if val_loader is not None else "N/A"
                logging.info(
                    f"Epoch {epoch}/{p['epochs']} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss_str} | LR: {optimizer.param_groups[0]['lr']:.2e}")

            # Early stopping logic
            # Se non c'è validation_loader, best_val_loss rimane inf, quindi l'early stopping non si attiva
            # a meno che non si modifichi per usare avg_train_loss.
            current_loss_for_stopping = avg_val_loss if val_loader is not None else avg_train_loss  # Scegli su cosa basare l'early stopping

            if current_loss_for_stopping < best_val_loss - 1e-6:
                best_val_loss = current_loss_for_stopping
                best_state_dict = self.state_dict()
                patience_counter = 0
                if p['verbose']:
                    logging.debug(f"New best loss: {best_val_loss:.4f}. Saving model state.")
            else:
                patience_counter += 1
                if p['verbose']:
                    logging.debug(f"Loss did not improve. Patience: {patience_counter}/{p['patience']}.")
                if patience_counter >= p['patience']:
                    logging.info(
                        f"Early stopping triggered at epoch {epoch} due to no improvement for {p['patience']} epochs.")
                    break

        if best_state_dict:
            logging.info(f"Loading best model state with loss: {best_val_loss:.4f}")
            self.load_state_dict(best_state_dict)
        else:
            logging.warning("No best model state was saved (e.g., training stopped or did not improve).")

    def predict_ite(self, X: np.ndarray) -> np.ndarray:
        X_np = convert_pd_to_np(X) if not isinstance(X, np.ndarray) else X
        return self.base.predict_ite(X_np)

    def predict_ate(self, X: np.ndarray) -> float:
        X_np = convert_pd_to_np(X) if not isinstance(X, np.ndarray) else X
        return self.base.predict_ate(X_np)

    def predict_mu_hat(self, X: np.ndarray) -> np.ndarray:
        X_np = convert_pd_to_np(X) if not isinstance(X, np.ndarray) else X
        if hasattr(self.base, 'predict_mu_hat'):
            # Se predict_mu_hat si aspetta il device, passalo. Altrimenti, rimuovilo.
            # return self.base.predict_mu_hat(X_np, self.device)
            return self.base.predict_mu_hat(X_np)  # Assumendo che predict_mu_hat non prenda device
        else:
            logging.warning("predict_mu_hat not implemented in the base model. Using fallback via mu_and_embedding.")
            self.base.eval()
            # BCAUSS.mu_and_embedding dovrebbe gestire internamente lo scaling di X se necessario
            xt = torch.tensor(X_np, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                mu_preds, _ = self.base.mu_and_embedding(xt)

            mu_preds_np = mu_preds.cpu().numpy()
            # Se le predizioni di mu_and_embedding sono scalate e devono essere de-scalate:
            if self.base.y_scaler is not None and hasattr(self.base.y_scaler, 'inverse_transform') and \
                    self.base.params.get('scale_preds', False):  # scale_preds è un param di BCAUSS

                y0_scaled = mu_preds_np[:, 0].reshape(-1, 1)
                y1_scaled = mu_preds_np[:, 1].reshape(-1, 1)
                try:
                    y0_original = self.base.y_scaler.inverse_transform(y0_scaled)
                    y1_original = self.base.y_scaler.inverse_transform(y1_scaled)
                    return np.concatenate([y0_original, y1_original], axis=1)
                except Exception as e:
                    logging.error(f"Error during y_scaler.inverse_transform: {e}. Returning scaled mu_hat.")
                    return mu_preds_np
            else:
                return mu_preds_np
