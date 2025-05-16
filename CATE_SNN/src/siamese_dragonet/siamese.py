import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import device as torch_device
from torch.amp import autocast, GradScaler  # autocast e GradScaler sono mantenuti se use_amp è True
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

# Assumendo che DynamicContrastiveCausalDS e first_item siano definiti altrove
try:
    from src.contrastive import DynamicContrastiveCausalDS, first_item
except ImportError:
    logging.warning("src.contrastive not found, using dummy placeholders.")


    # Definizione di placeholder per DynamicContrastiveCausalDS
    class DynamicContrastiveCausalDS:  # type: ignore
        def __init__(self, X, T, Y, mu0, mu1, bs):
            self.X, self.T, self.Y, self.mu0, self.mu1, self.bs = X, T, Y, mu0, mu1, bs
            # Calcola un dummy threshold, assicurandosi che mu0 e mu1 non siano None e abbiano dati
            mu0_mean = np.nanmean(mu0) if mu0 is not None and mu0.size > 0 else 0.0
            mu1_mean = np.nanmean(mu1) if mu1 is not None and mu1.size > 0 else 0.0
            if np.isnan(mu0_mean) or np.isinf(mu0_mean): mu0_mean = 0.0
            if np.isnan(mu1_mean) or np.isinf(mu1_mean): mu1_mean = 0.0

            # Calcola thr, gestendo il caso in cui mu0 o mu1 siano None
            if mu0 is not None and mu1 is not None and mu0.size > 0 and mu1.size > 0:
                diff = np.abs(mu0_mean - mu1_mean)
                self.thr = float(diff / 2.0) if not (np.isnan(diff) or np.isinf(diff)) else 0.5
            else:
                self.thr = 0.5  # Valore di default se mu0 o mu1 non sono validi

            if np.isnan(self.thr) or np.isinf(self.thr) or self.thr < 0:  # Aggiunto controllo per thr < 0
                logging.warning(
                    f"Calculated dummy thr is invalid (NaN/Inf/Negative: {self.thr}). mu0_mean: {mu0_mean}, mu1_mean: {mu1_mean}. Resetting to 0.5.")
                self.thr = 0.5
            # logging.info(f"Dummy DynamicContrastiveCausalDS initialized. mu0_mean: {mu0_mean}, mu1_mean: {mu1_mean}, Calculated thr: {self.thr}")

        def __len__(self):
            if self.X is None or len(self.X) == 0: return 0  # Gestisce X vuoto
            # Assicurati che bs sia almeno 1 per evitare divisione per zero se bs è 0
            effective_bs = max(1, self.bs)
            return len(self.X) // effective_bs + (1 if len(self.X) % effective_bs else 0)

        def __getitem__(self, idx):
            batch_size = max(1, self.bs)

            start_idx = idx * batch_size
            if self.X is None or start_idx >= len(self.X):  # Controllo aggiunto per len(self.X) == 0
                return [torch.empty(0) for _ in range(7)]

            end_idx = min(start_idx + batch_size, len(self.X))
            actual_batch_size = end_idx - start_idx

            if actual_batch_size <= 0:
                return [torch.empty(0) for _ in range(7)]

            num_features = self.X.shape[1] if self.X.ndim > 1 and self.X.shape[1] > 0 else 1

            x1_np = self.X[start_idx:end_idx]
            y1_np = self.Y[start_idx:end_idx]
            t1_np = self.T[start_idx:end_idx]

            if len(self.X) >= actual_batch_size:
                perm_indices = np.random.choice(len(self.X), actual_batch_size,
                                                replace=(len(self.X) < actual_batch_size))
                x2_np = self.X[perm_indices]
                y2_np = self.Y[perm_indices]
                t2_np = self.T[perm_indices]
            else:
                x2_np = x1_np.copy()
                y2_np = y1_np.copy()
                t2_np = t1_np.copy()

            labels_np = np.random.randint(0, 2, size=(actual_batch_size, 1)).astype(np.float32)

            if np.isnan(x1_np).any() or np.isnan(y1_np).any() or np.isnan(t1_np).any() or \
                    np.isnan(x2_np).any() or np.isnan(y2_np).any() or np.isnan(t2_np).any():
                logging.warning(f"NaNs found in NumPy arrays sampled by __getitem__ at idx {idx}.")

            return (torch.from_numpy(x1_np.astype(np.float32)), torch.from_numpy(y1_np.astype(np.float32)),
                    torch.from_numpy(t1_np.astype(np.float32)),
                    torch.from_numpy(x2_np.astype(np.float32)), torch.from_numpy(y2_np.astype(np.float32)),
                    torch.from_numpy(t2_np.astype(np.float32)),
                    torch.from_numpy(labels_np))


    first_item = lambda x: x[0]  # type: ignore

from src.models.dragonnet import DragonNet
from src.models.utils import dragonnet_loss_binarycross, convert_pd_to_np, make_tarreg_loss


class SiameseDragonNet(nn.Module):
    """
    Siamese network that uses DragonNet as backbone for contrastive causal learning.
    """

    def __init__(self,
                 input_dim: int,
                 ds_class: type = DynamicContrastiveCausalDS,
                 margin: float = 1.0,  # Questo 'margin' viene da Hydra (cfg.margin)
                 lambda_ctr: float = 1.0,
                 device: torch_device = None,
                 **user_params):
        super().__init__()

        if isinstance(device, str):
            self.device = torch_device(device)
        elif device is None:
            self.device = torch_device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.base = DragonNet(input_dim=input_dim,
                              neurons_per_layer=user_params.get('neurons_per_layer', 200),
                              targeted_reg=user_params.get('targeted_reg', True),
                              ratio=user_params.get('ratio', 1.0))
        self.base.to(self.device)

        self.ds_class = ds_class
        # self.margin ora è fisso al valore passato da Hydra
        self.margin = margin
        if not isinstance(self.margin, (float, int)) or np.isnan(self.margin) or np.isinf(
                self.margin) or self.margin < 0:
            logging.error(f"Valore di margin iniziale da Hydra non valido: {self.margin}. Impostazione a default 1.0.")
            self.margin = 1.0
        logging.info(f"SiameseDragonNet initialized with fixed margin: {self.margin}")

        self.lambda_ctr = lambda_ctr
        self.params = {
            'val_split': user_params.get('val_split', 0.2),
            'batch_size': user_params.get('batch_size', 128),
            'optim': user_params.get('optim', 'adam'),
            'lr': user_params.get('lr', 1e-4),
            'momentum': user_params.get('momentum', 0.9),
            'epochs': user_params.get('epochs', 100),
            'patience': user_params.get('patience', 20),
            'clip_norm': user_params.get('clip_norm', 1.0),
            'use_amp': user_params.get('use_amp', False),
            'verbose': user_params.get('verbose', True),
            'warmup_epochs_base': user_params.get('warmup_epochs_base', 0)
        }

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0: logging.warning("Empty tensor 'x' in embed"); return x
        if torch.isnan(x).any() or torch.isinf(x).any():
            logging.warning("NaN/Inf detected in input to embed function (x)")
        x = x.to(self.device)

        x_repr = self.base.fc1(x)
        if torch.isnan(x_repr).any() or torch.isinf(x_repr).any():
            logging.warning(
                f"NaN/Inf detected after self.base.fc1 in embed. Input x norm: {torch.norm(x).item() if x.numel() > 0 else 'N/A'}")
            return torch.nan_to_num(x_repr)
        x_repr = F.elu(x_repr)

        x_repr = self.base.fc2(x_repr)
        if torch.isnan(x_repr).any() or torch.isinf(x_repr).any():
            logging.warning("NaN/Inf detected after self.base.fc2 in embed")
            return torch.nan_to_num(x_repr)
        x_repr = F.elu(x_repr)

        x_repr_out = self.base.fc3(x_repr)
        if torch.isnan(x_repr_out).any() or torch.isinf(x_repr_out).any():
            logging.warning("NaN/Inf detected after self.base.fc3 in embed (output)")
            return torch.nan_to_num(x_repr_out)
        x_repr_out = F.elu(x_repr_out)
        return x_repr_out

    def mu_and_embedding(self, x: torch.Tensor):
        if x.numel() == 0: logging.warning("Empty tensor 'x' in mu_and_embedding"); return torch.empty(0, 2,
                                                                                                       device=self.device), torch.empty_like(
            x)
        if torch.isnan(x).any() or torch.isinf(x).any():
            logging.warning("NaN/Inf detected in input to mu_and_embedding function (x)")
        x = x.to(self.device)
        rep = self.embed(x)

        if torch.isnan(rep).any() or torch.isinf(rep).any():
            logging.warning("NaN/Inf detected in rep from self.embed in mu_and_embedding")

        preds = self.base(x)
        if torch.isnan(preds).any() or torch.isinf(preds).any():
            logging.warning(f"NaN/Inf detected in preds from self.base(x) in mu_and_embedding. preds: {preds}")
            preds = torch.nan_to_num(preds)

        if preds.ndim < 2 or preds.shape[1] < 2:
            logging.error(
                f"preds in mu_and_embedding has unexpected shape: {preds.shape}. Expected at least 2 columns.")
            dummy_mu = torch.zeros((x.shape[0], 2), device=self.device, dtype=x.dtype)
            return dummy_mu, rep

        mu0 = preds[:, 0:1]
        mu1 = preds[:, 1:2]
        mu = torch.cat([mu0, mu1], dim=1)
        if torch.isnan(mu).any() or torch.isinf(mu).any():
            logging.warning(f"NaN/Inf detected in mu in mu_and_embedding. mu: {mu}")
        return mu, rep

    def compute_loss(self, X: torch.Tensor, T: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        if X.numel() == 0:
            logging.error("X is empty in compute_loss")
            return torch.tensor(float('nan'), device=self.device,
                                dtype=X.dtype if isinstance(X, torch.Tensor) else torch.float32)
        if T.numel() == 0: logging.error("T is empty in compute_loss"); return torch.tensor(float('nan'),
                                                                                            device=self.device,
                                                                                            dtype=T.dtype)
        if Y.numel() == 0: logging.error("Y is empty in compute_loss"); return torch.tensor(float('nan'),
                                                                                            device=self.device,
                                                                                            dtype=Y.dtype)

        if torch.isnan(X).any() or torch.isinf(X).any(): logging.warning(
            f"NaN/Inf in X for compute_loss. X norm: {torch.norm(X).item()}")
        if torch.isnan(T).any() or torch.isinf(T).any(): logging.warning("NaN/Inf in T for compute_loss")
        if torch.isnan(Y).any() or torch.isinf(Y).any(): logging.warning("NaN/Inf in Y for compute_loss")

        X, T, Y = X.to(self.device), T.to(self.device), Y.to(self.device)
        concat_true = torch.cat([Y, T], dim=1)

        concat_pred = self.base(X)

        if torch.isnan(concat_pred).any() or torch.isinf(concat_pred).any():
            logging.error(f"CRITICAL: NaN/Inf detected in concat_pred from self.base(X) in compute_loss.")
            return torch.tensor(float('nan'), device=self.device, dtype=X.dtype)

        if concat_pred.ndim < 2 or concat_pred.shape[1] < 3:
            logging.error(
                f"CRITICAL: concat_pred has unexpected shape {concat_pred.shape} in compute_loss. Expected at least 3 columns.")
            return torch.tensor(float('nan'), device=self.device, dtype=X.dtype)

        t_pred_raw = concat_pred[:, 2]
        if torch.isnan(t_pred_raw).any() or torch.isinf(t_pred_raw).any():
            logging.error(
                f"CRITICAL: NaN/Inf in t_pred_raw (propensity part of concat_pred) in compute_loss. t_pred_raw: {t_pred_raw}")
        elif (t_pred_raw < -1e-5).any() or (t_pred_raw > 1.0 + 1e-5).any():
            logging.warning(
                f"WARNING: t_pred_raw (concat_pred[:, 2]) is outside [0,1] range in compute_loss BEFORE smoothing. Values: {t_pred_raw[(t_pred_raw < -1e-5) | (t_pred_raw > 1.0 + 1e-5)]}")

        if self.base.targeted_reg:
            loss_fn = make_tarreg_loss(self.base.ratio, dragonnet_loss=dragonnet_loss_binarycross)
        else:
            loss_fn = dragonnet_loss_binarycross

        calculated_loss = loss_fn(concat_true, concat_pred)
        if torch.isnan(calculated_loss).any() or torch.isinf(calculated_loss).any():
            logging.error(f"CRITICAL: Loss calculated by loss_fn is NaN/Inf in compute_loss. Loss: {calculated_loss}")
        return calculated_loss

    def contrastive_loss(self, h1: torch.Tensor, h2: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if h1.numel() == 0 or h2.numel() == 0:
            logging.warning("Empty h1 or h2 in contrastive_loss")
            return torch.tensor(0.0, device=self.device,
                                dtype=h1.dtype if h1.numel() > 0 else (h2.dtype if h2.numel() > 0 else torch.float32))

        h1_safe = torch.nan_to_num(h1, nan=0.0, posinf=1e6, neginf=-1e6)
        h2_safe = torch.nan_to_num(h2, nan=0.0, posinf=1e6, neginf=-1e6)

        if torch.isnan(labels).any() or torch.isinf(labels).any(): logging.warning(
            "NaN/Inf in labels for contrastive_loss")

        d = torch.norm(h1_safe - h2_safe, p=2, dim=1)

        if torch.isnan(d).any() or torch.isinf(d).any():
            logging.warning(
                f"NaN/Inf in distance 'd' in contrastive_loss after nan_to_num on h1/h2. d: {d}. h1_safe norm: {torch.norm(h1_safe).item()}, h2_safe norm: {torch.norm(h2_safe).item()}")
            return torch.tensor(float('nan'), device=h1.device, dtype=h1.dtype)

        # Usa self.margin che è stato impostato in __init__ (fisso)
        # e controllato all'inizio di ogni epoca nel metodo fit.
        current_margin_to_use = self.margin
        if not isinstance(current_margin_to_use, (float, int)) or np.isnan(current_margin_to_use) or np.isinf(
                current_margin_to_use) or current_margin_to_use < 0:
            logging.error(
                f"self.margin is invalid ({current_margin_to_use}) inside contrastive_loss. Using default 1.0 for this calculation.")
            current_margin_to_use = 1.0

        relu_term = F.relu(current_margin_to_use - d)
        sim = labels.float() * d.pow(2)
        dis = (1 - labels.float()) * relu_term.pow(2)

        if torch.isinf(sim).any() or torch.isinf(dis).any():
            logging.warning(
                f"Inf detected in sim or dis. sim_mean: {sim.mean().item() if sim.numel() > 0 else 'N/A'}, dis_mean: {dis.mean().item() if dis.numel() > 0 else 'N/A'}")
            sim = torch.clamp(sim, max=1e12)
            dis = torch.clamp(dis, max=1e12)

        contr_loss = 0.5 * torch.mean(sim + dis)
        if torch.isnan(contr_loss).any() or torch.isinf(contr_loss).any():
            logging.error(
                f"CRITICAL: Contrastive loss is NaN/Inf. margin used: {current_margin_to_use}, d mean: {d.mean().item() if d.numel() > 0 else 'N/A'}")
        return contr_loss

    def fit(self, X, T, Y, best_model_path=None):
        if not isinstance(X, np.ndarray): X, T, Y = convert_pd_to_np(X, T, Y)
        if np.isnan(X).any() or np.isinf(X).any(): logging.critical("NaN/Inf in input X to fit method!")
        if np.isnan(T).any() or np.isinf(T).any(): logging.critical("NaN/Inf in input T to fit method!")
        if np.isnan(Y).any() or np.isinf(Y).any(): logging.critical("NaN/Inf in input Y to fit method!")

        N = X.shape[0]
        if N == 0:
            logging.error("Input data X is empty. Cannot fit the model.")
            return

        p = self.params
        scaler_enabled = p['use_amp'] and self.device.type == 'cuda'
        if scaler_enabled:
            logging.info("Mixed precision (use_amp=True) was configured by user params.")
        scaler_enabled = False
        if p['use_amp']:
            logging.info("Forcing mixed precision to False for NaN debugging, overriding user param 'use_amp'.")

        if p['warmup_epochs_base'] > 0:
            logging.info(f"Warmup DragonNet for {p['warmup_epochs_base']} epochs")
            self.base.fit(X, T, Y)

        idx = np.random.permutation(N)

        actual_val_split = p['val_split']
        if N * actual_val_split < 1 and N > 0:
            logging.warning(
                f"Validation split {actual_val_split} results in <1 sample for N={N}. Disabling validation.")
            actual_val_split = 0

        split_idx = int(N * (1 - actual_val_split))
        tr_idx, va_idx = idx[:split_idx], idx[split_idx:]

        if len(tr_idx) == 0:
            logging.error("Training set is empty after split. Cannot fit the model.")
            return

        mu_tr_np_raw = self.base.predict_mu_hat(X[tr_idx])
        if mu_tr_np_raw is None or mu_tr_np_raw.size == 0:
            logging.error("predict_mu_hat returned None or empty for training set. Using zeros.")
            mu_tr_np = np.zeros((len(tr_idx), 2))
        else:
            if np.isnan(mu_tr_np_raw).any() or np.isinf(mu_tr_np_raw).any():
                logging.critical(
                    f"NaN/Inf detected in mu_tr_np_raw. Mean before nan_to_num: {np.nanmean(mu_tr_np_raw) if mu_tr_np_raw.size > 0 else 'N/A'}")
            mu_tr_np = np.nan_to_num(mu_tr_np_raw, nan=0.0, posinf=1e6, neginf=-1e6)
        mu0_tr, mu1_tr = mu_tr_np[:, 0], mu_tr_np[:, 1]

        mu0_va, mu1_va = None, None
        if len(va_idx) > 0:
            mu_va_np_raw = self.base.predict_mu_hat(X[va_idx])
            if mu_va_np_raw is None or mu_va_np_raw.size == 0:
                logging.error("predict_mu_hat returned None or empty for validation set. Setting mu_va to None.")
            else:
                if np.isnan(mu_va_np_raw).any() or np.isinf(mu_va_np_raw).any():
                    logging.critical(
                        f"NaN/Inf detected in mu_va_np_raw. Mean before nan_to_num: {np.nanmean(mu_va_np_raw) if mu_va_np_raw.size > 0 else 'N/A'}")
                mu_va_np = np.nan_to_num(mu_va_np_raw, nan=0.0, posinf=1e6, neginf=-1e6)
                mu0_va, mu1_va = mu_va_np[:, 0], mu_va_np[:, 1]
        else:
            logging.info("Validation set is empty.")

        effective_batch_size_ds = max(1, p['batch_size'])
        train_ds = self.ds_class(X[tr_idx], T[tr_idx], Y[tr_idx], mu0_tr, mu1_tr, bs=effective_batch_size_ds)

        # Rimosso il logging di train_ds.thr qui perché self.margin non viene più aggiornato da esso.
        # Il valore di self.margin è quello fisso da Hydra.

        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=first_item, num_workers=0)
        val_loader = None
        if len(va_idx) > 0 and mu0_va is not None and mu1_va is not None:
            val_ds = self.ds_class(X[va_idx], T[va_idx], Y[va_idx], mu0_va, mu1_va, bs=effective_batch_size_ds)
            val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=first_item, num_workers=0)
        else:
            logging.info("Validation loader not created.")

        Optim = Adam if p['optim'] == 'adam' else SGD
        optim_kwargs = {'lr': p['lr']}
        if p['optim'] == 'sgd': optim_kwargs['momentum'] = p['momentum']
        optimizer = Optim(self.parameters(), **optim_kwargs)

        scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=max(3, p['patience'] // 2), min_lr=1e-8)

        scaler = GradScaler(enabled=scaler_enabled)

        best_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(1, p['epochs'] + 1):
            self.train()
            total_train_loss = 0.0
            num_train_batches = 0

            # self.margin è ora fisso e non viene aggiornato da train_ds.thr
            # Controllo di sicurezza per self.margin all'inizio dell'epoca
            if not isinstance(self.margin, (float, int)) or np.isnan(self.margin) or np.isinf(
                    self.margin) or self.margin < 0:
                logging.error(
                    f"CRITICAL: self.margin (fisso da Hydra) è invalido ({self.margin}) all'inizio dell'epoca {epoch}. Reimpostazione a 1.0.")
                self.margin = 1.0

            for batch_idx, batch_data in enumerate(train_loader):
                if batch_data is None or not isinstance(batch_data, (tuple, list)) or len(batch_data) != 7:
                    logging.error(f"Epoch {epoch}, Batch {batch_idx}: Invalid batch data format. Skipping batch.")
                    continue

                x1, y1, t1, x2, y2, t2, labels = batch_data
                if any(not isinstance(t, torch.Tensor) or t.numel() == 0 for t in [x1, y1, t1, x2, y2, t2, labels]):
                    logging.warning(f"Skipping batch {batch_idx} in train_loader due to one or more empty tensors.")
                    continue

                x1_d, y1_d, t1_d, x2_d, y2_d, t2_d, labels_d = [
                    b.to(self.device) for b in (x1, y1, t1, x2, y2, t2, labels)
                ]

                optimizer.zero_grad(set_to_none=True)

                with autocast(device_type=self.device.type, enabled=scaler_enabled):
                    h1 = self.embed(x1_d)
                    h2 = self.embed(x2_d)
                    ctr_loss = self.contrastive_loss(h1, h2, labels_d)

                XY_d = torch.cat([x1_d, x2_d], 0)
                TT_d = torch.cat([t1_d, t2_d], 0)
                YY_d = torch.cat([y1_d, y2_d], 0)
                base_loss = self.compute_loss(XY_d, TT_d, YY_d)

                base_loss_is_nan = torch.isnan(base_loss)
                ctr_loss_is_nan = torch.isnan(ctr_loss)

                if base_loss_is_nan:
                    logging.error(f"Epoch {epoch}, Batch {batch_idx}: base_loss is NaN.")
                if ctr_loss_is_nan:
                    logging.error(f"Epoch {epoch}, Batch {batch_idx}: ctr_loss is NaN.")

                loss = base_loss.float() + self.lambda_ctr * ctr_loss.float()

                if torch.isnan(loss):
                    logging.error(
                        f"Epoch {epoch}, Batch {batch_idx}: Total loss is NaN. base_loss: {base_loss.item() if not base_loss_is_nan else 'NaN'}, ctr_loss: {ctr_loss.item() if not ctr_loss_is_nan else 'NaN'}. Skipping backward.")
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.parameters(), p['clip_norm'])
                scaler.step(optimizer)
                scaler.update()

                total_train_loss += loss.item()
                num_train_batches += 1

            avg_tr_loss = total_train_loss / num_train_batches if num_train_batches > 0 else float('inf')
            avg_val_loss = float('inf')

            if val_loader:
                self.eval()
                val_total_loss, val_num_batches = 0.0, 0
                with torch.no_grad():
                    for batch_idx_val, batch_data_val in enumerate(val_loader):
                        if not isinstance(batch_data_val, (tuple, list)) or len(batch_data_val) != 7:
                            logging.error(
                                f"Epoch {epoch}, Val Batch {batch_idx_val}: Invalid batch data format. Skipping.")
                            continue

                        x1_v, y1_v, t1_v, x2_v, y2_v, t2_v, labels_v = batch_data_val
                        if any(not isinstance(t, torch.Tensor) or t.numel() == 0 for t in
                               [x1_v, y1_v, t1_v, x2_v, y2_v, t2_v, labels_v]):
                            logging.warning(f"Skipping empty tensor in batch {batch_idx_val} in val_loader")
                            continue

                        x1_v_d, y1_v_d, t1_v_d, x2_v_d, y2_v_d, t2_v_d, labels_v_d = [
                            b.to(self.device) for b in (x1_v, y1_v, t1_v, x2_v, y2_v, t2_v, labels_v)
                        ]

                        h1_val = self.embed(x1_v_d)
                        h2_val = self.embed(x2_v_d)
                        ctr_loss_val = self.contrastive_loss(h1_val, h2_val, labels_v_d)

                        XY_val_d = torch.cat([x1_v_d, x2_v_d], 0)
                        TT_val_d = torch.cat([t1_v_d, t2_v_d], 0)
                        YY_val_d = torch.cat([y1_v_d, y2_v_d], 0)
                        base_loss_val = self.compute_loss(XY_val_d, TT_val_d, YY_val_d)

                        if not torch.isnan(base_loss_val) and not torch.isnan(ctr_loss_val):
                            current_val_loss = base_loss_val.float() + self.lambda_ctr * ctr_loss_val.float()
                            val_total_loss += current_val_loss.item()
                            val_num_batches += 1
                        else:
                            logging.warning(
                                f"Epoch {epoch}, Val Batch {batch_idx_val}: Validation loss component is NaN. base_val: {base_loss_val.item() if not torch.isnan(base_loss_val) else 'NaN'}, ctr_val: {ctr_loss_val.item() if not torch.isnan(ctr_loss_val) else 'NaN'}")

                avg_val_loss = val_total_loss / val_num_batches if val_num_batches > 0 else float('inf')

            scheduler_metric = avg_val_loss if val_loader and avg_val_loss != float('inf') else (
                best_loss if best_loss != float('inf') else avg_tr_loss)
            if scheduler_metric != float('inf'):
                scheduler.step(scheduler_metric)

            current_epoch_eval_loss = avg_val_loss if val_loader and avg_val_loss != float('inf') else avg_tr_loss

            if current_epoch_eval_loss != float('inf') and current_epoch_eval_loss < best_loss - 1e-6:
                best_loss = current_epoch_eval_loss
                patience_counter = 0
                if best_model_path:
                    try:
                        best_state = self.state_dict()
                        torch.save(best_state, best_model_path)
                    except Exception as e:
                        logging.error(f"Error saving model: {e}")
            elif current_epoch_eval_loss != float('inf'):
                patience_counter += 1

            if patience_counter >= p['patience']:
                logging.info(f"Early stopping at epoch {epoch} due to no improvement for {p['patience']} epochs.")
                break

            if p['verbose']:
                # self.margin qui sarà il valore fisso da Hydra
                logging.info(
                    f"Epoch {epoch}/{p['epochs']}: train_loss={avg_tr_loss:.4f}, val_loss={avg_val_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.1e}, margin={self.margin:.4f}")

        if best_state:
            try:
                self.load_state_dict(best_state)
                logging.info(f"Loaded best model state with loss: {best_loss:.4f}")
            except Exception as e:
                logging.error(f"Error loading best model state: {e}")
        else:
            logging.warning("No best model state was saved or loaded during training.")

    def predict_ite(self, X):
        if not isinstance(X, np.ndarray):
            X_np = convert_pd_to_np(X)
        else:
            X_np = X
        if X_np is None or X_np.size == 0:
            logging.error("Input X to predict_ite is empty or None after conversion.")
            return np.array([])
        return self.base.predict_ite(X_np)

    def predict_ate(self, X):
        if not isinstance(X, np.ndarray):
            X_np = convert_pd_to_np(X)
        else:
            X_np = X
        if X_np is None or X_np.size == 0:
            logging.error("Input X to predict_ate is empty or None after conversion.")
            return np.nan
        return self.base.predict_ate(X_np)

    def predict_mu_hat(self, X):
        if not isinstance(X, np.ndarray):
            X_np = convert_pd_to_np(X)
        else:
            X_np = X
        if X_np is None or X_np.size == 0:
            logging.error("Input X to predict_mu_hat is empty or None after conversion.")
            return np.full((0, 2), np.nan) if X_np is None else np.full((X_np.shape[0] if X_np is not None else 0, 2),
                                                                        np.nan)
        return self.base.predict_mu_hat(X_np)

