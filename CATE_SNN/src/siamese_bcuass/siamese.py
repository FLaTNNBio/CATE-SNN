#!/usr/bin/env python3
import logging
import numpy as np
import torch
from torch import nn
from torch.amp import autocast
from torch.amp.grad_scaler import GradScaler
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from src.models.utils import convert_pd_to_np
from src.contrastive import first_item
from src.metrics import PEHE_with_ite


class SiameseBCAUSS(nn.Module):
    """
    Siamese wrapper for BCAUSS with integrated fit logic and ablation support.

    Key update:
    - if X_val_np and true_ite_val are provided, model selection / early stopping
      is performed on validation PEHE (sqrt=True), while training still optimizes
      the composite loss.
    - avoids the internal train/val split when an external validation set is given.
    """

    def __init__(
            self,
            base_model: nn.Module,
            ds_class: type[Dataset],
            margin: float = 1.0,
            lambda_ctr: float = 1.0,
            **user_params
    ):
        super().__init__()
        self.base = base_model
        p = {
            'val_split': 0.2,
            'batch_size': 128,
            'optim': 'adam',
            'lr': 1e-4,
            'momentum': 0.9,
            'epochs': 100,
            'patience': 20,
            'clip_norm': 1.0,
            'use_amp': False,
            'verbose': True,
            'update_ite_freq': 1,
            'warmup_epochs_base': 0,
            'pairing_strategy': 'dynamic_ite'
        }
        p.update(user_params)
        self.params = p
        self.ds_class = ds_class
        self.margin = margin
        self.lambda_ctr = lambda_ctr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base.to(self.device)

    def contrastive_loss(self, h1: torch.Tensor, h2: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        d = torch.norm(h1 - h2, p=2, dim=1)
        loss_sim = d.pow(2)
        loss_dis = torch.clamp(self.margin - d, min=0).pow(2)
        loss = labels * loss_sim + (1 - labels) * loss_dis
        return loss.mean()

    def fit(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        X_val_np: np.ndarray | None = None,
        T_val_np: np.ndarray | None = None,
        Y_val_np: np.ndarray | None = None,
        true_ite_val: np.ndarray | None = None,
        best_model_path=None
    ):
        if not all(isinstance(arr, np.ndarray) for arr in [X, T, Y]):
            X_np, T_np, Y_np = convert_pd_to_np(X, T, Y)
        else:
            X_np, T_np, Y_np = X, T, Y

        p = self.params
        N = X_np.shape[0]

        # ------------------------------------------------------------
        # Warm-up base model (optional)
        # ------------------------------------------------------------
        Y_flat = Y_np.reshape(-1, 1)
        if p['warmup_epochs_base'] > 0:
            self.base.fit(X_np, T_np, Y_flat, epochs=p['warmup_epochs_base'])

        self.base.to(self.device)

        # ------------------------------------------------------------
        # Train/validation split
        # - if external validation is provided, use it directly
        # - otherwise fallback to internal split
        # ------------------------------------------------------------
        external_validation = X_val_np is not None and T_val_np is not None and Y_val_np is not None

        if external_validation:
            Xtr_np, Ttr_np, Ytr_np = X_np, T_np, Y_np
            Xva_np, Tva_np, Yva_np = X_val_np, T_val_np, Y_val_np
        else:
            idx = np.random.permutation(N)
            split_idx = int(N * (1 - p['val_split']))
            tr_idx, va_idx = idx[:split_idx], idx[split_idx:]

            if len(tr_idx) == 0:
                return

            Xtr_np, Ttr_np, Ytr_np = X_np[tr_idx], T_np[tr_idx], Y_np[tr_idx]
            Xva_np, Tva_np, Yva_np = (
                X_np[va_idx], T_np[va_idx], Y_np[va_idx]
            ) if len(va_idx) > 0 else (None, None, None)

        # ------------------------------------------------------------
        # Initial pseudo-ITE estimates for pair construction
        # ------------------------------------------------------------
        with torch.no_grad():
            mu_tr_tensor, _ = self.base.mu_and_embedding(
                torch.tensor(Xtr_np, dtype=torch.float32, device=self.device)
            )
        mu_tr = mu_tr_tensor.detach()
        mu0_tr, mu1_tr = mu_tr[:, 0].cpu().numpy(), mu_tr[:, 1].cpu().numpy()

        mu0_va, mu1_va = None, None
        if Xva_np is not None and len(Xva_np) > 0:
            with torch.no_grad():
                mu_va_tensor, _ = self.base.mu_and_embedding(
                    torch.tensor(Xva_np, dtype=torch.float32, device=self.device)
                )
            mu_va = mu_va_tensor.detach()
            mu0_va, mu1_va = mu_va[:, 0].cpu().numpy(), mu_va[:, 1].cpu().numpy()

        # ------------------------------------------------------------
        # Datasets / loaders
        # ------------------------------------------------------------
        train_ds = self.ds_class(
            Xtr_np, Ttr_np, Ytr_np,
            mu0_tr, mu1_tr,
            bs=p['batch_size'],
            strategy=p['pairing_strategy']
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=1,
            shuffle=True,
            collate_fn=first_item,
            num_workers=0
        )

        val_loader = None
        if Xva_np is not None and mu0_va is not None and mu1_va is not None:
            val_strategy = 'static_ite' if p['pairing_strategy'] == 'dynamic_ite' else p['pairing_strategy']
            val_ds = self.ds_class(
                Xva_np, Tva_np, Yva_np,
                mu0_va, mu1_va,
                bs=p['batch_size'],
                strategy=val_strategy
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=1,
                shuffle=False,
                collate_fn=first_item,
                num_workers=0
            )

        # ------------------------------------------------------------
        # Optimizer / scheduler
        # ------------------------------------------------------------
        optim_cls = Adam if p['optim'] == 'adam' else SGD
        optim_kwargs = {'lr': p['lr']}
        if p['optim'] == 'sgd':
            optim_kwargs['momentum'] = p['momentum']
        optimizer = optim_cls(self.parameters(), **optim_kwargs)

        # scheduler will follow the same metric used for checkpoint selection
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        use_cuda_for_amp = p['use_amp'] and self.device.type == 'cuda'
        scaler = GradScaler(enabled=use_cuda_for_amp)

        best_monitor_metric = float('inf')
        patience_counter = 0
        best_state_dict = None

        for epoch in range(1, p['epochs'] + 1):
            self.train()
            total_train_loss = 0.0
            num_train_batches = 0

            for batch_data in train_loader:
                x1, y1_true, t1_true, x2, y2_true, t2_true, labels_ctr = batch_data
                x1, y1_true, t1_true = x1.to(self.device), y1_true.to(self.device), t1_true.to(self.device)
                x2, y2_true, t2_true = x2.to(self.device), y2_true.to(self.device), t2_true.to(self.device)
                labels_ctr = labels_ctr.to(self.device).float()

                with autocast(
                    device_type=self.device.type,
                    dtype=torch.float16 if use_cuda_for_amp else None,
                    enabled=use_cuda_for_amp
                ):
                    mu_preds1, h1 = self.base.mu_and_embedding(x1)
                    mu_preds2, h2 = self.base.mu_and_embedding(x2)

                    X_combined = torch.cat([x1, x2], dim=0)
                    T_combined_true = torch.cat([t1_true, t2_true], dim=0)
                    Y_combined_true = torch.cat([y1_true, y2_true], dim=0)

                    Y_combined_for_base_loss = Y_combined_true
                    if self.base.y_scaler is not None and hasattr(self.base.y_scaler, 'transform'):
                        try:
                            y_np_scaled = self.base.y_scaler.transform(
                                Y_combined_true.cpu().numpy().reshape(-1, 1)
                            )
                            Y_combined_for_base_loss = torch.tensor(
                                y_np_scaled,
                                dtype=torch.float32,
                                device=self.device
                            ).reshape(Y_combined_true.shape)
                        except Exception:
                            pass

                    base_loss = self.base.compute_loss(X_combined, T_combined_true, Y_combined_for_base_loss)
                    ctr_loss = self.contrastive_loss(h1, h2, labels_ctr)
                    loss = base_loss + self.lambda_ctr * ctr_loss

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.parameters(), p['clip_norm'])
                scaler.step(optimizer)
                scaler.update()

                total_train_loss += loss.item()
                num_train_batches += 1

            avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else 0.0

            # --------------------------------------------------------
            # Update pseudo-ITE estimates for pair construction
            # train only
            # --------------------------------------------------------
            if epoch % p['update_ite_freq'] == 0:
                with torch.no_grad():
                    mu_train_tensor, _ = self.base.mu_and_embedding(
                        torch.tensor(Xtr_np, dtype=torch.float32, device=self.device)
                    )
                mu_train_detached = mu_train_tensor.detach()
                mu0_train_updated = mu_train_detached[:, 0].cpu().numpy()
                mu1_train_updated = mu_train_detached[:, 1].cpu().numpy()

                if hasattr(train_ds, 'update_ite_estimates'):
                    train_ds.update_ite_estimates(mu0_train_updated, mu1_train_updated)

            # --------------------------------------------------------
            # Validation
            # --------------------------------------------------------
            avg_val_loss = float('inf')
            val_pehe = float('inf')

            if val_loader is not None:
                self.eval()
                total_val_loss = 0.0
                num_val_batches = 0

                with torch.no_grad():
                    for batch_data_val in val_loader:
                        x1_val, y1_val_true, t1_val_true, x2_val, y2_val_true, t2_val_true, labels_ctr_val = batch_data_val
                        x1_val, y1_val_true, t1_val_true = x1_val.to(self.device), y1_val_true.to(self.device), t1_val_true.to(self.device)
                        x2_val, y2_val_true, t2_val_true = x2_val.to(self.device), y2_val_true.to(self.device), t2_val_true.to(self.device)
                        labels_ctr_val = labels_ctr_val.to(self.device).float()

                        with autocast(
                            device_type=self.device.type,
                            dtype=torch.float16 if use_cuda_for_amp else None,
                            enabled=use_cuda_for_amp
                        ):
                            _, h1_val = self.base.mu_and_embedding(x1_val)
                            _, h2_val = self.base.mu_and_embedding(x2_val)

                            X_combined_val = torch.cat([x1_val, x2_val], dim=0)
                            T_combined_val_true = torch.cat([t1_val_true, t2_val_true], dim=0)
                            Y_combined_val_true = torch.cat([y1_val_true, y2_val_true], dim=0)

                            Y_combined_for_base_loss_val = Y_combined_val_true
                            if self.base.y_scaler is not None and hasattr(self.base.y_scaler, 'transform'):
                                try:
                                    y_np_scaled_val = self.base.y_scaler.transform(
                                        Y_combined_val_true.cpu().numpy().reshape(-1, 1)
                                    )
                                    Y_combined_for_base_loss_val = torch.tensor(
                                        y_np_scaled_val,
                                        dtype=torch.float32,
                                        device=self.device
                                    ).reshape(Y_combined_val_true.shape)
                                except Exception:
                                    pass

                            base_loss_val = self.base.compute_loss(
                                X_combined_val,
                                T_combined_val_true,
                                Y_combined_for_base_loss_val
                            )
                            ctr_loss_val = self.contrastive_loss(h1_val, h2_val, labels_ctr_val)
                            val_loss_batch = base_loss_val + self.lambda_ctr * ctr_loss_val

                        if not (torch.isnan(val_loss_batch) or torch.isinf(val_loss_batch)):
                            total_val_loss += val_loss_batch.item()
                            num_val_batches += 1

                avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else float('inf')

                # ----------------------------------------------------
                # Validation PEHE for IHDP model selection
                # ----------------------------------------------------
                if true_ite_val is not None:
                    with torch.no_grad():
                        pred_ite_val = self.predict_ite(Xva_np).reshape(-1)
                    val_pehe = PEHE_with_ite(
                        true_ite_val.reshape(-1),
                        pred_ite_val,
                        sqrt=True
                    )

            # --------------------------------------------------------
            # Monitor metric
            # - use validation PEHE if available
            # - else fallback to validation loss
            # --------------------------------------------------------
            if true_ite_val is not None and val_loader is not None:
                monitor_metric = val_pehe
            elif val_loader is not None:
                monitor_metric = avg_val_loss
            else:
                monitor_metric = avg_train_loss

            scheduler.step(monitor_metric)

            if monitor_metric < best_monitor_metric - 1e-6:
                best_monitor_metric = monitor_metric
                best_state_dict = {
                    k: v.detach().cpu().clone()
                    for k, v in self.state_dict().items()
                }
                patience_counter = 0
                if best_model_path is not None:
                    torch.save(best_state_dict, best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= p['patience']:
                    break

        if best_state_dict is not None:
            self.load_state_dict(best_state_dict)

    def predict_ite(self, X: np.ndarray) -> np.ndarray:
        X_np = convert_pd_to_np(X) if not isinstance(X, np.ndarray) else X
        return self.base.predict_ite(X_np)

    def predict_ate(self, X: np.ndarray) -> float:
        X_np = convert_pd_to_np(X) if not isinstance(X, np.ndarray) else X
        return self.base.predict_ate(X_np)