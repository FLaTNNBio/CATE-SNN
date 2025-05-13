import torch
import numpy as np
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

class SiameseBCAUSS(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        margin: float,
        lambda_ctr: float,
        **fit_params
    ):
        super().__init__()
        self.base = base_model
        self.margin = margin
        self.lambda_ctr = lambda_ctr

        # Parametri di fit con default
        self.fit_params = {
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
        }
        self.fit_params.update(fit_params)
        self.device = getattr(self.base, 'device', torch.device('cpu'))

    def contrastive_loss(self, h1: torch.Tensor, h2: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
        d = torch.norm(h1 - h2, p=2, dim=1)
        loss_sim = d.pow(2)
        loss_dis = torch.clamp(self.margin - d, min=0).pow(2)
        return (l * loss_sim + (1 - l) * loss_dis).mean()

    def step(
        self,
        batch,
        scaler: GradScaler,
        opt: torch.optim.Optimizer,
        clip_norm: float,
        use_amp: bool
    ) -> tuple:
        x1, y1, t1, x2, y2, t2, l = batch
        x1, y1, t1 = x1.to(self.device), y1.to(self.device), t1.to(self.device)
        x2, y2, t2 = x2.to(self.device), y2.to(self.device), t2.to(self.device)
        l = l.to(self.device)

        with autocast(enabled=use_amp):
            mu1, h1 = self.base.mu_and_embedding(x1)
            mu2, h2 = self.base.mu_and_embedding(x2)
            base_loss = self.base.compute_loss(
                torch.cat([x1, x2]),
                torch.cat([t1, t2]),
                torch.cat([y1, y2])
            )
            ctr_loss = self.contrastive_loss(h1, h2, l.float())
            loss = base_loss + self.lambda_ctr * ctr_loss

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if clip_norm > 0:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(self.parameters(), clip_norm)
        scaler.step(opt)
        scaler.update()

        return loss.item(), base_loss.item(), ctr_loss.item()

    def evaluate(self, loader: DataLoader) -> float:
        self.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for batch in loader:
                x1, y1, t1, x2, y2, t2, l = [t.to(self.device) for t in batch]
                mu1, h1 = self.base.mu_and_embedding(x1)
                mu2, h2 = self.base.mu_and_embedding(x2)
                base_loss = self.base.compute_loss(
                    torch.cat([x1, x2]),
                    torch.cat([t1, t2]),
                    torch.cat([y1, y2])
                )
                ctr_loss = self.contrastive_loss(h1, h2, l.float())
                total += (base_loss + self.lambda_ctr * ctr_loss).item()
                count += 1
        return total / max(1, count)

    def fit(
        self,
        X: np.ndarray,
        T: np.ndarray,
        Y: np.ndarray,
        device: torch.device,
        ds_class,
        collate_fn
    ):
        # Split
        N = X.shape[0]
        idx = np.random.permutation(N)
        split = int(N * (1 - self.fit_params['val_split']))
        tr_idx, va_idx = idx[:split], idx[split:]

        # Stime iniziali mu
        X_tr = torch.tensor(X[tr_idx], dtype=torch.float32, device=device)
        with torch.no_grad():
            mu_tr, _ = self.base.mu_and_embedding(X_tr)
        mu0_tr, mu1_tr = mu_tr[:,0].cpu().numpy(), mu_tr[:,1].cpu().numpy()

        X_va = torch.tensor(X[va_idx], dtype=torch.float32, device=device)
        with torch.no_grad():
            mu_va, _ = self.base.mu_and_embedding(X_va)
        mu0_va, mu1_va = mu_va[:,0].cpu().numpy(), mu_va[:,1].cpu().numpy()

        # Dataset + Loader
        train_ds = ds_class(X[tr_idx], T[tr_idx], Y[tr_idx], mu0_tr, mu1_tr)
        val_ds   = ds_class(X[va_idx], T[va_idx], Y[va_idx], mu0_va, mu1_va)
        train_loader = DataLoader(train_ds, batch_size=self.fit_params['batch_size'], shuffle=True, collate_fn=collate_fn)
        val_loader   = DataLoader(val_ds,   batch_size=self.fit_params['batch_size'], shuffle=False, collate_fn=collate_fn)

        # Ottimizzatore
        if self.fit_params['optim'].lower() == 'sgd':
            optimizer = SGD(self.parameters(), lr=self.fit_params['lr'], momentum=self.fit_params['momentum'])
        else:
            optimizer = Adam(self.parameters(), lr=self.fit_params['lr'])
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=self.fit_params['verbose'])
        scaler    = GradScaler(enabled=self.fit_params['use_amp'])

        # Train loop
        best_val, patience_cnt, best_state = float('inf'), 0, None
        for epoch in range(1, self.fit_params['epochs']+1):
            self.train()
            for batch in train_loader:
                self.step(batch, scaler, optimizer, self.fit_params['clip_norm'], self.fit_params['use_amp'])
            val_loss = self.evaluate(val_loader)
            scheduler.step(val_loss)
            if self.fit_params['verbose']:
                print(f"Epoch {epoch:03d} — val_loss: {val_loss:.4f}")
            if val_loss < best_val - 1e-6:
                best_val, best_state, patience_cnt = val_loss, {k:v.cpu() for k,v in self.state_dict().items()}, 0
            else:
                patience_cnt += 1
                if patience_cnt >= self.fit_params['patience']:
                    if self.fit_params['verbose']:
                        print(f"Early stopping at epoch {epoch}")
                    break
        if best_state is not None:
            self.load_state_dict(best_state)

    def predict_ite(self, X_np: np.ndarray) -> np.ndarray:
        return self.base.predict_ite(X_np)

    def predict_ate(self, X_np: np.ndarray, *args) -> float:
        return self.base.predict_ate(X_np)

    def predict_mu_hat(self, X, device: torch.device) -> tuple:
        X_np = X.values if hasattr(X, 'values') else X
        xt = torch.tensor(X_np, dtype=torch.float32, device=self.device)
        self.eval()
        with torch.no_grad(): mu, _ = self.base.mu_and_embedding(xt)
        mu_np = mu.cpu().numpy()
        return mu_np[:,0], mu_np[:,1]
