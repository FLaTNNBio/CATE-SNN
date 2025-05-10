#!/usr/bin/env python3
"""
Siamese‑BCAUSS con Hydra (PyTorch) + CodeCarbon
----------------------------------------------
• Contrastive loss con soglia dinamica sul 20° percentile delle CATE
• Mixed‑precision (AMP) opzionale
• Gradient clipping
• Log CSV per **ogni epoca** (`train_loss.csv`) e per replica (`metrics.csv`)
• Grafico `loss_curve.png` (media su tutte le repliche)
• CodeCarbon: `emissions.csv` nella cartella di output Hydra

Esempio:
```bash
python siamese_bcauss_hydra.py batch=512 lr=5e-5 epochs=30 use_amp=true
```
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
    from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from codecarbon import EmissionsTracker

from src.data_loader import DataLoader as CFLoader
from src.metrics import eps_ATE_diff, PEHE_with_ite
from src.models.bcauss import BCAUSS


###############################################################################
# helper                                                                      #
###############################################################################

def compute_tau_threshold(mu0, mu1, perc=20, sample=100_000):
    tau = mu1 - mu0
    sample = min(sample, len(tau))
    idx1 = np.random.randint(0, len(tau), size=sample)
    idx2 = np.random.randint(0, len(tau), size=sample)
    return float(np.percentile(np.abs(tau[idx1] - tau[idx2]), perc))


def make_pairs(X, T, Y, mu0, mu1, thr, n_pairs):
    ite = mu1 - mu0;
    N = len(X)
    idx_a = np.random.randint(0, N, size=n_pairs)
    idx_b, lab = [], []
    for i in idx_a:
        if np.random.rand() < .5:  # simile
            cand = np.where(np.abs(ite - ite[i]) < thr)[0]
            lab.append(1)
        else:  # dissimile
            cand = np.where(np.abs(ite - ite[i]) >= thr)[0]
            lab.append(0)
        # fallback in caso di array vuoto
        if cand.size == 0:
            logging.warning(f"No candidates for index {i} (replica), using fallback all except self.")
            cand = np.delete(np.arange(N), i)
        idx_b.append(np.random.choice(cand))
    idx_b = np.array(idx_b)
    return (X[idx_a], Y[idx_a], T[idx_a],
            X[idx_b], Y[idx_b], T[idx_b],
            np.array(lab, np.int64))


def first_item(batch):
    return batch[0]


###############################################################################
# dataset                                                                     #
###############################################################################

class ContrastiveCausalDS(Dataset):
    def __init__(self, X, T, Y, m0, m1, bs=256, perc=20):
        self.X, self.T, self.Y, self.bs = X, T, Y, bs
        self.thr = compute_tau_threshold(m0, m1, perc)
        self.mu0, self.mu1 = m0, m1

    def __len__(self):
        return 250  # batch virtuali per epoca

    def __getitem__(self, _):
        tpl = make_pairs(self.X, self.T, self.Y,
                         self.mu0, self.mu1, self.thr, self.bs)
        tf = lambda a: torch.tensor(a, dtype=torch.float32)
        x1, y1, t1, x2, y2, t2, lab = map(tf, tpl)
        return x1, y1, t1, x2, y2, t2, lab.long()


###############################################################################
# siamese wrapper                                                             #
###############################################################################

class SiameseBCAUSS(torch.nn.Module):
    def __init__(self, base, margin, lambda_ctr):
        super().__init__()
        self.base = base
        self.margin = margin
        self.lambda_ctr = lambda_ctr

    def contrastive_loss(self, h1, h2, l):
        d = torch.norm(h1 - h2, dim=1)
        return ((l * d ** 2) + ((1 - l) * torch.clamp(self.margin - d, 0) ** 2)).mean()

    def step(self, b, scaler, opt, clip, amp):
        x1, y1, t1, x2, y2, t2, l = b
        with autocast(enabled=amp):
            mu1, h1 = self.base.mu_and_embedding(x1)
            mu2, h2 = self.base.mu_and_embedding(x2)
            base = self.base.compute_loss(torch.cat([x1, x2]), torch.cat([t1, t2]), torch.cat([y1, y2]))
            ctr = self.contrastive_loss(h1, h2, l.float())
            loss = base + self.lambda_ctr * ctr
        scaler.scale(loss).backward()
        if clip:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        return loss.item(), base.item(), ctr.item()

    @torch.no_grad()
    def predict_ite(self, X_np, device):
        self.eval()
        xt = torch.tensor(X_np, dtype=torch.float32, device=device)
        mu, _ = self.base.mu_and_embedding(xt)
        return (mu[:, 1] - mu[:, 0]).cpu().numpy()


###############################################################################
# Hydra main                                                                 #
###############################################################################

@hydra.main(config_path="../configs", config_name="default", version_base="1.3")
def run(cfg: DictConfig):
    print("Config:\n" + OmegaConf.to_yaml(cfg))

    # reproducibilità
    os.environ['PYTHONHASHSEED'] = str(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = cfg.device if torch.cuda.is_available() else "cpu"
    out_dir = Path.cwd()

    # CSV init
    (out_dir / "train_loss.csv").write_text("replica,epoch,loss,base,ctr\n")
    (out_dir / "metrics.csv").write_text("replica,eps_ate,pehe,co2_kg\n")

    tracker = EmissionsTracker(output_dir=str(out_dir), log_level="error")
    tracker.start()

    # load IHDP
    data = CFLoader.get_loader('IHDP').load()
    (X_tr, T_tr, YF_tr, _, m0_tr, m1_tr,
     X_te, _, _, _, m0_te, m1_te) = data

    results = []
    loss_per_epoch = []
    for idx in range(cfg.n_reps):
        logging.info(f"Replica {idx + 1}/{cfg.n_reps}")
        Xtr, Ttr, Ytr = X_tr[:, :, idx].astype(np.float32), T_tr[:, idx, None], YF_tr[:, idx, None]
        m0, m1 = m0_tr[:, idx], m1_tr[:, idx]

        ds = ContrastiveCausalDS(Xtr, Ttr, Ytr, m0, m1, bs=cfg.batch, perc=cfg.percentile)
        loader = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=first_item,
                            num_workers=cfg.num_workers, pin_memory=True)

        base = BCAUSS(input_dim=Xtr.shape[1])
        model = SiameseBCAUSS(base, cfg.margin, cfg.lambda_ctr).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        scaler = GradScaler(enabled=cfg.use_amp)

        # training per epoca
        for ep in range(1, cfg.epochs + 1):
            model.train()
            log = []
            for b in loader:
                b = [t.to(device) for t in b]
                l, bl, cl = model.step(b, scaler, opt, cfg.clip_norm, cfg.use_amp)
                log.append([l, bl, cl])
            l, bl, cl = np.mean(log, 0)
            loss_per_epoch.append([idx, ep, l])
            if ep == 1 or ep % 10 == 0:
                print(f"rep{idx + 1} ep{ep:02d} loss={l:.1f} base={bl:.1f} ctr={cl:.2f}")
            with open(out_dir / "train_loss.csv", "a", newline="") as f:
                csv.writer(f).writerow([idx, ep, round(l, 6), round(bl, 6), round(cl, 6)])

        # fine training replica: evaluation + metrics logging
        true_ite = m1_te[:, idx] - m0_te[:, idx]
        ite_pred = model.predict_ite(X_te[:, :, idx].astype(np.float32), device)
        eps = eps_ATE_diff(ite_pred.mean(), true_ite.mean())
        pehe = PEHE_with_ite(ite_pred, true_ite, sqrt=True)
        # registra su metrics.csv
        with open(out_dir / "metrics.csv", "a", newline="") as f2:
            csv.writer(f2).writerow([idx, round(eps, 6), round(pehe, 6), ""])
        # aggiungi ai risultati per la media
        results.append((eps, pehe))

    # dopo tutte le repliche, ferma tracker e salva pesi le repliche, ferma tracker e salva pesi
    total_co2 = tracker.stop() or 0.0
    torch.save(model.base.state_dict(), out_dir / "model_final.pt")

    # calcola metriche medie su tutte le repliche e le aggiunge a metrics.csv
    eps_vals, pehe_vals = zip(*results)
    mean_eps = np.mean(eps_vals)
    mean_pehe = np.mean(pehe_vals)
    with open(out_dir / "metrics.csv", "a", newline="") as f2:
        csv.writer(f2).writerow(["AVERAGE", round(mean_eps, 6), round(mean_pehe, 6), round(total_co2, 6)])

    # genera grafico loss media per epoca(model.base.state_dict(), out_dir/"model_final.pt")

    # genera grafico loss media per epoca
    df_loss = pd.read_csv(out_dir / "train_loss.csv")
    plt.figure()
    for name, grp in df_loss.groupby('replica'):
        plt.plot(grp['epoch'], grp['loss'], color='gray', alpha=0.3)
    mean_loss = df_loss.groupby('epoch')['loss'].mean()
    plt.plot(mean_loss.index, mean_loss.values, color='blue', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.savefig(out_dir / "loss_curve.png")
    plt.close()

    print(f"Model saved to {out_dir / 'model_final.pt'}")
    print(f"Total CO2 emitted: {total_co2:.6f} kg")
    print("Done.")


if __name__ == "__main__":
    run()
