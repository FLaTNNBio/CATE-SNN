# --- bcauss_torch.py -------------------------------------------------
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.model import Model  # tua classe base
from src.models.utils import convert_pd_to_np  # helper CausalForge


class EpsilonLayer(nn.Module):
    def forward(self, t_pred):
        return torch.rand_like(t_pred)  # uniforme [0,1) come in Keras


class BCAUSS(Model, nn.Module):
    # -----------------------------------------------------------------
    def __init__(self, **user_params):
        nn.Module.__init__(self)
        self.build(user_params)

    # -----------------------------------------------------------------
    def _init_weights(self):
        """Normal(0,0.05) come in Keras"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.05)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # -----------------------------------------------------------------
    def build(self, user_params):
        # ---------- parametri default ----------
        p = {
            'neurons_per_layer': 200,
            'act_fn': 'relu',
            'reg_l2': 0.01,
            'verbose': True,
            'val_split': 0.22,
            'ratio': 1.0,
            'optim': 'sgd',
            'epochs': 500,
            'learning_rate': 1e-5,
            'momentum': 0.9,
            'use_bce': False,
            'norm_bal_term': True,
            'use_targ_term': False,
            'b_ratio': 1.0,
            'bs_ratio': 1.0,
            'scale_preds': True
        }
        if 'input_dim' not in user_params:
            raise ValueError("input_dim must be specified!")
        p.update(user_params)
        self.params = p

        # ---------- rete ----------
        D, N = p['input_dim'], p['neurons_per_layer']
        act = nn.ReLU if p['act_fn'] == 'relu' else nn.Identity

        # Representation tower
        self.repr_net = nn.Sequential(
            nn.Linear(D, N), act(),
            nn.Linear(N, N), act(),
            nn.Linear(N, N), act()
        )

        # Heads
        self.t_head = nn.Sequential(nn.Linear(N, 1), nn.Sigmoid())
        h = N // 2
        self.y0_net = nn.Sequential(nn.Linear(N, h), act(),
                                    nn.Linear(h, h), act(),
                                    nn.Linear(h, 1))
        self.y1_net = nn.Sequential(nn.Linear(N, h), act(),
                                    nn.Linear(h, h), act(),
                                    nn.Linear(h, 1))
        self.eps_layer = EpsilonLayer()

        self._init_weights()  # ← same init as Keras
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.y_scaler = None

    # -----------------------------------------------------------------
    def forward(self, x):
        z = self.repr_net(x)
        t_pred = self.t_head(z)
        y0 = self.y0_net(z)
        y1 = self.y1_net(z)
        eps = self.eps_layer(t_pred)
        return t_pred, y0, y1, eps

    # -----------------------------------------------------------------
    def compute_loss(self, x, t_true, y_true):
        p = self.params
        bs = x.size(0)

        t_pred, y0, y1, eps = self.forward(x)
        t_smooth = (t_pred + 1e-3) / 1.002

        # -------- (i) factual MSE --------
        loss0 = ((1 - t_true) * (y_true - y0).pow(2)).sum()
        loss1 = (t_true * (y_true - y1).pow(2)).sum()
        reg_loss = loss0 + loss1

        # -------- (ii) BCE sui trattamenti --------
        if p['use_bce']:
            bce = F.binary_cross_entropy(t_smooth, t_true, reduction='sum')
            reg_loss = reg_loss + bce

        # -------- (iii) targeted reg --------
        if p['use_targ_term']:
            y_pred = t_true * y1 + (1 - t_true) * y0
            h_term = t_true / t_smooth - (1 - t_true) / (1 - t_smooth)
            y_pert = y_pred + eps * h_term
            targ = p['ratio'] * ((y_true - y_pert).pow(2)).sum()
            reg_loss = reg_loss + targ

        # -------- (iv) balancing --------
        w1 = t_true / t_smooth
        w0 = (1 - t_true) / (1 - t_smooth)
        if p['norm_bal_term']:
            ones_mean = (w1 * x).sum(0) / w1.sum(0)
            zeros_mean = (w0 * x).sum(0) / w0.sum(0)
        else:
            ones_mean = (w1 * x).sum(0)
            zeros_mean = (w0 * x).sum(0)
        bal = F.mse_loss(zeros_mean, ones_mean, reduction='sum')
        reg_loss = reg_loss + p['b_ratio'] * bal

        # -------- (v) L2 come in Keras --------
        l2_term = 0.0
        for param in self.parameters():
            if param.ndim > 1:  # escludi bias
                l2_term = l2_term + (param ** 2).sum()
        reg_loss = reg_loss + p['reg_l2'] * l2_term

        if torch.isnan(reg_loss):
            pass
            #raise ValueError("NaN in loss")

        return reg_loss

    # -----------------------------------------------------------------
    def fit(self, X, treatment, y, epochs=None):  # <-- MODIFICA QUI: aggiungi epochs=None
        X, treatment, y = convert_pd_to_np(X, treatment, y)
        N = X.shape[0]
        p = self.params  # Parametri interni di BCAUSS

        # Determina il numero di epoche per questo specifico run di fit
        # Se 'epochs' viene passato come argomento, usa quello, altrimenti usa p['epochs'] di BCAUSS
        num_epochs_to_run = epochs if epochs is not None else p['epochs']

        # ----- Inizio della logica di scaling di y (assicurati sia consistente) -----
        # Se y deve essere scalato, e lo scaler non è ancora stato fittato (es. prima chiamata a fit)
        # O se vuoi ri-fittare lo scaler ogni volta (meno comune per y_scaler se fit è chiamato più volte
        # con dati diversi, ma per il warmup è ok fittarlo qui).
        # Y viene passato come Y_flat, quindi è già (N,1)
        y_original_for_scaler = y  # Salva una copia se necessario per fittare lo scaler
        if p['scale_preds']:
            if self.y_scaler is None:  # Fitta solo se non esiste già
                self.y_scaler = StandardScaler()
                self.y_scaler.fit(y_original_for_scaler)  # y_original_for_scaler deve essere (N,1)

            y = self.y_scaler.transform(y_original_for_scaler)  # y ora è scalato
        # ----- Fine della logica di scaling di y -----

        # ----- Inizio logica scaling di X (se implementata come suggerito precedentemente) -----
        # Esempio:
        # if p.get('scale_inputs', False): # Assumendo un parametro 'scale_inputs'
        #     if self.x_scaler is None:
        #         self.x_scaler = StandardScaler()
        #         # Fitta self.x_scaler su X_train (devi definire X_train qui o passarlo)
        #         # Potrebbe essere necessario splittare X qui per fittare solo sul training set di questo fit
        #         # temp_idx_split = np.random.permutation(N)
        #         # temp_tr_idx = temp_idx_split[:int(N * (1-p['val_split']))]
        #         # self.x_scaler.fit(X[temp_tr_idx])
        #     X = self.x_scaler.transform(X)
        # ----- Fine logica scaling di X -----

        X_t = torch.tensor(X, dtype=torch.float32)
        # Assicurati che treatment sia (N,1) se necessario per t_head
        t_t = torch.tensor(treatment.reshape(-1, 1), dtype=torch.float32)
        y_t = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)  # y è già (N,1) o lo diventa qui

        # split (la logica di split è già presente nel tuo codice BCAUSS)
        idx = np.random.permutation(N)
        split_point = int(N * (1 - p['val_split']))
        tr, va = idx[:split_point], idx[split_point:]

        # Batch size: usa p['bs_ratio'] come nel tuo codice originale BCAUSS
        # Se vuoi usare un batch_size fisso invece di bs_ratio, dovresti aggiungere un parametro
        # 'batch_size' a p e usarlo qui. Per ora, mantengo la tua logica originale.
        effective_train_size = len(tr)
        bs = int(effective_train_size * p['bs_ratio'])
        if bs == 0 and effective_train_size > 0: bs = effective_train_size  # Evita batch size 0 se ci sono dati

        # DataLoaders (la logica è già presente nel tuo codice BCAUSS)
        # Assicurati che il batch_size (bs) sia gestito correttamente se N è piccolo.
        if effective_train_size > 0:
            train_loader = DataLoader(TensorDataset(X_t[tr], t_t[tr], y_t[tr]),
                                      batch_size=bs if bs > 0 else 1, shuffle=True)  # Aggiunto bs > 0 else 1
        else:
            train_loader = []  # o gestisci il caso di training set vuoto

        if len(va) > 0:
            val_bs = int(len(va) * p['bs_ratio'])  # Potresti voler un val_batch_size diverso
            val_loader = DataLoader(TensorDataset(X_t[va], t_t[va], y_t[va]),
                                    batch_size=val_bs if val_bs > 0 else 1, shuffle=False)
        else:
            val_loader = []

        # optimizer (la logica è già presente)
        optim_cls = Adam if p['optim'] == 'adam' else SGD
        optim_kwargs = {'lr': p['learning_rate'],
                        'momentum': p['momentum']} if p['optim'] == 'sgd' else \
            {'lr': p['learning_rate']}
        optimizer = optim_cls(self.parameters(), **optim_kwargs)
        scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                      factor=0.5, patience=5,  # Considera di rendere 'patience' un param
                                      )

        best_val, patience_counter = float('inf'), 0  # rinominato patience a patience_counter
        best_state = None  # Inizializza best_state
        self.to(self.device)
        # X_t, t_t, y_t = None, None, None # Libera RAM host - fallo dopo aver creato i dataloader se N è molto grande

        if not train_loader:  # Se non ci sono dati di training
            if p['verbose']:
                print("BCAUSS Warmup: No training data provided or training set is empty.")
            return

        for epoch in range(1, num_epochs_to_run + 1):  # <-- MODIFICA QUI: usa num_epochs_to_run
            # ---- train ----
            self.train()
            tl = 0.0
            train_batches = 0
            for xb, tb, yb in train_loader:
                xb, tb, yb = xb.to(self.device), tb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = self.compute_loss(xb, tb, yb)
                if not torch.isnan(loss):  # Aggiunto controllo NaN prima di backward
                    loss.backward()
                    optimizer.step()
                    tl += loss.item()
                else:
                    print(f"Warning: NaN loss detected in BCAUSS epoch {epoch}. Skipping batch.")  # Avviso
                train_batches += 1
            tl /= train_batches if train_batches > 0 else 1

            # ---- val ----
            vl = 0.0
            if val_loader:  # Solo se ci sono dati di validazione
                self.eval()
                val_batches = 0
                with torch.no_grad():
                    for xb, tb, yb in val_loader:
                        xb, tb, yb = xb.to(self.device), tb.to(self.device), yb.to(self.device)
                        loss_val = self.compute_loss(xb, tb, yb)
                        if not torch.isnan(loss_val):
                            vl += loss_val.item()
                        else:
                            print(f"Warning: NaN validation loss detected in BCAUSS epoch {epoch}.")
                        val_batches += 1
                vl /= val_batches if val_batches > 0 else 1
                scheduler.step(vl)  # Step dello scheduler solo se c'è una val loss
            else:  # Se non c'è val_loader, puoi usare la training loss per lo scheduler o non usarlo
                scheduler.step(tl)  # Esempio: usa training loss se non c'è validation
                vl = tl  # Imposta vl=tl per la stampa e l'early stopping se non c'è validazione

            if p['verbose']:
                print(
                    f"BCAUSS Warmup Epoch {epoch}/{num_epochs_to_run}  train={tl:.4f}  val={vl:.4f}  lr={optimizer.param_groups[0]['lr']:.2e}")

            if vl < best_val - 1e-6:  # Early stopping basato su vl
                best_val = vl
                best_state = self.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                # Rendi la patience dell'early stopping un parametro, es p['early_stopping_patience']
                if patience_counter >= p.get('early_stopping_patience', 40):
                    if p['verbose']:
                        print(f"BCAUSS Warmup: Early stopping at epoch {epoch}.")
                    break

        if best_state:  # Carica il best_state solo se è stato salvato qualcosa
            self.load_state_dict(best_state)

    # -----------------------------------------------------------------
    def mu_and_embedding(self, x):
        z = self.repr_net(x)
        y0 = self.y0_net(z)
        y1 = self.y1_net(z)
        mu = torch.cat([y0, y1], dim=1)
        return mu, z

    # -----------------------------------------------------------------
    @property
    def support_ite(self):
        return True

    # -----------------------------------------------------------------
    def predict_ite(self, X):
        X = convert_pd_to_np(X)
        self.eval()
        xt = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            _, y0, y1, _ = self.forward(xt)
        y0_np, y1_np = y0.cpu().numpy(), y1.cpu().numpy()
        if self.params['scale_preds']:
            y0_np = self.y_scaler.inverse_transform(y0_np)
            y1_np = self.y_scaler.inverse_transform(y1_np)
        return (y1_np - y0_np).squeeze()

    def predict_ate(self, X, *args):
        return float(np.mean(self.predict_ite(X)))

