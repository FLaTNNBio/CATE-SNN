import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np


# --- EpsilonLayer -----------------------------------------------------------
class EpsilonLayer(nn.Module):
    """Custom layer che impara uno scalare epsilon e lo replica per batch."""

    def __init__(self):
        super().__init__()
        # inizializzato a zero per essere simile a RandomNormal~0
        self.epsilon = nn.Parameter(torch.zeros(1, 1))

    def forward(self, t_pred: torch.Tensor) -> torch.Tensor:
        # t_pred: (batch_size, 1)
        return self.epsilon * torch.ones_like(t_pred)


# --- Loss functions (da CausalML) ------------------------------------------
def regression_loss(concat_true: torch.Tensor,
                    concat_pred: torch.Tensor) -> torch.Tensor:
    y_true = concat_true[:, 0]
    t_true = concat_true[:, 1]
    y0_pred, y1_pred = concat_pred[:, 0], concat_pred[:, 1]
    loss0 = ((1.0 - t_true) * (y_true - y0_pred) ** 2).sum()
    loss1 = (t_true * (y_true - y1_pred) ** 2).sum()
    return loss0 + loss1


def binary_classification_loss(concat_true: torch.Tensor,
                               concat_pred: torch.Tensor) -> torch.Tensor:
    t_true = concat_true[:, 1]
    t_pred = concat_pred[:, 2]
    # smoothing come in Keras
    t_pred = (t_pred + 0.001) / 1.002
    return F.binary_cross_entropy(t_pred, t_true, reduction='sum')


def dragonnet_loss_binarycross(concat_true: torch.Tensor,
                               concat_pred: torch.Tensor) -> torch.Tensor:
    return regression_loss(concat_true, concat_pred) + \
        binary_classification_loss(concat_true, concat_pred)


def make_tarreg_loss(ratio: float = 1.0,
                     dragonnet_loss_fn=dragonnet_loss_binarycross):
    """Targeted regularization loss wrapper."""

    def tarreg_loss(concat_true, concat_pred):
        vanilla = dragonnet_loss_fn(concat_true, concat_pred)
        y_true = concat_true[:, 0]
        t_true = concat_true[:, 1]
        y0, y1 = concat_pred[:, 0], concat_pred[:, 1]
        t_pred = concat_pred[:, 2]
        eps = concat_pred[:, 3]
        t_pred = (t_pred + 0.01) / 1.02
        y_pred = t_true * y1 + (1 - t_true) * y0
        h = t_true / t_pred - (1 - t_true) / (1 - t_pred)
        y_pert = y_pred + eps * h
        targ_reg = ((y_true - y_pert) ** 2).sum()
        return vanilla + ratio * targ_reg

    return tarreg_loss


# --- DragonNet model (solo le parti modificate) ----------------------------------
class DragonNet(nn.Module):
    def __init__(self,
                 input_dim: int,
                 neurons_per_layer: int = 200,
                 reg_l2: float = 0.01,
                 targeted_reg: bool = True,
                 ratio: float = 1.0):
        super().__init__()
        self.targeted_reg = targeted_reg
        self.ratio = ratio

        # Representation
        self.fc1 = nn.Linear(input_dim, neurons_per_layer)
        self.fc2 = nn.Linear(neurons_per_layer, neurons_per_layer)
        self.fc3 = nn.Linear(neurons_per_layer, neurons_per_layer)

        # Propensity head
        self.t_pred = nn.Linear(neurons_per_layer, 1)

        # Potential outcomes heads
        h = neurons_per_layer // 2
        self.y0_fc1 = nn.Linear(neurons_per_layer, h)
        self.y1_fc1 = nn.Linear(neurons_per_layer, h)
        self.y0_fc2 = nn.Linear(h, h)
        self.y1_fc2 = nn.Linear(h, h)
        self.y0_pred = nn.Linear(h, 1)
        self.y1_pred = nn.Linear(h, 1)

        # Epsilon layer
        self.epsilon_layer = EpsilonLayer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # shared representation
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))

        # propensity
        t = torch.sigmoid(self.t_pred(x))

        # y0 / y1 heads
        y0 = F.elu(self.y0_fc1(x))
        y0 = F.elu(self.y0_fc2(y0))
        y0 = self.y0_pred(y0)

        y1 = F.elu(self.y1_fc1(x))
        y1 = F.elu(self.y1_fc2(y1))
        y1 = self.y1_pred(y1)

        # epsilon
        eps = self.epsilon_layer(t)  # t è già sul dispositivo corretto da forward(x)

        # concat [y0, y1, propensity, epsilon]
        return torch.cat([y0, y1, t, eps], dim=1)

    def predict_ite(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            # Determina il dispositivo del modello
            target_device = next(self.parameters()).device
            # Converte NumPy in tensore e lo sposta sul dispositivo corretto
            X_t = torch.from_numpy(X.astype(np.float32)).to(target_device)

            preds = self.forward(X_t)
            # Sposta il risultato sulla CPU prima di convertirlo in NumPy
            return (preds[:, 1] - preds[:, 0]).cpu().numpy()

    def predict_ate(self, X, treatment=None, y=None) -> float:
        # predict_ite ora gestisce correttamente il dispositivo
        return float(self.predict_ite(X).mean())

        # All'interno della classe DragonNet nel file src/models/dragonnet.py
        # Aggiungi questo nuovo metodo, ad esempio dopo predict_ite o predict_ate

    def predict_mu_hat(self, X: np.ndarray) -> np.ndarray:
        """
            Predice i risultati potenziali mu0_hat(X) e mu1_hat(X).

            Args:
                X (np.ndarray): Features di input.

            Returns:
                np.ndarray: Un array NumPy di shape (N, 2) dove la prima colonna
                            è mu0_hat e la seconda colonna è mu1_hat.
            """
        self.eval()  # Imposta il modello in modalità valutazione
        with torch.no_grad():  # Disabilita il calcolo dei gradienti
            # Determina il dispositivo del modello
            target_device = next(self.parameters()).device
            # Converte NumPy in tensore e lo sposta sul dispositivo corretto
            X_t = torch.from_numpy(X.astype(np.float32)).to(target_device)

            # Il metodo forward restituisce [y0, y1, t_pred, eps]
            # y0 (mu0_hat) è all'indice 0
            # y1 (mu1_hat) è all'indice 1
            all_predictions = self.forward(X_t)

            mu0_predictions = all_predictions[:, 0:1]  # Shape: (batch_size, 1)
            mu1_predictions = all_predictions[:, 1:2]  # Shape: (batch_size, 1)

            # Concatena per ottenere shape (batch_size, 2)
            mu_hat = torch.cat([mu0_predictions, mu1_predictions], dim=1)

            # Sposta il risultato sulla CPU e converti in NumPy
            return mu_hat.cpu().numpy()

    def fit(self,
            X: np.ndarray,
            treatment: np.ndarray,
            y: np.ndarray,
            val_split: float = 0.2,
            batch_size: int = 64,
            epochs: int = 100,
            learning_rate: float = 1e-5,
            momentum: float = 0.9,
            use_adam: bool = True,
            adam_epochs: int = 30,
            adam_lr: float = 1e-3):
        # prepara i dati (questi tensori inizialmente sono su CPU)
        X_t = torch.from_numpy(X.astype(np.float32))
        t_t = torch.from_numpy(treatment.astype(np.float32))
        y_t = torch.from_numpy(y.astype(np.float32))
        concat_true = torch.cat([y_t, t_t], dim=1)

        dataset = TensorDataset(X_t, concat_true)
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        loss_fn = (make_tarreg_loss(self.ratio)
                   if self.targeted_reg
                   else dragonnet_loss_binarycross)

        if use_adam:
            opt = Adam(self.parameters(), lr=adam_lr)
            sched = ReduceLROnPlateau(opt, 'min', factor=0.5, patience=5,
                                      min_lr=0.0)  # min_lr era 0.0, forse volevi un valore più piccolo come 1e-8
            best, patience = float('inf'), 0
            for epoch_idx in range(adam_epochs):  # rinominato _ a epoch_idx per chiarezza
                train_l = self._train_epoch(train_loader, loss_fn, opt)
                val_l = self._eval_epoch(val_loader, loss_fn)
                sched.step(val_l)  # Solitamente si usa val_l per lo scheduler ReduceLROnPlateau
                if val_l < best - 1e-6:  # Aggiunta una piccola tolleranza per 'best'
                    best, patience = val_l, 0
                else:
                    patience += 1
                if patience >= 20:  # Aumentata la pazienza per Adam, 2 era molto poco
                    print(f"Early stopping Adam at epoch {epoch_idx + 1}")
                    break

        opt = SGD(self.parameters(), lr=learning_rate, momentum=momentum, nesterov=True)
        sched = ReduceLROnPlateau(opt, 'min', factor=0.5, patience=10,
                                  min_lr=1e-8)  # Aumentata pazienza e settato min_lr
        best, patience = float('inf'), 0
        for epoch_idx in range(epochs):  # rinominato _ a epoch_idx per chiarezza
            train_l = self._train_epoch(train_loader, loss_fn, opt)
            val_l = self._eval_epoch(val_loader, loss_fn)
            sched.step(val_l)  # Solitamente si usa val_l
            if val_l < best - 1e-6:  # Aggiunta una piccola tolleranza
                best, patience = val_l, 0
            else:
                patience += 1
            if patience >= 40:  # La pazienza di 40 sembra ragionevole
                print(f"Early stopping SGD at epoch {epoch_idx + 1}")
                break

    def _train_epoch(self, loader, loss_fn, optimizer):
        self.train()
        total_loss = 0.0
        # Determina il dispositivo del modello UNA VOLTA all'inizio dell'epoca
        target_device = next(self.parameters()).device

        for xb, yb in loader:
            # Sposta i dati del batch sul dispositivo corretto
            xb = xb.to(target_device)
            yb = yb.to(target_device)

            optimizer.zero_grad()
            preds = self.forward(xb)  # xb è ora sul dispositivo corretto
            loss = loss_fn(yb, preds)  # yb e preds sono ora sullo stesso dispositivo

            if torch.isnan(loss):
                print(f"Attenzione: NaN loss rilevata durante il training. Valore loss: {loss.item()}")
                # Potresti voler gestire questo caso in modo più specifico, es. saltando il batch
                # o interrompendo l'addestramento se accade troppo spesso.
                # Per ora, saltiamo l'aggiornamento per questo batch se la loss è NaN.
                continue  # Salta il resto del ciclo per questo batch

            loss.backward()
            optimizer.step()
            total_loss += loss.item()  # .item() è preferibile a float() per tensori scalari

        return total_loss / len(loader) if len(loader) > 0 else 0.0  # Ritorna la loss media

    def _eval_epoch(self, loader, loss_fn):
        self.eval()
        total_loss = 0.0
        # Determina il dispositivo del modello UNA VOLTA
        target_device = next(self.parameters()).device

        with torch.no_grad():
            for xb, yb in loader:
                # Sposta i dati del batch sul dispositivo corretto
                xb = xb.to(target_device)
                yb = yb.to(target_device)

                preds = self.forward(xb)
                loss = loss_fn(yb, preds)

                if torch.isnan(loss):
                    print(f"Attenzione: NaN loss rilevata durante la validazione. Valore loss: {loss.item()}")
                    # Se la loss di validazione è NaN, potresti volerla trattare come infinito
                    # o un valore molto alto per evitare che venga scelta come "best loss".
                    # Per ora, la aggiungiamo come se fosse un valore valido (ma NaN propaga).
                    # Sarebbe meglio gestirla, ad es. ritornando float('inf') se accade.
                total_loss += loss.item()  # .item() è preferibile

        return total_loss / len(loader) if len(loader) > 0 else 0.0  # Ritorna la loss media
