import datetime
import itertools
import logging
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

# Importa il modello DualHeadSiameseCATE e la funzione contrastive_loss_batch
from SiameseTARNet import DualHeadSiameseCATE, contrastive_loss_batch
from paper.IHDPLoader import IHDPLoader

###############################################################################
# Imposta il sistema di log
###############################################################################
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = os.path.join("logs", timestamp)
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "experiment.log")

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)

logging.info("Avvio script")


###############################################################################
# Funzione per calcolare le statistiche globali (mean e std) del train set
###############################################################################
def compute_global_stats(dataset, device="cpu"):
    all_x = []
    for i in range(len(dataset)):
        # dataset[i] già ritorna un torch.Tensor per x_i
        x_i, t_i, Yf_i, *others = dataset[i]  # x_i: [d, R]
        # Trasponi direttamente il tensor, senza ricostruirne uno nuovo
        x_T = x_i.T  # [R, d]
        all_x.append(x_T)
    # Concatena lungo la prima dimensione (tutte le righe R)
    X_cat = torch.cat(all_x, dim=0)  # [N, d]
    mean_global = X_cat.mean(dim=0).to(device)
    std_global = X_cat.std(dim=0).to(device)
    return mean_global, std_global


###############################################################################
# Funzioni di generazione delle coppie
###############################################################################
def create_pairs(x, t, y):
    t = t.view(-1)
    treated_idx = (t == 1).nonzero(as_tuple=True)[0]
    control_idx = (t == 0).nonzero(as_tuple=True)[0]
    if treated_idx.numel() == 0 or control_idx.numel() == 0:
        return []
    positive_pairs = []
    negative_pairs = []
    # Coppie positive per i trattati
    if treated_idx.numel() > 1:
        treated_shuffled = treated_idx[torch.randperm(treated_idx.size(0))]
        for i in range(0, treated_shuffled.size(0) - 1, 2):
            idx_a = treated_shuffled[i].item()
            idx_b = treated_shuffled[i + 1].item()
            positive_pairs.append(("positive", (idx_a, idx_b)))
    # Coppie positive per i controlli
    if control_idx.numel() > 1:
        control_shuffled = control_idx[torch.randperm(control_idx.size(0))]
        for i in range(0, control_shuffled.size(0) - 1, 2):
            idx_a = control_shuffled[i].item()
            idx_b = control_shuffled[i + 1].item()
            positive_pairs.append(("positive", (idx_a, idx_b)))
    # Coppie negative (tra trattati e controlli)
    n_neg = min(treated_idx.numel(), control_idx.numel())
    if n_neg > 0:
        treated_shuffled = treated_idx[torch.randperm(treated_idx.size(0))]
        control_shuffled = control_idx[torch.randperm(control_idx.size(0))]
        for i in range(n_neg):
            idx_t = treated_shuffled[i].item()
            idx_c = control_shuffled[i].item()
            negative_pairs.append(("negative", (idx_t, idx_c)))
    pairs = []
    i_pos, i_neg = 0, 0
    while i_pos < len(positive_pairs) or i_neg < len(negative_pairs):
        if i_pos < len(positive_pairs):
            pairs.append(positive_pairs[i_pos])
            i_pos += 1
        if i_neg < len(negative_pairs):
            pairs.append(negative_pairs[i_neg])
            i_neg += 1

    return pairs


def create_pairs_batch(x, t, y, w, pairs, max_pairs=None):
    if max_pairs is not None and len(pairs) > max_pairs:
        pairs = pairs[:max_pairs]
    if len(pairs) == 0:
        return None

    x1_list, z1_list, y1_list, w1_list = [], [], [], []
    x2_list, z2_list, y2_list, w2_list = [], [], [], []
    same_list = []

    for label, (idx1, idx2) in pairs:
        x1_list.append(x[idx1].unsqueeze(0))
        z1_list.append(t[idx1].unsqueeze(0))
        y1_list.append(y[idx1].unsqueeze(0))
        w1_list.append(w[idx1].unsqueeze(0))  # aggiungi il peso corrispondente

        x2_list.append(x[idx2].unsqueeze(0))
        z2_list.append(t[idx2].unsqueeze(0))
        y2_list.append(y[idx2].unsqueeze(0))
        w2_list.append(w[idx2].unsqueeze(0))  # aggiungi il peso corrispondente

        same_indicator = 1.0 if label == "positive" else 0.0
        same_list.append(same_indicator)

    x1_batch = torch.cat(x1_list, dim=0)
    z1_batch = torch.cat(z1_list, dim=0)
    y1_batch = torch.cat(y1_list, dim=0)
    w1_batch = torch.cat(w1_list, dim=0)
    x2_batch = torch.cat(x2_list, dim=0)
    z2_batch = torch.cat(z2_list, dim=0)
    y2_batch = torch.cat(y2_list, dim=0)
    w2_batch = torch.cat(w2_list, dim=0)
    same_batch = torch.tensor(same_list, device=x.device).float()

    return x1_batch, z1_batch, y1_batch, w1_batch, x2_batch, z2_batch, y2_batch, w2_batch, same_batch


def train(
        model,
        num_epochs,
        train_loader,
        realizations_to_use,
        mean_global,
        std_global,
        margin=1.0,
        max_pairs=None,
        lr=1e-3,
        device="cpu",
        alpha=1.0,  # peso della contrastive loss
        beta=1.0  # peso della factual loss
):
    training_losses = []

    # Ottimizzatori separati
    optimizer_phi = optim.Adam(
        list(model.Phi1.parameters()) + list(model.Phi2.parameters()),
        lr=lr
    )
    optimizer_hyp = optim.Adam(
        list(model.h0.parameters()) + list(model.h1.parameters()),
        lr=lr
    )

    for epoch in range(num_epochs):
        epoch_total_loss = 0.0
        epoch_factual_loss = 0.0
        epoch_contrast_loss = 0.0

        for x, t, Yf, *_, w_train in train_loader:
            x, t, Yf, w_train = x.to(device), t.to(device), Yf.to(device), w_train.to(device)

            batch_total_loss = 0.0
            batch_factual_loss = 0.0
            batch_contrast_loss = 0.0

            for i in range(realizations_to_use):
                x_i = x[:, :, i]
                t_i = t[:, i].unsqueeze(1)
                y_i = Yf[:, i].unsqueeze(1)
                w_i = w_train[:, i].unsqueeze(1)

                # Normalizzo
                x_i = (x_i - mean_global) / (std_global + 1e-6)

                # Creo le coppie (positive/negative)
                pairs = create_pairs(x_i, t_i, y_i)
                batch_pairs = create_pairs_batch(x_i, t_i, y_i, w_i, pairs, max_pairs)
                if batch_pairs is None:
                    continue

                x1, t1, y1, w1, x2, t2, y2, w2, same = batch_pairs

                # ========== Contrastive ==========
                # Embedding: Phi1 per i trattati, Phi2 per i controlli
                e1 = t1 * model.Phi1(x1) + (1 - t1) * model.Phi2(x1)
                e2 = t2 * model.Phi1(x2) + (1 - t2) * model.Phi2(x2)

                optimizer_phi.zero_grad()
                c_loss = contrastive_loss_batch(e1, e2, same, margin)
                c_loss.backward()
                optimizer_phi.step()

                # ========== Factual ==========
                optimizer_hyp.zero_grad()
                # stacco e dal grafo di Phi
                e1d, e2d = e1.detach(), e2.detach()
                pred1 = t1 * model.h1(e1d) + (1 - t1) * model.h0(e1d)
                pred2 = t2 * model.h1(e2d) + (1 - t2) * model.h0(e2d)
                f_loss = ((w1 * (pred1 - y1) ** 2 + w2 * (pred2 - y2) ** 2) / 2).mean()
                f_loss.backward()
                optimizer_hyp.step()

                batch_contrast_loss += c_loss.item()
                batch_factual_loss += f_loss.item()
                batch_total_loss += alpha * c_loss.item() + beta * f_loss.item()

            # Medie su tutte le realizzazioni
            norm = float(realizations_to_use)
            batch_total_loss /= norm
            batch_contrast_loss /= norm
            batch_factual_loss /= norm

            epoch_total_loss += batch_total_loss
            epoch_contrast_loss += batch_contrast_loss
            epoch_factual_loss += batch_factual_loss

        n_batches = len(train_loader)
        avg_total_loss = epoch_total_loss / n_batches
        avg_factual_loss = epoch_factual_loss / n_batches
        avg_contrast_loss = epoch_contrast_loss / n_batches

        logging.info(f"Epoch {epoch + 1}: Total Loss = {avg_total_loss:.4f} | " +
                     f"Factual Loss = {avg_factual_loss:.4f} | Contrastive Loss = {avg_contrast_loss:.4f}")
        training_losses.append(avg_total_loss)

    return training_losses


###############################################################################
# Funzione per plottare l'andamento della loss
###############################################################################
def plot_training_losses(training_losses):
    import matplotlib.pyplot as plt
    epochs = range(1, len(training_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, training_losses, marker='o', linestyle='-', label='Loss Totale')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Andamento della Loss durante il Training")
    plt.legend()
    plt.grid(True)
    plt.show()


###############################################################################
# Funzioni di valutazione (evaluation)
###############################################################################
def evaluate_model(model, dataloader, device="cpu", realizations_to_use=100, mean_global=None, std_global=None):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    batches = 0

    for x, t, Yf, *_ in dataloader:
        x, t, Yf = x.to(device), t.to(device), Yf.to(device)
        batch_loss = 0.0
        for i in range(realizations_to_use):
            xi = x[:, :, i]
            ti = t[:, i].unsqueeze(1)
            y_true = Yf[:, i].unsqueeze(1)
            if mean_global is not None:
                xi = (xi - mean_global) / (std_global + 1e-6)
            e, y_pred = model.forward_once(xi, ti)
            batch_loss += criterion(y_pred, y_true).item()
        total_loss += batch_loss / realizations_to_use
        batches += 1

    return total_loss / max(batches, 1)


def evaluate_pehe(model, dataloader, device="cpu"):
    model.eval()
    mse = 0.0
    n = 0
    with torch.no_grad():
        for x, t, Yf, *_ in dataloader:
            x, t, Yf = x.to(device), t.to(device), Yf.to(device)
            xi = x[:, :, 0]
            t0 = torch.zeros_like(xi[:, :1])
            t1 = torch.ones_like(t0)
            _, y0 = model.forward_once(xi, t0)
            _, y1 = model.forward_once(xi, t1)
            tau_hat = (y1 - y0).squeeze()
            tau_true = Yf[:, 1] - Yf[:, 0]
            mse += ((tau_hat - tau_true) ** 2).sum().item()
            n += xi.size(0)
    return mse / max(n, 1)


###############################################################################
# Run e Grid Search
###############################################################################
def run_experiment(config, train_loader, test_loader, mean_global, std_global, device="cpu"):
    hidden_dim = config.get("hidden_dim", 100)
    margin = config.get("margin", 1.0)
    max_pairs = config.get("max_pairs", None)
    realizations_to_use = config.get("realizations_to_use", 100)
    lr = config.get("learning_rate", 1e-3)
    num_epochs = config.get("num_epochs", 30)

    model = DualHeadSiameseCATE(input_dim=25, hidden_dim=hidden_dim, margin=margin).to(device)

    training_losses = train(
        model=model,
        num_epochs=num_epochs,
        train_loader=train_loader,
        realizations_to_use=realizations_to_use,
        mean_global=mean_global,
        std_global=std_global,
        margin=margin,
        max_pairs=max_pairs,
        lr=lr,
        device=device,
        alpha=0.1,  # scegli il peso della contrastive loss
        beta=0.9  # scegli il peso della factual loss
    )

    # Plotta l'andamento della loss totale
    plot_training_losses(training_losses)

    test_loss = evaluate_model(
        model,
        test_loader,
        device=device,
        realizations_to_use=realizations_to_use,
        mean_global=mean_global,
        std_global=std_global
    )
    return test_loss, model


def grid_search(param_grid, train_loader, test_loader, mean_global, std_global, device="cpu"):
    keys = list(param_grid.keys())
    values = list(param_grid.values())

    best_config = None
    best_score = float("inf")
    best_model_state = None

    for combo in itertools.product(*values):
        config = dict(zip(keys, combo))
        logging.info(f"Provo config: {config}")

        score, model = run_experiment(config, train_loader, test_loader, mean_global, std_global, device=device)
        logging.info(f"Risultato test MSE: {score:.4f} per config: {config}")

        if score < best_score:
            best_score = score
            best_config = config
            best_model_state = model.state_dict()

    return best_config, best_score, best_model_state


def evaluate_pehe_avg(model, X_test, T_test, YF_test, mu0_test, mu1_test, device="cpu"):
    model.eval()
    n_units, d, n_real = X_test.shape  # ad esempio, [n_units, d, n_real]
    ite_estimates = []

    with torch.no_grad():
        for i in range(n_real):
            x_i = X_test[:, :, i].to(device)  # [n_units, d]
            # Calcola tcf invertendo il trattamento
            t_i = T_test[:, i].unsqueeze(1).to(device)
            tcf_i = 1 - t_i  # controfattuale

            # Usa il metodo forward_once per ottenere le predizioni
            # Supponiamo che forward_once restituisca una tupla (qualcosa, prediction)
            _, pred_cf = model.forward_once(x_i, tcf_i)

            # Nel caso in cui tu debba calcolare la differenza per stimare l'effetto,
            # Potresti usare pred_cf come controfattuale.
            ite_estimates.append(pred_cf.squeeze())

        ite_estimates = torch.stack(ite_estimates, dim=1)  # [n_units, n_real]
        ite_mean = ite_estimates.mean(dim=1)

        # Calcola gli effetti veri, ad esempio come media su mu1_test - mu0_test (a seconda di come sono organizzati)
        # Supponiamo che tau_true sia la media di mu1 - mu0 per ogni unità:
        tau_true = (mu1_test - mu0_test).mean(dim=1)

        pehe = torch.mean((ite_mean - tau_true) ** 2).item()

    return pehe


def evaluate_ate_avg(model, X, T, device="cpu"):
    """
    Calcola l'ATE stimato mediando sugli effetti individuali per unità,
    aggregando su tutte le repliche.
    X: tensor di dimensioni [n_units, d, n_real]
    T: tensor di dimensioni [n_units, n_real]
    """
    model.eval()
    n_units, d, n_real = X.shape
    with torch.no_grad():
        ate_estimates = []
        for i in range(n_real):
            x_i = X[:, :, i].to(device)  # [n_units, d]
            t_i = T[:, i].unsqueeze(1).to(device)  # [n_units, 1]
            treatment_zeros = torch.zeros(n_units, 1, device=device)
            treatment_ones = torch.ones(n_units, 1, device=device)
            # Usa il metodo forward_once, come hai già fatto
            _, pred_0 = model.forward_once(x_i, treatment_zeros)
            _, pred_1 = model.forward_once(x_i, treatment_ones)
            # Effetto stimato per ciascuna unità per questa replica
            ITE_i = (pred_1 - pred_0).squeeze()  # [n_units]
            ate_estimates.append(ITE_i)
        ate_estimates = torch.stack(ate_estimates, dim=1)  # [n_units, n_real]
        # Calcola la media per unità
        ate_per_unit = ate_estimates.mean(dim=1)  # [n_units]
        # L'ATE stimato globale è la media su tutte le unità
        ate_pred = ate_per_unit.mean().item()
    return ate_pred


# params aggiuntivi:
#   mu0_all:   [n_units, n_real]  mean potential outcome t=0
#   mu1_all:   [n_units, n_real]  mean potential outcome t=1
def evaluate_pehe_all_realizations(model, X, mu0_all, mu1_all, device):
    """
    X:        [n_units, d, n_real]
    mu0_all:  [n_units, n_real] mean potential outcome t=0 (noiseless)
    mu1_all:  [n_units, n_real] mean potential outcome t=1 (noiseless)
    """
    model.eval()
    n_units, _, n_real = X.shape

    # Stima τ per ogni unità e replica
    tau_preds = torch.zeros(n_units, n_real, device=device)
    zeros = torch.zeros(n_units, 1, device=device)
    ones = torch.ones(n_units, 1, device=device)

    with torch.no_grad():
        for i in range(n_real):
            x_i = X[:, :, i].to(device)
            _, p0 = model.forward_once(x_i, zeros)
            _, p1 = model.forward_once(x_i, ones)
            tau_preds[:, i] = (p1 - p0).squeeze()

    # τ̂ mediasu le repliche
    tau_hat = tau_preds.mean(dim=1)  # [n_units]
    # τ⁰ vero (noiseless)
    tau_true = (mu1_all - mu0_all).mean(dim=1).to(device)

    mse_pehe = torch.mean((tau_hat - tau_true) ** 2).item()
    rmse_pehe = mse_pehe ** 0.5
    return mse_pehe, rmse_pehe


###############################################################################
# MAIN
###############################################################################
# Esempio di utilizzo nel main:
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Usando device: {device}")

    # Carica i dataset
    train_dataset = IHDPLoader(is_train=True)
    test_dataset = IHDPLoader(is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, drop_last=False)

    # Calcola mean/std globali sul train set
    mean_global, std_global = compute_global_stats(train_dataset, device=device)

    # Grid search
    param_grid = {
        "hidden_dim": [100],
        "margin": [0.1],
        "max_pairs": [256],
        "realizations_to_use": [100],
        "learning_rate": [1e-3],
        "num_epochs": [30],
    }
    best_config, best_score, best_state = grid_search(
        param_grid,
        train_loader,
        test_loader,
        mean_global, std_global,
        device=device
    )

    # Ricostruisci il modello migliore e carica i pesi
    best_model = DualHeadSiameseCATE(
        input_dim=25,
        hidden_dim=best_config["hidden_dim"],
        margin=best_config["margin"]
    ).to(device)
    best_model.load_state_dict(best_state)
    best_model.eval()

    # Carica TUTTI i tensori dal test set (no loader)
    X_test, T_test, YF_test, YCF_test, mu0_test, mu1_test, _, _ = test_dataset.load()

    # Spostali tutti sulla device
    X_test = X_test.to(device)  # [n_units, d, n_real]
    T_test = T_test.to(device)  # [n_units, n_real]
    YF_test = YF_test.to(device)
    YCF_test = YCF_test.to(device)
    mu0_test = mu0_test.to(device)
    mu1_test = mu1_test.to(device)

    # Ricostruisci i due potenziali outcomes
    # Y0_all[i,r] = Y0_factual se T=0, altrimenti Y0_counterfactual
    Y0_all = torch.where(T_test == 0, YF_test, YCF_test)
    Y1_all = torch.where(T_test == 1, YF_test, YCF_test)

    # 1) ATE
    ate_pred = evaluate_ate_avg(best_model, X_test, T_test, device=device)
    true_ate = (mu1_test - mu0_test).mean().item()
    ate_error = abs(ate_pred - true_ate)
    logging.info(f"ATE predetto = {ate_pred:.4f} | ATE vero = {true_ate:.4f} | errore ATE = {ate_error:.4f}")

    # 2) PEHE su tutte le realizzazioni (noiseless ground‑truth = mu1_test/mu0_test)
    mse_pehe, rmse_pehe = evaluate_pehe_all_realizations(
        best_model,
        X_test,
        mu0_test,
        mu1_test,
        device=device
    )
    logging.info(f"PEHE MSE = {mse_pehe:.4f}, PEHE RMSE = {rmse_pehe:.4f}")
