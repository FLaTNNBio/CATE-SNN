#!/usr/bin/env python
import torch
import torch.optim as optim
import torch.nn as nn
from scipy.stats import wasserstein_distance
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from paper.IHDPLoader import IHDPLoader
from paper.TARNet import TARNet

# Imposta il seed per la riproducibilità
torch.manual_seed(1337)

# Carica i dataset
full_train_dataset = IHDPLoader(is_train=True)
test_dataset = IHDPLoader(is_train=False)

# Se il dataset training deve essere suddiviso in training e validazione:
total_units = len(full_train_dataset)  # ad esempio, 672 unità
# Per esempio, se vuoi usare il 70% per training e il 30% per validazione:
train_size = int(0.7 * total_units)
valid_size = total_units - train_size

train_dataset, valid_dataset = random_split(full_train_dataset, [train_size, valid_size])
print("Numero unità train:", len(train_dataset))
print("Numero unità valid:", len(valid_dataset))
print("Numero unità test:", len(test_dataset))

# Crea i DataLoader per ogni split
train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True, drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size=100, shuffle=False, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, drop_last=False)

# Verifica le dimensioni dei dati (utilizzando il dataset full, per esempio)
X_train, T_train, YF_train, YCF_train, mu_0_train, mu_1_train, u_train, w_train = full_train_dataset.load()
print("Shape:", X_train.shape, T_train.shape, YF_train.shape, YCF_train.shape,
      mu_0_train.shape, mu_1_train.shape, u_train.shape, w_train.shape)
print("Numero di batch - train_loader:", len(train_loader))
print("Numero di batch - valid_loader:", len(valid_loader))
print("Numero di batch - test_loader:", len(test_loader))

# Visualizza le forme dei tensori nel primo batch del training
train_iter = iter(train_loader)
first_batch = next(train_iter)
print("Forme dei tensori nel primo batch:")
for tensor in first_batch:
    print(tensor.shape)

# Imposta il device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


##############################################
# Inizializzazione del modello TARNet
##############################################
# Gli input hanno shape: X_train: [n_samples, 25, 1000], T_train e YF_train: [n_samples, 1000]
# Assicurati che il modello sia predisposto per gestire questi input.

##############################################
# Funzione di training
##############################################
def train(train_dataset, train_loader, num_epochs, alpha, model, optimizer):
    # Nota: i dati sono già caricati nel dataset fornito
    _, _, YF_train, _, _, _, _, _ = train_dataset.load()
    training_losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for sample_num, batched_factual_sample in enumerate(train_loader):
            # x: [batch_size, 25, 1000]; t, Yf, w_train: [batch_size, 1000]
            x, t, Yf, _, _, _, w_train = batched_factual_sample
            x, t, Yf, w_train = x.to(device), t.to(device), Yf.to(device), w_train.to(device)
            optimizer.zero_grad()

            loss_batch = 0.0
            # Loop sulle 1000 realizzazioni (o repliche)
            for i in range(100):
                realization_x = x[:, :, i]  # [batch_size, 25]
                realization_t = t[:, i].unsqueeze(1)  # [batch_size, 1]
                realization_Yf = Yf[:, i].unsqueeze(1)  # [batch_size, 1]
                realization_w = w_train[:, i].unsqueeze(1)  # [batch_size, 1]

                predictions = model(realization_x, realization_t)
                squared_errors = (predictions - realization_Yf) ** 2
                factual_loss = (realization_w * squared_errors).mean()

                # Calcolo del termine IPM con Wasserstein (esempio)
                IPM_term = 0.0
                with torch.no_grad():
                    phi_x = model.phi(realization_x)
                    # Separiamo i dati trattati e di controllo
                    phi_x_treated = phi_x[realization_t.squeeze() == 1].cpu().numpy()
                    phi_x_control = phi_x[realization_t.squeeze() == 0].cpu().numpy()
                    if phi_x_treated.size > 0 and phi_x_control.size > 0:
                        for dim in range(phi_x_treated.shape[1]):
                            IPM_term += wasserstein_distance(phi_x_treated[:, dim],
                                                             phi_x_control[:, dim])
                        IPM_term /= phi_x_treated.shape[1]

                loss = factual_loss + alpha * IPM_term
                loss.backward()
                optimizer.step()
                loss_batch += loss.item()
            loss_batch /= 1000  # Media sulle 1000 repliche nel batch
            epoch_loss += loss_batch
        epoch_loss /= len(train_loader)
        print(f"Epoch {epoch + 1}, Epoch Loss: {epoch_loss}")
        training_losses.append(epoch_loss)
    return training_losses


##############################################
# Funzione di evaluation
##############################################
def evaluate_model(model, dataloader):
    model.eval()
    total_loss = 0.0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for x, t, y_f, _, _, _, _ in dataloader:
            x, t, y_f = x.to(device), t.to(device), y_f.to(device)
            batch_loss = 0.0
            for i in range(100):
                realization_x = x[:, :, i]
                realization_t = t[:, i].unsqueeze(1)
                realization_y = y_f[:, i].unsqueeze(1)
                y_pred = model(realization_x, realization_t)
                loss = criterion(y_pred, realization_y)
                batch_loss += loss.item()
            batch_loss /= 1000.0
            total_loss += batch_loss
    avg_loss = total_loss / len(dataloader)
    print("Evaluation Loss:", avg_loss)
    return avg_loss


##############################################
# Main: Training, Evaluation e Calcolo delle metriche
##############################################
print("Inizio training del modello TARNet...")
lr = 1e-3
num_epochs = 30
alpha = 1
input_dim = 25
hidden_dim = 100

model = TARNet(input_dim=input_dim, hidden_dim=hidden_dim)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

training_losses = train(full_train_dataset, train_loader, num_epochs, alpha, model, optimizer)
plt.plot(training_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()

# Valutazione (puoi usare il valid_loader o test_loader)
evaluate_model(model, valid_loader)

# Calcolo degli ITE e delle metriche (usiamo il dataset completo di training per il calcolo delle metriche)
X_train, T_train, YF_train, _, mu_0_train, mu_1_train, _, _ = full_train_dataset.load()
X_train, T_train, YF_train = X_train.to(device), T_train.to(device), YF_train.to(device)
# X_train: [672, 25, 1000], T_train: [672, 1000], YF_train: [672, 1000]

# Calcolo degli ITE: ciclo su 1000 repliche (o ad un numero desiderato, ad esempio 1000)
ITE_accumulated = torch.zeros_like(YF_train).float()
for i in range(100):
    realization_x = X_train[:, :, i]  # [672, 25]
    realization_t = T_train[:, i].unsqueeze(1)  # [672, 1]
    realization_tcf = 1 - realization_t  # Invertiamo il trattamento per il controfattuale
    realization_Yf = YF_train[:, i].unsqueeze(1)  # [672, 1]

    predictions_cf = model(realization_x, realization_tcf)

    ITE_vals = torch.zeros_like(realization_Yf)
    # Per soggetti trattati: ITE = outcome osservato - controfattuale stimato
    ITE_vals[realization_t == 1] = realization_Yf[realization_t == 1] - predictions_cf[realization_t == 1]
    # Per soggetti non trattati: ITE = controfattuale stimato - outcome osservato
    ITE_vals[realization_t == 0] = predictions_cf[realization_t == 0] - realization_Yf[realization_t == 0]

    ITE_accumulated[:, i] = ITE_vals.squeeze()

# Calcola l'effetto medio per ogni unità (media sulle repliche)
ITE_avg = ITE_accumulated.mean(dim=1)
ITE_avg_np = ITE_avg.detach().cpu().numpy()

# Calcolo dell'ATE stimato: media degli ITE su tutte le unità
ATE_stimato = ITE_avg_np.mean()
print("Average Treatment Effect (ATE) stimato:", ATE_stimato)

# Calcola il PEHE, se hai le stime "vere" (mu_1_train e mu_0_train)
tau_true = (mu_1_train - mu_0_train).to(device)
tau_true_avg = tau_true.mean(dim=1)
PEHE = torch.mean((ITE_avg - tau_true_avg) ** 2)
print("PEHE:", PEHE.item())

# Visualizzazione degli ITE per unità
plt.figure(figsize=(12, 6))
plt.plot(ITE_avg_np, label='Average ITE across 1000 Realizations', marker='o', linestyle='-', markersize=4)
plt.xlabel('Unit Index')
plt.ylabel('Average ITE')
plt.title('Predicted Individual Treatment Effect (ITE) Across Units')
plt.legend()
plt.grid(True)
plt.show()
