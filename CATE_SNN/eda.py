import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # oppure 'MacOSX' se funziona meglio sul tuo Mac
import matplotlib.pyplot as plt
import seaborn as sns

# Carica il dataset IHDP utilizzando la classe IHDPLoader.
# Il dataset IHDP è molto usato in studi di inferenza causale e contiene:
#   - X: covariate (le caratteristiche di ciascun soggetto), qui con forma (n, d, R) dove:
#         n = numero di unità (es. 672)
#         d = numero di covariate (es. 25)
#         R = numero di realizzazioni (es. 1000) – ogni unità ha più "campionamenti"
#   - T: indicatore di trattamento (0 = controllo, 1 = trattato) per ogni unità e realizzazione
#   - YF: outcome "fattuale" osservato (ossia, l’outcome per il trattamento effettivamente ricevuto)
#   - YCF: outcome "controfattuale" (non osservato, ma disponibile nel dataset per confronto)
#   - mu0 e mu1: stime medie dei potenziali outcome per controllo e trattamento
#   - u: probabilità marginale del trattamento per ogni realizzazione
#   - w: pesi di ribilanciamento calcolati a partire da T (spesso utilizzati per correggere lo sbilanciamento)
from paper.IHDPLoader import IHDPLoader

# Carica il dataset di training
loader = IHDPLoader(is_train=True)
X_train, T_train, YF_train, YCF_train, mu_0_train, mu_1_train, u_train, w_train = loader.load()

# Converti in numpy per agevolare l'analisi con numpy/pandas/matplotlib
X_train_np = X_train.numpy()    # Forma: (672, 25, 1000)
T_train_np = T_train.numpy()    # Forma: (672, 1000)
YF_train_np = YF_train.numpy()  # Forma: (672, 1000)
YCF_train_np = YCF_train.numpy()# Forma: (672, 1000)
mu0_np = mu_0_train.numpy()     # Forma: (672, 1000)
mu1_np = mu_1_train.numpy()     # Forma: (672, 1000)
u_np = u_train.numpy()          # Forma: (1, 1000)
w_np = w_train.numpy()          # Forma: (672, 1000)

# Stampa le shape dei tensori per avere un quadro generale della struttura dei dati
print("Forme dei tensori:")
print("X_train:", X_train_np.shape)
print("T_train:", T_train_np.shape)
print("YF_train:", YF_train_np.shape)
print("YCF_train:", YCF_train_np.shape)
print("mu0:", mu0_np.shape)
print("mu1:", mu1_np.shape)
print("u:", u_np.shape)
print("w:", w_np.shape)

####################################
# ANALISI STATISTICA DELLE COVARIATE
####################################
# Le covariate sono contenute in X_train, con shape (672, 25, 1000).
# Poiché 1000 rappresenta le realizzazioni (o campionamenti),
# per iniziare analizziamo la prima realizzazione (indice 0) – questo fornisce
# una "istantanea" delle covariate per ogni unità.
X0 = X_train_np[:, :, 0]  # Forma: (672, 25)

print("\nStatistiche delle covariate (prima realizzazione):")
print("Min per covariata:", X0.min(axis=0))
print("Max per covariata:", X0.max(axis=0))
print("Media per covariata:", X0.mean(axis=0))
print("Std per covariata:", X0.std(axis=0))

# Visualizza le distribuzioni delle prime 5 covariate
plt.figure(figsize=(10, 6))
for j in range(min(5, X0.shape[1])):
    sns.histplot(X0[:, j], kde=True, label=f'Covariata {j+1}', bins=20)
plt.title("Distribuzione delle prime 5 covariate (prima realizzazione)")
plt.xlabel("Valore della covariata")
plt.legend()
plt.show()  # Se plt.show() dà errore, usa plt.savefig('eda_covariate.png') e apri l'immagine

####################################
# ANALISI DEL TRATTAMENTO
####################################
# T_train contiene gli indicatori di trattamento e ha shape (672, 1000).
# Esaminiamo la distribuzione per la prima realizzazione.
T0 = T_train_np[:, 0]
print("\nDistribuzione del trattamento (prima realizzazione):")
unique, counts = np.unique(T0, return_counts=True)
print(dict(zip(unique, counts)))

plt.figure(figsize=(6,4))
plt.bar(unique, counts, tick_label=["Controllo (0)", "Trattato (1)"])
plt.title("Distribuzione del trattamento (prima realizzazione)")
plt.xlabel("Indicatore di trattamento")
plt.ylabel("Frequenza")
plt.show()  # Usa plt.savefig se necessario

####################################
# ANALISI DEGLI OUTCOME
####################################
# YF_train contiene gli outcome osservati ("fattuali") con shape (672, 1000).
# Consideriamo la prima realizzazione per analizzare la distribuzione degli outcome.
Y0 = YF_train_np[:, 0]
print("\nStatistiche degli outcome fattuali (prima realizzazione):")
print("Min:", Y0.min())
print("Max:", Y0.max())
print("Media:", Y0.mean())
print("Std:", Y0.std())

plt.figure(figsize=(8, 5))
sns.histplot(Y0, kde=True, bins=30)
plt.title("Distribuzione degli outcome fattuali (prima realizzazione)")
plt.xlabel("Outcome osservato")
plt.ylabel("Frequenza")
plt.show()

####################################
# CONFRONTO TRA OUTCOME OSSERVATO E POTENZIALI (mu0, mu1)
####################################
# mu0 e mu1 sono le stime dei potenziali outcome per controllo e trattamento, con shape (672, 1000).
# Per la realizzazione 0, confrontiamo gli outcome osservati e le stime.
mu0_0 = mu0_np[:, 0]
mu1_0 = mu1_np[:, 0]

plt.figure(figsize=(10, 6))
plt.scatter(Y0, mu0_0, alpha=0.6, label="Controllo (mu0)")
plt.scatter(Y0, mu1_0, alpha=0.6, label="Trattato (mu1)")
plt.xlabel("Outcome osservato (YF, prima realizzazione)")
plt.ylabel("Potenziale outcome stimato")
plt.title("Confronto tra outcome osservato e stimato (mu0/mu1)")
plt.legend()
plt.show()

####################################
# ANALISI DI PROBABILITÀ DI TRATTAMENTO E PESI
####################################
print("\nValori medi della probabilità di trattamento 'u':", u_np)
plt.figure(figsize=(8, 4))
# u_np è di forma (1, 1000): traccia la probabilità di trattamento per ciascuna realizzazione
plt.plot(u_np[0], marker='o', linestyle='-', label="Probabilità di trattamento (u)")
plt.title("Andamento della probabilità di trattamento per realizzazione")
plt.xlabel("Indice di realizzazione")
plt.ylabel("u")
plt.legend()
plt.show()

####################################
# OSSERVAZIONI FINALI SULLA STRUTTURA DEL DATASET
####################################
print("\nOsservazioni EDA:")
print(f"- X_train ha forma: {X_train_np.shape} (n, d, R), dove n = numero di unità, d = numero di covariate, R = realizzazioni.")
print(f"- T_train ha forma: {T_train_np.shape} (n, R), dove ogni elemento è 0 o 1, indicante il trattamento.")
print(f"- YF_train e YCF_train hanno forma: {YF_train_np.shape}, con outcome osservati e controfattuali.")
print(f"- mu0 e mu1 hanno forma: {mu0_np.shape}; rappresentano stime dei potenziali outcome per controllo e trattamento.")
print(f"- u ha forma: {u_np.shape} ed indica la probabilità marginale di trattamento per ogni realizzazione.")
print(f"- w ha forma: {w_np.shape} e fornisce i pesi per il ribilanciamento degli esempi.")
