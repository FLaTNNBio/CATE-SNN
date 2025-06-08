# script_demonstrate_cate_implications.py

import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # O 'Qt5Agg' se preferisci
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
import pandas as pd

# -----------------------------------------------------------------------------
# 0) Impostazioni generali
# -----------------------------------------------------------------------------
NUM_NEIGHBORS = 5      # numero di vicini per personalizzazione
SUBSET_SIZE = 500      # campione per analisi Δtau
TAU_PERCENTILE = 20    # percentuale per Δtau_thr
EPSILON = 0.02         # tolleranza per hard negatives
GENDER_COL = 0         # indice colonna 'gender' in covariates
AGE_COL = 1            # indice colonna 'age' (se presente)

OUTPUT_DIR = "cate_implications_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# 1) Carica dati salvati
# -----------------------------------------------------------------------------
# embeddings.npy: shape (n_samples, embedding_dim)
# treatment_labels.npy: shape (n_samples,), 0/1
# ITE_proxy.npy: shape (n_samples,), proxy di tau (ad esempio mu1 - mu0)
# covariates.npy: shape (n_samples, p), contiene almeno la colonna 'gender'
embeddings = np.load("embeddings_trial4_rep1000.npy")
treatments = np.load("treatment_labels.npy")
ite_proxy = np.load("ITE_proxy.npy")
X = np.load("covariates.npy")

n_samples, embedding_dim = embeddings.shape

# -----------------------------------------------------------------------------
# 2) Personalizzazione dell’ITE: nearest-neighbors nel latent space
# -----------------------------------------------------------------------------
nn_model = NearestNeighbors(n_neighbors=NUM_NEIGHBORS + 1, metric="euclidean").fit(embeddings)
distances, indices = nn_model.kneighbors(embeddings)

# Crea DataFrame con nearest neighbors
nn_records = []
for i in range(n_samples):
    for rank, j in enumerate(indices[i][1:], start=1):  # skip self, rank da 1
        nn_records.append({
            "unit_idx": i,
            "neighbor_rank": rank,
            "neighbor_idx": int(j),
            "distance": float(distances[i][rank]),
            "tau_u": float(ite_proxy[i]),
            "tau_neighbor": float(ite_proxy[j]),
            "treatment_u": int(treatments[i]),
            "treatment_neighbor": int(treatments[j])
        })
df_nn = pd.DataFrame(nn_records)
df_nn.to_csv(os.path.join(OUTPUT_DIR, "nearest_neighbors.csv"), index=False)

print("Esempio nearest neighbors (prime 3 unità):")
for i in range(3):
    subset = df_nn[df_nn["unit_idx"] == i]
    print(f"Unità {i}: tau_proxy = {ite_proxy[i]:.3f}, trattamento = {treatments[i]}")
    for _, r in subset.iterrows():
        if r["neighbor_rank"] > NUM_NEIGHBORS:
            break
        print(f"  Rank {int(r['neighbor_rank'])}: idx {int(r['neighbor_idx'])}, "
              f"dist {r['distance']:.3f}, tau_neighbor {r['tau_neighbor']:.3f}, "
              f"treatment {int(r['treatment_neighbor'])}")
    print()

# -----------------------------------------------------------------------------
# 3) Interpretabilità: PCA e t-SNE
# -----------------------------------------------------------------------------
pca = PCA(n_components=2)
Z_pca = pca.fit_transform(embeddings)
plt.figure(figsize=(6, 5))
plt.scatter(Z_pca[treatments == 1, 0], Z_pca[treatments == 1, 1],
            c='tab:blue', label='Trattati', alpha=0.6, s=20)
plt.scatter(Z_pca[treatments == 0, 0], Z_pca[treatments == 0, 1],
            c='tab:orange', label='Controlli', alpha=0.6, s=20)
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("PCA 2D: embedding trattati vs controlli")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "PCA2D_trattati_controlli.png"), dpi=200)
plt.close()

tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
Z_tsne = tsne.fit_transform(embeddings)
plt.figure(figsize=(6, 5))
plt.scatter(Z_tsne[treatments == 1, 0], Z_tsne[treatments == 1, 1],
            c='tab:blue', label='Trattati', alpha=0.6, s=20)
plt.scatter(Z_tsne[treatments == 0, 0], Z_tsne[treatments == 0, 1],
            c='tab:orange', label='Controlli', alpha=0.6, s=20)
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title("t-SNE 2D: embedding trattati vs controlli")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "tSNE2D_trattati_controlli.png"), dpi=200)
plt.close()

# -----------------------------------------------------------------------------
# 4) Robustezza alla violazione della positività: propensity score
# -----------------------------------------------------------------------------
ps_model = LogisticRegression(solver='liblinear')
ps_model.fit(X, treatments)
ps_pred = ps_model.predict_proba(X)[:, 1]

plt.figure(figsize=(6, 4))
plt.hist(ps_pred[treatments == 1], bins=20, alpha=0.6, label='T=1', color='tab:blue')
plt.hist(ps_pred[treatments == 0], bins=20, alpha=0.6, label='T=0', color='tab:orange')
plt.xlabel("Propensity Score")
plt.ylabel("Frequenza")
plt.title("Distribuzione propensity score: trattati vs controlli")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "propensity_histogram.png"), dpi=200)
plt.close()

plt.figure(figsize=(6, 5))
sc = plt.scatter(Z_pca[:, 0], Z_pca[:, 1], c=ps_pred, cmap='coolwarm', s=20)
plt.colorbar(sc, label="Propensity Score")
plt.title("PCA embedding colorato per propensity score")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "PCA_propensity.png"), dpi=200)
plt.close()

# -----------------------------------------------------------------------------
# 5) Mining dinamico degli "hard negatives": analisi Δtau_proxy
# -----------------------------------------------------------------------------
subset_idx = np.random.choice(n_samples, size=min(n_samples, SUBSET_SIZE), replace=False)
Z_sub = embeddings[subset_idx]
tau_sub = ite_proxy[subset_idx]

Delta = np.abs(tau_sub.reshape(-1, 1) - tau_sub.reshape(1, -1))
flat_Delta = Delta.flatten()
tau_thr = np.percentile(flat_Delta, TAU_PERCENTILE)
print(f"Soglia Δtau ({TAU_PERCENTILE}° percentile) = {tau_thr:.4f}")

hard_neg_mask = (Delta > tau_thr) & (Delta <= tau_thr + EPSILON)
hard_pairs = np.argwhere(hard_neg_mask)
print(f"Numero di possibili hard negatives: {len(hard_pairs)}")

hard_records = []
for i, j in hard_pairs[:10]:
    hard_records.append({
        "unit_i": int(subset_idx[i]),
        "unit_j": int(subset_idx[j]),
        "Delta_tau": float(Delta[i, j])
    })
df_hard = pd.DataFrame(hard_records)
df_hard.to_csv(os.path.join(OUTPUT_DIR, "hard_negatives_examples.csv"), index=False)

plt.figure(figsize=(6, 4))
plt.hist(flat_Delta, bins=50, density=True, color='tab:green', alpha=0.7)
plt.axvline(tau_thr, color='red', linestyle='--', label=f"Percentile {TAU_PERCENTILE}%")
plt.xlabel("Δtau_proxy")
plt.ylabel("Densità")
plt.title("Distribuzione Δtau_proxy (sottoinsieme)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "DeltaTau_distribution.png"), dpi=200)
plt.close()

# -----------------------------------------------------------------------------
# 6) Supporto decision making: mostrare casi analoghi per un esempio i
# -----------------------------------------------------------------------------
i = 10  # esempio di unità
neighbors = indices[i][1:NUM_NEIGHBORS+1]
print(f"Casi analoghi per unità {i} (tau={ite_proxy[i]:.3f}, treatment={treatments[i]}):")
for rank, j in enumerate(neighbors, start=1):
    dist = np.linalg.norm(embeddings[i] - embeddings[j])
    print(f"  Rank {rank}: idx {j}, dist {dist:.3f}, tau={ite_proxy[j]:.3f}, treatment={treatments[j]}, X={X[j]}")

# -----------------------------------------------------------------------------
# 7) Fairness / Auditing: analisi per sottogruppi demografici (gender)
# -----------------------------------------------------------------------------
gender = X[:, GENDER_COL].astype(int)  # 0 per F, 1 per M

plt.figure(figsize=(6, 5))
plt.scatter(Z_pca[gender == 0, 0], Z_pca[gender == 0, 1],
            c='magenta', label='Genere 0 (F)', alpha=0.6, s=20)
plt.scatter(Z_pca[gender == 1, 0], Z_pca[gender == 1, 1],
            c='cyan', label='Genere 1 (M)', alpha=0.6, s=20)
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("PCA embedding colorato per genere")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "PCA_gender.png"), dpi=200)
plt.close()

df_tau = pd.DataFrame({
    'tau_proxy': ite_proxy,
    'gender': np.where(gender == 0, 'F', 'M')
})
plt.figure(figsize=(5, 4))
sns.boxplot(x='gender', y='tau_proxy', data=df_tau, palette=['magenta', 'cyan'])
plt.title("Boxplot tau_proxy per genere")
plt.xlabel("Genere")
plt.ylabel("tau_proxy")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.savefig(os.path.join(OUTPUT_DIR, "Boxplot_tau_by_gender.png"), dpi=200)
plt.close()

def compute_linear_mmd(emb, tr):
    z_t = emb[tr == 1]
    z_c = emb[tr == 0]
    if len(z_t) < 2 or len(z_c) < 2:
        return np.nan
    return np.sum((z_t.mean(axis=0) - z_c.mean(axis=0)) ** 2)

mmd_gender = {}
for g in [0, 1]:
    idxs = np.where(gender == g)[0]
    emb_g = embeddings[idxs]
    tr_g = treatments[idxs]
    mmd_gender[g] = compute_linear_mmd(emb_g, tr_g)
    print(f"MMD lineare per genere {g}: {mmd_gender[g]:.6f}")

plt.figure(figsize=(6, 4))
sns.kdeplot(ps_pred[gender == 0], label='Ps gender=0 (F)', shade=True, color='magenta')
sns.kdeplot(ps_pred[gender == 1], label='Ps gender=1 (M)', shade=True, color='cyan')
plt.xlabel("Propensity Score")
plt.ylabel("Densità")
plt.title("KDE propensity score per genere")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "KDE_ps_by_gender.png"), dpi=200)
plt.close()

print("\n*** Analisi completata. Tutti i risultati (plot e CSV) si trovano in:", OUTPUT_DIR)
