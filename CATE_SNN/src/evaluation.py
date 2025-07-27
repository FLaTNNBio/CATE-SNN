# evaluate_siamese_updated.py

import os
import numpy as np
import torch
import pandas as pd
import matplotlib

matplotlib.use('TkAgg')  # O 'Qt5Agg' se preferisci
import matplotlib.pyplot as plt
import seaborn as sns

from itertools import combinations
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from pathlib import Path

# NEW IMPORTS
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegressionCV
# END NEW IMPORTS

# Importa le classi del progetto (regola i percorsi se necessario)
from src.models.bcauss import BCAUSS
from src.siamese_bcuass.siamese import SiameseBCAUSS
from src.data_loader import DataLoader as CFLoader


# -----------------------------------------------------------------------------
# 1) Funzioni ausiliarie per MMD, distanze, propensity e metriche
# -----------------------------------------------------------------------------

def compute_linear_mmd_np(embeddings: np.ndarray, treatments: np.ndarray) -> float:
    """
    Calcola l’MMD^2 lineare tra embedding di trattati e controlli:
      MMD^2 = || mean(z[T=1]) - mean(z[T=0]) ||^2
    """
    z_t = embeddings[treatments == 1]
    z_c = embeddings[treatments == 0]
    if len(z_t) == 0 or len(z_c) == 0:
        return np.nan
    mean_t = z_t.mean(axis=0)
    mean_c = z_c.mean(axis=0)
    return float(np.sum((mean_t - mean_c) ** 2))


def compute_rbf_mmd_np(embeddings: np.ndarray, treatments: np.ndarray, sigma: float = 1.0) -> float:
    """
    Calcola l’MMD^2 con kernel RBF (gaussiano) tra embedding di trattati e controlli.
    """

    def rbf_kernel(A: np.ndarray, B: np.ndarray, sigma: float) -> np.ndarray:
        sq_A = np.sum(A ** 2, axis=1)[:, np.newaxis]
        sq_B = np.sum(B ** 2, axis=1)[np.newaxis, :]
        sq_dists = sq_A + sq_B - 2 * A.dot(B.T)
        return np.exp(-sq_dists / (2 * sigma ** 2))

    z_t = embeddings[treatments == 1]
    z_c = embeddings[treatments == 0]
    m, n = len(z_t), len(z_c)
    if m < 2 or n < 2:
        return np.nan

    K_tt = rbf_kernel(z_t, z_t, sigma)
    K_cc = rbf_kernel(z_c, z_c, sigma)
    K_tc = rbf_kernel(z_t, z_c, sigma)

    sum_tt = (np.sum(K_tt) - np.trace(K_tt)) / (m * (m - 1))
    sum_cc = (np.sum(K_cc) - np.trace(K_cc)) / (n * (n - 1))
    sum_tc = np.sum(K_tc) / (m * n)

    return float(sum_tt + sum_cc - 2 * sum_tc)


def compute_poly_mmd_np(embeddings: np.ndarray, treatments: np.ndarray,
                        degree: int = 2, c: float = 1.0) -> float:
    """
    Calcola l’MMD^2 con kernel polinomiale tra embedding di trattati e controlli:
      k(x,y) = (x·y + c)^degree
    """
    z_t = embeddings[treatments == 1]
    z_c = embeddings[treatments == 0]
    m, n = len(z_t), len(z_c)
    if m < 1 or n < 1:
        return np.nan

    # Kernel tra trattati-trattati
    K_tt = (z_t.dot(z_t.T) + c) ** degree
    # Kernel tra controlli-controlli
    K_cc = (z_c.dot(z_c.T) + c) ** degree
    # Kernel tra trattati-controlli
    K_tc = (z_t.dot(z_c.T) + c) ** degree

    # Rimozione termini diagonali e normalizzazione
    sum_tt = (np.sum(K_tt) - np.trace(K_tt)) / (m * (m - 1)) if m > 1 else 0.0
    sum_cc = (np.sum(K_cc) - np.trace(K_cc)) / (n * (n - 1)) if n > 1 else 0.0
    sum_tc = np.sum(K_tc) / (m * n)

    return float(sum_tt + sum_cc - 2 * sum_tc)


def compute_smd(x: np.ndarray, w: np.ndarray, t: np.ndarray) -> float:
    """
    Calcola lo Standardized Mean Difference (SMD) per una singola covariata.
    """
    # Filtra i dati per i gruppi trattati e controlli
    x1 = x[t == 1]
    w1 = w[t == 1]
    x0 = x[t == 0]
    w0 = w[t == 0]

    # Calcola le medie pesate
    m1 = np.average(x1, weights=w1)
    m0 = np.average(x0, weights=w0)

    # Calcola le varianze pesate (attenzione: np.average con weights per la varianza
    # richiede un po' di attenzione per essere uncorrected sample variance.
    # Qui usiamo la formula standard per SMD.)
    # Varianza per i trattati
    if len(x1) > 1 and np.sum(w1) > 0:
        v1 = np.sum(w1 * (x1 - m1) ** 2) / (np.sum(w1) * (1 - np.sum(w1 ** 2) / np.sum(w1) ** 2))
    else:
        v1 = 0.0

    # Varianza per i controlli
    if len(x0) > 1 and np.sum(w0) > 0:
        v0 = np.sum(w0 * (x0 - m0) ** 2) / (np.sum(w0) * (1 - np.sum(w0 ** 2) / np.sum(w0) ** 2))
    else:
        v0 = 0.0

    # Se le varianze sono zero o molto piccole, il denominatore può essere problematico.
    # Aggiungi un piccolo epsilon per stabilità numerica o gestisci il caso.
    denominator = np.sqrt((v1 + v0) / 2)
    if denominator < 1e-8:  # Evita divisione per zero
        return 0.0  # Se le varianze sono quasi zero, i gruppi sono identici

    return abs(m1 - m0) / denominator


def compute_pairwise_distances(embeddings: np.ndarray, treatments: np.ndarray, max_pairs: int = 100_000):
    """
    Calcola distanze intra-gruppo (treat-treat, control-control) e
    inter-gruppo (treat-control), campionando fino a max_pairs per ciascuna categoria.
    Ritorna (dists_tt, dists_cc, dists_tc).
    """
    idx_t = np.where(treatments == 1)[0]
    idx_c = np.where(treatments == 0)[0]
    z_t = embeddings[idx_t]
    z_c = embeddings[idx_c]

    # Campioniamo coppie intra-gruppo
    pairs_t = np.array(list(combinations(range(len(z_t)), 2)))
    pairs_c = np.array(list(combinations(range(len(z_c)), 2)))
    if len(pairs_t) > max_pairs:
        idxs = np.random.choice(len(pairs_t), max_pairs, replace=False)
        pairs_t = pairs_t[idxs]
    if len(pairs_c) > max_pairs:
        idxs = np.random.choice(len(pairs_c), max_pairs, replace=False)
        pairs_c = pairs_c[idxs]

    dists_tt = np.linalg.norm(z_t[pairs_t[:, 0]] - z_t[pairs_t[:, 1]], axis=1)
    dists_cc = np.linalg.norm(z_c[pairs_c[:, 0]] - z_c[pairs_c[:, 1]], axis=1)

    # Distanze inter-gruppo
    D_tc = cdist(z_t, z_c)
    dists_tc = D_tc.flatten()
    if len(dists_tc) > max_pairs:
        dists_tc = np.random.choice(dists_tc, max_pairs, replace=False)

    return dists_tt, dists_cc, dists_tc


# -----------------------------------------------------------------------------
# 2) Funzione principale: estrazione embedding e metriche di confronto
# -----------------------------------------------------------------------------

def evaluate_siamese(
        trial_id: int,
        rep_index: int,
        weights_dir: str,
        save_dir: str = None
):
    """
    Valuta il modello Siamese su IHDP (replica rep_index):
      • Estrae embedding Φ(x) per tutti i punti di training
      • Calcola:
         - Mean(propensity) per T=1 e T=0
         - Wasserstein distance tra distribuzioni di propensity score
         - MMD (lineare, RBF, polinomiale)
      • Genera plots: PCA/t-SNE, distribuzioni distanze intra/inter, heatmap, boxplot
      • Calcola:
         - Silhouette score su embeddings
         - Accuracy di un classificatore su embeddings per predire T
         - Max SMD sui covariati originali dopo weighting (con Love Plot)
    """

    # A) Carica IHDP
    loader = CFLoader.get_loader('IHDP')
    (X_tr_all, T_tr_all, X_val_all,
     T_val_all, _, _,  # se vuoi ripetere su validation/test, ma qui prendiamo solo train
     _, _, _, _, _, _) = loader.load()

    # Estrai la replica di train
    Xrep = X_tr_all[:, :, rep_index].astype(np.float32)  # (n_train, dim_X)
    Trep = T_tr_all[:, rep_index].astype(np.int64)  # (n_train,)
    n_train, input_dim = Xrep.shape

    # B) Ricostruisci modello Siamese identico al training
    base_model = BCAUSS(input_dim=input_dim)
    siamese = SiameseBCAUSS(
        base_model=base_model,
        ds_class=None,
        margin=0.75,
        lambda_ctr=1.0,
        lr=0.0002570883195054887,
        batch_size=32,
        epochs=100,
        val_split=0.2,
        update_ite_freq=3,
        warmup_epochs_base=0
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    siamese.to(device)

    # C) Carica i pesi
    weights_path = Path(weights_dir) / f"weights_trial_{trial_id}_rep_{rep_index + 1}.pth"
    if not weights_path.is_file():
        raise FileNotFoundError(f"Pesi non trovati in: {weights_path}")
    siamese.load_state_dict(torch.load(weights_path, map_location=device))
    siamese.eval()

    # D) Estrai embedding Φ(x) su tutto il train set
    X_tensor = torch.from_numpy(Xrep).to(device)
    with torch.no_grad():
        _, z_tensor = siamese.base.mu_and_embedding(X_tensor)
    embeddings = z_tensor.cpu().numpy()  # (n_train, embed_dim)

    # Salva embeddings e tratamenti se richiesto
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, f"embeddings_trial{trial_id}_rep{rep_index + 1}.npy"), embeddings)
        np.save(os.path.join(save_dir, f"treatments_trial{trial_id}_rep{rep_index + 1}.npy"), Trep)

    # -----------------------------------------------------------------------------
    #  E1) Propensity Score con regressione logistica su Xrep → T
    # -----------------------------------------------------------------------------
    ps_model = LogisticRegression(solver='liblinear')
    ps_model.fit(Xrep, Trep)
    ps_pred = ps_model.predict_proba(Xrep)[:, 1]  # valori in [0,1]

    # Calcolo di mean(propensity) su gruppi T=1 e T=0
    mean_ps_treated = ps_pred[Trep == 1].mean() if np.sum(Trep == 1) > 0 else np.nan
    mean_ps_control = ps_pred[Trep == 0].mean() if np.sum(Trep == 0) > 0 else np.nan

    # -----------------------------------------------------------------------------
    #  E2) Wasserstein Distance tra distribuzioni di propensity score
    # -----------------------------------------------------------------------------
    ps_t = ps_pred[Trep == 1]
    ps_c = ps_pred[Trep == 0]
    if len(ps_t) >= 1 and len(ps_c) >= 1:
        wass_ps = wasserstein_distance(ps_t, ps_c)
    else:
        wass_ps = np.nan

    # -----------------------------------------------------------------------------
    #  E3) MMD su embedding: lineare, RBF e polinomiale
    # -----------------------------------------------------------------------------
    mmd_lin = compute_linear_mmd_np(embeddings, Trep)
    mmd_rbf = compute_rbf_mmd_np(embeddings, Trep, sigma=1.0)
    mmd_poly = compute_poly_mmd_np(embeddings, Trep, degree=2, c=1.0)

    # Stampa su console (o salva su file CSV se preferisci)
    print(f"> Trial {trial_id}, Replica {rep_index + 1}:")
    print(f"  • Mean Propensity (T=1 / T=0): {mean_ps_treated:.4f} / {mean_ps_control:.4f}")
    print(f"  • Wasserstein(propensity) = {wass_ps:.6f}")
    print(f"  • MMD(linear) = {mmd_lin:.6e}")
    print(f"  • MMD(RBF, σ=1) = {mmd_rbf:.6e}")
    print(f"  • MMD(Poly, deg=2, c=1) = {mmd_poly:.6e}\n")

    # -------------------------------------------------------------------------
    # E5) Metriche di separabilità e balance aggiuntive
    # -------------------------------------------------------------------------
    # 1) Silhouette score sullo spazio di embedding
    sil_score = silhouette_score(embeddings, Trep)
    print(f"  • Silhouette score (0=overlap, 1=separazione perfetta): {sil_score:.4f}")

    # 2) Classificatore su embeddings per predire T
    clf_emb = LogisticRegressionCV(cv=5, max_iter=1000, solver='liblinear').fit(embeddings, Trep)
    acc_emb = clf_emb.score(embeddings, Trep)
    print(f"  • Accuracy predizione T da embedding: {acc_emb:.4f}")

    # 3) Standardized Mean Difference (SMD) sui covariati Xrep dopo weighting
    #    calcola i pesi da ps_pred
    # Troncamento dei propensity scores per stabilità dei pesi IPTW
    epsilon = 1e-3  # Valore di trimming
    ps_pred_trimmed = np.clip(ps_pred, epsilon, 1 - epsilon)
    w = Trep / ps_pred_trimmed + (1 - Trep) / (1 - ps_pred_trimmed)

    smds = [compute_smd(Xrep[:, j], w, Trep) for j in range(Xrep.shape[1])]
    max_smd_weighted = np.max(smds)
    print(f"  • Max SMD sui covariati dopo weighting: {max_smd_weighted:.4f}")

    # Calcola SMD non pesato
    smd_unw = [compute_smd(Xrep[:, j], np.ones_like(w), Trep) for j in range(Xrep.shape[1])]
    max_smd_unweighted = np.max(smd_unw)
    print(f"  • Max SMD sui covariati non pesati: {max_smd_unweighted:.4f}\n")

    # -----------------------------------------------------------------------------
    #  E4) Plot e analisi visive (come prima: PCA, t-SNE, istogramma distanze, heatmap, boxplot)
    # -----------------------------------------------------------------------------

    # Love Plot (SMD prima vs dopo weighting)
    plt.figure(figsize=(10, 7))
    plt.scatter(smd_unw, range(Xrep.shape[1]), label='SMD Non Pesato', alpha=0.7, color='tab:red')
    plt.scatter(smds, range(Xrep.shape[1]), label='SMD Pesato (IPTW)', alpha=0.7, color='tab:blue')
    plt.axvline(x=0.1, color='green', linestyle='--', label='Target SMD < 0.1')
    plt.xlabel("Standardized Mean Difference (SMD)")
    plt.ylabel("Covariata Index")
    plt.title(f"Love Plot: Covariate Balance (Trial {trial_id}, Replica {rep_index + 1})")
    plt.legend()
    plt.grid(True)
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"LovePlot_trial{trial_id}_rep{rep_index + 1}.png"),
                    dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

    # PCA 2D
    pca = PCA(n_components=2)
    Z_pca = pca.fit_transform(embeddings)
    plt.figure(figsize=(6, 5))
    plt.scatter(Z_pca[Trep == 1, 0], Z_pca[Trep == 1, 1],
                c='tab:blue', label='Trattati', alpha=0.6, s=20)
    plt.scatter(Z_pca[Trep == 0, 0], Z_pca[Trep == 0, 1],
                c='tab:orange', label='Controlli', alpha=0.6, s=20)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("PCA 2D: embedding trattati vs controlli")
    plt.legend()
    plt.grid(True)
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"PCA2D_trial{trial_id}_rep{rep_index + 1}.png"), dpi=300,
                    bbox_inches='tight')
    else:
        plt.show()
    plt.close()

    # t-SNE 2D
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    Z_tsne = tsne.fit_transform(embeddings)
    plt.figure(figsize=(6, 5))
    plt.scatter(Z_tsne[Trep == 1, 0], Z_tsne[Trep == 1, 1],
                c='tab:blue', label='Trattati', alpha=0.6, s=20)
    plt.scatter(Z_tsne[Trep == 0, 0], Z_tsne[Trep == 0, 1],
                c='tab:orange', label='Controlli', alpha=0.6, s=20)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("t-SNE 2D: embedding trattati vs controlli")
    plt.legend()
    plt.grid(True)
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"tSNE2D_trial{trial_id}_rep{rep_index + 1}.png"), dpi=300,
                    bbox_inches='tight')
    else:
        plt.show()
    plt.close()

    # Istogramma delle distanze intra-/inter-
    dists_tt, dists_cc, dists_tc = compute_pairwise_distances(embeddings, Trep, max_pairs=100_000)
    bins = np.linspace(0, np.percentile(dists_tc, 99), 100)
    plt.figure(figsize=(8, 5))
    plt.hist(dists_tt, bins=bins, alpha=0.5, label='Distanze TT', density=True, color='tab:blue')
    plt.hist(dists_cc, bins=bins, alpha=0.5, label='Distanze CC', density=True, color='tab:orange')
    plt.hist(dists_tc, bins=bins, alpha=0.5, label='Distanze TC', density=True, color='tab:green')
    plt.xlabel("Distanza Euclidea")
    plt.ylabel("Densità")
    plt.title("Istogramma distanze: TT vs CC vs TC")
    plt.legend()
    plt.grid(True)
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"Distanze_hist_trial{trial_id}_rep{rep_index + 1}.png"),
                    dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

    # Heatmap correlazione prime 20 dimensioni
    dim_to_plot = min(20, embeddings.shape[1])
    corr_matrix = np.corrcoef(embeddings[:, :dim_to_plot].T)
    plt.figure(figsize=(7, 6))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, square=True,
                cbar_kws={'shrink': 0.5}, xticklabels=False, yticklabels=False)
    plt.title(f"Heatmap correlazione prime {dim_to_plot} dim (trial {trial_id})")
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"HeatmapCorr_trial{trial_id}_rep{rep_index + 1}.png"),
                    dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

    # Box-plot prime 10 dimensioni embedding (T=1 vs T=0)
    records = []
    for dim in range(min(dim_to_plot, 10)):  # Plotting up to 10 dimensions for clarity
        for i in range(n_train):
            records.append({
                'dimension': f'dim_{dim}',
                'value': embeddings[i, dim],
                'group': 'Trattati' if Trep[i] == 1 else 'Controlli'
            })
    df_embed = pd.DataFrame.from_records(records)
    # No need for subset as we limit in the loop now
    # subset = df_embed[df_embed['dimension'].isin([f'dim_{i}' for i in range(min(dim_to_plot, 10))])]

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='dimension', y='value', hue='group', data=df_embed,  # Use df_embed directly
                palette=['tab:blue', 'tab:orange'])
    plt.xticks(rotation=45)
    plt.title("Boxplot prime 10 dimensioni (T=1 vs T=0)")
    plt.ylabel("Valore embedding")
    plt.xlabel("Dimensione latente")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"BoxplotDims_trial{trial_id}_rep{rep_index + 1}.png"),
                    dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

    # Logging dei risultati in un CSV
    results = {
        'trial_id': trial_id,
        'rep_index': rep_index + 1,
        'mean_ps_treated': mean_ps_treated,
        'mean_ps_control': mean_ps_control,
        'wasserstein_ps': wass_ps,
        'mmd_linear': mmd_lin,
        'mmd_rbf': mmd_rbf,
        'mmd_poly': mmd_poly,
        'silhouette_score': sil_score,
        'acc_pred_T_from_embedding': acc_emb,
        'max_smd_unweighted': max_smd_unweighted,
        'max_smd_weighted': max_smd_weighted
    }

    if save_dir:
        results_df = pd.DataFrame([results])
        csv_path = os.path.join(save_dir, "evaluation_summary.csv")
        # Se il file non esiste, crea con header; altrimenti, aggiungi senza header
        if not os.path.exists(csv_path):
            results_df.to_csv(csv_path, index=False, mode='w', header=True)
        else:
            results_df.to_csv(csv_path, index=False, mode='a', header=False)

    print("Valutazione completata.\n")


# -----------------------------------------------------------------------------
# 3) Esempio di invocazione
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    TRIAL_ID = 4
    REP_INDEX = 999  # 0-based, corrisponde a “rep_1000”
    WEIGHTS_DIR = "siamese_bcuass/saved_weights"
    SAVE_DIR = "evaluation_results"  # Cartella per salvare immagini e .npy

    evaluate_siamese(
        trial_id=TRIAL_ID,
        rep_index=REP_INDEX,
        weights_dir=WEIGHTS_DIR,
        save_dir=SAVE_DIR
    )