# -*- coding: utf-8 -*-
"""
Script per la valutazione di un modello SiameseBCAUSS addestrato sul dataset IHDP.

Questo script esegue le seguenti operazioni:
1. Carica una specifica replica del dataset IHDP e i pesi di un modello pre-addestrato.
2. Estrae gli embedding (rappresentazioni latenti) dei dati di training.
3. Calcola una serie di metriche per valutare il bilanciamento dei gruppi
   trattati e di controllo nello spazio latente e nello spazio originale (pesato).
   Le metriche includono:
    - Propensity Score Analysis (media e distanza di Wasserstein).
    - Maximum Mean Discrepancy (MMD) con kernel lineare, RBF e polinomiale.
    - Silhouette Score per la separabilità dei cluster.
    - Accuratezza di un classificatore logistico nel predire il trattamento dagli embedding.
    - Standardized Mean Difference (SMD) sulle covariate originali.
4. Genera e salva una serie di visualizzazioni per l'analisi qualitativa:
    - PCA e t-SNE degli embedding.
    - Istogrammi delle distanze intra- e inter-gruppo.
    - Love Plot per visualizzare il bilanciamento delle covariate.
    - Heatmap di correlazione e Boxplot delle dimensioni latenti.
5. Salva i risultati numerici in un file CSV per un'analisi aggregata.

L'intera logica è incapsulata nella classe `SiameseEvaluator` per una migliore
organizzazione e riusabilità.
"""

from itertools import combinations
from pathlib import Path
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# Imposta il backend di Matplotlib per evitare problemi in alcuni ambienti
matplotlib.use('TkAgg')

# Importa le classi del progetto (assicurati che 'src' sia nel python path)
try:
    from src.models.bcauss import BCAUSS
    from src.siamese_bcuass.siamese import SiameseBCAUSS
    from src.data_loader import DataLoader as CFLoader
except ImportError as e:
    print(f"Errore: impossibile importare i moduli del progetto. Assicurati che la cartella 'src' sia accessibile.")
    print(f"Dettagli: {e}")
    # Esci se i moduli custom non sono trovati
    exit()


# =============================================================================
# 2. FUNZIONI AUSILIARIE PER LE METRICHE (HELPER FUNCTIONS)
# Queste funzioni sono "pure", cioè non dipendono dallo stato di una classe.
# =============================================================================

def compute_linear_mmd_np(embeddings: np.ndarray, treatments: np.ndarray) -> float:
    """
    Calcola il Maximum Mean Discrepancy (MMD^2) con kernel lineare.

    Questa metrica misura la distanza tra le medie degli embedding dei gruppi
    trattati e di controllo. MMD^2 = || mean(z_treated) - mean(z_control) ||^2.

    Args:
        embeddings (np.ndarray): Matrice degli embedding (n_samples, n_features).
        treatments (np.ndarray): Vettore binario dei trattamenti (n_samples,).

    Returns:
        float: Il valore di MMD^2 lineare. Restituisce np.nan se uno dei due
               gruppi è vuoto.
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
    Calcola il MMD^2 con kernel RBF (Gaussiano).

    Questa è una versione più potente del MMD che cattura differenze di ordine
    superiore tra le distribuzioni, non solo le medie.

    Args:
        embeddings (np.ndarray): Matrice degli embedding.
        treatments (np.ndarray): Vettore dei trattamenti.
        sigma (float): Parametro di larghezza (bandwidth) del kernel RBF.

    Returns:
        float: Il valore di MMD^2 RBF. Restituisce np.nan se uno dei gruppi
               ha meno di 2 campioni.
    """
    z_t = embeddings[treatments == 1]
    z_c = embeddings[treatments == 0]
    m, n = len(z_t), len(z_c)

    if m < 2 or n < 2:
        return np.nan

    def rbf_kernel(A: np.ndarray, B: np.ndarray, sigma_val: float) -> np.ndarray:
        """Calcola la matrice del kernel RBF tra due set di vettori."""
        sq_dists = cdist(A, B, 'sqeuclidean')
        return np.exp(-sq_dists / (2 * sigma_val ** 2))

    K_tt = rbf_kernel(z_t, z_t, sigma)
    K_cc = rbf_kernel(z_c, z_c, sigma)
    K_tc = rbf_kernel(z_t, z_c, sigma)

    # Formula non-biasata per MMD^2
    sum_tt = (np.sum(K_tt) - np.trace(K_tt)) / (m * (m - 1))
    sum_cc = (np.sum(K_cc) - np.trace(K_cc)) / (n * (n - 1))
    sum_tc = np.sum(K_tc) / (m * n)

    return float(sum_tt + sum_cc - 2 * sum_tc)


def compute_poly_mmd_np(embeddings: np.ndarray, treatments: np.ndarray, degree: int = 2, c: float = 1.0) -> float:
    """
    Calcola il MMD^2 con kernel polinomiale.

    k(x, y) = (x·y + c)^degree

    Args:
        embeddings (np.ndarray): Matrice degli embedding.
        treatments (np.ndarray): Vettore dei trattamenti.
        degree (int): Grado del polinomio.
        c (float): Termine di offset.

    Returns:
        float: Il valore di MMD^2 polinomiale. Restituisce np.nan se uno dei
               gruppi ha meno di 2 campioni.
    """
    z_t = embeddings[treatments == 1]
    z_c = embeddings[treatments == 0]
    m, n = len(z_t), len(z_c)

    if m < 2 or n < 2:
        return np.nan

    K_tt = (z_t @ z_t.T + c) ** degree
    K_cc = (z_c @ z_c.T + c) ** degree
    K_tc = (z_t @ z_c.T + c) ** degree

    sum_tt = (np.sum(K_tt) - np.trace(K_tt)) / (m * (m - 1))
    sum_cc = (np.sum(K_cc) - np.trace(K_cc)) / (n * (n - 1))
    sum_tc = np.sum(K_tc) / (m * n)

    return float(sum_tt + sum_cc - 2 * sum_tc)


def compute_smd(x: np.ndarray, t: np.ndarray, w: np.ndarray = None) -> float:
    """
    Calcola lo Standardized Mean Difference (SMD) per una singola covariata.

    L'SMD misura la differenza tra le medie di due gruppi in unità di
    deviazione standard. Un valore < 0.1 è generalmente considerato un buon
    bilanciamento.

    Args:
        x (np.ndarray): Vettore della covariata (n_samples,).
        t (np.ndarray): Vettore binario dei trattamenti (n_samples,).
        w (np.ndarray, optional): Pesi per ogni campione (es. IPTW).
                                  Se None, tutti i pesi sono 1.

    Returns:
        float: Il valore assoluto dell'SMD.
    """
    if w is None:
        w = np.ones_like(t)

    x1, w1 = x[t == 1], w[t == 1]
    x0, w0 = x[t == 0], w[t == 0]

    if len(x1) == 0 or len(x0) == 0:
        return 0.0

    # Calcolo medie e varianze (non pesate per la formula standard dell'SMD)
    m1, m0 = np.mean(x1), np.mean(x0)
    v1, v0 = np.var(x1, ddof=1), np.var(x0, ddof=1)

    # Varianza pooled per il denominatore
    pooled_std = np.sqrt((v1 + v0) / 2)

    if pooled_std < 1e-8:  # Evita divisione per zero
        return 0.0

    return float(np.abs(m1 - m0) / pooled_std)


def compute_pairwise_distances(embeddings: np.ndarray, treatments: np.ndarray, max_pairs: int = 100_000) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcola le distanze euclidee a coppie.

    Distingue tra:
    - Intra-gruppo Trattati (T-T)
    - Intra-gruppo Controlli (C-C)
    - Inter-gruppo (T-C)

    Per efficienza, campiona un numero massimo di coppie se queste sono troppe.

    Args:
        embeddings (np.ndarray): Matrice degli embedding.
        treatments (np.ndarray): Vettore dei trattamenti.
        max_pairs (int): Numero massimo di coppie da campionare per categoria.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple contenente i tre array
        di distanze (dists_tt, dists_cc, dists_tc).
    """
    idx_t = np.where(treatments == 1)[0]
    idx_c = np.where(treatments == 0)[0]
    z_t = embeddings[idx_t]
    z_c = embeddings[idx_c]

    # Campiona coppie per distanze intra-gruppo
    def sample_pairs(n_points, max_p):
        pairs = np.array(list(combinations(range(n_points), 2)))
        if len(pairs) > max_p:
            idxs = np.random.choice(len(pairs), max_p, replace=False)
            return pairs[idxs]
        return pairs

    pairs_t = sample_pairs(len(z_t), max_pairs)
    pairs_c = sample_pairs(len(z_c), max_pairs)

    dists_tt = np.linalg.norm(z_t[pairs_t[:, 0]] - z_t[pairs_t[:, 1]], axis=1) if len(pairs_t) > 0 else np.array([])
    dists_cc = np.linalg.norm(z_c[pairs_c[:, 0]] - z_c[pairs_c[:, 1]], axis=1) if len(pairs_c) > 0 else np.array([])

    # Calcola e campiona distanze inter-gruppo
    if len(z_t) > 0 and len(z_c) > 0:
        dists_tc = cdist(z_t, z_c).flatten()
        if len(dists_tc) > max_pairs:
            dists_tc = np.random.choice(dists_tc, max_pairs, replace=False)
    else:
        dists_tc = np.array([])

    return dists_tt, dists_cc, dists_tc


# =============================================================================
# 3. CLASSE PRINCIPALE PER LA VALUTAZIONE
# =============================================================================

class SiameseEvaluator:
    """
    Classe per orchestrare la valutazione di un modello SiameseBCAUSS.
    """

    def __init__(self, trial_id: int, rep_index: int, weights_dir: str, save_dir: str):
        """
        Inizializza l'evaluator con i parametri della valutazione.

        Args:
            trial_id (int): ID dell'esperimento (trial).
            rep_index (int): Indice della replica del dataset (0-based).
            weights_dir (str): Directory contenente i pesi del modello.
            save_dir (str): Directory dove salvare output (immagini, CSV).
        """
        self.trial_id = trial_id
        self.rep_index = rep_index
        self.weights_dir = Path(weights_dir)
        self.save_dir = Path(save_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Placeholder per dati e risultati
        self.X_rep, self.T_rep = None, None
        self.model = None
        self.embeddings = None
        self.results = {}

        # Crea la cartella di output se non esiste
        self.save_dir.mkdir(parents=True, exist_ok=True)
        print(f"--- Inizializzazione Valutazione per Trial {self.trial_id}, Replica {self.rep_index + 1} ---")
        print(f"Output salvati in: {self.save_dir.resolve()}")

    def _load_data_and_model(self):
        """Carica il dataset IHDP e il modello Siamese con i pesi specificati."""
        print("1. Caricamento dati e modello...")

        # Carica la replica specifica del dataset IHDP
        loader = CFLoader.get_loader('IHDP')
        (X_tr_all, T_tr_all, _, _, _, _, _, _, _, _, _, _) = loader.load()
        self.X_rep = X_tr_all[:, :, self.rep_index].astype(np.float32)
        self.T_rep = T_tr_all[:, self.rep_index].astype(np.int64)
        _, input_dim = self.X_rep.shape

        # Ricostruisci l'architettura del modello
        base_model = BCAUSS(input_dim=input_dim)
        self.model = SiameseBCAUSS(
            base_model=base_model,
            # I parametri seguenti sono fittizi, non influenzano l'inferenza
            # ma sono richiesti dal costruttore della classe.
            ds_class=None, margin=0.75, lambda_ctr=1.0, lr=1e-4,
            batch_size=32, epochs=1, val_split=0.2,
            update_ite_freq=1, warmup_epochs_base=0
        )
        self.model.to(self.device)

        # Carica i pesi addestrati
        weights_path = self.weights_dir / f"weights_trial_{self.trial_id}_rep_{self.rep_index + 1}.pth"
        if not weights_path.is_file():
            raise FileNotFoundError(f"File dei pesi non trovato: {weights_path}")
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()
        print("   ... Dati e modello caricati con successo.")

    def _extract_embeddings(self):
        """Estrae gli embedding dal modello caricato."""
        print("2. Estrazione degli embedding...")
        X_tensor = torch.from_numpy(self.X_rep).to(self.device)
        with torch.no_grad():
            # La funzione `mu_and_embedding` potrebbe restituire una tupla
            out = self.model.base.mu_and_embedding(X_tensor)
            z_tensor = out[-1] if isinstance(out, (tuple, list)) else out
        self.embeddings = z_tensor.cpu().numpy()

        # Salva gli embedding per analisi future
        np.save(self.save_dir / f"embeddings_trial{self.trial_id}_rep{self.rep_index + 1}.npy", self.embeddings)
        np.save(self.save_dir / f"treatments_trial{self.trial_id}_rep{self.rep_index + 1}.npy", self.T_rep)
        print(f"   ... Embedding estratti (shape: {self.embeddings.shape}) e salvati.")

    def _calculate_metrics(self):
        """Calcola tutte le metriche di valutazione e le salva nel dizionario `self.results`."""
        print("3. Calcolo delle metriche di valutazione...")

        # a) Propensity Score (PS) analysis
        ps_model = LogisticRegression(solver='liblinear', max_iter=1000).fit(self.X_rep, self.T_rep)
        ps_pred = ps_model.predict_proba(self.X_rep)[:, 1]
        ps_t = ps_pred[self.T_rep == 1]
        ps_c = ps_pred[self.T_rep == 0]

        self.results['mean_ps_treated'] = ps_t.mean() if len(ps_t) > 0 else np.nan
        self.results['mean_ps_control'] = ps_c.mean() if len(ps_c) > 0 else np.nan
        self.results['wasserstein_ps'] = wasserstein_distance(ps_t, ps_c) if (
                    len(ps_t) > 0 and len(ps_c) > 0) else np.nan

        # b) MMD su embedding
        self.results['mmd_linear'] = compute_linear_mmd_np(self.embeddings, self.T_rep)
        self.results['mmd_rbf'] = compute_rbf_mmd_np(self.embeddings, self.T_rep, sigma=1.0)
        self.results['mmd_poly'] = compute_poly_mmd_np(self.embeddings, self.T_rep, degree=2, c=1.0)

        # c) Silhouette Score
        self.results['silhouette_score'] = silhouette_score(self.embeddings, self.T_rep)

        # d) Accuratezza predizione T da embedding
        clf_emb = LogisticRegressionCV(cv=5, max_iter=1000, solver='liblinear').fit(self.embeddings, self.T_rep)
        self.results['acc_pred_T_from_embedding'] = clf_emb.score(self.embeddings, self.T_rep)

        # e) Standardized Mean Difference (SMD)
        smd_unweighted = [compute_smd(self.X_rep[:, j], self.T_rep) for j in range(self.X_rep.shape[1])]
        self.results['max_smd_unweighted'] = np.max(smd_unweighted)

        # Calcola pesi IPTW per SMD pesato
        epsilon = 1e-3  # Troncamento per stabilità
        ps_trimmed = np.clip(ps_pred, epsilon, 1 - epsilon)
        weights = self.T_rep / ps_trimmed + (1 - self.T_rep) / (1 - ps_trimmed)
        smd_weighted = [compute_smd(self.X_rep[:, j], self.T_rep, w=weights) for j in range(self.X_rep.shape[1])]
        self.results['max_smd_weighted'] = np.max(smd_weighted)

        print("   ... Metriche calcolate:")
        for key, val in self.results.items():
            print(f"      - {key}: {val:.4f}")

        # Salva dati per il Love Plot
        self.smd_data = {'unweighted': smd_unweighted, 'weighted': smd_weighted}

    def _generate_visualizations(self):
        """Genera e salva tutti i grafici per l'analisi qualitativa."""
        print("4. Generazione delle visualizzazioni...")

        # Impostazioni comuni per i plot
        common_title_suffix = f"(Trial {self.trial_id}, Rep {self.rep_index + 1})"
        save_suffix = f"_trial{self.trial_id}_rep{self.rep_index + 1}.png"

        # a) PCA Plot
        pca = PCA(n_components=2)
        Z_pca = pca.fit_transform(self.embeddings)
        plt.figure(figsize=(8, 7))
        sns.scatterplot(x=Z_pca[:, 0], y=Z_pca[:, 1], hue=self.T_rep, palette=['tab:orange', 'tab:blue'], alpha=0.7)
        plt.title(f"PCA degli Embedding {common_title_suffix}")
        plt.xlabel("Componente Principale 1")
        plt.ylabel("Componente Principale 2")
        plt.legend(title='Gruppo', labels=['Controllo', 'Trattato'])
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(self.save_dir / f"PCA_plot{save_suffix}", dpi=300, bbox_inches='tight')
        plt.close()

        # b) t-SNE Plot
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
        Z_tsne = tsne.fit_transform(self.embeddings)
        plt.figure(figsize=(8, 7))
        sns.scatterplot(x=Z_tsne[:, 0], y=Z_tsne[:, 1], hue=self.T_rep, palette=['tab:orange', 'tab:blue'], alpha=0.7)
        plt.title(f"t-SNE degli Embedding {common_title_suffix}")
        plt.xlabel("Dimensione t-SNE 1")
        plt.ylabel("Dimensione t-SNE 2")
        plt.legend(title='Gruppo', labels=['Controllo', 'Trattato'])
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(self.save_dir / f"tSNE_plot{save_suffix}", dpi=300, bbox_inches='tight')
        plt.close()

        # c) Istogramma delle distanze
        dists_tt, dists_cc, dists_tc = compute_pairwise_distances(self.embeddings, self.T_rep)
        plt.figure(figsize=(10, 6))
        sns.histplot(dists_tc, bins=50, color='tab:green', label='Trattato-Controllo (Inter)', stat='density',
                     alpha=0.6)
        sns.histplot(dists_tt, bins=50, color='tab:blue', label='Trattato-Trattato (Intra)', stat='density', alpha=0.6)
        sns.histplot(dists_cc, bins=50, color='tab:orange', label='Controllo-Controllo (Intra)', stat='density',
                     alpha=0.6)
        plt.title(f"Distribuzione delle Distanze Euclidee {common_title_suffix}")
        plt.xlabel("Distanza Euclidea")
        plt.ylabel("Densità")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(self.save_dir / f"Distances_hist{save_suffix}", dpi=300, bbox_inches='tight')
        plt.close()

        # d) Love Plot
        plt.figure(figsize=(10, 8))
        y_ticks = range(self.X_rep.shape[1])
        plt.scatter(self.smd_data['unweighted'], y_ticks, color='tab:red', alpha=0.7, label='SMD Non Pesato')
        plt.scatter(self.smd_data['weighted'], y_ticks, color='tab:blue', alpha=0.7, label='SMD Pesato (IPTW)')
        plt.axvline(x=0.1, color='green', linestyle='--', label='Soglia di bilanciamento (0.1)')
        plt.title(f"Love Plot: Bilanciamento Covariate {common_title_suffix}")
        plt.xlabel("Standardized Mean Difference (SMD)")
        plt.ylabel("Indice Covariata")
        plt.yticks(y_ticks, [f'Cov_{i}' for i in y_ticks])
        plt.legend()
        plt.grid(True, axis='x', linestyle='--', alpha=0.6)
        plt.savefig(self.save_dir / f"Love_plot{save_suffix}", dpi=300, bbox_inches='tight')
        plt.close()

        print("   ... Visualizzazioni salvate.")

    def _save_results_to_csv(self):
        """Salva il dizionario dei risultati in un file CSV."""
        print("5. Salvataggio dei risultati su CSV...")

        # Aggiungi info identificative al dizionario dei risultati
        summary = {
            'trial_id': self.trial_id,
            'rep_index': self.rep_index + 1,
            **self.results
        }

        results_df = pd.DataFrame([summary])
        csv_path = self.save_dir / "evaluation_summary.csv"

        # Scrivi con header se il file non esiste, altrimenti aggiungi in coda
        file_exists = csv_path.is_file()
        results_df.to_csv(csv_path, mode='a', header=not file_exists, index=False)
        print(f"   ... Risultati aggiunti a {csv_path.resolve()}")

    def run(self):
        """
        Esegue l'intero pipeline di valutazione:
        caricamento -> estrazione -> calcolo metriche -> visualizzazione -> salvataggio.
        """
        try:
            self._load_data_and_model()
            self._extract_embeddings()
            self._calculate_metrics()
            self._generate_visualizations()
            self._save_results_to_csv()
            print(f"--- Valutazione per Trial {self.trial_id}, Replica {self.rep_index + 1} COMPLETATA ---\n")
        except Exception as e:
            print(f"\nERRORE durante la valutazione per Trial {self.trial_id}, Replica {self.rep_index + 1}:")
            print(f"Dettagli: {e}")
            # Rilancia l'eccezione per un debugging più approfondito se necessario
            # raise e


# =============================================================================
# 4. BLOCCO DI ESECUZIONE
# =============================================================================
if __name__ == "__main__":
    # --- CONFIGURAZIONE DELL'ESECUZIONE ---
    # Modifica questi parametri per eseguire la valutazione desiderata
    CONFIG = {
        "TRIAL_ID": 4,
        "REP_INDEX": 999,  # Indice 0-based, quindi 999 -> replica 1000
        "WEIGHTS_DIR": "siamese_bcuass/saved_weights",
        "SAVE_DIR": "evaluation_results_refactored"
    }
    # -----------------------------------------

    # Istanzia e avvia il processo di valutazione
    evaluator = SiameseEvaluator(
        trial_id=CONFIG["TRIAL_ID"],
        rep_index=CONFIG["REP_INDEX"],
        weights_dir=CONFIG["WEIGHTS_DIR"],
        save_dir=CONFIG["SAVE_DIR"]
    )
    evaluator.run()

    # Esempio: eseguire la valutazione su più repliche
    # print("\n--- ESECUZIONE SU REPLICHE MULTIPLE ---")
    # for rep_idx in range(5): # Esegue per le prime 5 repliche
    #     evaluator_multi = SiameseEvaluator(
    #         trial_id=CONFIG["TRIAL_ID"],
    #         rep_index=rep_idx,
    #         weights_dir=CONFIG["WEIGHTS_DIR"],
    #         save_dir=CONFIG["SAVE_DIR"]
    #     )
    #     evaluator_multi.run()

