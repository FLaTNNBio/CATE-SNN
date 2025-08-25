# -*- coding: utf-8 -*-
"""
Script standalone per estrarre embedding e calcolare MMD da un modello
SiameseBCAUSS pre-addestrato sul dataset IHDP.

Questo script esegue le seguenti operazioni:
1. Carica una specifica replica del dataset IHDP.
2. Ricostruisce l'architettura del modello e carica i pesi salvati.
3. Estrae gli embedding (rappresentazioni latenti) per i dati della replica.
4. Calcola il Maximum Mean Discrepancy (MMD) con kernel lineare.
5. Salva su file sia gli embedding che i vettori di trattamento corrispondenti.
"""

import numpy as np
import torch
from pathlib import Path

# Assicurati che il percorso a 'src' sia corretto o che 'src' sia nel PYTHONPATH
try:
    from src.models.bcauss import BCAUSS
    from src.siamese_bcuass.siamese import SiameseBCAUSS
    from src.data_loader import DataLoader as CFLoader
except ImportError as e:
    print(f"Errore: impossibile importare i moduli del progetto. Assicurati che la cartella 'src' sia accessibile.")
    print(f"Dettagli: {e}")
    exit()


def compute_linear_mmd_np(embeddings: np.ndarray, treatments: np.ndarray) -> float:
    """
    Calcola il Maximum Mean Discrepancy (MMD^2) con kernel lineare.
    """
    treated = embeddings[treatments == 1]
    control = embeddings[treatments == 0]
    if len(treated) == 0 or len(control) == 0:
        print("Attenzione: uno dei due gruppi (trattati o controlli) è vuoto.")
        return np.nan

    mean_t = treated.mean(axis=0)
    mean_c = control.mean(axis=0)
    return float(np.sum((mean_t - mean_c) ** 2))


def extract_and_evaluate(trial_id: int, rep_index: int, weights_dir: str, save_dir: Path):
    """
    Funzione principale che orchestra il caricamento, l'estrazione e la valutazione.
    """
    print(f"--- Avvio estrazione per Trial {trial_id}, Replica {rep_index + 1} ---")

    # Crea la cartella di output se non esiste
    save_dir.mkdir(parents=True, exist_ok=True)

    # A) Carica i dati del dataset IHDP
    print("1. Caricamento dati IHDP...")
    loader = CFLoader.get_loader('IHDP')
    (X_tr_all, T_tr_all, _, _, _, _, _, _, _, _, _, _) = loader.load()

    # B) Estrai la replica desiderata (rep_index è 0-based)
    X_rep = X_tr_all[:, :, rep_index].astype(np.float32)
    T_rep = T_tr_all[:, rep_index].astype(np.int64)
    _, input_dim = X_rep.shape
    print(f"   ... Dati caricati. Shape X: {X_rep.shape}, Shape T: {T_rep.shape}")

    # C) Ricostruisci il modello Siamese con la stessa architettura usata in training
    print("2. Ricostruzione del modello...")
    base_model = BCAUSS(input_dim=input_dim)
    siamese = SiameseBCAUSS(
        base_model=base_model,
        # I parametri seguenti sono fittizi e servono solo per l'inizializzazione
        ds_class=None, margin=1.0, lambda_ctr=1.0, lr=1e-4,
        batch_size=128, epochs=1, val_split=0.2,
        update_ite_freq=1, warmup_epochs_base=0
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    siamese.to(device)

    # D) Carica i pesi del modello pre-addestrato
    weights_path = Path(weights_dir) / f"weights_trial_{trial_id}_rep_{rep_index + 1}.pth"
    if not weights_path.is_file():
        raise FileNotFoundError(f"File dei pesi non trovato: '{weights_path}'")

    siamese.load_state_dict(torch.load(weights_path, map_location=device))
    siamese.eval()
    print(f"   ... Modello e pesi caricati da '{weights_path}'")

    # E) Estrai gli embedding Φ(x)
    print("3. Estrazione degli embedding...")
    X_tensor = torch.from_numpy(X_rep).float().to(device)
    with torch.no_grad():
        # --- FIX APPLICATO QUI ---
        # La funzione può restituire un singolo tensore o una tupla.
        # Questo codice gestisce entrambi i casi in modo robusto, prendendo
        # sempre l'ultimo elemento se è una tupla/lista.
        output = siamese.base.mu_and_embedding(X_tensor)
        if isinstance(output, (tuple, list)):
            z_tensor = output[-1]
        else:
            z_tensor = output

    embeddings = z_tensor.cpu().numpy()
    print(f"   ... Embedding estratti. Shape: {embeddings.shape}")

    # F) Calcola l'MMD lineare
    mmd_val = compute_linear_mmd_np(embeddings, T_rep)
    print(f"4. Calcolo MMD: MMD lineare = {mmd_val:.6f}")

    # G) Salva embedding e trattamenti su file .npy
    embed_path = save_dir / f"embeddings_trial{trial_id}_rep{rep_index + 1}.npy"
    treat_path = save_dir / f"treatments_trial{trial_id}_rep{rep_index + 1}.npy"

    np.save(embed_path, embeddings)
    np.save(treat_path, T_rep)
    print(f"5. Output salvati:")
    print(f"   - Embedding in: '{embed_path}'")
    print(f"   - Trattamenti in: '{treat_path}'")

    print("--- Estrazione completata. ---\n")
    return embeddings, T_rep, mmd_val


if __name__ == "__main__":

    # --- CONFIGURAZIONE ---
    TRIAL_ID = 4
    REP_INDEX = 999  # Corrisponde alla replica 1000 nei nomi dei file
    WEIGHTS_DIR = "../saved_weights"
    SAVE_DIR = Path("../../extraction_results")  # Cartella per salvare gli output
    # --------------------

    try:
        extract_and_evaluate(
            trial_id=TRIAL_ID,
            rep_index=REP_INDEX,
            weights_dir=WEIGHTS_DIR,
            save_dir=SAVE_DIR
        )
    except Exception as e:
        print(f"\nERRORE durante l'esecuzione: {e}")

