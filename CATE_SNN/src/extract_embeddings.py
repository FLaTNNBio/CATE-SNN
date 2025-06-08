# estrai_embeddings_e_mmd.py

import numpy as np
import torch
from pathlib import Path

# 1) Importa le tue classi (assicurati che siamese.py e bcauss.py siano nel path corretto)
from src.models.bcauss import BCAUSS
from src.siamese_bcuass.siamese import SiameseBCAUSS
from src.data_loader import DataLoader as CFLoader

def compute_linear_mmd_np(embeddings: np.ndarray, treatments: np.ndarray) -> float:
    treated = embeddings[treatments == 1]
    control = embeddings[treatments == 0]
    if len(treated) == 0 or len(control) == 0:
        raise ValueError("Non ci sono esempi né di trattati né di controlli.")
    mean_t = treated.mean(axis=0)
    mean_c = control.mean(axis=0)
    return float(np.sum((mean_t - mean_c) ** 2))

def extract_embeddings_and_mmd(trial_id: int, rep_index: int, weights_dir: str):
    # A) Carica IHDP
    loader = CFLoader.get_loader('IHDP')
    (X_tr_all, T_tr_all, _,
     _, _, _,
     _, _, _, _, _, _) = loader.load()
    # X_tr_all: (n_samples, n_covariate, n_reps)
    # T_tr_all: (n_samples, n_reps)

    # B) Estrai la replica desiderata (rep_index è 0-based)
    Xrep = X_tr_all[:, :, rep_index].astype(np.float32)  # (n_samples, input_dim)
    Trep = T_tr_all[:, rep_index].astype(np.int64)       # (n_samples,)

    n_samples, input_dim = Xrep.shape

    # C) Ricostruisci il base_model e il SiameseBCAUSS con gli stessi iperparametri usati in train
    base_model = BCAUSS(input_dim=input_dim)
    siamese = SiameseBCAUSS(
        base_model=base_model,
        ds_class=None,          # serve solo per compatibilità, non useremo il dataset
        margin=1.0,             # lo stesso margin che hai usato in training
        lambda_ctr=1.0,         # lo stesso lambda_ctr che hai usato
        lr=1e-4,                # serve solo per inizializzare internamente, non useremo ottimizzatore
        batch_size=128,         # idem
        epochs=100,             # idem
        val_split=0.2,          # idem
        update_ite_freq=1,      # idem
        warmup_epochs_base=0    # idem
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    siamese.to(device)

    # D) Carica i pesi specifici per trial=4, rep_index=999 (in filename rep=rep_index+1=1000)
    weights_path = Path(weights_dir) / f"weights_trial_{trial_id}_rep_{rep_index+1}.pth"
    if not weights_path.is_file():
        raise FileNotFoundError(f"Non ho trovato '{weights_path}'")
    siamese.load_state_dict(torch.load(weights_path, map_location=device))
    siamese.eval()

    # E) Ottieni gli embedding Φ(x) dal base_model
    X_tensor = torch.from_numpy(Xrep).float().to(device)
    with torch.no_grad():
        _, z_tensor = siamese.base.mu_and_embedding(X_tensor)
    embeddings = z_tensor.cpu().numpy()  # (n_samples, embed_dim)

    # F) Calcola l'MMD lineare
    mmd_val = compute_linear_mmd_np(embeddings, Trep)
    print(f"Trial {trial_id}, replica {rep_index+1}: MMD lineare = {mmd_val:.6f}")

    # (Opzionale) Salva embeddings in un file .npy
    np.save(f"embeddings_trial{trial_id}_rep{rep_index+1}.npy", embeddings)
    print(f"Embeddings salvati in 'embeddings_trial{trial_id}_rep{rep_index+1}.npy'")

    return embeddings, mmd_val

if __name__ == "__main__":
    # Qui impostiamo trial_id=4, rep_index=999, weights_dir come da te indicato
    extract_embeddings_and_mmd(
        trial_id=4,
        rep_index=999,
        weights_dir="siamese_bcuass/saved_weights"
    )
    rep_index = 999

    # 2) Carica IHDP
    loader = CFLoader.get_loader('IHDP')
    (X_tr_all, T_tr_all, _,  # Carica anche altri output, ma ci serve solo T_tr_all
     _, _, _,  # (ignoriamo YF_tr_all, m0_tr_all, m1_tr_all)
     _, _, _, _, _, _) = loader.load()

    # 3) Estrai il vettore dei trattamenti per quella replica
    #    T_tr_all ha shape (n_samples, n_reps)
    Trep = T_tr_all[:, rep_index].astype(np.int64)  # forma (n_samples,)

    # 4) Salva su disco in un file .npy
    out_path = Path("treatment_labels.npy")
    np.save(out_path, Trep)
    print(f"Salvato vettore T di lunghezza {len(Trep)} in '{out_path}'.")
