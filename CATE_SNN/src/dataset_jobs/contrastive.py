import logging
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from torch.amp import GradScaler  # PyTorch ≥ 2.1
except ImportError:
    from torch.cuda.amp import GradScaler  # type: ignore


class DynamicContrastiveCausalDS(torch.utils.data.Dataset):
    def __init__(self, X, T, Y, base_model, n_pairs=10000,
                 perc=60, sample=100_000,
                 min_thr=0.1, max_thr=0.5, smooth=0.7):
        self.X = X
        self.T = T
        self.Y = Y
        self.model = base_model
        self.n_pairs = n_pairs
        self.perc = perc
        self.sample = sample
        self.min_thr = min_thr
        self.max_thr = max_thr
        self.smooth = smooth
        self.prev_thr = None

        # initial threshold
        with torch.no_grad():
            mu0, mu1, _ = self.model.mu_and_embedding(self.X)
        self.thr = self.compute_tau_threshold(mu0, mu1)
        self._build_pairs()

    def compute_tau_threshold(self, mu0_hat, mu1_hat):
        tau_vals = (mu1_hat - mu0_hat).cpu().numpy().ravel()
        n = min(self.sample, len(tau_vals))
        if n == 0:
            return self.min_thr
        idx1 = np.random.randint(0, len(tau_vals), size=n)
        idx2 = np.random.randint(0, len(tau_vals), size=n)
        diffs = np.abs(tau_vals[idx1] - tau_vals[idx2])
        raw_thr = float(np.percentile(diffs, self.perc))
        clamped = max(min(raw_thr, self.max_thr), self.min_thr)
        if self.prev_thr is None:
            thr = clamped
        else:
            thr = self.smooth * self.prev_thr + (1 - self.smooth) * clamped
        self.prev_thr = thr
        return thr

    def _build_pairs(self):
        # safe pairs generation
        with torch.no_grad():  # <--- SOLUZIONE
            mu0, mu1, _ = self.model.mu_and_embedding(self.X)
        pairs, labels = make_pairs_from_hat(
            self.X, self.T, self.Y,
            mu0.cpu().numpy(), mu1.cpu().numpy(),
            self.thr, self.n_pairs
        )
        if len(pairs) == 0:
            # fallback single pair
            i, j = 0, 1 if len(mu0) > 1 else 0
            pairs = [(self.X[i], self.X[j])]
            labels = [int(abs((mu1[i] - mu0[i]) - (mu1[j] - mu0[j])) < self.thr)]
            logging.warning("No contrastive pairs generated: using fallback pair.")
        self.pairs = pairs
        self.labels = labels

    def update_threshold(self):
        with torch.no_grad():  # <--- SOLUZIONE
            mu0, mu1, _ = self.model.mu_and_embedding(self.X)
        self.thr = self.compute_tau_threshold(mu0, mu1)
        self._build_pairs()
        logging.info(f"Updated dynamic threshold: {self.thr:.4f}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        x1, x2 = self.pairs[idx]
        label = self.labels[idx]
        return x1, x2, label


def compute_tau_threshold(mu0_hat, mu1_hat, perc=20, sample=100_000, min_thr=0.1):
    """
    Compute the dynamic threshold at the specified percentile of absolute differences
    between two ITE estimate distributions.
    """
    # difference between ITE arrays
    tau_vals = mu1_hat - mu0_hat
    actual_sample_size = min(sample, len(tau_vals))
    if actual_sample_size == 0:
        logging.warning("Cannot compute tau threshold with 0 samples.")
        return 0.1
    idx1 = np.random.randint(0, len(tau_vals), size=actual_sample_size)
    idx2 = np.random.randint(0, len(tau_vals), size=actual_sample_size)
    diffs = np.abs(tau_vals[idx1] - tau_vals[idx2])
    # percentile-based threshold
    thr = float(np.percentile(diffs, perc))
    # clamp to a minimum
    return max(thr, min_thr)


def make_pairs_from_hat(X, T, Y, mu0_hat, mu1_hat, thr, n_pairs, seed=None):
    """
    Versione finale e corretta che gestisce correttamente i tipi di dato.
    """
    if seed is not None:
        np.random.seed(seed)

    ite_hat = mu1_hat - mu0_hat
    N = ite_hat.shape[0]
    if N < 2:
        return [], []

    # Calcola tutte le coppie possibili una sola volta
    all_indices_i, all_indices_j = np.triu_indices(N, k=1)
    if len(all_indices_i) == 0:
        return [], []

    ite_diffs = np.abs(ite_hat[all_indices_i] - ite_hat[all_indices_j])

    similar_indices = np.where(ite_diffs < thr)[0]
    dissimilar_indices = np.where(ite_diffs >= thr)[0]

    # --- Inizializza gli array di indici come array NumPy di interi ---
    # QUESTA È LA CORREZIONE CHIAVE
    final_sim_indices = np.array([], dtype=int)
    final_dissim_indices = np.array([], dtype=int)
    half = n_pairs // 2

    if len(similar_indices) > 0:
        n_sim = min(half, len(similar_indices))
        final_sim_indices = np.random.choice(similar_indices, n_sim, replace=False)

    if len(dissimilar_indices) > 0:
        n_dissim = min(n_pairs - len(final_sim_indices), len(dissimilar_indices))
        final_dissim_indices = np.random.choice(dissimilar_indices, n_dissim, replace=False)

    final_indices = np.concatenate([final_sim_indices, final_dissim_indices])
    if len(final_indices) == 0:
        return [], []

    idx_a = all_indices_i[final_indices]
    idx_b = all_indices_j[final_indices]

    pairs = [(X[i], X[j]) for i, j in zip(idx_a, idx_b)]

    labels = np.concatenate([
        np.ones(len(final_sim_indices), dtype=int),
        np.zeros(len(final_dissim_indices), dtype=int)
    ])

    return pairs, labels


def first_item(batch):
    return batch[0]



