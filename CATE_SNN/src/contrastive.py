import logging
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler


def compute_tau_threshold(mu0_hat, mu1_hat, perc=20, sample=100_000):
    """
    Compute the dynamic threshold at the specified percentile of absolute differences
    between two ITE estimate distributions.
    """
    tau_vals = mu1_hat - mu0_hat
    actual_sample_size = min(sample, len(tau_vals))
    if actual_sample_size == 0:
        logging.warning("Cannot compute tau threshold with 0 samples.")
        return 0.1
    idx1 = np.random.randint(0, len(tau_vals), size=actual_sample_size)
    idx2 = np.random.randint(0, len(tau_vals), size=actual_sample_size)
    diffs = np.abs(tau_vals[idx1] - tau_vals[idx2])
    return float(np.percentile(diffs, perc))


def make_pairs_from_hat(X, T, Y, mu0_hat, mu1_hat, thr, n_pairs, seed=None):
    if seed is not None:
        np.random.seed(seed)

    ite_hat = mu1_hat - mu0_hat
    N = ite_hat.shape[0]
    if N == 0:
        empty_shape = (0,) + X.shape[1:]
        return (np.zeros(empty_shape, dtype=X.dtype), np.zeros((0,) + Y.shape[1:], dtype=Y.dtype),
                np.zeros((0,) + T.shape[1:], dtype=T.dtype), np.zeros(empty_shape, dtype=X.dtype),
                np.zeros((0,) + Y.shape[1:], dtype=Y.dtype), np.zeros((0,) + T.shape[1:], dtype=T.dtype),
                np.array([], dtype=np.int64))

    n_pairs = min(n_pairs, N)
    half = n_pairs // 2
    sim_pairs, diss_pairs = [], []
    used = set()

    def add_pair(i, j, label, container):
        if i == j: return
        key = (min(i, j), max(i, j))
        if key in used: return
        used.add(key)
        container.append((i, j, label))

    # Similar pairs
    attempts = 0
    while len(sim_pairs) < half and attempts < n_pairs * 5:
        i = np.random.randint(N)
        diffs = np.abs(ite_hat - ite_hat[i])
        cand = np.where(diffs < thr)[0]
        cand = cand[cand != i]
        if cand.size > 0:
            j = np.random.choice(cand)
            add_pair(i, j, 1, sim_pairs)
        attempts += 1

    # Dissimilar pairs (Hard negatives fix)
    attempts = 0
    while len(diss_pairs) < n_pairs - half and attempts < n_pairs * 5:
        i = np.random.randint(N)
        diffs = np.abs(ite_hat - ite_hat[i])
        cand = np.where(diffs >= thr)[0]
        cand = cand[cand != i]
        if cand.size > 0:
            # Scegliamo un candidato a caso tra i dissimili invece che il più vicino con argmin
            j = np.random.choice(cand)
            add_pair(i, j, 0, diss_pairs)
        attempts += 1

    pairs = sim_pairs + diss_pairs
    if not pairs:
        empty_shape = (0,) + X.shape[1:]
        return (np.zeros(empty_shape, dtype=X.dtype), np.zeros((0,) + Y.shape[1:], dtype=Y.dtype),
                np.zeros((0,) + T.shape[1:], dtype=T.dtype), np.zeros(empty_shape, dtype=X.dtype),
                np.zeros((0,) + Y.shape[1:], dtype=Y.dtype), np.zeros((0,) + T.shape[1:], dtype=T.dtype),
                np.array([], dtype=np.int64))

    np.random.shuffle(pairs)
    idx_a, idx_b, labels = zip(*pairs)
    return (X[np.array(idx_a)], Y[np.array(idx_a)], T[np.array(idx_a)],
            X[np.array(idx_b)], Y[np.array(idx_b)], T[np.array(idx_b)],
            np.array(labels, dtype=np.int64))


def make_random_pairs(X, T, Y, n_pairs, seed=None):
    if seed is not None:
        np.random.seed(seed)

    N = X.shape[0]
    n_pairs = min(n_pairs, N)
    if N < 2:
        return make_pairs_from_hat(X, T, Y, np.zeros(N), np.zeros(N), 0, 0)  # Fallback empty

    idx_a = np.random.randint(0, N, size=n_pairs)
    idx_b = np.random.randint(0, N, size=n_pairs)

    same = idx_a == idx_b
    while np.any(same):
        idx_b[same] = np.random.randint(0, N, size=np.sum(same))
        same = idx_a == idx_b

    labels = np.random.randint(0, 2, size=n_pairs)
    return (X[idx_a], Y[idx_a], T[idx_a],
            X[idx_b], Y[idx_b], T[idx_b],
            labels.astype(np.int64))


def make_pairs_from_covariates(X, T, Y, n_pairs, seed=None, quantile=0.2):
    if seed is not None:
        np.random.seed(seed)

    N = X.shape[0]
    n_pairs = min(n_pairs, N)
    if N < 2:
        return make_pairs_from_hat(X, T, Y, np.zeros(N), np.zeros(N), 0, 0)

    # Euclidean distance matrix
    dmat = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)

    sim_pairs, diss_pairs = [], []
    used = set()
    half = n_pairs // 2

    # Avoid zeros (self-distance)
    positive_dmat = dmat[dmat > 0]
    if len(positive_dmat) == 0:
        return make_random_pairs(X, T, Y, n_pairs, seed)

    sim_thr = np.quantile(positive_dmat, quantile)
    dis_thr = np.quantile(positive_dmat, 1 - quantile)

    def add_pair(i, j, label, container):
        if i == j: return
        key = (min(i, j), max(i, j))
        if key in used: return
        used.add(key)
        container.append((i, j, label))

    attempts = 0
    while len(sim_pairs) < half and attempts < 10 * n_pairs:
        i = np.random.randint(N)
        cand = np.where((dmat[i] <= sim_thr) & (np.arange(N) != i))[0]
        if cand.size > 0:
            j = np.random.choice(cand)
            add_pair(i, j, 1, sim_pairs)
        attempts += 1

    attempts = 0
    while len(diss_pairs) < n_pairs - half and attempts < 10 * n_pairs:
        i = np.random.randint(N)
        cand = np.where(dmat[i] >= dis_thr)[0]
        cand = cand[cand != i]
        if cand.size > 0:
            j = np.random.choice(cand)
            add_pair(i, j, 0, diss_pairs)
        attempts += 1

    pairs = sim_pairs + diss_pairs
    if not pairs:
        return make_random_pairs(X, T, Y, n_pairs, seed)

    np.random.shuffle(pairs)
    idx_a, idx_b, labels = zip(*pairs)
    return (X[np.array(idx_a)], Y[np.array(idx_a)], T[np.array(idx_a)],
            X[np.array(idx_b)], Y[np.array(idx_b)], T[np.array(idx_b)],
            np.array(labels, dtype=np.int64))


def first_item(batch):
    return batch[0]


class ContrastiveCausalDS(Dataset):
    """Unified dataset that supports multiple pairing strategies for ablation."""

    def __init__(
            self, X_all, T_all, Y_all,
            mu0_hat=None, mu1_hat=None,
            bs=256, strategy="dynamic_ite",
            perc=20, sample_for_thr_calc=100_000, seed=42
    ):
        self.X_all = X_all
        self.T_all = T_all
        self.Y_all = Y_all
        self.bs = bs
        self.strategy = strategy
        self.perc = perc
        self.sample_for_thr_calc = sample_for_thr_calc
        self.seed = seed
        self.epoch = 0  # Aggiunto contatore dell'epoca

        if mu0_hat is not None and mu1_hat is not None and mu0_hat.size and mu1_hat.size:
            self.current_mu0_hat = mu0_hat
            self.current_mu1_hat = mu1_hat
        else:
            self.current_mu0_hat = np.zeros(X_all.shape[0])
            self.current_mu1_hat = np.zeros(X_all.shape[0])

        if self.strategy in ["dynamic_ite", "static_ite"]:
            self.update_threshold()
            if self.strategy == "static_ite":
                # Cache list di batch statici invece di un solo set di copie
                self.static_batches = [
                    self._make_ite_pairs(seed=self.seed + k)
                    for k in range(len(self))
                ]

    def update_threshold(self):
        self.thr = compute_tau_threshold(
            self.current_mu0_hat, self.current_mu1_hat,
            self.perc, self.sample_for_thr_calc
        )

    def update_ite_estimates(self, mu0_hat, mu1_hat):
        self.current_mu0_hat = mu0_hat
        self.current_mu1_hat = mu1_hat
        self.epoch += 1  # Incrementa per rinnovare il seed
        if self.strategy == "dynamic_ite":
            self.update_threshold()
        # static_ite DOES NOT regenerate pairs.

    def __len__(self):
        return int(np.ceil(self.X_all.shape[0] / self.bs))

    def __getitem__(self, idx):
        # Utilizziamo l'epoca per diversificare il seed ai successivi ricaricamenti dinamici
        dynamic_seed = self.epoch * 100000 + idx

        if self.strategy == "dynamic_ite":
            x1, y1, t1, x2, y2, t2, lab = self._make_ite_pairs(seed=dynamic_seed)
        elif self.strategy == "static_ite":
            # Scorre i batch caccheati per mantenere diversificazione inter-epoca
            x1, y1, t1, x2, y2, t2, lab = self.static_batches[idx % len(self.static_batches)]
        elif self.strategy == "random":
            x1, y1, t1, x2, y2, t2, lab = self._make_random_pairs(seed=dynamic_seed)
        elif self.strategy == "covariate":
            x1, y1, t1, x2, y2, t2, lab = self._make_covariate_pairs(seed=dynamic_seed)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        return (
            torch.tensor(x1, dtype=torch.float32), torch.tensor(y1, dtype=torch.float32),
            torch.tensor(t1, dtype=torch.float32),
            torch.tensor(x2, dtype=torch.float32), torch.tensor(y2, dtype=torch.float32),
            torch.tensor(t2, dtype=torch.float32),
            torch.tensor(lab, dtype=torch.long)
        )

    def _make_ite_pairs(self, seed=None):
        return make_pairs_from_hat(self.X_all, self.T_all, self.Y_all, self.current_mu0_hat, self.current_mu1_hat,
                                   self.thr, self.bs, seed=seed)

    def _make_random_pairs(self, seed=None):
        return make_random_pairs(self.X_all, self.T_all, self.Y_all, self.bs, seed=seed)

    def _make_covariate_pairs(self, seed=None):
        return make_pairs_from_covariates(self.X_all, self.T_all, self.Y_all, self.bs, seed=seed)