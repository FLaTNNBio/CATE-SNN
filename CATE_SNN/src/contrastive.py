#!/usr/bin/env python3
import logging
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from torch.amp import GradScaler  # PyTorch ≥ 2.1
except ImportError:
    from torch.cuda.amp import GradScaler  # type: ignore


def compute_tau_threshold(mu0_hat, mu1_hat, perc=20, sample=100_000):
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
    return float(np.percentile(diffs, perc))


def make_pairs_from_hat(X, T, Y, mu0_hat, mu1_hat, thr, n_pairs, seed=None):
    """
    Generate a balanced set of similar and dissimilar pairs based on ITE estimates.
    Returns:
      x1,y1,t1, x2,y2,t2, labels  (all numpy arrays)
    """
    if seed is not None:
        np.random.seed(seed)

    ite_hat = mu1_hat - mu0_hat
    N = ite_hat.shape[0]
    if N == 0:
        logging.warning("Empty dataset for pairing.")
        empty_shape = (0,) + X.shape[1:]
        return (np.zeros(empty_shape, dtype=X.dtype),
                np.zeros((0,) + Y.shape[1:], dtype=Y.dtype),
                np.zeros((0,) + T.shape[1:], dtype=T.dtype),
                np.zeros(empty_shape, dtype=X.dtype),
                np.zeros((0,) + Y.shape[1:], dtype=Y.dtype),
                np.zeros((0,) + T.shape[1:], dtype=T.dtype),
                np.array([], dtype=np.int64))

    # cap n_pairs to available samples
    n_pairs = min(n_pairs, N)
    half = n_pairs // 2
    sim_pairs, diss_pairs = [], []
    used = set()

    def add_pair(i, j, label, container):
        if i == j:
            return
        key = (min(i, j), max(i, j))
        if key in used:
            return
        used.add(key)
        container.append((i, j, label))

    # similar pairs
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

    # dissimilar pairs (hard negatives)
    attempts = 0
    while len(diss_pairs) < n_pairs - half and attempts < n_pairs * 5:
        i = np.random.randint(N)
        diffs = np.abs(ite_hat - ite_hat[i])
        cand = np.where(diffs >= thr)[0]
        cand = cand[cand != i]
        if cand.size > 0:
            j = cand[np.argmin(np.abs(ite_hat[cand] - ite_hat[i]))]
            add_pair(i, j, 0, diss_pairs)
        attempts += 1

    pairs = sim_pairs + diss_pairs
    if not pairs:
        logging.warning("No pairs generated.")
        empty_shape = (0,) + X.shape[1:]
        return (np.zeros(empty_shape, dtype=X.dtype),
                np.zeros((0,) + Y.shape[1:], dtype=Y.dtype),
                np.zeros((0,) + T.shape[1:], dtype=T.dtype),
                np.zeros(empty_shape, dtype=X.dtype),
                np.zeros((0,) + Y.shape[1:], dtype=Y.dtype),
                np.zeros((0,) + T.shape[1:], dtype=T.dtype),
                np.array([], dtype=np.int64))

    np.random.shuffle(pairs)
    idx_a, idx_b, labels = zip(*pairs)
    idx_a = np.array(idx_a, dtype=int)
    idx_b = np.array(idx_b, dtype=int)
    labels = np.array(labels, dtype=np.int64)

    return (
        X[idx_a], Y[idx_a], T[idx_a],
        X[idx_b], Y[idx_b], T[idx_b],
        labels
    )


def first_item(batch):
    return batch[0]


class DynamicContrastiveCausalDS(Dataset):
    """Dataset that generates contrastive pairs each call, robust to small N."""
    def __init__(
        self,
        X_all,
        T_all,
        Y_all,
        mu0_hat,
        mu1_hat,
        bs=256,
        perc=20,
        sample_for_thr_calc=100_000
    ):
        self.X_all = X_all
        self.T_all = T_all
        self.Y_all = Y_all
        self.bs = bs
        self.perc = perc
        self.sample_for_thr_calc = sample_for_thr_calc

        if mu0_hat is not None and mu1_hat is not None and mu0_hat.size and mu1_hat.size:
            self.current_mu0_hat = mu0_hat
            self.current_mu1_hat = mu1_hat
        else:
            logging.error("Empty initial ITE estimates, using zeros.")
            self.current_mu0_hat = np.zeros(X_all.shape[0])
            self.current_mu1_hat = np.zeros(X_all.shape[0])
        self.update_threshold()

    def update_threshold(self):
        self.thr = compute_tau_threshold(
            self.current_mu0_hat,
            self.current_mu1_hat,
            self.perc,
            self.sample_for_thr_calc
        )
        logging.info(f"Updated dynamic threshold: {self.thr:.4f}")

    def update_ite_estimates(self, mu0_hat, mu1_hat):
        if mu0_hat is not None and mu1_hat is not None and mu0_hat.size and mu1_hat.size:
            self.current_mu0_hat = mu0_hat
            self.current_mu1_hat = mu1_hat
            self.update_threshold()
        else:
            logging.warning(f"Empty ITE update, threshold remains {self.thr}.")

    def __len__(self):
        return int(np.ceil(self.X_all.shape[0] / self.bs))

    def __getitem__(self, idx):
        # Generate pairs
        x1, y1, t1, x2, y2, t2, lab = make_pairs_from_hat(
            self.X_all,
            self.T_all,
            self.Y_all,
            self.current_mu0_hat,
            self.current_mu1_hat,
            self.thr,
            self.bs,
            seed=idx
        )
        # to torch
        return (
            torch.tensor(x1, dtype=torch.float32),
            torch.tensor(y1, dtype=torch.float32),
            torch.tensor(t1, dtype=torch.float32),
            torch.tensor(x2, dtype=torch.float32),
            torch.tensor(y2, dtype=torch.float32),
            torch.tensor(t2, dtype=torch.float32),
            torch.tensor(lab, dtype=torch.long)
        )
