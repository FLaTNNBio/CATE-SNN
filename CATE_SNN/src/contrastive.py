import logging
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from torch.amp import GradScaler  # PyTorch ≥ 2.1
except ImportError:
    from torch.cuda.amp import GradScaler  # type: ignore


def compute_tau_threshold(mu0_or_mu0_hat, mu1_or_mu1_hat, perc=20, sample=100_000):
    tau_or_tau_hat = mu1_or_mu1_hat - mu0_or_mu0_hat
    actual_sample_size = min(sample, len(tau_or_tau_hat))
    if actual_sample_size == 0:
        logging.warning("Cannot compute tau threshold with 0 samples for tau_or_tau_hat.")
        return 0.1
    idx = np.random.randint(0, len(tau_or_tau_hat), size=actual_sample_size)
    diffs = np.abs(tau_or_tau_hat[idx] - tau_or_tau_hat[np.roll(idx, 1)])
    return float(np.percentile(diffs, perc))


def make_pairs_from_hat(X, T, Y, mu0_hat, mu1_hat, thr, n_pairs, seed=None):
    """
    Generate a fixed quota of similar and dissimilar pairs based on ITE distance.
    """
    if seed is not None:
        np.random.seed(seed)
    ite_hat = mu1_hat - mu0_hat
    N = len(ite_hat)
    desired_sim = n_pairs // 2
    desired_diss = n_pairs - desired_sim
    sim_pairs = []
    diss_pairs = []
    used = set()

    # helper to avoid duplicates
    def add_pair(i, j, label, container):
        key = (min(i, j), max(i, j))
        if key not in used:
            used.add(key)
            container.append((i, j, label))

    all_indices = np.arange(N)
    # generate similar pairs
    trials = 0
    while len(sim_pairs) < desired_sim and trials < n_pairs * 5:
        i = np.random.randint(0, N)
        diffs = np.abs(ite_hat - ite_hat[i])
        candidates = np.where(diffs < thr)[0]
        candidates = candidates[candidates != i]
        if candidates.size:
            j = np.random.choice(candidates)
            add_pair(i, j, 1, sim_pairs)
        trials += 1
    # generate dissimilar pairs
    trials = 0
    while len(diss_pairs) < desired_diss and trials < n_pairs * 5:
        i = np.random.randint(0, N)
        diffs = np.abs(ite_hat - ite_hat[i])
        candidates = np.where(diffs >= thr)[0]
        candidates = candidates[candidates != i]
        if candidates.size:
            # hardest negative first
            j = candidates[np.argmin(diffs[candidates])]
            add_pair(i, j, 0, diss_pairs)
        trials += 1

    pairs = sim_pairs + diss_pairs
    np.random.shuffle(pairs)
    if not pairs:
        logging.warning("make_pairs_from_hat generated 0 pairs.")
        return (np.empty((0,) + X.shape[1:], dtype=X.dtype),)*6 + (np.array([], dtype=np.int64),)

    idx_a, idx_b, labels = zip(*pairs)
    idx_a = np.array(idx_a); idx_b = np.array(idx_b); labels = np.array(labels)

    return (
        X[idx_a], Y[idx_a], T[idx_a],
        X[idx_b], Y[idx_b], T[idx_b],
        labels
    )


def first_item(batch):
    return batch[0]


class DynamicContrastiveCausalDS(Dataset):
    def __init__(
        self,
        X_all_replica,
        T_all_replica,
        Y_all_replica,
        initial_mu0_hat,
        initial_mu1_hat,
        bs=256,
        perc=20,
        sample_for_thr_calc=100_000
    ):
        self.X_all = X_all_replica
        self.T_all = T_all_replica
        self.Y_all = Y_all_replica
        self.bs = bs
        self.perc = perc
        self.sample_for_thr_calc = sample_for_thr_calc
        if initial_mu0_hat.size and initial_mu1_hat.size:
            self.current_mu0_hat = initial_mu0_hat
            self.current_mu1_hat = initial_mu1_hat
        else:
            logging.error("Initial mu_hat empty, using zeros.")
            self.current_mu0_hat = np.zeros_like(Y_all.squeeze())
            self.current_mu1_hat = np.zeros_like(Y_all.squeeze())
        self.update_threshold()

    def update_threshold(self):
        self.thr = compute_tau_threshold(
            self.current_mu0_hat, self.current_mu1_hat,
            self.perc, self.sample_for_thr_calc
        )
        logging.info(f"Updated dynamic threshold: {self.thr:.4f}")

    def update_ite_estimates(self, new_mu0_hat, new_mu1_hat):
        if new_mu0_hat.size and new_mu1_hat.size:
            self.current_mu0_hat = new_mu0_hat
            self.current_mu1_hat = new_mu1_hat
            self.update_threshold()
        else:
            logging.warning("Empty new_mu_hat, threshold unchanged.")

    def __len__(self):
        # Number of virtual batches per epoch
        return int(np.ceil(len(self.X_all) / self.bs))

    def __getitem__(self, idx):
        # ignore idx, generate fresh pairs each call
        return make_pairs_from_hat(
            self.X_all, self.T_all, self.Y_all,
            self.current_mu0_hat, self.current_mu1_hat,
            self.thr, self.bs,
            seed=None
        )
