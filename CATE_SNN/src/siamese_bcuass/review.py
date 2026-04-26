import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import truncnorm
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------
# 1. UTILITIES
# ---------------------------------------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------
# 2. GENERAZIONE DEL DATASET SIMULATO PARAMETRICO
# ---------------------------------------------------------
def generate_psychiatric_data(
    n_samples: int = 5000,
    overlap_strength: float = 1.0,
    confounding_strength: float = 1.0,
    seed: int = 42,
):
    """
    overlap_strength:
        > 1.0  => lower overlap (propensity più estrema)
        = 1.0  => baseline
        < 1.0  => higher overlap (propensity più morbida)

    confounding_strength:
        > 1.0  => stronger confounding
        = 1.0  => baseline
        < 1.0  => weaker confounding
    """
    set_seed(seed)

    # X1: Depression severity (Truncated Normal)
    a, b = (0 - 15) / 5, (30 - 15) / 5
    X1 = truncnorm.rvs(a, b, loc=15, scale=5, size=n_samples)

    # X2: Age (Truncated Normal)
    a, b = (18 - 45) / 25, (85 - 45) / 25
    X2 = truncnorm.rvs(a, b, loc=45, scale=25, size=n_samples)

    # X3: Substance abuse
    p_X3 = np.clip(0.2 + 0.015 * X1, 0, 1)
    X3 = np.random.binomial(1, p_X3)

    # X4: Access to care
    p_X4 = sigmoid(-1.5 + 0.03 * X2)
    X4 = np.random.binomial(1, p_X4)

    X = np.column_stack([X1, X2, X3, X4]).astype(np.float32)

    # -------------------------------------------------
    # Treatment assignment:
    # confounding_strength amplifica i coefficienti
    # overlap_strength rende il logit più o meno "ripido"
    # -------------------------------------------------
    treatment_logit = (
        -1.2
        + confounding_strength * (
            0.30 * X1
            + 0.02 * X2
            - 0.8 * X3
            + 1.1 * X4
        )
    )
    treatment_logit = overlap_strength * treatment_logit
    p_T = sigmoid(treatment_logit)
    T = np.random.binomial(1, p_T).astype(np.float32)

    # -------------------------------------------------
    # Outcome model:
    # confounding_strength amplifica la parte prognostica
    # il treatment effect resta eterogeneo come nel tuo scenario
    # -------------------------------------------------
    base_risk = (
        -0.5
        + confounding_strength * (
            -0.4 * (X1 / 30)
            + 0.03 * X2
            - 0.8 * X3
            + 1.0 * X4
        )
    )

    treatment_effect = 0.6 - 0.2 * X3

    y0_prob = sigmoid(base_risk)
    y1_prob = sigmoid(base_risk + treatment_effect)
    y_prob = T * y1_prob + (1 - T) * y0_prob
    Y = np.random.binomial(1, y_prob).astype(np.float32)

    true_ite = (y1_prob - y0_prob).astype(np.float32)

    # low-support mask basata sulla propensity vera
    low_support_mask = ((p_T < 0.1) | (p_T > 0.9)).astype(np.float32)

    return {
        "X": torch.tensor(X, dtype=torch.float32),
        "T": torch.tensor(T, dtype=torch.float32).unsqueeze(1),
        "Y": torch.tensor(Y, dtype=torch.float32).unsqueeze(1),
        "true_ite": torch.tensor(true_ite, dtype=torch.float32).unsqueeze(1),
        "p_T": torch.tensor(p_T, dtype=torch.float32).unsqueeze(1),
        "low_support_mask": torch.tensor(low_support_mask, dtype=torch.float32).unsqueeze(1),
    }


# ---------------------------------------------------------
# 3. ARCHITETTURA DI BASE
# ---------------------------------------------------------
class HermesBase(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.head_0 = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        self.head_1 = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        phi = self.encoder(x)
        y0 = self.head_0(phi)
        y1 = self.head_1(phi)
        return phi, y0, y1


# ---------------------------------------------------------
# 4. TRAINING
# ---------------------------------------------------------
def train_hermes_with_noise_injection(
    X_train,
    T_train,
    Y_train,
    true_ite_train,
    X_test,
    true_ite_test,
    low_support_test,
    noise_std: float = 0.0,
    epochs: int = 100,
    warm_up: int = 20,
    lr: float = 1e-3,
    lambda_ctr: float = 0.5,
    margin: float = 1.0,
    device: str = "cpu",
):
    model = HermesBase(input_dim=X_train.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    X_train = X_train.to(device)
    T_train = T_train.to(device)
    Y_train = Y_train.to(device)
    true_ite_train = true_ite_train.to(device)

    X_test = X_test.to(device)
    true_ite_test = true_ite_test.to(device)
    low_support_test = low_support_test.to(device)

    history = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        phi, y0, y1 = model(X_train)

        # factual loss
        y_pred = T_train * y1 + (1 - T_train) * y0
        loss_factual = mse_loss(y_pred, Y_train)
        loss_total = loss_factual

        # contrastive loss after warm-up
        if epoch >= warm_up:
            with torch.no_grad():
                estimated_ite = y1 - y0

                if noise_std > 0:
                    noise = torch.randn_like(estimated_ite) * noise_std
                    estimated_ite = estimated_ite + noise

                ite_diffs = torch.abs(estimated_ite - estimated_ite.T)
                tau_thr = torch.quantile(ite_diffs, 0.20)

                pos_mask = (ite_diffs <= tau_thr).float()
                neg_mask = (ite_diffs > tau_thr).float()

            phi_dist_sq = torch.sum((phi.unsqueeze(1) - phi.unsqueeze(0)) ** 2, dim=2)
            phi_dist = torch.sqrt(phi_dist_sq + 1e-8)

            idx = torch.randperm(len(X_train))[: min(100, len(X_train))]
            pos_sub = pos_mask[idx][:, idx]
            neg_sub = neg_mask[idx][:, idx]
            dist_sq_sub = phi_dist_sq[idx][:, idx]
            dist_sub = phi_dist[idx][:, idx]

            loss_pos = torch.sum(pos_sub * dist_sq_sub)
            loss_neg = torch.sum(neg_sub * torch.relu(margin - dist_sub) ** 2)
            loss_ctr = (loss_pos + loss_neg) / (len(idx) ** 2)

            loss_total = loss_total + lambda_ctr * loss_ctr

        loss_total.backward()
        optimizer.step()

        # evaluation
        model.eval()
        with torch.no_grad():
            _, y0_te, y1_te = model(X_test)
            ite_hat_test = y1_te - y0_te

            rmse_all = torch.sqrt(torch.mean((ite_hat_test - true_ite_test) ** 2)).item()

            low_mask_bool = low_support_test.squeeze(1) > 0.5
            if low_mask_bool.sum() > 0:
                rmse_low = torch.sqrt(
                    torch.mean((ite_hat_test[low_mask_bool] - true_ite_test[low_mask_bool]) ** 2)
                ).item()
            else:
                rmse_low = float("nan")

        history.append({
            "epoch": epoch,
            "phase": "warmup" if epoch < warm_up else "contrastive",
            "loss_total": float(loss_total.item()),
            "loss_factual": float(loss_factual.item()),
            "rmse_all": rmse_all,
            "rmse_low_support": rmse_low,
        })

    return model, history


# ---------------------------------------------------------
# 5. SIMPLE BASELINE (FACTUAL ONLY)
# ---------------------------------------------------------
def train_factual_only(
    X_train,
    T_train,
    Y_train,
    X_test,
    true_ite_test,
    low_support_test,
    epochs: int = 100,
    lr: float = 1e-3,
    device: str = "cpu",
):
    model = HermesBase(input_dim=X_train.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    X_train = X_train.to(device)
    T_train = T_train.to(device)
    Y_train = Y_train.to(device)

    X_test = X_test.to(device)
    true_ite_test = true_ite_test.to(device)
    low_support_test = low_support_test.to(device)

    history = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        _, y0, y1 = model(X_train)
        y_pred = T_train * y1 + (1 - T_train) * y0
        loss = mse_loss(y_pred, Y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            _, y0_te, y1_te = model(X_test)
            ite_hat_test = y1_te - y0_te
            rmse_all = torch.sqrt(torch.mean((ite_hat_test - true_ite_test) ** 2)).item()

            low_mask_bool = low_support_test.squeeze(1) > 0.5
            if low_mask_bool.sum() > 0:
                rmse_low = torch.sqrt(
                    torch.mean((ite_hat_test[low_mask_bool] - true_ite_test[low_mask_bool]) ** 2)
                ).item()
            else:
                rmse_low = float("nan")

        history.append({
            "epoch": epoch,
            "phase": "factual_only",
            "loss_total": float(loss.item()),
            "loss_factual": float(loss.item()),
            "rmse_all": rmse_all,
            "rmse_low_support": rmse_low,
        })

    return model, history


# ---------------------------------------------------------
# 6. SPLIT HELPER
# ---------------------------------------------------------
def split_tensors(data_dict, test_size=0.3, seed=42):
    n = data_dict["X"].shape[0]
    idx = np.arange(n)

    tr_idx, te_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )

    train = {k: v[tr_idx] for k, v in data_dict.items()}
    test = {k: v[te_idx] for k, v in data_dict.items()}
    return train, test


# ---------------------------------------------------------
# 7. GRID EXPERIMENT
# ---------------------------------------------------------
def run_systematic_robustness_experiment(
    output_dir="robustness_outputs",
    n_reps=5,
    n_samples=2000,
    epochs=100,
    warm_up=20,
    device="cpu",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    overlap_grid = {
        "high": 0.6,
        "medium": 1.0,
        "low": 1.8,
    }

    confounding_grid = {
        "low": 0.5,
        "medium": 1.0,
        "high": 1.5,
    }

    model_configs = [
        {"name": "FactualOnly", "noise_std": None},
        {"name": "HERMES", "noise_std": 0.0},
        {"name": "HERMES_Noise2.0", "noise_std": 2.0},
    ]

    results = []
    history_rows = []

    for rep in range(n_reps):
        for overlap_name, overlap_strength in overlap_grid.items():
            for conf_name, conf_strength in confounding_grid.items():

                print(f"\n=== Rep {rep} | overlap={overlap_name} | confounding={conf_name} ===")

                data = generate_psychiatric_data(
                    n_samples=n_samples,
                    overlap_strength=overlap_strength,
                    confounding_strength=conf_strength,
                    seed=42 + rep,
                )
                train, test = split_tensors(data, test_size=0.3, seed=100 + rep)

                low_support_frac = float(test["low_support_mask"].mean().item())

                for cfg in model_configs:
                    name = cfg["name"]

                    if name == "FactualOnly":
                        model, history = train_factual_only(
                            X_train=train["X"],
                            T_train=train["T"],
                            Y_train=train["Y"],
                            X_test=test["X"],
                            true_ite_test=test["true_ite"],
                            low_support_test=test["low_support_mask"],
                            epochs=epochs,
                            lr=1e-3,
                            device=device,
                        )
                    else:
                        model, history = train_hermes_with_noise_injection(
                            X_train=train["X"],
                            T_train=train["T"],
                            Y_train=train["Y"],
                            true_ite_train=train["true_ite"],
                            X_test=test["X"],
                            true_ite_test=test["true_ite"],
                            low_support_test=test["low_support_mask"],
                            noise_std=cfg["noise_std"],
                            epochs=epochs,
                            warm_up=warm_up,
                            lr=1e-3,
                            lambda_ctr=0.5,
                            margin=1.0,
                            device=device,
                        )

                    final = history[-1]

                    results.append({
                        "rep": rep,
                        "model": name,
                        "overlap": overlap_name,
                        "confounding": conf_name,
                        "overlap_strength": overlap_strength,
                        "confounding_strength": conf_strength,
                        "low_support_frac": low_support_frac,
                        "final_rmse_all": final["rmse_all"],
                        "final_rmse_low_support": final["rmse_low_support"],
                    })

                    for h in history:
                        history_rows.append({
                            "rep": rep,
                            "model": name,
                            "overlap": overlap_name,
                            "confounding": conf_name,
                            "epoch": h["epoch"],
                            "phase": h["phase"],
                            "loss_total": h["loss_total"],
                            "loss_factual": h["loss_factual"],
                            "rmse_all": h["rmse_all"],
                            "rmse_low_support": h["rmse_low_support"],
                        })

    # save raw
    raw_path = output_dir / "robustness_raw_results.csv"
    with open(raw_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    hist_path = output_dir / "robustness_training_history.csv"
    with open(hist_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history_rows[0].keys())
        writer.writeheader()
        writer.writerows(history_rows)

    # aggregate summary
    summary = {}
    for row in results:
        key = (row["model"], row["overlap"], row["confounding"])
        summary.setdefault(key, {"rmse_all": [], "rmse_low": [], "low_support_frac": []})
        summary[key]["rmse_all"].append(row["final_rmse_all"])
        summary[key]["rmse_low"].append(row["final_rmse_low_support"])
        summary[key]["low_support_frac"].append(row["low_support_frac"])

    summary_rows = []
    for (model, overlap, conf), vals in summary.items():
        summary_rows.append({
            "model": model,
            "overlap": overlap,
            "confounding": conf,
            "rmse_all_mean": float(np.nanmean(vals["rmse_all"])),
            "rmse_all_std": float(np.nanstd(vals["rmse_all"])),
            "rmse_low_support_mean": float(np.nanmean(vals["rmse_low"])),
            "rmse_low_support_std": float(np.nanstd(vals["rmse_low"])),
            "low_support_frac_mean": float(np.nanmean(vals["low_support_frac"])),
        })

    summary_path = output_dir / "robustness_summary_results.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSaved:\n  - {raw_path}\n  - {hist_path}\n  - {summary_path}")
    return results, history_rows, summary_rows


# ---------------------------------------------------------
# 8. MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Running systematic robustness experiment...")
    run_systematic_robustness_experiment(
        output_dir="robustness_outputs",
        n_reps=5,
        n_samples=2000,
        epochs=100,
        warm_up=20,
        device=device,
    )