# -*- coding: utf-8 -*-
"""
Script principale per addestrare e valutare modelli CATE siamesi
su dataset IHDP, confrontando diverse strategie di training.
"""
import datetime
import inspect
import itertools
import os
import logging
import copy
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from SiameseTARNet import DualHeadSiameseCATE, contrastive_loss_batch
from paper.IHDPLoader import IHDPLoader, BalancedBatchSampler

# ====================================================================
# CONFIGURAZIONE LOGGING
# ====================================================================
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(seed)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = os.path.join("logs", timestamp)
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "experiment.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ],
    datefmt='%Y-%m-%d %H:%M:%S'
)
logging.info(f"Avvio script - Log salvati in: {log_file}")


# ====================================================================
# HELPER: STATS, COPPIE
# ====================================================================
def compute_global_stats(dataset, device=torch.device("cpu")):
    if isinstance(dataset, torch.utils.data.Subset):
        original = dataset.dataset
        indices = dataset.indices
    else:
        original = dataset
        indices = range(len(dataset))

    all_reps = []
    for idx in indices:
        x_i, *_ = original[idx]
        all_reps.append(x_i)
    Xcat = torch.cat([xr.t() for xr in all_reps], dim=0)
    mean_global = Xcat.mean(dim=0)
    std_global = Xcat.std(dim=0) + 1e-8
    logging.info(f"Computed stats on {Xcat.shape[0]} samples: mean/std shape {mean_global.shape}")
    return mean_global.to(device), std_global.to(device)


def create_pairs(x, t, y):
    t = t.view(-1)
    treated_idx = torch.where(t == 1)[0]
    control_idx = torch.where(t == 0)[0]
    if treated_idx.numel() < 1 or control_idx.numel() < 1:
        return []
    pos_pairs = []
    if treated_idx.numel() > 1:
        tr_shuf = treated_idx[torch.randperm(len(treated_idx))]
        for i in range(0, len(tr_shuf) - 1, 2):
            pos_pairs.append(("positive", (tr_shuf[i].item(), tr_shuf[i + 1].item())))
    if control_idx.numel() > 1:
        ct_shuf = control_idx[torch.randperm(len(control_idx))]
        for i in range(0, len(ct_shuf) - 1, 2):
            pos_pairs.append(("positive", (ct_shuf[i].item(), ct_shuf[i + 1].item())))
    neg_pairs = []
    n_neg = min(len(treated_idx), len(control_idx))
    if n_neg > 0:
        tr_neg = treated_idx[torch.randperm(len(treated_idx))[:n_neg]]
        ct_neg = control_idx[torch.randperm(len(control_idx))[:n_neg]]
        for i in range(n_neg):
            neg_pairs.append(("negative", (tr_neg[i].item(), ct_neg[i].item())))
    pairs = []
    for i in range(max(len(pos_pairs), len(neg_pairs))):
        if i < len(pos_pairs): pairs.append(pos_pairs[i])
        if i < len(neg_pairs): pairs.append(neg_pairs[i])
    return pairs


def create_pairs_batch(x, t, y, w, pairs, max_pairs=None):
    if not pairs:
        return None
    if max_pairs is not None and len(pairs) > max_pairs:
        pairs = [pairs[i] for i in torch.randperm(len(pairs))[:max_pairs].tolist()]

    x1l, z1l, y1l, w1l, x2l, z2l, y2l, w2l, sl = [], [], [], [], [], [], [], [], []
    B = x.size(0)
    for label, (i1, i2) in pairs:
        if i1 >= B or i2 >= B:
            continue
        x1l.append(x[i1].unsqueeze(0))
        z1l.append(t[i1].unsqueeze(0))
        y1l.append(y[i1].unsqueeze(0))
        w1l.append(w[i1].unsqueeze(0))
        x2l.append(x[i2].unsqueeze(0))
        z2l.append(t[i2].unsqueeze(0))
        y2l.append(y[i2].unsqueeze(0))
        w2l.append(w[i2].unsqueeze(0))
        sl.append(1.0 if label == "positive" else 0.0)
    if not x1l:
        return None
    x1b = torch.cat(x1l)
    z1b = torch.cat(z1l)
    y1b = torch.cat(y1l)
    w1b = torch.cat(w1l)
    x2b = torch.cat(x2l)
    z2b = torch.cat(z2l)
    y2b = torch.cat(y2l)
    w2b = torch.cat(w2l)
    sb = torch.tensor(sl, device=x.device).unsqueeze(1)
    return x1b, z1b, y1b, w1b, x2b, z2b, y2b, w2b, sb


# ====================================================================
# VALUTAZIONE: MSE reale
# ====================================================================
def evaluate_model(model, dataloader, device, realizations_to_use, mean_global, std_global):
    model.eval()
    criterion = nn.MSELoss(reduction='sum')
    total_mse = 0.0
    total_n = 0
    with torch.no_grad():
        for x, t, Yf, *_ in dataloader:
            x, t, Yf = x.to(device), t.to(device), Yf.to(device)
            B, d, R = x.shape
            N = min(realizations_to_use, R)
            if N == 0:
                continue
            batch_mse = 0.0
            for i in range(N):
                xi = x[:, :, i]
                ti = t[:, i].unsqueeze(1)
                yi = Yf[:, i].unsqueeze(1)
                xi_norm = (xi - mean_global) / std_global
                _, y_pred = model.forward_once(xi_norm, ti)
                batch_mse += criterion(y_pred, yi).item()
            total_mse += batch_mse / N
            total_n += B
    return total_mse / max(total_n, 1)


@torch.no_grad()
def evaluate_causal_metrics(model, X_norm, mu0_all, mu1_all, T=None, device="cpu"):
    model.eval()
    N, D, R = X_norm.shape
    zeros = torch.zeros(N, 1, device=device)
    ones = torch.ones(N, 1, device=device)

    pehe_vals = torch.zeros(R, device=device)
    ate_err_vals = torch.zeros(R, device=device)
    taus_pred = torch.zeros(R, device=device)
    taus_true = torch.zeros(R, device=device)
    tau_preds_mat = torch.zeros(N, R, device=device)

    if T is not None:
        att_err_vals = torch.zeros(R, device=device)
        treated_idx = torch.where(T[:, 0] == 1)[0]
    else:
        att_err_vals = None

    for r in range(R):
        x_r = X_norm[:, :, r].to(device)
        if hasattr(model, 'forward_once'):
            _, y0 = model.forward_once(x_r, zeros)
            _, y1 = model.forward_once(x_r, ones)
        else:
            y0, y1 = model.predict_potential_outcomes(x_r)

        tau_pred = (y1 - y0).squeeze()
        tau_true = (mu1_all[:, r] - mu0_all[:, r]).to(device).squeeze()
        tau_preds_mat[:, r] = tau_pred

        pehe_vals[r] = torch.sqrt(((tau_pred - tau_true) ** 2).mean())
        ate_err_vals[r] = torch.abs(tau_pred.mean() - tau_true.mean())

        if T is not None:
            if treated_idx.numel() > 0:
                att_err_vals[r] = torch.abs(
                    torch.mean(tau_pred[treated_idx]) - torch.mean(tau_true[treated_idx])
                )
            else:
                att_err_vals[r] = torch.tensor(float('nan'), device=device)

        taus_pred[r] = torch.mean(tau_pred)
        taus_true[r] = torch.mean(tau_true)

    result = {
        'pehe_mean': pehe_vals.mean().item(),
        'pehe_std': pehe_vals.std().item(),
        'ate_err_mean': ate_err_vals.mean().item(),
        'ate_err_std': ate_err_vals.std().item(),
    }
    if T is not None:
        result.update({
            'att_err_mean': att_err_vals.mean().item(),
            'att_err_std': att_err_vals.std().item()
        })

    # nuove metriche
    result['mae_ate'] = torch.mean(torch.abs(taus_pred - taus_true)).item()
    result['pehe_nested_mean'] = pehe_vals.mean().item()
    result['pehe_global'] = torch.sqrt(
        torch.mean((tau_preds_mat - (mu1_all.to(device) - mu0_all.to(device))) ** 2)).item()

    return result


# ====================================================================
# TRAINING: GRADIENTI CONDIVISI
# ====================================================================
def train_shared_gradients(
        model,
        num_epochs,
        train_loader,
        X_val_norm, T_val, mu0_val, mu1_val,
        realizations_to_use,
        mean_global, std_global,
        mu_y, sigma_y,
        optimizer_name="adam",
        lr=1e-3,
        momentum=0.9,
        weight_decay=0.0,
        alpha=0.1, beta=0.9,
        patience=10,
        margin=1.0, max_pairs=None,
        device="cpu",
        use_scheduler=False,
        scheduler_patience=10,
        scheduler_factor=0.1, warmup_epochs=5,
):
    """
    Training con gradienti condivisi e validazione tramite metriche causali.
    """
    # ----- Optimizer + Scheduler -----
    if optimizer_name.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(),
                              lr=lr, momentum=momentum,
                              weight_decay=weight_decay)
    elif optimizer_name.lower() == "adam":
        optimizer = optim.Adam(model.parameters(),
                               lr=lr,
                               weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer '{optimizer_name}' non supportato")

    scheduler = None
    if use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_factor,
            patience=scheduler_patience,
            verbose=True
        )

    # ----- Storici e best model -----
    history = {'total': [], 'contrastive': [], 'factual': []}
    best_val = float('inf')
    epochs_no_improve = 0
    best_state = copy.deepcopy(model.state_dict())

    # ======================= LOOP EPOCHS =======================
    for epoch in range(1, num_epochs + 1):
        model.train()
        sum_total = sum_contrast = sum_factual = 0.0
        n_batches = 0

        # ---- TRAIN ----
        for x, t, Yf, *_, w in train_loader:
            # normalizza outcome
            Yf = (Yf - mu_y) / sigma_y
            if w is None or w.numel() == 0:
                w = torch.ones_like(Yf)

            x = x.to(device)
            t = t.to(device)
            Yf = Yf.to(device)
            w = w.to(device)
            B, d, R = x.shape
            N = min(realizations_to_use, R)
            if N == 0:
                continue

            optimizer.zero_grad()
            c_sum = torch.tensor(0., device=device)
            f_sum = torch.tensor(0., device=device)
            real_ct = 0

            # loop sulle repliche
            for i in range(N):
                xi = x[:, :, i]
                ti = t[:, i].unsqueeze(1)
                yi = Yf[:, i].unsqueeze(1)
                wi = w[:, i].unsqueeze(1)

                xi_norm = (xi - mean_global) / std_global
                pairs = create_pairs(xi_norm, ti, yi)
                if not pairs:
                    continue
                batch = create_pairs_batch(xi_norm, ti, yi, wi, pairs, max_pairs)
                if batch is None:
                    continue

                x1, t1, y1, w1, x2, t2, y2, w2, same = batch
                e1, p1 = model.forward_once(x1, t1)
                e2, p2 = model.forward_once(x2, t2)

                c = contrastive_loss_batch(e1, e2, same, margin)
                f = ((w1 * (p1 - y1) ** 2 + w2 * (p2 - y2) ** 2) / 2).mean()

                c_sum += c
                f_sum += f
                real_ct += 1

            if real_ct == 0:
                continue

            c_batch = c_sum / real_ct
            f_batch = f_sum / real_ct

            total_loss = alpha * c_batch + beta * f_batch

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            sum_contrast += c_batch.item()
            sum_factual += f_batch.item()
            sum_total += total_loss.item()
            n_batches += 1

        if n_batches == 0:
            logging.warning(f"Epoch {epoch}: nessun batch processato")
            continue

        # ----- Media loss di training -----
        avg_contrast = sum_contrast / n_batches
        avg_factual = sum_factual / n_batches
        avg_total = sum_total / n_batches
        history['contrastive'].append(avg_contrast)
        history['factual'].append(avg_factual)
        history['total'].append(avg_total)

        # ----- VALIDAZIONE CAUSALE -----
        model.eval()
        with torch.no_grad():
            met = evaluate_causal_metrics(
                model,
                X_val_norm, mu0_val, mu1_val,
                T=T_val,
                device=device
            )
        # ----- VALIDAZIONE CAUSALE -----
        model.eval()
        with torch.no_grad():
            met = evaluate_causal_metrics(
                model,
                X_val_norm, mu0_val, mu1_val,
                T=T_val,
                device=device
            )
        # metriche normalizzate
        pehe_n      = met['pehe_mean']
        ate_n       = met['ate_err_mean']
        att_n       = met.get('att_err_mean', float('nan'))
        mae_ate_n   = met['mae_ate']
        pehe_nest_n = met['pehe_nested_mean']
        pehe_glob_n = met['pehe_global']

        # metriche raw
        pehe_r      = pehe_n      * sigma_y
        ate_r       = ate_n       * sigma_y
        att_r       = att_n       * sigma_y
        mae_ate_r   = mae_ate_n   * sigma_y
        pehe_nest_r = pehe_nest_n * sigma_y
        pehe_glob_r = pehe_glob_n * sigma_y

        logging.info(
            f"Epoch {epoch}/{num_epochs}\n"
            f"  TRAIN_LOSS: C={avg_contrast:.4f} | F={avg_factual:.4f} | T={avg_total:.4f}\n"
            f"  VAL_METRICS (norm):  "
            f"PEHE={pehe_n:.4f} | ATE-err={ate_n:.4f} | ATT-err={att_n:.4f} | "
            f"MAE-ATE={mae_ate_n:.4f} | PEHE-nested={pehe_nest_n:.4f} | PEHE-global={pehe_glob_n:.4f}\n"
            f"  VAL_METRICS (raw):   "
            f"PEHE={pehe_r:.4f} | ATE-err={ate_r:.4f} | ATT-err={att_r:.4f} | "
            f"MAE-ATE={mae_ate_r:.4f} | PEHE-nested={pehe_nest_r:.4f} | PEHE-global={pehe_glob_r:.4f}"
        )
        # ----- Scheduler + Early-Stopping -----
        if scheduler:
            scheduler.step(pehe_n)

        current = pehe_n
        if not (math.isnan(current) or math.isinf(current)) and current < best_val:
            best_val = current
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logging.info("Early stopping")
                break

    # ----- Ripristino del miglior modello -----
    model.load_state_dict(best_state)
    return history, best_state


# ====================================================================
# PLOTTING
# ====================================================================
def plot_training_losses_shared(loss_history_dict, plot_log_dir, plot_suffix=""):
    num_epochs = len(loss_history_dict.get('total', []))
    if num_epochs == 0:
        return
    epochs = range(1, num_epochs + 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, loss_history_dict['total'], marker='.', label='Total')
    ax.plot(epochs, loss_history_dict['contrastive'], marker='o', label='Contrastive')
    ax.plot(epochs, loss_history_dict['factual'], marker='s', label='Factual')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale('log')
    ax.set_title(f"Shared Gradients{plot_suffix}")
    ax.legend()
    ax.grid(True)
    path = os.path.join(plot_log_dir, f"training_loss_shared{plot_suffix}.png")
    fig.savefig(path)
    plt.close(fig)
    logging.info(f"Plot saved to {path}")


# ====================================================================
# EXPERIMENT & GRID SEARCH
# ====================================================================
def run_experiment(config, train_loader,
                   X_val_norm, T_val, mu0_val, mu1_val,
                   mean_global, std_global,
                   model_class, train_function, mu_y, sigma_y,
                   device="cpu", log_dir_run=None):
    try:
        arch_params = {k: config[k] for k in
                       ['hidden_dim', 'rep_layers', 'rep_activation',
                        'hyp_layers', 'hyp_activation']}
        model = model_class(input_dim=25, **arch_params).to(device)
    except Exception as e:
        logging.error(f"Model instantiation failed: {e}")
        return float('inf'), None

    train_func_name = train_function.__name__
    logging.info(f"Running {train_func_name} on {model_class.__name__}")

    all_params = dict(
        model=model,
        num_epochs=config.get("num_epochs"),
        train_loader=train_loader,
        val_loader=val_loader,
        X_val_norm=X_val_norm,
        T_val=T_val,
        mu0_val=mu0_val,
        mu1_val=mu1_val,
        realizations_to_use=config.get("realizations_to_use"),
        mean_global=mean_global,
        std_global=std_global,
        mu_y=mu_y,
        sigma_y=sigma_y,
        optimizer_name=config.get("optimizer"),
        lr=config.get("learning_rate"),
        momentum=config.get("momentum"),
        weight_decay=config.get("weight_decay"),
        alpha=config.get("alpha"),
        beta=config.get("beta"),
        margin=config.get("margin"),
        max_pairs=config.get("max_pairs"),
        patience=config.get("patience"),
        monitor_metric=config.get("monitor_metric"),
        use_scheduler=config.get("use_scheduler"),
        scheduler_patience=config.get("scheduler_patience"),
        scheduler_factor=config.get("scheduler_factor"),
        device=device
    )

    sig = inspect.signature(train_function)
    train_params = {k: v for k, v in all_params.items() if k in sig.parameters}

    plot_func = plot_training_losses_shared if train_func_name == "train_shared_gradients" else None
    plot_suffix = "_shared" if train_func_name == "train_shared_gradients" else ""

    try:
        loss_history, best_state = train_function(**train_params)
        if plot_func:
            plot_func(loss_history, log_dir_run or log_dir, plot_suffix)
    except Exception as e:
        logging.error(f"Training failed: {e}", exc_info=True)
        return float('inf'), None

    if not best_state:
        return float('inf'), None

    model.load_state_dict(best_state)
    model.eval()
    try:
        metrics = evaluate_causal_metrics(
            model, X_val_norm, mu0_val, mu1_val,
            T=T_val if 'att' in config.get('monitor_metric', '') else None,
            device=device
        )
        if config["monitor_metric"] == 'pehe_rmse':
            score = metrics['pehe_mean']
        elif config["monitor_metric"] == 'ate_error':
            score = metrics['ate_err_mean']
        elif config["monitor_metric"] == 'att_error':
            score = metrics['att_err_mean']
        else:
            score = float('inf')
        if math.isnan(score) or math.isinf(score):
            score = float('inf')
        if log_dir_run:
            torch.save(best_state, os.path.join(log_dir_run, "best_model.pt"))
        return score, model
    except Exception as e:
        logging.error(f"Validation failed: {e}", exc_info=True)
        return float('inf'), None


def grid_search(param_grid, train_loader,
                X_val_norm, T_val, mu0_val, mu1_val,
                mean_global, std_global, mu_y, sigma_y,
                model_class, train_function,
                device="cpu", log_dir_grid=None):
    best_cfg, best_score, best_state = None, float('inf'), None
    keys, vals = zip(*param_grid.items())
    combos = [dict(zip(keys, v)) for v in itertools.product(*vals)]
    grid_id = f"{model_class.__name__}_{train_function.__name__}"
    log_dir_grid = log_dir_grid or log_dir
    results = []

    for i, cfg in enumerate(combos, 1):
        run_dir = os.path.join(log_dir_grid, f"run_{i}_{grid_id}")
        os.makedirs(run_dir, exist_ok=True)
        logging.info(f"Grid run {i}/{len(combos)}: {cfg}")
        score, model = run_experiment(
            cfg, train_loader,
            X_val_norm, T_val, mu0_val, mu1_val,
            mean_global, std_global,
            model_class, train_function, mu_y, sigma_y,
            device, run_dir
        )
        if model is None:
            continue
        weights = os.path.join(run_dir, "best_model.pt")
        results.append((i, score, cfg, weights))
        if score < best_score:
            best_score, best_cfg, best_state = score, cfg, copy.deepcopy(model.state_dict())
            logging.info(f"*** New best: {best_score:.4f}")
    df = pd.DataFrame([{"Run": r, "Score": s, "Config": str(cfg), "Weights": w}
                       for r, s, cfg, w in results])
    df.to_csv(os.path.join(log_dir_grid, f"grid_{grid_id}.csv"), index=False)
    logging.info(f"Grid done. Best score: {best_score:.4f}")
    return best_cfg, best_score, best_state


# ====================================================================
# MAIN
# ====================================================================
if __name__ == "__main__":
    logging.info("Main started")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    # 1) Load data
    trainval_ds = IHDPLoader(is_train=True)
    test_ds = IHDPLoader(is_train=False)

    # 2) Split train/val
    n_tv = len(trainval_ds)
    n_train = int(0.8 * n_tv)
    perm = torch.randperm(n_tv)
    train_idx, val_idx = perm[:n_train].tolist(), perm[n_train:].tolist()
    train_ds = torch.utils.data.Subset(trainval_ds, train_idx)
    val_ds = torch.utils.data.Subset(trainval_ds, val_idx)

    # 2.1) Target stats on train
    X_all, T_all, YF_all, YCF_all, mu0_all, mu1_all, _, _ = trainval_ds.load()
    YF_train = YF_all[train_idx]
    mu_y = YF_train.mean()
    sigma_y = YF_train.std(unbiased=False) + 1e-8
    logging.info(f"Target mu={mu_y:.4f}, sigma={sigma_y:.4f}")

    # 3) DataLoaders
    loader_args = dict(num_workers=0, pin_memory=device.type == 'cuda',
                       worker_init_fn=seed_worker, generator=g)
    treatments_train = T_all[train_idx, 0].cpu().numpy()
    sampler = BalancedBatchSampler(treatments_train, batch_size=1024)
    train_loader = DataLoader(train_ds, batch_sampler=sampler, **loader_args)
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False, **loader_args)
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False, **loader_args)

    # 4) Covariate stats
    mean_cpu, std_cpu = compute_global_stats(train_ds, device=torch.device('cpu'))
    mean_global, std_global = mean_cpu.to(device), std_cpu.to(device)

    # 5) Prepare val tensors
    X_all, T_all, YF_all, YCF_all, mu0_all, mu1_all, _, _ = trainval_ds.load()
    X_val = X_all[val_idx].to(device)
    T_val = T_all[val_idx].to(device)
    mu0_val = mu0_all[val_idx].to(device)
    mu1_val = mu1_all[val_idx].to(device)
    mu0_val = (mu0_val - mu_y) / sigma_y
    mu1_val = (mu1_val - mu_y) / sigma_y
    mean_r, std_r = mean_global.view(1, -1, 1), std_global.view(1, -1, 1)
    X_val_norm = (X_val - mean_r) / std_r

    # 5b) τ‐true std
    with torch.no_grad():
        tau_true = (mu1_val - mu0_val)
        tau_std_norm = tau_true.mean(dim=1).std().item()
    logging.info(f"tau_true std norm={tau_std_norm:.4f}, raw={tau_std_norm * sigma_y:.4f}")

    # 6) Grid search
    param_grid = {
        "optimizer": ["adam", 'sgd'], "momentum": [0.9], "learning_rate": [5e-4],
        "weight_decay": [0], "use_scheduler": [False],
        "scheduler_patience": [10], "scheduler_factor": [0.5],
        "alpha": [0.1], "beta": [0.9], "margin": [.5],
        "max_pairs": [None], "realizations_to_use": [1000],
        "num_epochs": [1000], "patience": [30],
        "monitor_metric": ['ate_error'],
        "hidden_dim": [200], "rep_layers": [5], "rep_activation": ['relu', 'elu'],
        "hyp_layers": [5], "hyp_activation": ['relu', 'elu']
    }
    best_cfg, best_score, best_state = grid_search(
        param_grid=param_grid,
        train_loader=train_loader,
        X_val_norm=X_val_norm,
        T_val=T_val,
        mu0_val=mu0_val,
        mu1_val=mu1_val,
        mean_global=mean_global,
        std_global=std_global,
        mu_y=mu_y,
        sigma_y=sigma_y,
        model_class=DualHeadSiameseCATE,
        train_function=train_shared_gradients,
        device=device,
        log_dir_grid=log_dir
    )

    # 7) Final test evaluation
    logging.info("Phase 7: Final Evaluation")
    X_test, T_test, YF_test, YCF_test, mu0_test, mu1_test, _, _ = test_ds.load()
    X_test, T_test = X_test.to(device), T_test.to(device)
    mu0_test, mu1_test = mu0_test.to(device), mu1_test.to(device)
    mu0_test_norm = (mu0_test - mu_y) / sigma_y
    mu1_test_norm = (mu1_test - mu_y) / sigma_y
    X_test_norm = (X_test - mean_r) / std_r

    model = DualHeadSiameseCATE(
        input_dim=X_test.shape[1],
        hidden_dim=best_cfg['hidden_dim'],
        rep_layers=best_cfg['rep_layers'],
        rep_activation=best_cfg['rep_activation'],
        hyp_layers=best_cfg['hyp_layers'],
        hyp_activation=best_cfg['hyp_activation']
    ).to(device)
    model.load_state_dict(best_state)
    model.eval()

    metrics_test = evaluate_causal_metrics(
        model, X_test_norm, mu0_test_norm, mu1_test_norm,
        T=T_test, device=device
    )
    # Normalized
    p_m, p_s = metrics_test['pehe_mean'], metrics_test['pehe_std']
    a_m, a_s = metrics_test['ate_err_mean'], metrics_test['ate_err_std']
    t_m, t_s = metrics_test['att_err_mean'], metrics_test['att_err_std']
    mae_ate_test = metrics_test['mae_ate']
    pehe_nest_test = metrics_test['pehe_nested_mean']
    pehe_glob_test = metrics_test['pehe_global']

    # Raw (riportati alle unità originali moltiplicando per sigma_y)
    p_m_raw = p_m * sigma_y
    p_s_raw = p_s * sigma_y
    a_m_raw = a_m * sigma_y
    a_s_raw = a_s * sigma_y
    t_m_raw = t_m * sigma_y
    t_s_raw = t_s * sigma_y
    mae_ate_raw = mae_ate_test * sigma_y
    pehe_nest_raw = pehe_nest_test * sigma_y
    pehe_glob_raw = pehe_glob_test * sigma_y

    logging.info(
        f"Test (norm):  PEHE={p_m:.4f}±{p_s:.4f} | ATE-err={a_m:.4f}±{a_s:.4f} | ATT-err={t_m:.4f}±{t_s:.4f} | "
        f"MAE-ATE={mae_ate_test:.4f} | PEHE-nested={pehe_nest_test:.4f} | PEHE-global={pehe_glob_test:.4f}\n"
        f"Test (raw):   PEHE={p_m_raw:.4f}±{p_s_raw:.4f} | ATE-err={a_m_raw:.4f}±{a_s_raw:.4f} | "
        f"ATT-err={t_m_raw:.4f}±{t_s_raw:.4f} | MAE-ATE={mae_ate_raw:.4f} | "
        f"PEHE-nested={pehe_nest_raw:.4f} | PEHE-global={pehe_glob_raw:.4f}"
    )

    logging.info("===== Script finished =====")
