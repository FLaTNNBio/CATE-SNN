import argparse
import os
import pandas as pd
import yaml
import numpy as np
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--optuna", required=True, help="Path to optuna_aggregated_metrics.csv")
    ap.add_argument("--yaml", required=True, help="Path to default.yaml")
    ap.add_argument("--out", default="figs", help="Output directory")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # --- read optuna results ---
    df = pd.read_csv(args.optuna)
    df["pehe"] = df["pehe"].astype(float)
    df["activation"] = df["activation"].astype(str).str.lower()
    df["optim"] = df["optim"].astype(str).str.lower()

    # --- detect axis column ---
    if "batch_size" in df.columns:
        axis_col = "batch_size"
    elif "batch" in df.columns:
        axis_col = "batch"
    elif "pair_pct" in df.columns:
        axis_col = "pair_pct"
    else:
        raise ValueError("CSV optuna non contiene né 'batch_size', né 'batch', né 'pair_pct'.")

    df[axis_col] = df[axis_col].astype(float)

    # --- read yaml config ---
    with open(args.yaml, "r") as f:
        y = yaml.safe_load(f) or {}
    y_margin = float(y.get("siamese", {}).get("margin"))
    y_act    = (y.get("bcauss_params", {}).get("act_fn") or "").lower()
    y_optim  = (y.get("bcauss_params", {}).get("optim") or "").lower()
    if axis_col == "pair_pct":
        y_axis_val = float(y.get("siamese", {}).get("pair_pct"))
    else:
        y_axis_val = float(y.get("batch") or y.get("batch_size"))

    # --- subset for same face (activation, optim) ---
    face_df = df[(df["activation"] == y_act) & (df["optim"] == y_optim)].copy()
    if face_df.empty:
        raise ValueError("Nessun trial in optuna_aggregated_metrics.csv con stessa (activation, optim) della YAML.")

    # --- scatter plot ---
    plt.figure()
    sc = plt.scatter(face_df["margin"], face_df[axis_col], c=face_df["pehe"],
                     cmap="viridis", s=60, alpha=0.8)
    plt.colorbar(sc, label="PEHE (√)")
    plt.xlabel("margin")
    plt.ylabel(axis_col)
    plt.title(f"Scatter PEHE — face: {y_act}, {y_optim}")

    # highlight YAML config
    plt.scatter([y_margin], [y_axis_val], color="red", marker="x", s=120, label="YAML config")
    plt.legend()
    plt.tight_layout()
    out_png = os.path.join(args.out, "yaml_vs_best.png")
    plt.savefig(out_png)
    plt.close()

    print(f"✅ Salvato scatter plot in {out_png}")

if __name__ == "__main__":
    main()
