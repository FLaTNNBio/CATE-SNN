import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def _mesh_for_heatmap(df, xcol, ycol, zcol):
    xs = sorted(df[xcol].unique())
    ys = sorted(df[ycol].unique())
    X, Y = np.meshgrid(xs, ys)
    Z = np.full_like(X, np.nan, dtype=float)
    lut = {(row[xcol], row[ycol]): row[zcol] for _, row in df.iterrows()}
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            if (x, y) in lut:
                Z[i, j] = lut[(x, y)]
    return X, Y, Z

def plot_heatmaps(optuna_csv, yaml_path, out_dir):
    df = pd.read_csv(optuna_csv)
    needed = ["margin","pair_pct","activation","optim","pehe"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in {optuna_csv}. Found: {list(df.columns)}")

    # parse YAML (optional)
    y_margin = y_pair = y_act = y_optim = None
    if yaml_path and os.path.exists(yaml_path):
        try:
            with open(yaml_path, "r") as f:
                y = yaml.safe_load(f) or {}
            y_margin = y.get("siamese", {}).get("margin")
            y_pair   = y.get("siamese", {}).get("pair_pct")
            y_act    = y.get("bcauss_params", {}).get("act_fn")
            y_optim  = y.get("bcauss_params", {}).get("optim")
        except Exception:
            pass

    ensure_dir(out_dir)
    best_rows = []
    for (act, opt), g in df.groupby(["activation","optim"]):
        g2 = g.copy()
        g2["pehe"] = g2["pehe"].astype(float)
        X, Y, Z = _mesh_for_heatmap(g2, "margin", "pair_pct", "pehe")
        plt.figure()
        plt.pcolormesh(X, Y, Z, shading="nearest")
        plt.xlabel("margin"); plt.ylabel("pair_pct")
        plt.title(f"PEHE heatmap — activation={act}, optim={opt}")
        cb = plt.colorbar(); cb.set_label("pehe")
        # mark YAML config if matches facet
        if (y_margin is not None) and (y_pair is not None) and ((y_act is None) or (y_act == act)) and ((y_optim is None) or (y_optim == opt)):
            plt.scatter([y_margin], [y_pair], marker="x")
        plt.tight_layout()
        fname = f"heatmap_pehe_{act}_{opt}.png".replace("/","-")
        plt.savefig(os.path.join(out_dir, fname))
        plt.close()

        # best row in this facet
        best_idx = g2["pehe"].idxmin()
        best_rows.append(df.loc[best_idx])

    pd.DataFrame(best_rows).to_csv(os.path.join(out_dir, "best_per_facet.csv"), index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--optuna", required=True, help="Path to optuna_aggregated_metrics.csv")
    ap.add_argument("--yaml", required=False, help="Path to default.yaml (to mark configuration)")
    ap.add_argument("--out", default="figs", help="Output directory")
    args = ap.parse_args()
    plot_heatmaps(args.optuna, args.yaml, args.out)
    print(f"Saved figures to: {os.path.abspath(args.out)}")

if __name__ == "__main__":
    main()
