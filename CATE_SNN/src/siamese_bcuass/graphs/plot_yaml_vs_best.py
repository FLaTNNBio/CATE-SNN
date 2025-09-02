import argparse
import os
import pandas as pd
import yaml
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best", required=True, help="Path to best_per_facet.csv")
    ap.add_argument("--yaml", required=True, help="Path to default.yaml")
    ap.add_argument("--out", default="figs", help="Output directory")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # leggi best_per_facet
    best_df = pd.read_csv(args.best)
    best_df["pehe"] = best_df["pehe"].astype(float)

    # leggi config yaml
    with open(args.yaml, "r") as f:
        y = yaml.safe_load(f) or {}
    y_margin = y.get("siamese", {}).get("margin")
    y_pair   = y.get("siamese", {}).get("pair_pct")
    y_act    = y.get("bcauss_params", {}).get("act_fn")
    y_optim  = y.get("bcauss_params", {}).get("optim")

    # trova la faccia corrispondente
    mask = (best_df["activation"] == y_act) & (best_df["optim"] == y_optim)
    if not mask.any():
        print("⚠️ Config YAML non trovata in best_per_facet.csv.")
        return
    best_row = best_df[mask].iloc[0]
    best_pehe = best_row["pehe"]

    # costruiamo piccolo dataframe confronto
    compare = pd.DataFrame([
        {"config": "yaml_config", "pehe": None, "margin": y_margin, "pair_pct": y_pair},
        {"config": "best_in_facet", "pehe": best_pehe, "margin": best_row["margin"], "pair_pct": best_row["pair_pct"]}
    ])

    # bar chart (solo pehe best, yaml non ha pehe se non lo cerchi in optuna_aggregated_metrics.csv)
    plt.figure()
    plt.bar(compare["config"], compare["pehe"])
    plt.ylabel("PEHE (√)")
    plt.title(f"Confronto: YAML vs Best ({y_act}, {y_optim})")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "04_yaml_vs_best.png"))
    plt.close()

    compare.to_csv(os.path.join(args.out, "yaml_vs_best.csv"), index=False)
    print(f"Salvati grafico e CSV in {args.out}")

if __name__ == "__main__":
    main()
