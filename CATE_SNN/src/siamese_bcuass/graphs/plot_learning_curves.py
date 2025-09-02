
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def plot_learning_curves(log_csv, out_dir):
    df = pd.read_csv(log_csv)
    needed = ["epoch","train_base","train_ctr","train_total"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in {log_csv}. Found: {list(df.columns)}")

    epochs = df["epoch"].values

    # Total
    plt.figure()
    plt.plot(epochs, df["train_total"].values, label="train_total")
    if "val_total" in df.columns:
        plt.plot(epochs, df["val_total"].values, label="val_total")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curves: Total Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "01_learning_curves_total.png"))
    plt.close()

    # Components
    plt.figure()
    plt.plot(epochs, df["train_base"].values, label="train_base")
    plt.plot(epochs, df["train_ctr"].values, label="train_ctr")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curves: Base vs Contrastive")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "02_learning_curves_components.png"))
    plt.close()

    # Optional embedding dynamics
    if set(["pos_dist","neg_dist","viol_rate"]).intersection(df.columns):
        plt.figure()
        if "pos_dist" in df.columns:
            plt.plot(epochs, df["pos_dist"].values, label="pos_dist")
        if "neg_dist" in df.columns:
            plt.plot(epochs, df["neg_dist"].values, label="neg_dist")
        if "viol_rate" in df.columns:
            plt.plot(epochs, df["viol_rate"].values, label="margin_violation_rate")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title("Embedding Dynamics")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "03_embedding_dynamics.png"))
        plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Path to training_log.csv")
    ap.add_argument("--out", default="figs", help="Output directory")
    args = ap.parse_args()
    ensure_dir(args.out)
    plot_learning_curves(args.log, args.out)
    print(f"Saved figures to: {os.path.abspath(args.out)}")

if __name__ == "__main__":
    main()
