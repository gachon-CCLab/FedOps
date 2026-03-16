"""
missingness_robustness.py — Simulate missing data and measure performance degradation.

Two types of missingness:
  1. Feature missingness: randomly zero-out a fraction of sequence timesteps
  2. Client dropout:      simulate a hospital going offline (only 2/3 hospitals train)

Produces:
  - Table: AUROC × missingness rate (0%, 10%, 20%, 30%) per horizon
  - Plot:  AUROC vs missingness rate curves

Usage:
  python experiments/missingness_robustness.py \
    --checkpoint checkpoints/fedtft_best.pth \
    --rates 0.0 0.1 0.2 0.3 \
    --output results/significance/missingness.json

Run from FedTFT_paper/ root directory.
"""

import os
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score

_DEFAULT_DATA_ROOT = "patient_level_split/last_npy_data"
GLOBAL_DIR    = _DEFAULT_DATA_ROOT + "/GlobalData"
HOSPITAL_DIR  = _DEFAULT_DATA_ROOT + "/HospitalsData"
STATIC_SHAPE  = (14,)
SEQ_SHAPE     = (192, 25)
TARGET_SHAPE  = (3,)
HORIZON_NAMES = ["1h", "1d", "1w"]


def load_memmap_data(path, shape, dtype=np.float32):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    size = os.path.getsize(path)
    per  = np.prod(shape) * np.dtype(dtype).itemsize
    n    = size // per
    return np.memmap(path, dtype=dtype, mode="r", shape=(n,) + shape)


def apply_feature_missingness(seq_data, rate, seed=42):
    """
    Randomly zero-out `rate` fraction of timesteps for each sample.
    Returns a new numpy array (copy).
    """
    rng = np.random.default_rng(seed)
    seq = seq_data.copy().astype(np.float32)
    T   = seq.shape[1]
    n_mask = int(T * rate)
    if n_mask == 0:
        return seq
    for i in range(len(seq)):
        ts_idx = rng.choice(T, size=n_mask, replace=False)
        seq[i, ts_idx, :] = 0.0
    return seq


def evaluate_model(model, device, static_data, seq_data, targets_data,
                   thresholds, batch_size=32):
    from model_fedtft import TFTDataset
    loader = DataLoader(
        TFTDataset(list(zip(static_data, seq_data, targets_data))),
        batch_size=batch_size, shuffle=False)
    thresh_t = torch.tensor(thresholds, device=device)
    all_probs, all_preds, all_targets = [], [], []
    model.eval()
    with torch.no_grad():
        for sx, seqx, y in loader:
            sx, seqx, y = sx.to(device), seqx.to(device), y.to(device)
            probs  = torch.sigmoid(model(sx, seqx))
            preds  = (probs >= thresh_t).long()
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    probs_np   = np.concatenate(all_probs)
    preds_np   = np.concatenate(all_preds)
    targets_np = np.concatenate(all_targets)
    results = {}
    for i, h in enumerate(HORIZON_NAMES):
        try:
            auc = roc_auc_score(targets_np[:, i], probs_np[:, i])
        except Exception:
            auc = float('nan')
        f1  = f1_score(targets_np[:, i], preds_np[:, i], zero_division=0)
        tp  = int(((preds_np[:,i]==1)&(targets_np[:,i]==1)).sum())
        tn  = int(((preds_np[:,i]==0)&(targets_np[:,i]==0)).sum())
        fp  = int(((preds_np[:,i]==1)&(targets_np[:,i]==0)).sum())
        fn  = int(((preds_np[:,i]==0)&(targets_np[:,i]==1)).sum())
        spec = tn / (tn + fp + 1e-8)
        results[h] = {"auroc": float(auc), "f1": float(f1), "specificity": float(spec)}
    return results


def plot_missingness(all_results, rates, metric, outdir):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors  = ["#2196F3", "#4CAF50", "#F44336"]
    for h_idx, horizon in enumerate(HORIZON_NAMES):
        y = [all_results[r][horizon].get(metric, float('nan')) for r in rates]
        ax.plot([r*100 for r in rates], y, marker="o",
                label=horizon, color=colors[h_idx])
    ax.set_xlabel("Missingness rate (%)")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Robustness to Feature Missingness — {metric.upper()}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks([r*100 for r in rates])
    plt.tight_layout()
    out = os.path.join(outdir, f"missingness_{metric}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",  default="patient_level_split/last_npy_data",
                        help="Path to last_npy_data root")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rates", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3])
    parser.add_argument("--thresholds_json", default=None,
                        help="JSON file with [t0,t1,t2] thresholds")
    parser.add_argument("--output", default="results/significance/missingness.json")
    parser.add_argument("--hidden_dim", type=int, default=64)
    args = parser.parse_args()
    global GLOBAL_DIR, HOSPITAL_DIR
    GLOBAL_DIR   = os.path.join(args.data_root, "GlobalData")
    HOSPITAL_DIR = os.path.join(args.data_root, "HospitalsData")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from model_fedtft import TFTPredictor
    model = TFTPredictor(
        input_dim=25, static_dim=14, hidden_dim=args.hidden_dim, output_dim=3
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Thresholds
    if args.thresholds_json and os.path.exists(args.thresholds_json):
        with open(args.thresholds_json) as f:
            thresholds = json.load(f)
    else:
        thresholds = [0.5, 0.5, 0.5]
    print(f"Thresholds: {thresholds}")

    # Load test data
    static_data  = load_memmap_data(os.path.join(GLOBAL_DIR, "static_data.npy"), STATIC_SHAPE)
    seq_data     = load_memmap_data(os.path.join(GLOBAL_DIR, "sequence_data.npy"), SEQ_SHAPE)
    targets_data = load_memmap_data(os.path.join(GLOBAL_DIR, "targets.npy"), TARGET_SHAPE)
    print(f"Test samples: {len(static_data)}")

    # ── 1) Feature missingness sweep ───────────────────────────────
    print("\n=== Feature Missingness Sweep ===")
    print(f"{'Rate':>6}  " + "  ".join(f"AUC_{h:>3}" for h in HORIZON_NAMES))
    print("-" * 40)

    all_results = {}
    for rate in args.rates:
        seq_masked = apply_feature_missingness(seq_data, rate, seed=42)
        results    = evaluate_model(model, device, static_data, seq_masked,
                                    targets_data, thresholds)
        all_results[rate] = results
        auc_str = "  ".join(f"{results[h]['auroc']:.4f}" for h in HORIZON_NAMES)
        print(f"  {rate*100:4.0f}%  {auc_str}")

    print("\nResults (JSON save disabled):")
    print({str(k): v for k, v in all_results.items()})

    # Plots
    outdir = os.path.dirname(args.output)
    for metric in ["auroc", "f1", "specificity"]:
        plot_missingness(all_results, args.rates, metric, outdir)

    # ── 2) Summary table ───────────────────────────────────────────
    print("\n=== Robustness Summary Table ===")
    print(f"{'Horizon':<6} {'0%':>8} {'10%':>8} {'20%':>8} {'30%':>8}  {'Δ (0%→30%)':>12}")
    print("-" * 55)
    for h in HORIZON_NAMES:
        vals = [all_results[r][h]['auroc'] for r in args.rates]
        delta = vals[-1] - vals[0] if not np.isnan(vals[-1]) else float('nan')
        row = f"{h:<6} " + " ".join(f"{v:8.4f}" for v in vals)
        row += f"  {delta:+12.4f}"
        print(row)


if __name__ == "__main__":
    main()
