"""
sensitivity_analysis.py — Sensitivity to sequence length and model capacity.

Experiments:
  1. Sequence length: 96 / 192 / 288 timesteps
  2. Hidden dimension: 32 / 64 / 128
  3. Dropout: 0.05 / 0.1 / 0.2

For each config, loads the global test set, runs inference with a re-trained
checkpoint (if available), and reports AUROC per horizon.

This is a POST-HOC analysis script — it reads already-trained checkpoints
from results/sensitivity/<config_name>/checkpoint.pth

Usage:
  python experiments/sensitivity_analysis.py --mode seq_len
  python experiments/sensitivity_analysis.py --mode hidden_dim
  python experiments/sensitivity_analysis.py --mode all
  python experiments/sensitivity_analysis.py --plot_only  # just plot existing results
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

_DEFAULT_DATA_ROOT = "patient_level_split/last_npy_data"
GLOBAL_DIR     = _DEFAULT_DATA_ROOT + "/GlobalData"
STATIC_SHAPE   = (14,)
TARGETS_SHAPE  = (3,)
HORIZON_NAMES  = ["1h", "1d", "1w"]
RESULT_BASE    = "results/sensitivity"


def load_memmap_data(path, shape, dtype=np.float32):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    size = os.path.getsize(path)
    per  = np.prod(shape) * np.dtype(dtype).itemsize
    n    = size // per
    return np.memmap(path, dtype=dtype, mode="r", shape=(n,) + shape)


def evaluate_checkpoint(checkpoint_path, seq_len, hidden_dim, thresholds=None):
    """
    Load a trained checkpoint and evaluate on the global test set.
    Returns dict with auroc, f1, spec per horizon.
    """
    from sklearn.metrics import roc_auc_score, f1_score
    from model_fedtft import TFTPredictor, TFTDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = TFTPredictor(
        input_dim=25, static_dim=STATIC_SHAPE[-1],
        hidden_dim=hidden_dim, output_dim=3).to(device)

    if not os.path.exists(checkpoint_path):
        print(f"  [SKIP] Checkpoint not found: {checkpoint_path}")
        return None

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    seq_shape  = (seq_len, 25)
    static_t   = load_memmap_data(os.path.join(GLOBAL_DIR, "static_data.npy"), STATIC_SHAPE)
    seq_t      = load_memmap_data(os.path.join(GLOBAL_DIR, "sequence_data.npy"), (192, 25))
    targets    = load_memmap_data(os.path.join(GLOBAL_DIR, "targets.npy"), TARGETS_SHAPE)

    # Truncate / pad sequence to target length
    if seq_len < 192:
        seq_t = seq_t[:, -seq_len:, :]   # take last seq_len timesteps
    elif seq_len > 192:
        pad  = np.zeros((len(seq_t), seq_len-192, 25), dtype=seq_t.dtype)
        seq_t = np.concatenate([pad, seq_t], axis=1)

    thresholds = thresholds or [0.5, 0.5, 0.5]
    thresh_t   = torch.tensor(thresholds, device=device)

    loader = DataLoader(TFTDataset(list(zip(static_t, seq_t, targets))), batch_size=32)

    all_probs, all_preds, all_targets = [], [], []
    with torch.no_grad():
        for sx, seqx, y in loader:
            sx, seqx, y = sx.to(device), seqx.to(device), y.to(device)
            logits = model(sx, seqx)
            probs  = torch.sigmoid(logits)
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
        tp  = int(((preds_np[:, i]==1)&(targets_np[:, i]==1)).sum())
        tn  = int(((preds_np[:, i]==0)&(targets_np[:, i]==0)).sum())
        fp  = int(((preds_np[:, i]==1)&(targets_np[:, i]==0)).sum())
        fn  = int(((preds_np[:, i]==0)&(targets_np[:, i]==1)).sum())
        spec = tn / (tn + fp + 1e-8)
        results[h] = {"auroc": auc, "f1": f1, "specificity": spec}
    return results


def sensitivity_sweep(param_name, param_values, default_seq_len=192, default_hid=64):
    """Run sweep over a single parameter, others fixed at default."""
    all_configs = []
    for v in param_values:
        if param_name == "seq_len":
            config = {"seq_len": v, "hidden_dim": default_hid}
            label  = f"seqlen{v}"
        elif param_name == "hidden_dim":
            config = {"seq_len": default_seq_len, "hidden_dim": v}
            label  = f"hid{v}"
        else:
            raise ValueError(param_name)

        ckpt_path = os.path.join(RESULT_BASE, param_name, label, "checkpoint.pth")
        out_dir   = os.path.join(RESULT_BASE, param_name, label)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n  {param_name}={v} → {ckpt_path}")
        results = evaluate_checkpoint(
            ckpt_path, config["seq_len"], config["hidden_dim"])
        if results is None:
            print(f"  [WARN] No checkpoint for {label}, skipping.")
            results = {h: {"auroc": None, "f1": None, "specificity": None}
                       for h in HORIZON_NAMES}

        all_configs.append({"label": label, "value": v, "metrics": results})
    return all_configs


def plot_sensitivity(param_name, configs, metric="auroc", outdir=None):
    outdir = outdir or os.path.join(RESULT_BASE, param_name)
    os.makedirs(outdir, exist_ok=True)

    x_vals  = [c["value"] for c in configs]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors  = ["#2196F3", "#4CAF50", "#F44336"]

    for h_idx, horizon in enumerate(HORIZON_NAMES):
        y_vals = []
        for c in configs:
            m = c["metrics"].get(horizon, {})
            v = m.get(metric)
            y_vals.append(v if v is not None else float('nan'))
        ax.plot(x_vals, y_vals, marker="o", label=horizon, color=colors[h_idx])

    ax.set_xlabel(param_name.replace("_", " ").title())
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Sensitivity to {param_name.replace('_', ' ').title()} — {metric.upper()}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if param_name == "seq_len":
        ax.set_xticks(x_vals)
        ax.set_xticklabels([str(v) for v in x_vals])
    plt.tight_layout()
    out = os.path.join(outdir, f"sensitivity_{param_name}_{metric}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="patient_level_split/last_npy_data",
                        help="Path to last_npy_data root")
    parser.add_argument("--mode", default="all",
                        choices=["seq_len", "hidden_dim", "all"])
    parser.add_argument("--metric", default="auroc")
    parser.add_argument("--plot_only", action="store_true")
    args = parser.parse_args()
    global GLOBAL_DIR
    GLOBAL_DIR = os.path.join(args.data_root, "GlobalData")

    print("=== Sensitivity Analysis ===")

    if args.mode in ("seq_len", "all"):
        seq_len_vals = [96, 192, 288]
        print(f"\n--- Sequence length sweep: {seq_len_vals} ---")  # 96 / 192 (default) / 288
        if args.plot_only:
            print("  [INFO] --plot_only fallback: evaluating checkpoints directly (no JSON cache).")
        configs = sensitivity_sweep("seq_len", seq_len_vals)
        plot_sensitivity("seq_len", configs, metric=args.metric)

    if args.mode in ("hidden_dim", "all"):
        hid_vals = [32, 64, 128]
        print(f"\n--- Hidden dim sweep: {hid_vals} ---")
        if args.plot_only:
            print("  [INFO] --plot_only fallback: evaluating checkpoints directly (no JSON cache).")
        configs = sensitivity_sweep("hidden_dim", hid_vals)
        plot_sensitivity("hidden_dim", configs, metric=args.metric)

    print("\nSensitivity analysis complete.")


if __name__ == "__main__":
    main()
