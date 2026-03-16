"""
compute_ablation_ci.py — Bootstrap AUROC 95% CI for R1–R4 ablation checkpoints,
Row A (FedAvg + decoupled heads, controlled A/B), and SoTA baselines.

Ablation rows (Table III):
  R1: FedAvg  + fixed mu=0   + shared head     → model_fedtft.py
  R2: FedProx + fixed mu=1e-5 + shared head    → model_fedtft.py
  R3: FedProx + FVWA          + shared head    → model_fedtft.py
  R4: FedProx + FVWA          + decoupled heads (FedTFT) → model_fedtft_hdfp.py
   A: FedAvg  + fixed mu=0   + decoupled heads (A/B)    → model_fedtft_hdfp.py

Outputs: results/ablation_ci.json
Run from FedTFT_paper/ root.
"""
import os, sys, json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.abspath("."))

_DEFAULT_DATA_ROOT = "patient_level_split/last_npy_data"
GLOBAL_DIR   = _DEFAULT_DATA_ROOT + "/GlobalData"
STATIC_SHAPE = (14,)
SEQ_SHAPE    = (192, 25)
TARGET_SHAPE = (3,)
HORIZONS     = ["1h", "1d", "1w"]
N_BOOT       = 2000
SEED         = 42

def load_memmap(path, shape, dtype=np.float32):
    size = os.path.getsize(path)
    per  = np.prod(shape) * np.dtype(dtype).itemsize
    n    = size // per
    return np.memmap(path, dtype=dtype, mode="r", shape=(n,) + shape)

def bootstrap_auroc_ci(probs, targets, n_boot=N_BOOT, seed=SEED):
    """Returns (mean_auroc, ci_low, ci_high) per horizon, then mean across horizons."""
    rng = np.random.default_rng(seed)
    n   = len(targets)
    results = {}
    for i, h in enumerate(HORIZONS):
        pt  = targets[:, i]
        pp  = probs[:, i]
        try:
            point = roc_auc_score(pt, pp)
        except Exception:
            results[h] = {"point": float("nan"), "ci_95": [float("nan"), float("nan")]}
            continue
        boot_aucs = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            try:
                boot_aucs.append(roc_auc_score(pt[idx], pp[idx]))
            except Exception:
                pass
        boot_aucs = np.array(boot_aucs)
        results[h] = {
            "point": float(point),
            "ci_95": [float(np.percentile(boot_aucs, 2.5)),
                      float(np.percentile(boot_aucs, 97.5))]
        }
    # Mean across horizons
    mean_pt = float(np.nanmean([results[h]["point"] for h in HORIZONS]))
    mean_lo = float(np.nanmean([results[h]["ci_95"][0] for h in HORIZONS]))
    mean_hi = float(np.nanmean([results[h]["ci_95"][1] for h in HORIZONS]))
    results["mean"] = {"point": mean_pt, "ci_95": [mean_lo, mean_hi]}
    return results

def get_probs_from_checkpoint(ckpt_path, device, use_hdfp=False):
    """Load checkpoint and run inference on global test set.

    use_hdfp=True  → model_fedtft_hdfp.py (TFTPredictor_FedTFT, horizon-decoupled heads) — R4, Row A
    use_hdfp=False → model_fedtft.py       (TFTPredictor, shared head)                   — R1, R2, R3
    """
    if use_hdfp:
        from model_fedtft_hdfp import TFTPredictor_FedTFT as ModelCls, TFTDataset
        model = ModelCls(input_dim=25, static_dim=14, hidden_dim=64).to(device)
    else:
        from model_fedtft import TFTPredictor as ModelCls, TFTDataset
        model = ModelCls(input_dim=25, static_dim=14, hidden_dim=64, output_dim=3).to(device)

    sd = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(sd, strict=False)
    model.eval()

    static  = load_memmap(os.path.join(GLOBAL_DIR, "static_data.npy"),   STATIC_SHAPE)
    seq     = load_memmap(os.path.join(GLOBAL_DIR, "sequence_data.npy"), SEQ_SHAPE)
    targets = load_memmap(os.path.join(GLOBAL_DIR, "targets.npy"),       TARGET_SHAPE)

    ds     = TFTDataset(list(zip(static, seq, targets)))
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    probs_list = []
    with torch.no_grad():
        for sx, seqx, _ in loader:
            sx, seqx = sx.to(device), seqx.to(device)
            probs_list.append(torch.sigmoid(model(sx, seqx)).cpu().numpy())
    probs = np.concatenate(probs_list)
    return probs, np.array(targets)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="patient_level_split/last_npy_data",
                        help="Path to last_npy_data root")
    args = parser.parse_args()
    global GLOBAL_DIR
    GLOBAL_DIR = os.path.join(args.data_root, "GlobalData")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_ci = {}

    # ── R1–R4 + Row A from checkpoints ─────────────────────────────
    # (label, filename, use_hdfp)
    # R1-R3: shared head       → model_fedtft.py
    # R4, A: decoupled heads   → model_fedtft_hdfp.py
    ablation_checkpoints = [
        ("R1", "ablation_R1_fedavg_best.pth",          False),
        ("R2", "ablation_R2_fedprox_best.pth",         False),
        ("R3", "ablation_R3_fvwa_best.pth",            False),
        ("R4", "ablation_R4_fedtft_best.pth",          True),
        ("A",  "ablation_A_fedavg_decoupled_best.pth", True),
    ]

    targets_global = None

    for label, fname, use_hdfp in ablation_checkpoints:
        ckpt = os.path.join(GLOBAL_DIR, fname)
        print(f"\nProcessing {label}: {ckpt}")
        if not os.path.exists(ckpt):
            print(f"  MISSING: {ckpt}")
            continue
        try:
            probs, targets = get_probs_from_checkpoint(ckpt, device, use_hdfp=use_hdfp)
            if targets_global is None:
                targets_global = targets
            ci = bootstrap_auroc_ci(probs, targets)
            all_ci[label] = ci
            mean = ci["mean"]
            print(f"  Mean AUROC={mean['point']:.4f} CI=[{mean['ci_95'][0]:.4f},{mean['ci_95'][1]:.4f}]")
            for h in HORIZONS:
                print(f"    {h}: {ci[h]['point']:.4f} [{ci[h]['ci_95'][0]:.4f},{ci[h]['ci_95'][1]:.4f}]")
        except Exception as e:
            print(f"  ERROR: {e}")

    # ── SoTA baselines from prediction files ────────────────────────
    pred_files = {
        "FEDFormer":    "results/predictions/fedformer_preds.npz",
        "iTransformer": None,  # not available
        "PatchTST":     "results/predictions/patchtst_preds.npz",
        "DLinear":      "results/predictions/dlinear_preds.npz",
    }

    for label, fpath in pred_files.items():
        if fpath is None or not os.path.exists(fpath):
            print(f"\n{label}: prediction file not found, skipping")
            continue
        print(f"\nProcessing {label}: {fpath}")
        d = np.load(fpath)
        probs   = d["probs"]
        targets = d["targets"]
        ci = bootstrap_auroc_ci(probs, targets)
        all_ci[label] = ci
        mean = ci["mean"]
        print(f"  Mean AUROC={mean['point']:.4f} CI=[{mean['ci_95'][0]:.4f},{mean['ci_95'][1]:.4f}]")

    print("\nAblation CI results (JSON save disabled):")
    print(all_ci)

if __name__ == "__main__":
    main()
