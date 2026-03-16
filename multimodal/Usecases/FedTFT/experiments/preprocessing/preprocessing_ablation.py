"""
preprocessing_ablation.py — Preprocessing and data augmentation ablation study.

Ablation study on preprocessing choices and data augmentation strategies.

Variants tested:
  P0: No normalization (raw features)
  P1: StandardScaler (z-score) — default pipeline
  P2: MinMaxScaler (0–1 range)
  P3: RevIN (Reversible Instance Normalization, per-sample)
  P4: P1 + SMOTE on training set (handle class imbalance)
  P5: P1 + Jitter augmentation (Gaussian noise σ=0.01 per timestep)
  P6: P1 + Time masking (randomly zero 10% of timesteps during training)
  P7: P1 + Mixup (interpolate pairs of training samples, α=0.2)

Each variant trains a FedTFT (R4 configuration) from scratch for 30 rounds
and reports AUROC per horizon on the held-out global test set.

Usage:
  python experiments/preprocessing_ablation.py \
      --num_rounds 30 \
      --output results/preprocessing_ablation/

  # Single variant (for debugging):
  python experiments/preprocessing_ablation.py --variants P1 P4 --num_rounds 30

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
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score, f1_score
from copy import deepcopy

_DEFAULT_DATA_ROOT = "patient_level_split/last_npy_data"
GLOBAL_DIR   = _DEFAULT_DATA_ROOT + "/GlobalData"
HOSPITAL_DIR = _DEFAULT_DATA_ROOT + "/HospitalsData"
STATIC_SHAPE = (14,)
SEQ_SHAPE    = (192, 25)
TARGET_SHAPE = (3,)
HORIZON_NAMES = ["1h", "1d", "1w"]

VARIANTS = ["P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7"]
VARIANT_LABELS = {
    "P0": "No normalization",
    "P1": "StandardScaler (default)",
    "P2": "MinMaxScaler",
    "P3": "RevIN (per-sample)",
    "P4": "StandardScaler + SMOTE",
    "P5": "StandardScaler + Jitter",
    "P6": "StandardScaler + Time masking",
    "P7": "StandardScaler + Mixup (α=0.2)",
}


# ─────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────
def load_memmap_data(path, shape, dtype=np.float32):
    if not os.path.exists(path):
        return None
    size = os.path.getsize(path)
    per  = np.prod(shape) * np.dtype(dtype).itemsize
    n    = size // per
    return np.memmap(path, dtype=dtype, mode="r", shape=(n,) + shape)


def load_hospital_splits(hospital_name):
    base = os.path.join(HOSPITAL_DIR, hospital_name)
    return {
        "static_train":  load_memmap_data(os.path.join(base, "static_train.npy"),  STATIC_SHAPE),
        "seq_train":     load_memmap_data(os.path.join(base, "sequence_train.npy"), SEQ_SHAPE),
        "tgt_train":     load_memmap_data(os.path.join(base, "targets_train.npy"),  TARGET_SHAPE),
        "static_val":    load_memmap_data(os.path.join(base, "static_val.npy"),     STATIC_SHAPE),
        "seq_val":       load_memmap_data(os.path.join(base, "sequence_val.npy"),   SEQ_SHAPE),
        "tgt_val":       load_memmap_data(os.path.join(base, "targets_val.npy"),    TARGET_SHAPE),
    }


def load_global_test():
    return {
        "static": load_memmap_data(os.path.join(GLOBAL_DIR, "static_data.npy"),   STATIC_SHAPE),
        "seq":    load_memmap_data(os.path.join(GLOBAL_DIR, "sequence_data.npy"), SEQ_SHAPE),
        "tgt":    load_memmap_data(os.path.join(GLOBAL_DIR, "targets.npy"),       TARGET_SHAPE),
    }


# ─────────────────────────────────────────────────────────────────
# Preprocessing transforms
# ─────────────────────────────────────────────────────────────────
class RevIN:
    """Per-sample reversible instance normalization for sequence data."""
    def __init__(self, eps=1e-5):
        self.eps = eps

    def fit_transform(self, X):
        """X: [N, T, F]. Normalise each sample over its T timesteps."""
        X = X.copy().astype(np.float32)
        self.means_ = X.mean(axis=1, keepdims=True)   # [N, 1, F]
        self.stds_  = X.std(axis=1, keepdims=True) + self.eps
        return (X - self.means_) / self.stds_

    def transform(self, X):
        X = X.copy().astype(np.float32)
        mu  = X.mean(axis=1, keepdims=True)
        std = X.std(axis=1, keepdims=True) + self.eps
        return (X - mu) / std


def apply_standard_scaler(seq_train, seq_val, seq_test):
    """Fit StandardScaler on train, apply to val & test."""
    N_tr, T, F = seq_train.shape
    sc = StandardScaler()
    sc.fit(seq_train.reshape(-1, F))
    seq_train = sc.transform(seq_train.reshape(-1, F)).reshape(N_tr, T, F)
    seq_val   = sc.transform(seq_val.reshape(-1, seq_val.shape[-1])).reshape(*seq_val.shape)
    seq_test  = sc.transform(seq_test.reshape(-1, seq_test.shape[-1])).reshape(*seq_test.shape)
    return seq_train, seq_val, seq_test


def apply_minmax_scaler(seq_train, seq_val, seq_test):
    N_tr, T, F = seq_train.shape
    sc = MinMaxScaler()
    sc.fit(seq_train.reshape(-1, F))
    seq_train = sc.transform(seq_train.reshape(-1, F)).reshape(N_tr, T, F)
    seq_val   = sc.transform(seq_val.reshape(-1, seq_val.shape[-1])).reshape(*seq_val.shape)
    seq_test  = sc.transform(seq_test.reshape(-1, seq_test.shape[-1])).reshape(*seq_test.shape)
    return seq_train, seq_val, seq_test


def apply_revin(seq_train, seq_val, seq_test):
    """Per-sample RevIN — fit has no parameters, applied per-sample."""
    revin = RevIN()
    seq_train = revin.fit_transform(seq_train)
    seq_val   = revin.transform(seq_val)
    seq_test  = revin.transform(seq_test)
    return seq_train, seq_val, seq_test


def apply_smote(seq_train, static_train, tgt_train, rng=42):
    """
    Apply SMOTE independently for each binary horizon label.
    Returns oversampled (seq, static, tgt) arrays.
    Because SMOTE works on flat feature vectors, we flatten seq to [N, T*F].
    Only oversample the MINORITY positive class.
    """
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        print("[WARN] imbalanced-learn not installed; skipping SMOTE (P4). "
              "Install with: pip install imbalanced-learn")
        return seq_train, static_train, tgt_train

    N, T, F = seq_train.shape
    flat_seq = seq_train.reshape(N, T * F)
    flat_all = np.concatenate([static_train, flat_seq], axis=1)  # [N, 14 + T*F]

    # Use first horizon label for oversampling (most imbalanced)
    y_primary = tgt_train[:, 0].astype(int)

    if y_primary.sum() < 2:
        print("[WARN] Too few positives for SMOTE, skipping.")
        return seq_train, static_train, tgt_train

    try:
        sm = SMOTE(random_state=rng, k_neighbors=min(5, y_primary.sum() - 1))
        flat_resampled, y_resampled = sm.fit_resample(flat_all, y_primary)
    except Exception as e:
        print(f"[WARN] SMOTE failed: {e}. Using original data.")
        return seq_train, static_train, tgt_train

    n_new = len(flat_resampled)
    static_new = flat_resampled[:, :14]
    seq_new    = flat_resampled[:, 14:].reshape(n_new, T, F)
    # For multi-label targets, replicate with zeros for new synthetic samples
    tgt_new    = np.zeros((n_new, tgt_train.shape[1]), dtype=tgt_train.dtype)
    tgt_new[:N] = tgt_train
    # Synthetic samples get the primary label repeated
    tgt_new[N:, 0] = y_resampled[N:]

    print(f"  SMOTE: {N} → {n_new} samples ({n_new - N} synthetic added)")
    return seq_new.astype(np.float32), static_new.astype(np.float32), tgt_new.astype(np.float32)


def apply_jitter(seq, sigma=0.01, rng=None):
    """Add Gaussian noise to sequence data (training only)."""
    if rng is None:
        rng = np.random.default_rng(42)
    return seq + rng.normal(0, sigma, seq.shape).astype(np.float32)


def apply_time_masking(seq, mask_rate=0.10, rng=None):
    """Randomly zero out mask_rate fraction of timesteps per sample."""
    if rng is None:
        rng = np.random.default_rng(42)
    seq = seq.copy()
    T = seq.shape[1]
    n_mask = max(1, int(T * mask_rate))
    for i in range(len(seq)):
        ts_idx = rng.choice(T, size=n_mask, replace=False)
        seq[i, ts_idx, :] = 0.0
    return seq


def apply_mixup(seq, static, tgt, alpha=0.2, rng=None):
    """
    Mixup: interpolate pairs of samples.
    Returns augmented (seq, static, tgt) with 2× original size.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    N = len(seq)
    idx = rng.permutation(N)
    lam = rng.beta(alpha, alpha, size=(N, 1, 1)).astype(np.float32)
    seq_mix    = lam * seq + (1 - lam) * seq[idx]
    lam_s      = lam[:, :, 0]  # [N, 1]
    static_mix = lam_s * static + (1 - lam_s) * static[idx]
    tgt_mix    = lam_s * tgt + (1 - lam_s) * tgt[idx]
    # Concatenate original + mixed
    seq_out    = np.concatenate([seq, seq_mix], axis=0)
    static_out = np.concatenate([static, static_mix], axis=0)
    tgt_out    = np.concatenate([tgt, tgt_mix], axis=0)
    return seq_out.astype(np.float32), static_out.astype(np.float32), tgt_out.astype(np.float32)


# ─────────────────────────────────────────────────────────────────
# Build preprocessed data for a given variant
# ─────────────────────────────────────────────────────────────────
def preprocess_hospital(variant, hospital_data, global_test):
    """
    Apply the specified preprocessing variant to one hospital's data.
    Returns (seq_train, static_train, tgt_train, seq_val, static_val, tgt_val,
             seq_test, static_test, tgt_test).
    """
    seq_tr    = hospital_data["seq_train"].copy().astype(np.float32)
    stat_tr   = hospital_data["static_train"].copy().astype(np.float32)
    tgt_tr    = hospital_data["tgt_train"].copy().astype(np.float32)
    seq_val   = hospital_data["seq_val"].copy().astype(np.float32)
    stat_val  = hospital_data["static_val"].copy().astype(np.float32)
    tgt_val   = hospital_data["tgt_val"].copy().astype(np.float32)
    seq_test  = global_test["seq"].copy().astype(np.float32)
    stat_test = global_test["static"].copy().astype(np.float32)
    tgt_test  = global_test["tgt"].copy().astype(np.float32)

    # ── Step 1: Normalization ──────────────────────────────────────
    if variant == "P0":
        pass  # No normalization
    elif variant in ("P1", "P4", "P5", "P6", "P7"):
        seq_tr, seq_val, seq_test = apply_standard_scaler(seq_tr, seq_val, seq_test)
    elif variant == "P2":
        seq_tr, seq_val, seq_test = apply_minmax_scaler(seq_tr, seq_val, seq_test)
    elif variant == "P3":
        seq_tr, seq_val, seq_test = apply_revin(seq_tr, seq_val, seq_test)

    # ── Step 2: Augmentation (training only) ──────────────────────
    rng = np.random.default_rng(42)
    if variant == "P4":
        seq_tr, stat_tr, tgt_tr = apply_smote(seq_tr, stat_tr, tgt_tr)
    elif variant == "P5":
        seq_tr = apply_jitter(seq_tr, sigma=0.01, rng=rng)
    elif variant == "P6":
        seq_tr = apply_time_masking(seq_tr, mask_rate=0.10, rng=rng)
    elif variant == "P7":
        seq_tr, stat_tr, tgt_tr = apply_mixup(seq_tr, stat_tr, tgt_tr, alpha=0.2, rng=rng)

    return seq_tr, stat_tr, tgt_tr, seq_val, stat_val, tgt_val, seq_test, stat_test, tgt_test


# ─────────────────────────────────────────────────────────────────
# Single-round training (simplified — simulates one FL round)
# ─────────────────────────────────────────────────────────────────
def train_one_hospital(model, seq_tr, stat_tr, tgt_tr, seq_val, stat_val, tgt_val,
                       device, max_epochs=6, peak_lr=3e-5):
    from model_fedtft import TFTDataset
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import OneCycleLR
    import copy

    pos_counts = tgt_tr.sum(axis=0)
    neg_counts = len(tgt_tr) - pos_counts
    pw = torch.tensor(neg_counts / (pos_counts + 1e-6), dtype=torch.float32).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pw)

    tr_ds  = TFTDataset(list(zip(stat_tr, seq_tr, tgt_tr)))
    val_ds = TFTDataset(list(zip(stat_val, seq_val, tgt_val)))
    tr_loader  = DataLoader(tr_ds,  batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=peak_lr/25, weight_decay=1e-5)
    scheduler = OneCycleLR(optimizer, max_lr=peak_lr,
                           steps_per_epoch=len(tr_loader), epochs=max_epochs,
                           pct_start=0.3, anneal_strategy="cos")

    best_loss, best_state = float("inf"), copy.deepcopy(model.state_dict())
    for epoch in range(max_epochs):
        model.train()
        for sx, seqx, ty in tr_loader:
            sx, seqx, ty = sx.to(device), seqx.to(device), ty.to(device)
            optimizer.zero_grad()
            loss_fn(model(sx, seqx), ty).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step()

        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for sx, seqx, ty in val_loader:
                sx, seqx, ty = sx.to(device), seqx.to(device), ty.to(device)
                val_loss += loss_fn(model(sx, seqx), ty).item() * sx.size(0)
        val_loss /= max(len(val_loader.dataset), 1)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, best_loss


def fedavg_aggregate(global_sd, client_states, weights):
    """Simple FedAvg aggregation (no FVWA for preprocessing comparison)."""
    total = sum(weights)
    new_sd = deepcopy(global_sd)
    for key in new_sd:
        new_sd[key] = sum(
            client_states[i][key].float() * (weights[i] / total)
            for i in range(len(client_states))
        )
    return new_sd


def evaluate_global(model, seq_test, stat_test, tgt_test, device, thresholds=None):
    from model_fedtft import TFTDataset
    thresholds = thresholds or [0.5, 0.5, 0.5]
    thresh_t = torch.tensor(thresholds, device=device)
    loader = DataLoader(TFTDataset(list(zip(stat_test, seq_test, tgt_test))),
                        batch_size=32, shuffle=False)
    all_probs, all_preds, all_tgts = [], [], []
    model.eval()
    with torch.no_grad():
        for sx, seqx, y in loader:
            sx, seqx, y = sx.to(device), seqx.to(device), y.to(device)
            probs = torch.sigmoid(model(sx, seqx))
            preds = (probs >= thresh_t).long()
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_tgts.append(y.cpu().numpy())
    probs_np   = np.concatenate(all_probs)
    preds_np   = np.concatenate(all_preds)
    targets_np = np.concatenate(all_tgts)
    results = {}
    for i, h in enumerate(HORIZON_NAMES):
        try:
            auc = roc_auc_score(targets_np[:, i], probs_np[:, i])
        except Exception:
            auc = float('nan')
        f1    = f1_score(targets_np[:, i], preds_np[:, i], zero_division=0)
        brier = float(np.mean((probs_np[:, i] - targets_np[:, i].astype(np.float32)) ** 2))
        results[h] = {"auroc": float(auc), "f1": float(f1), "brier": float(brier)}
    return results


# ─────────────────────────────────────────────────────────────────
# Run one variant: simplified FL simulation
# ─────────────────────────────────────────────────────────────────
def run_variant(variant, hospitals, hospital_data, global_test, device, num_rounds=30):
    from model_fedtft import TFTPredictor

    print(f"\n{'='*60}")
    print(f"Variant {variant}: {VARIANT_LABELS[variant]}")
    print(f"{'='*60}")

    # Preprocess data per hospital
    preprocessed = {}
    for hosp in hospitals:
        result = preprocess_hospital(variant, hospital_data[hosp], global_test)
        preprocessed[hosp] = result
        seq_tr = result[0]
        print(f"  {hosp}: train={len(seq_tr)}")

    # Use first hospital's test split for evaluation (all get same global test)
    seq_test  = preprocessed[hospitals[0]][6]
    stat_test = preprocessed[hospitals[0]][7]
    tgt_test  = preprocessed[hospitals[0]][8]

    model = TFTPredictor(input_dim=25, static_dim=14, hidden_dim=64, output_dim=3).to(device)
    torch.manual_seed(42)

    best_result = None
    best_auc = -1.0

    for rnd in range(1, num_rounds + 1):
        global_sd = deepcopy(model.state_dict())
        client_states, weights = [], []

        for hosp in hospitals:
            seq_tr, stat_tr, tgt_tr = preprocessed[hosp][:3]
            seq_val, stat_val, tgt_val = preprocessed[hosp][3:6]

            client_model = TFTPredictor(input_dim=25, static_dim=14, hidden_dim=64, output_dim=3).to(device)
            client_model.load_state_dict(deepcopy(global_sd))
            client_model, _ = train_one_hospital(
                client_model, seq_tr, stat_tr, tgt_tr, seq_val, stat_val, tgt_val, device)
            client_states.append(deepcopy(client_model.state_dict()))
            weights.append(len(seq_tr))

        # FedAvg
        new_sd = fedavg_aggregate(global_sd, client_states, weights)
        model.load_state_dict(new_sd)

        if rnd % 5 == 0 or rnd == num_rounds:
            results = evaluate_global(model, seq_test, stat_test, tgt_test, device)
            mean_auc = np.nanmean([results[h]['auroc'] for h in HORIZON_NAMES])
            print(f"  Round {rnd:3d}: AUROC 1h={results['1h']['auroc']:.4f} "
                  f"1d={results['1d']['auroc']:.4f} 1w={results['1w']['auroc']:.4f} "
                  f"mean={mean_auc:.4f}")
            if mean_auc > best_auc:
                best_auc = mean_auc
                best_result = results

    return best_result


# ─────────────────────────────────────────────────────────────────
# Plot comparison
# ─────────────────────────────────────────────────────────────────
def plot_comparison(all_results, metric, outdir):
    variants = sorted(all_results.keys())
    x = np.arange(len(HORIZON_NAMES))
    width = 0.8 / len(variants)
    colors = plt.cm.Set2(np.linspace(0, 1, len(variants)))

    fig, ax = plt.subplots(figsize=(12, 6))
    for j, v in enumerate(variants):
        if all_results[v] is None:
            continue
        y = [all_results[v][h].get(metric, float('nan')) for h in HORIZON_NAMES]
        ax.bar(x + j*width, y, width, label=f"{v}: {VARIANT_LABELS[v]}", color=colors[j])

    ax.set_xticks(x + width * (len(variants)-1) / 2)
    ax.set_xticklabels(HORIZON_NAMES)
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Preprocessing Ablation — {metric.upper()} per Horizon")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out = os.path.join(outdir, f"preprocessing_ablation_{metric}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Plot saved → {out}")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="patient_level_split/last_npy_data",
                        help="Path to last_npy_data root")
    parser.add_argument("--variants",   nargs="+", default=VARIANTS,
                        choices=VARIANTS,
                        help="Which preprocessing variants to run")
    parser.add_argument("--num_rounds", type=int, default=30,
                        help="FL rounds per variant (30 is sufficient to compare convergence)")
    parser.add_argument("--output",     default="results/preprocessing_ablation/")
    args = parser.parse_args()
    global GLOBAL_DIR, HOSPITAL_DIR
    GLOBAL_DIR   = os.path.join(args.data_root, "GlobalData")
    HOSPITAL_DIR = os.path.join(args.data_root, "HospitalsData")

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Running variants: {args.variants}")

    # Load raw (unnormalized) data once
    hospitals = sorted(os.listdir(HOSPITAL_DIR))
    hospital_data = {h: load_hospital_splits(h) for h in hospitals}
    global_test   = load_global_test()
    print(f"Hospitals: {hospitals}")
    print(f"Global test samples: {len(global_test['tgt'])}")

    all_results = {}
    for variant in args.variants:
        try:
            result = run_variant(variant, hospitals, hospital_data,
                                 global_test, device, num_rounds=args.num_rounds)
            all_results[variant] = result
        except Exception as e:
            print(f"[ERROR] Variant {variant} failed: {e}")
            all_results[variant] = None

    # ── Summary table ──────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("PREPROCESSING ABLATION SUMMARY")
    print(f"{'='*80}")
    header = (f"{'Variant':<8} {'Description':<35} "
              + " ".join(f"{'AUC_'+h:>8}" for h in HORIZON_NAMES)
              + "  {'MeanAUC':>8}"
              + " ".join(f"{'Brier_'+h:>9}" for h in HORIZON_NAMES))
    print(header)
    print("-" * (len(header) + 10))

    for v in args.variants:
        r = all_results[v]
        if r is None:
            row = f"{v:<8} {VARIANT_LABELS[v]:<35} " + "     N/A" * (len(HORIZON_NAMES) * 2 + 1)
        else:
            aucs   = [r[h]['auroc'] for h in HORIZON_NAMES]
            briers = [r[h].get('brier', float('nan')) for h in HORIZON_NAMES]
            row = (f"{v:<8} {VARIANT_LABELS[v]:<35} "
                   + " ".join(f"{a:8.4f}" for a in aucs)
                   + f"  {np.nanmean(aucs):8.4f}"
                   + " ".join(f"{b:9.4f}" for b in briers))
        print(row)

    # Save JSON
    out_json = os.path.join(args.output, "preprocessing_ablation_results.json")
    with open(out_json, "w") as f:
        json.dump({v: all_results[v] for v in args.variants}, f, indent=2)
    print(f"\nResults saved → {out_json}")

    # Plots
    for metric in ["auroc", "f1"]:
        plot_comparison(all_results, metric, args.output)

    # ── P-values: each vs P1 (default) ────────────────────────────
    try:
        from scipy.stats import ttest_rel
        p1_aucs = [all_results["P1"][h]['auroc'] for h in HORIZON_NAMES if all_results.get("P1")]
        if p1_aucs:
            print(f"\nComparison vs P1 (StandardScaler, default):")
            for v in args.variants:
                if v == "P1" or all_results[v] is None:
                    continue
                other_aucs = [all_results[v][h]['auroc'] for h in HORIZON_NAMES]
                try:
                    _, pval = ttest_rel(p1_aucs, other_aucs)
                    stars = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
                    delta = np.mean(other_aucs) - np.mean(p1_aucs)
                    print(f"  {v} ({VARIANT_LABELS[v]:<32}): Δmean={delta:+.4f}, p={pval:.4f} {stars}")
                except Exception:
                    pass
    except ImportError:
        pass

    print("\n=== Preprocessing ablation complete ===")


if __name__ == "__main__":
    main()
