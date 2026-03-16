"""
heterogeneity_analysis.py — Robustness under data heterogeneity and FL settings.


Experiments:
  1. Partial participation: vary fraction of hospitals participating each round (1/3, 2/3, 3/3).
  2. Client imbalance: vary ratio of training sizes (1:1:1, 1:2:5, 1:5:20).
     Warm-start from R4 checkpoint, 10 fine-tuning rounds.
  3. Label heterogeneity (IID vs non-IID): sub-sample positive-class fraction per hospital.
     Warm-start from R4 checkpoint, 10 fine-tuning rounds.
  4. Round count sensitivity: 5, 10, 20, 30 fine-tuning rounds from R4 warm-start.

Metrics: AUROC (mean over horizons), F1, Brier — global hold-out test set.

Usage:
  python experiments/analysis/heterogeneity_analysis.py \\
      --checkpoint patient_level_split/last_npy_data/GlobalData/ablation_R4_fedtft_best.pth \\
      --output results/heterogeneity/ \\
      --num_rounds 10

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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score
from copy import deepcopy

_DEFAULT_DATA_ROOT = "patient_level_split/last_npy_data"
HOSPITAL_DIR  = _DEFAULT_DATA_ROOT + "/HospitalsData"
GLOBAL_DIR    = _DEFAULT_DATA_ROOT + "/GlobalData"
STATIC_SHAPE  = (14,)
SEQ_SHAPE     = (192, 25)
TARGET_SHAPE  = (3,)
HORIZON_NAMES = ["1h", "1d", "1w"]

R4_CHECKPOINT_DEFAULT = (
    "patient_level_split/last_npy_data/GlobalData/ablation_R4_fedtft_best.pth"
)


# ──────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────
def load_memmap(path, shape, dtype=np.float32):
    if not os.path.exists(path):
        return None
    size = os.path.getsize(path)
    per  = np.prod(shape) * np.dtype(dtype).itemsize
    n    = size // per
    return np.memmap(path, dtype=dtype, mode="r", shape=(n,) + shape)


def load_hospital(name):
    base = os.path.join(HOSPITAL_DIR, name)
    return {
        "seq_train":    load_memmap(os.path.join(base, "sequence_train.npy"), SEQ_SHAPE),
        "static_train": load_memmap(os.path.join(base, "static_train.npy"),   STATIC_SHAPE),
        "tgt_train":    load_memmap(os.path.join(base, "targets_train.npy"),  TARGET_SHAPE),
        "seq_val":      load_memmap(os.path.join(base, "sequence_val.npy"),   SEQ_SHAPE),
        "static_val":   load_memmap(os.path.join(base, "static_val.npy"),     STATIC_SHAPE),
        "tgt_val":      load_memmap(os.path.join(base, "targets_val.npy"),    TARGET_SHAPE),
    }


def fit_pooled_scaler(hospital_data_dict):
    seqs = [d["seq_train"] for d in hospital_data_dict.values()
            if d["seq_train"] is not None]
    all_seq = np.concatenate(seqs)
    N, T, F = all_seq.shape
    sc = StandardScaler()
    sc.fit(all_seq.reshape(-1, F))
    return sc


def scale_seq(sc, x):
    n, t, f = x.shape
    return sc.transform(x.reshape(-1, f)).reshape(n, t, f).astype(np.float32)


# ──────────────────────────────────────────────────────────────────
# Model helpers
# ──────────────────────────────────────────────────────────────────
def build_model(checkpoint, device, hidden_dim=64):
    """Build TFTPredictor and warm-start from checkpoint."""
    from model_fedtft import TFTPredictor
    model = TFTPredictor(
        input_dim=25, static_dim=14, hidden_dim=hidden_dim, output_dim=3
    ).to(device)
    if checkpoint and os.path.exists(checkpoint):
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"    Warm-started from: {checkpoint}")
    else:
        print("    WARNING: checkpoint not found — using random init")
    return model


# ──────────────────────────────────────────────────────────────────
# Training helpers
# ──────────────────────────────────────────────────────────────────
def train_one_round(model, seq_tr, stat_tr, tgt_tr, device, max_epochs=6, lr=3e-5):
    from model_fedtft import TFTDataset
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import OneCycleLR

    if len(seq_tr) == 0:
        return model

    pos = tgt_tr.sum(axis=0)
    neg = len(tgt_tr) - pos
    pw  = torch.tensor(neg / (pos + 1e-6), dtype=torch.float32).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pw)

    ds     = TFTDataset(list(zip(stat_tr, seq_tr, tgt_tr)))
    loader = DataLoader(ds, batch_size=32, shuffle=True, drop_last=False)
    if len(loader) == 0:
        return model

    opt   = AdamW(model.parameters(), lr=lr / 25, weight_decay=1e-5)
    sched = OneCycleLR(opt, max_lr=lr, steps_per_epoch=len(loader),
                       epochs=max_epochs, pct_start=0.3, anneal_strategy="cos")

    for _ in range(max_epochs):
        model.train()
        for sx, seqx, ty in loader:
            sx, seqx, ty = sx.to(device), seqx.to(device), ty.to(device)
            opt.zero_grad()
            loss_fn(model(sx, seqx), ty).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
    return model


def fedavg_aggregate(global_sd, client_states, weights):
    total  = sum(weights)
    new_sd = deepcopy(global_sd)
    for key in new_sd:
        new_sd[key] = sum(cs[key].float() * (w / total)
                          for cs, w in zip(client_states, weights))
    return new_sd


def evaluate_global(model, sc, global_test, device, threshold=0.5):
    from model_fedtft import TFTDataset
    seq_s  = scale_seq(sc, global_test["seq"])
    ds     = TFTDataset(list(zip(global_test["static"], seq_s, global_test["tgt"])))
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    probs_list, tgt_list = [], []
    model.eval()
    with torch.no_grad():
        for sx, seqx, y in loader:
            sx, seqx, y = sx.to(device), seqx.to(device), y.to(device)
            probs_list.append(torch.sigmoid(model(sx, seqx)).cpu().numpy())
            tgt_list.append(y.cpu().numpy())
    probs_np   = np.concatenate(probs_list)
    targets_np = np.concatenate(tgt_list)
    preds_np   = (probs_np >= threshold).astype(int)

    results = {}
    for i, h in enumerate(HORIZON_NAMES):
        yt, yp, yprob = targets_np[:, i], preds_np[:, i], probs_np[:, i]
        try:
            auc = roc_auc_score(yt, yprob) if 0 < yt.sum() < len(yt) else float('nan')
        except Exception:
            auc = float('nan')
        f1    = f1_score(yt, yp, zero_division=0)
        brier = float(np.mean((yprob - yt.astype(np.float32)) ** 2))
        results[h] = {"auroc": float(auc), "f1": float(f1), "brier": float(brier)}
    return results


def run_fl_simulation(hospitals, hosp_data_scaled, sc, global_test, device,
                      checkpoint, participation_rate=1.0, sample_weights=None,
                      num_rounds=10, hidden_dim=64, seed=42,
                      eval_every=5):
    """
    Warm-start FL simulation from checkpoint, fine-tune for num_rounds rounds.

    participation_rate: fraction of hospitals participating each round (1.0 = all).
    sample_weights: dict {hospital_name: fraction_of_train_to_use}.
    eval_every: evaluate global model every N rounds (and always at final round).
    """
    torch.manual_seed(seed)
    rng_np = np.random.default_rng(seed)

    model = build_model(checkpoint, device, hidden_dim)
    n_clients = max(1, int(len(hospitals) * participation_rate))
    round_results = {}

    for rnd in range(1, num_rounds + 1):
        global_sd = deepcopy(model.state_dict())

        participating = rng_np.choice(hospitals, size=n_clients, replace=False).tolist()

        client_states, weights = [], []
        for hosp in participating:
            d = hosp_data_scaled[hosp]
            seq_tr, stat_tr, tgt_tr = d["seq_train"], d["static_train"], d["tgt_train"]

            frac  = sample_weights.get(hosp, 1.0) if sample_weights else 1.0
            n_use = max(1, int(len(seq_tr) * frac))
            idx   = rng_np.choice(len(seq_tr), size=n_use, replace=False)
            seq_u  = seq_tr[idx];  stat_u = stat_tr[idx]; tgt_u = tgt_tr[idx]

            from model_fedtft import TFTPredictor
            cm = TFTPredictor(
                input_dim=25, static_dim=14, hidden_dim=hidden_dim, output_dim=3
            ).to(device)
            cm.load_state_dict(deepcopy(global_sd))
            cm = train_one_round(cm, seq_u, stat_u, tgt_u, device)
            client_states.append(deepcopy(cm.state_dict()))
            weights.append(n_use)

        if client_states:
            model.load_state_dict(fedavg_aggregate(global_sd, client_states, weights))

        if rnd % eval_every == 0 or rnd == num_rounds:
            res = evaluate_global(model, sc, global_test, device)
            mean_auc = float(np.nanmean([res[h]["auroc"] for h in HORIZON_NAMES]))
            round_results[rnd] = {**res, "mean_auroc": mean_auc}
            print(f"      Round {rnd:3d}/{num_rounds}: mean_AUROC={mean_auc:.4f}")

    best_rnd = max(round_results, key=lambda r: round_results[r]["mean_auroc"])
    return round_results[best_rnd], round_results


# ──────────────────────────────────────────────────────────────────
# Experiment 2: Client imbalance (size ratio)
# ──────────────────────────────────────────────────────────────────
def exp_client_imbalance(hospitals, hosp_data_scaled, sc, global_test,
                         device, checkpoint, num_rounds):
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Client Size Imbalance")
    print(f"  (warm-start from R4, {num_rounds} fine-tuning rounds)")
    print("=" * 60)

    n_hosp = len(hospitals)
    scenarios = {
        "Balanced (1:1:1)":  [1.0, 1.0, 1.0],
        "Moderate (1:2:5)":  [0.2, 0.4, 1.0],
        "Severe (1:5:20)":   [0.05, 0.25, 1.0],
    }

    results = {}
    for label, fracs in scenarios.items():
        fracs_padded = (fracs * (n_hosp // len(fracs) + 1))[:n_hosp]
        sw = {h: f for h, f in zip(sorted(hospitals), fracs_padded)}
        print(f"\n  {label}")
        print(f"    Sample fractions: {sw}")
        best, _ = run_fl_simulation(
            hospitals, hosp_data_scaled, sc, global_test, device,
            checkpoint=checkpoint, sample_weights=sw, num_rounds=num_rounds)
        results[label] = best
        auc_str = " | ".join(f"{h}:{best[h]['auroc']:.4f}" for h in HORIZON_NAMES)
        print(f"  Best: {auc_str} | mean={best['mean_auroc']:.4f}")
    return results


# ──────────────────────────────────────────────────────────────────
# Experiment 3: Data heterogeneity (IID vs non-IID positive-class skew)
# ──────────────────────────────────────────────────────────────────
def subsample_to_pos_rate(seq_tr, stat_tr, tgt_tr, target_rate, rng):
    """
    Subsample (seq_tr, stat_tr, tgt_tr) so that the fraction of samples
    with at least one positive horizon equals target_rate.

    Strategy:
      - positive = sample has tgt > 0 for ANY horizon
      - If target_rate > current_rate: keep all positives, drop negatives
      - If target_rate < current_rate: keep all negatives, drop positives
      - If target_rate ≈ current_rate: keep all data
    Returns subsampled arrays.
    """
    pos_mask = tgt_tr.sum(axis=1) > 0
    pos_idx  = np.where(pos_mask)[0]
    neg_idx  = np.where(~pos_mask)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)
    n_total = n_pos + n_neg
    current_rate = n_pos / max(n_total, 1)

    tol = 0.02  # don't bother subsampling if already within 2% of target
    if abs(target_rate - current_rate) < tol:
        return seq_tr, stat_tr, tgt_tr

    if target_rate > current_rate:
        # Keep all positives, reduce negatives
        n_neg_keep = max(1, int(n_pos * (1 - target_rate) / target_rate))
        neg_keep   = rng.choice(neg_idx, size=min(n_neg_keep, n_neg), replace=False)
        keep = np.concatenate([pos_idx, neg_keep])
    else:
        # Keep all negatives, reduce positives
        n_pos_keep = max(1, int(n_neg * target_rate / (1 - target_rate)))
        pos_keep   = rng.choice(pos_idx, size=min(n_pos_keep, n_pos), replace=False)
        keep = np.concatenate([pos_keep, neg_idx])

    keep = np.sort(keep)
    actual_rate = tgt_tr[keep].sum(axis=1).clip(0, 1).mean()
    return seq_tr[keep], stat_tr[keep], tgt_tr[keep]


def run_fl_simulation_skew(hospitals, hosp_data_scaled, sc, global_test, device,
                            checkpoint, target_pos_rates,
                            num_rounds=10, hidden_dim=64, seed=42, eval_every=5):
    """
    FL simulation where each hospital's training data is subsampled to a
    target positive-class rate (simulating label distribution heterogeneity).
    target_pos_rates: dict {hospital_name: float rate in [0,1]}
    """
    torch.manual_seed(seed)
    rng_np = np.random.default_rng(seed)

    model = build_model(checkpoint, device, hidden_dim)
    round_results = {}

    for rnd in range(1, num_rounds + 1):
        global_sd = deepcopy(model.state_dict())
        client_states, weights = [], []

        for hosp in hospitals:
            d = hosp_data_scaled[hosp]
            seq_tr, stat_tr, tgt_tr = d["seq_train"], d["static_train"], d["tgt_train"]

            rate = target_pos_rates.get(hosp, None)
            if rate is not None:
                seq_u, stat_u, tgt_u = subsample_to_pos_rate(
                    seq_tr, stat_tr, tgt_tr, rate, rng_np)
            else:
                seq_u, stat_u, tgt_u = seq_tr, stat_tr, tgt_tr

            from model_fedtft import TFTPredictor
            cm = TFTPredictor(
                input_dim=25, static_dim=14, hidden_dim=hidden_dim, output_dim=3
            ).to(device)
            cm.load_state_dict(deepcopy(global_sd))
            cm = train_one_round(cm, seq_u, stat_u, tgt_u, device)
            client_states.append(deepcopy(cm.state_dict()))
            weights.append(len(seq_u))

        if client_states:
            model.load_state_dict(fedavg_aggregate(global_sd, client_states, weights))

        if rnd % eval_every == 0 or rnd == num_rounds:
            res = evaluate_global(model, sc, global_test, device)
            mean_auc = float(np.nanmean([res[h]["auroc"] for h in HORIZON_NAMES]))
            round_results[rnd] = {**res, "mean_auroc": mean_auc}
            print(f"      Round {rnd:3d}/{num_rounds}: mean_AUROC={mean_auc:.4f}")

    best_rnd = max(round_results, key=lambda r: round_results[r]["mean_auroc"])
    return round_results[best_rnd], round_results


def exp_label_skew(hospitals, hosp_data_scaled, sc, global_test,
                   device, checkpoint, num_rounds):
    """
    Simulate label heterogeneity by controlling the positive-class rate per hospital.

    IID:           all hospitals keep their natural positive rate (~15-20%).
    Mild non-IID:  largest hospital → 30% positive, mid → natural, smallest → 5%.
    Severe non-IID: largest → 50% positive, mid → natural, smallest → 2%.

    This directly simulates the Dirichlet α label-skew effect used in FL research:
    lower α = greater divergence in label distributions across clients.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Data Heterogeneity (Non-IID Positive-Class Skew)")
    print(f"  (warm-start from R4, {num_rounds} fine-tuning rounds)")
    print("=" * 60)

    # Sort hospitals by training size: index 0 = largest, -1 = smallest
    sorted_by_size = sorted(hospitals,
                            key=lambda h: len(hosp_data_scaled[h]["seq_train"]),
                            reverse=True)

    # Compute natural positive rates for logging
    for h in sorted_by_size:
        tgt = hosp_data_scaled[h]["tgt_train"]
        nat_rate = (tgt.sum(axis=1) > 0).mean()
        print(f"  {h}: n_train={len(tgt)}, natural_pos_rate={nat_rate:.3f}")

    # Scenarios: {hospital: target_pos_rate}
    # None = keep natural rate unchanged
    n = len(sorted_by_size)
    def make_rates(rates_for_sorted):
        return {h: rates_for_sorted[i] for i, h in enumerate(sorted_by_size)}

    if n == 1:
        scenarios = {
            "IID (natural)":    make_rates([None]),
            "Mild non-IID":     make_rates([0.30]),
            "Severe non-IID":   make_rates([0.50]),
        }
    elif n == 2:
        scenarios = {
            "IID (natural)":    make_rates([None, None]),
            "Mild non-IID":     make_rates([0.30, 0.05]),
            "Severe non-IID":   make_rates([0.50, 0.02]),
        }
    else:
        # 3+ hospitals: large=high rate, mid=natural, small=low rate
        mid_rates  = [None] * (n - 2)
        scenarios = {
            "IID (natural)":    make_rates([None]  + mid_rates + [None]),
            "Mild non-IID":     make_rates([0.30]  + mid_rates + [0.05]),
            "Severe non-IID":   make_rates([0.50]  + mid_rates + [0.02]),
        }

    results = {}
    for label, target_rates in scenarios.items():
        display = {h: (f"{r:.2f}" if r is not None else "natural")
                   for h, r in target_rates.items()}
        print(f"\n  {label}: {display}")
        best, _ = run_fl_simulation_skew(
            hospitals, hosp_data_scaled, sc, global_test, device,
            checkpoint=checkpoint, target_pos_rates=target_rates,
            num_rounds=num_rounds)
        results[label] = best
        auc_str = " | ".join(f"{h}:{best[h]['auroc']:.4f}" for h in HORIZON_NAMES)
        print(f"  Best: {auc_str} | mean={best['mean_auroc']:.4f}")
    return results


# ──────────────────────────────────────────────────────────────────
# Experiment 4: Round count sensitivity
# ──────────────────────────────────────────────────────────────────
def exp_round_count(hospitals, hosp_data_scaled, sc, global_test,
                    device, checkpoint):
    """
    Fine-tune from R6 for 5, 10, 20, 30 rounds and compare convergence speed.
    Uses all hospitals (1.0 participation), no sample imbalance.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Round Count Sensitivity (warm-start from R4)")
    print("=" * 60)

    round_counts = [5, 10, 20, 30]
    results = {}
    for nr in round_counts:
        print(f"\n  Rounds = {nr}")
        best, history = run_fl_simulation(
            hospitals, hosp_data_scaled, sc, global_test, device,
            checkpoint=checkpoint, participation_rate=1.0,
            num_rounds=nr, eval_every=max(1, nr // 5))
        results[str(nr)] = {"best": best, "history": history}
        auc_str = " | ".join(f"{h}:{best[h]['auroc']:.4f}" for h in HORIZON_NAMES)
        print(f"  Best after {nr} rounds: {auc_str} | mean={best['mean_auroc']:.4f}")
    return results


# ──────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────
def plot_bar_comparison(results_dict, metric, title, outdir, filename):
    labels = list(results_dict.keys())
    x      = np.arange(len(HORIZON_NAMES))
    width  = 0.7 / max(len(labels), 1)
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))

    fig, ax = plt.subplots(figsize=(10, 5))
    for j, label in enumerate(labels):
        res = results_dict[label]
        y   = [res[h].get(metric, float('nan')) for h in HORIZON_NAMES]
        ax.bar(x + j * width, y, width, label=label, color=colors[j], alpha=0.85)

    ax.set_xticks(x + width * (len(labels) - 1) / 2)
    ax.set_xticklabels(HORIZON_NAMES)
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out = os.path.join(outdir, filename)
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Plot saved → {out}")


def plot_convergence(round_results_dict, outdir):
    """AUROC learning curves for different round counts."""
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.tab10(np.linspace(0, 0.5, len(round_results_dict)))

    for j, (n_rounds, data) in enumerate(round_results_dict.items()):
        history = data["history"]
        rounds  = sorted(int(r) for r in history.keys())
        aucs    = [history[r]["mean_auroc"] for r in rounds]
        ax.plot(rounds, aucs, marker="o", label=f"{n_rounds} rounds", color=colors[j])

    ax.set_xlabel("Fine-tuning Round (from R4 warm-start)")
    ax.set_ylabel("Mean AUROC (global test)")
    ax.set_title("FedTFT Convergence: AUROC vs Fine-tuning Rounds")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(outdir, "convergence_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Plot saved → {out}")


def print_summary_table(all_results):
    print("\n" + "=" * 80)
    print("HETEROGENEITY ANALYSIS SUMMARY")
    print("=" * 80)
    for exp_name, exp_results in all_results.items():
        if exp_name == "convergence":
            continue
        print(f"\n  [{exp_name.upper()}]")
        print(f"  {'Setting':<35} " +
              "  ".join(f"AUROC_{h:>3}" for h in HORIZON_NAMES) +
              "  Mean_AUROC  Mean_Brier")
        print("  " + "-" * 78)
        for setting, res in exp_results.items():
            aucs   = [res[h]["auroc"] for h in HORIZON_NAMES]
            briers = [res[h]["brier"]  for h in HORIZON_NAMES]
            print(f"  {setting:<35} " +
                  " ".join(f"{a:8.4f}" for a in aucs) +
                  f"  {float(np.nanmean(aucs)):10.4f}"
                  f"  {float(np.nanmean(briers)):10.4f}")


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Heterogeneity analysis: imbalance, label skew, convergence (warm-start)")
    parser.add_argument("--data_root", default="patient_level_split/last_npy_data",
                        help="Path to last_npy_data root")
    parser.add_argument("--checkpoint", default=R4_CHECKPOINT_DEFAULT,
                        help="R4 checkpoint for warm-start (default: ablation_R4_fedtft_best.pth)")
    parser.add_argument("--num_rounds", type=int, default=10,
                        help="Fine-tuning FL rounds per scenario (default: 10)")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--output", default="results/heterogeneity/")
    parser.add_argument("--experiments", nargs="+",
                        default=["imbalance", "label_skew", "convergence"],
                        choices=["imbalance", "label_skew", "convergence"])
    args = parser.parse_args()
    global GLOBAL_DIR, HOSPITAL_DIR
    GLOBAL_DIR   = os.path.join(args.data_root, "GlobalData")
    HOSPITAL_DIR = os.path.join(args.data_root, "HospitalsData")

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Fine-tuning rounds per scenario: {args.num_rounds}")
    print(f"Experiments: {args.experiments}")

    # ── Note: Exp 1 (participation) already completed ──────────────
    print("\n[NOTE] Experiment 1 (Participation Rate) already completed.")
    print("  Results: results/heterogeneity/participation_auroc.png")
    print("           results/heterogeneity/participation_brier.png")

    # ── Load data ──────────────────────────────────────────────────
    hospitals = sorted(h for h in os.listdir(HOSPITAL_DIR)
                       if os.path.isdir(os.path.join(HOSPITAL_DIR, h)))
    if not hospitals:
        raise RuntimeError(f"No hospital dirs found in {HOSPITAL_DIR}")
    print(f"\nHospitals: {hospitals}")

    raw_data = {h: load_hospital(h) for h in hospitals}

    # Drop hospitals missing training data
    raw_data = {h: d for h, d in raw_data.items() if d["seq_train"] is not None}
    hospitals = list(raw_data.keys())

    sc = fit_pooled_scaler(raw_data)
    print(f"Scaler fit on {sum(len(d['seq_train']) for d in raw_data.values())} train samples")

    hosp_data_scaled = {}
    for hosp, d in raw_data.items():
        hosp_data_scaled[hosp] = {
            "seq_train":    scale_seq(sc, d["seq_train"]),
            "static_train": d["static_train"].copy().astype(np.float32),
            "tgt_train":    d["tgt_train"].copy().astype(np.float32),
            "seq_val":      scale_seq(sc, d["seq_val"]),
            "static_val":   d["static_val"].copy().astype(np.float32),
            "tgt_val":      d["tgt_val"].copy().astype(np.float32),
        }

    global_test = {
        "static": load_memmap(os.path.join(GLOBAL_DIR, "static_data.npy"),   STATIC_SHAPE),
        "seq":    load_memmap(os.path.join(GLOBAL_DIR, "sequence_data.npy"), SEQ_SHAPE),
        "tgt":    load_memmap(os.path.join(GLOBAL_DIR, "targets.npy"),       TARGET_SHAPE),
    }
    print(f"Global test: N={len(global_test['tgt'])}")

    all_results = {}

    # ── Experiment 2: Client imbalance ─────────────────────────────
    if "imbalance" in args.experiments:
        res = exp_client_imbalance(hospitals, hosp_data_scaled, sc, global_test,
                                   device, args.checkpoint, args.num_rounds)
        all_results["client_imbalance"] = res
        plot_bar_comparison(res, "auroc", "AUROC by Client Size Imbalance (warm-start)",
                            args.output, "imbalance_auroc.png")
        plot_bar_comparison(res, "brier", "Brier Score by Client Size Imbalance",
                            args.output, "imbalance_brier.png")

    # ── Experiment 3: Label heterogeneity ──────────────────────────
    if "label_skew" in args.experiments:
        res = exp_label_skew(hospitals, hosp_data_scaled, sc, global_test,
                             device, args.checkpoint, args.num_rounds)
        all_results["label_skew"] = res
        plot_bar_comparison(res, "auroc", "AUROC: IID vs Non-IID Label Distribution (warm-start)",
                            args.output, "label_skew_auroc.png")
        plot_bar_comparison(res, "brier", "Brier Score: IID vs Non-IID",
                            args.output, "label_skew_brier.png")

    # ── Experiment 4: Round count sensitivity ──────────────────────
    if "convergence" in args.experiments:
        res = exp_round_count(hospitals, hosp_data_scaled, sc, global_test,
                              device, args.checkpoint)
        all_results["convergence"] = res
        plot_convergence(res, args.output)

    # ── Summary & save ─────────────────────────────────────────────
    print_summary_table(all_results)

    print("\nResults (JSON save disabled):")
    print(all_results)


if __name__ == "__main__":
    main()
