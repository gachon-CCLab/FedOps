"""
feature_ablation.py — Ablation on input feature groups.

Feature groups (based on feature_names.pkl / feature_names.txt):
  - Cosinor features: circadian rhythm features (MESOR, amplitude, acrophase, R²) —
      identified by column indices matching "cosinor", "mesor", "amplitude", "acrophase", "r2"
  - Location features: entropy, variability, room indicators —
      identified by patterns "entropy", "variability", "place_"
  - Treatment-time features: clinical intervention timing —
      identified by patterns "treatment", "injection", "medication_time"

Variants:
  FA0: All features (default)               — baseline
  FA1: Remove cosinor features              — raw sensor only
  FA2: Remove location features             — no spatial context
  FA3: Remove treatment-time features       — mitigates confounding (R3.4c)
  FA4: Remove cosinor + location            — pure sensor + static demographics only
  FA5: Sensor features only (FA1 + FA2 + FA3)

Metrics: AUROC, F1, Brier score per horizon (1h, 1d, 1w)

Usage:
  python experiments/analysis/feature_ablation.py \\
      --checkpoint checkpoints/fedtft_best.pth \\
      --output results/feature_ablation/

Run from FedTFT_paper/ root directory.
"""

import os
import sys
# Add project root (FedTFT_paper/) to path so model_fedtft can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import json
import argparse
import pickle
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score
from copy import deepcopy

# ──────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────
_DEFAULT_DATA_ROOT = "patient_level_split/last_npy_data"
HOSPITAL_DIR  = _DEFAULT_DATA_ROOT + "/HospitalsData"
GLOBAL_DIR    = _DEFAULT_DATA_ROOT + "/GlobalData"
STATIC_SHAPE  = (14,)
SEQ_SHAPE     = (192, 25)
TARGET_SHAPE  = (3,)
HORIZON_NAMES = ["1h", "1d", "1w"]

N_SEQ_FEATURES = 25   # columns in each timestep of sequence

VARIANTS = {
    "FA0":  "All features (baseline)",
    "FA1":  "Remove cosinor/circadian features",
    "FA2":  "Remove location features",
    "FA3a": "Remove injection-time only (VE_tx_inj_time — targeted confound)",
    "FA3":  "Remove all treatment features (VE_tx, inj_time, Restrictive)",
    "FA4":  "Remove cosinor + location",
    "FA5":  "Sensor + demographics only (FA1+FA2+FA3)",
}

# ──────────────────────────────────────────────────────────────────
# Feature group detection
# ──────────────────────────────────────────────────────────────────

# Keywords identifying feature types in the sequence feature names
COSINOR_KEYWORDS    = ["cosinor", "mesor", "amplitude", "acrophase", "r2",
                       "circadian", "cos_", "sin_", "periodic", "phase"]
LOCATION_KEYWORDS   = ["entropy", "variability", "place_", "location",
                       "room", "ward", "hallway", "semantic"]
TREATMENT_KEYWORDS  = ["treatment", "injection", "medication_time",
                       "med_time", "admin_time", "therapy_time",
                       "procedure_time", "inject"]

# Static feature names (14-dim) — treatment features are in static, not sequence
STATIC_FEATURE_NAMES = [
    "HR_nunique", "nonwearing", "VE_tx", "VE_tx_inj_time",
    "Restrictive_Intervention", "sex", "DIG_3class", "DIG_4class",
    "DIG_withPsychosis", "day_of_week", "holidays",
    "place_hallway", "place_other", "place_ward",
]
TREATMENT_STATIC_KEYWORDS = ["ve_tx", "tx_inj", "restrictive"]


def load_feature_names():
    """Try to load feature names from pkl or txt. Returns list or None."""
    for path in ["feature_names.pkl", "full_feature_names.pkl", "full_feature_names_new.pkl"]:
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    names = pickle.load(f)
                if isinstance(names, list):
                    return names
            except Exception:
                pass

    for path in ["feature_names.txt"]:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    return [line.strip() for line in f if line.strip()]
            except Exception:
                pass
    return None


def get_feature_masks(feature_names, n_seq_features=N_SEQ_FEATURES):
    """
    Returns a dict with keys 'seq' and 'static', each a dict of
    {feature_type: set of column indices to ZERO OUT}.
    """
    seq_names = feature_names[:n_seq_features] if feature_names else []

    def seq_matching(keywords):
        if not seq_names:
            return set()
        return {i for i, name in enumerate(seq_names)
                if any(kw.lower() in name.lower() for kw in keywords)}

    def static_matching(keywords):
        return {i for i, name in enumerate(STATIC_FEATURE_NAMES)
                if any(kw.lower() in name.lower() for kw in keywords)}

    seq_masks = {
        "cosinor":   seq_matching(COSINOR_KEYWORDS),
        "location":  seq_matching(LOCATION_KEYWORDS),
        "treatment": seq_matching(TREATMENT_KEYWORDS),   # likely empty
    }
    static_masks = {
        "treatment": static_matching(TREATMENT_STATIC_KEYWORDS),  # VE_tx, Restrictive
    }
    return {"seq": seq_masks, "static": static_masks}


def mask_sequence(seq, zero_cols):
    """Zero out specified column indices in seq [N, T, F]. Returns modified copy."""
    if not zero_cols:
        return seq
    seq = seq.copy()
    for col in zero_cols:
        if col < seq.shape[2]:
            seq[:, :, col] = 0.0
    return seq


def mask_static(stat, zero_cols):
    """Zero out specified column indices in static [N, F]. Returns modified copy."""
    if not zero_cols:
        return stat
    stat = stat.copy()
    for col in zero_cols:
        if col < stat.shape[1]:
            stat[:, col] = 0.0
    return stat


def get_variant_masks(variant, feature_masks):
    """Return (seq_zero_cols, static_zero_cols) for this variant."""
    sm = feature_masks["seq"]
    st = feature_masks["static"]
    if variant == "FA0":
        return set(), set()
    elif variant == "FA1":
        return sm["cosinor"], set()
    elif variant == "FA2":
        return sm["location"], set()
    elif variant == "FA3a":
        # Remove only VE_tx_inj_time (static index 3) — targeted injection-time confound
        inj_time_idx = STATIC_FEATURE_NAMES.index("VE_tx_inj_time")
        return set(), {inj_time_idx}
    elif variant == "FA3":
        return sm.get("treatment", set()), st["treatment"]
    elif variant == "FA4":
        return sm["cosinor"] | sm["location"], set()
    elif variant == "FA5":
        return (sm["cosinor"] | sm["location"] | sm.get("treatment", set())), st["treatment"]
    return set(), set()


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


def load_all_train(hospitals):
    """Pool all hospital training data for fitting scaler."""
    seq_list, stat_list, tgt_list = [], [], []
    for hosp in hospitals:
        base = os.path.join(HOSPITAL_DIR, hosp)
        seq  = load_memmap(os.path.join(base, "sequence_train.npy"), SEQ_SHAPE)
        stat = load_memmap(os.path.join(base, "static_train.npy"),   STATIC_SHAPE)
        tgt  = load_memmap(os.path.join(base, "targets_train.npy"),  TARGET_SHAPE)
        if seq is not None:
            seq_list.append(seq); stat_list.append(stat); tgt_list.append(tgt)
    if not seq_list:
        raise RuntimeError("No training data found.")
    return (np.concatenate(seq_list),
            np.concatenate(stat_list),
            np.concatenate(tgt_list))


def fit_scaler(seq_train):
    N, T, F = seq_train.shape
    sc = StandardScaler()
    sc.fit(seq_train.reshape(-1, F))
    return sc


def scale(sc, x):
    n, t, f = x.shape
    return sc.transform(x.reshape(-1, f)).reshape(n, t, f).astype(np.float32)


# ──────────────────────────────────────────────────────────────────
# Training (local FL simulation per variant)
# ──────────────────────────────────────────────────────────────────
def train_one_round(model, seq_tr, stat_tr, tgt_tr, seq_val, stat_val, tgt_val,
                    device, max_epochs=6, lr=3e-5):
    from model_fedtft import TFTDataset
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import OneCycleLR

    pos = tgt_tr.sum(axis=0)
    neg = len(tgt_tr) - pos
    pw  = torch.tensor(neg / (pos + 1e-6), dtype=torch.float32).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pw)

    tr_ds  = TFTDataset(list(zip(stat_tr, seq_tr, tgt_tr)))
    val_ds = TFTDataset(list(zip(stat_val, seq_val, tgt_val)))
    tr_ld  = DataLoader(tr_ds,  batch_size=32, shuffle=True)
    val_ld = DataLoader(val_ds, batch_size=32, shuffle=False)

    opt   = AdamW(model.parameters(), lr=lr / 25, weight_decay=1e-5)
    sched = OneCycleLR(opt, max_lr=lr, steps_per_epoch=len(tr_ld),
                       epochs=max_epochs, pct_start=0.3, anneal_strategy="cos")

    best_loss, best_state = float("inf"), deepcopy(model.state_dict())
    for _ in range(max_epochs):
        model.train()
        for sx, seqx, ty in tr_ld:
            sx, seqx, ty = sx.to(device), seqx.to(device), ty.to(device)
            opt.zero_grad()
            loss_fn(model(sx, seqx), ty).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()

        model.eval(); vl = 0.0
        with torch.no_grad():
            for sx, seqx, ty in val_ld:
                sx, seqx, ty = sx.to(device), seqx.to(device), ty.to(device)
                vl += loss_fn(model(sx, seqx), ty).item() * sx.size(0)
        vl /= max(len(val_ds), 1)
        if vl < best_loss:
            best_loss = vl; best_state = deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model


def fedavg(global_sd, client_states, weights):
    total  = sum(weights)
    new_sd = deepcopy(global_sd)
    for key in new_sd:
        new_sd[key] = sum(cs[key].float() * (w / total)
                          for cs, w in zip(client_states, weights))
    return new_sd


# ──────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────
def evaluate(model, static, seq, targets, device, threshold=0.5):
    from model_fedtft import TFTDataset
    loader = DataLoader(TFTDataset(list(zip(static, seq, targets))),
                        batch_size=32, shuffle=False)
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


# ──────────────────────────────────────────────────────────────────
# Run one variant
# ──────────────────────────────────────────────────────────────────
def run_variant(variant, zero_seq_cols, zero_stat_cols, sc, hospitals, global_test, device,
                num_rounds=30, hidden_dim=64, patience=3, eval_every=5):
    from model_fedtft import TFTPredictor

    print(f"\n{'='*60}")
    print(f"  {variant}: {VARIANTS[variant]}")
    if zero_seq_cols:
        print(f"  Zeroing {len(zero_seq_cols)} seq columns: {sorted(zero_seq_cols)}")
    if zero_stat_cols:
        print(f"  Zeroing {len(zero_stat_cols)} static columns: {sorted(zero_stat_cols)}")
        print(f"    ({[STATIC_FEATURE_NAMES[i] for i in sorted(zero_stat_cols)]})")
    if not zero_seq_cols and not zero_stat_cols:
        print(f"  Using ALL features")
    print(f"  max_rounds={num_rounds}, patience={patience}, eval_every={eval_every}")
    print(f"{'='*60}")

    # Scale + mask global test
    seq_test_s  = scale(sc, global_test["seq"])
    seq_test_s  = mask_sequence(seq_test_s, zero_seq_cols).astype(np.float32)
    stat_test_m = mask_static(global_test["static"].copy().astype(np.float32), zero_stat_cols)

    torch.manual_seed(42)
    model = TFTPredictor(
        input_dim=N_SEQ_FEATURES, static_dim=14,
        hidden_dim=hidden_dim, output_dim=3
    ).to(device)

    best_result, best_auc = None, -1.0
    no_improve = 0  # consecutive evaluations without improvement

    for rnd in range(1, num_rounds + 1):
        global_sd = deepcopy(model.state_dict())
        client_states, weights = [], []

        for hosp in hospitals:
            base   = os.path.join(HOSPITAL_DIR, hosp)
            seq_tr = load_memmap(os.path.join(base, "sequence_train.npy"), SEQ_SHAPE)
            st_tr  = load_memmap(os.path.join(base, "static_train.npy"),   STATIC_SHAPE)
            tg_tr  = load_memmap(os.path.join(base, "targets_train.npy"),  TARGET_SHAPE)
            seq_vl = load_memmap(os.path.join(base, "sequence_val.npy"),   SEQ_SHAPE)
            st_vl  = load_memmap(os.path.join(base, "static_val.npy"),     STATIC_SHAPE)
            tg_vl  = load_memmap(os.path.join(base, "targets_val.npy"),    TARGET_SHAPE)

            if seq_tr is None:
                continue

            # Scale + mask seq and static
            s_tr   = scale(sc, seq_tr); s_tr = mask_sequence(s_tr, zero_seq_cols).astype(np.float32)
            s_vl   = scale(sc, seq_vl); s_vl = mask_sequence(s_vl, zero_seq_cols).astype(np.float32)
            st_tr_ = mask_static(st_tr.copy().astype(np.float32), zero_stat_cols)
            st_vl_ = mask_static(st_vl.copy().astype(np.float32), zero_stat_cols)

            cm = TFTPredictor(
                input_dim=N_SEQ_FEATURES, static_dim=14,
                hidden_dim=hidden_dim, output_dim=3
            ).to(device)
            cm.load_state_dict(deepcopy(global_sd))
            cm = train_one_round(cm, s_tr, st_tr_,
                                 tg_tr.copy().astype(np.float32),
                                 s_vl, st_vl_,
                                 tg_vl.copy().astype(np.float32), device)
            client_states.append(deepcopy(cm.state_dict()))
            weights.append(len(s_tr))

        if not client_states:
            continue

        model.load_state_dict(fedavg(global_sd, client_states, weights))

        if rnd % eval_every == 0 or rnd == num_rounds:
            results = evaluate(model, stat_test_m, seq_test_s,
                               global_test["tgt"], device)
            mean_auc = np.nanmean([results[h]["auroc"] for h in HORIZON_NAMES])
            aucs = " | ".join(f"{h}: {results[h]['auroc']:.4f}" for h in HORIZON_NAMES)
            print(f"  Round {rnd:3d}: {aucs} | mean={mean_auc:.4f}")
            if mean_auc > best_auc + 1e-4:
                best_auc = mean_auc; best_result = results
                no_improve = 0
            else:
                no_improve += 1
                print(f"  [early-stop counter: {no_improve}/{patience}]")
                if no_improve >= patience:
                    print(f"  Early stop at round {rnd} (no improvement for {patience} evals)")
                    break

    return best_result


# ──────────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────────
def plot_feature_ablation(all_results, metric, outdir):
    x      = np.arange(len(HORIZON_NAMES))
    labels = list(all_results.keys())
    width  = 0.7 / max(len(labels), 1)
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))

    fig, ax = plt.subplots(figsize=(12, 5))
    for j, v in enumerate(labels):
        if all_results[v] is None:
            continue
        y = [all_results[v][h].get(metric, float('nan')) for h in HORIZON_NAMES]
        ax.bar(x + j * width, y, width, label=f"{v}", color=colors[j], alpha=0.85)

    ax.set_xticks(x + width * (len(labels) - 1) / 2)
    ax.set_xticklabels(HORIZON_NAMES)
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Feature Ablation — {metric.upper()} per Horizon")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out = os.path.join(outdir, f"feature_ablation_{metric}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Plot saved → {out}")


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Feature group ablation: cosinor, location, treatment-time")
    parser.add_argument("--data_root", default="patient_level_split/last_npy_data",
                        help="Path to last_npy_data root")
    parser.add_argument("--variants", nargs="+", default=["FA0","FA1","FA2","FA3a","FA3","FA4","FA5"],
                        choices=list(VARIANTS.keys()))
    parser.add_argument("--num_rounds", type=int, default=30)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--output", default="results/feature_ablation/")
    args = parser.parse_args()
    global GLOBAL_DIR, HOSPITAL_DIR
    GLOBAL_DIR   = os.path.join(args.data_root, "GlobalData")
    HOSPITAL_DIR = os.path.join(args.data_root, "HospitalsData")

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load feature names ──────────────────────────────────────────
    feature_names = load_feature_names()
    if feature_names:
        print(f"Loaded {len(feature_names)} feature names")
        feature_masks = get_feature_masks(feature_names)
        for group, idxs in feature_masks["seq"].items():
            print(f"  seq/{group}: {len(idxs)} features → indices {sorted(idxs)}")
        for group, idxs in feature_masks["static"].items():
            names = [STATIC_FEATURE_NAMES[i] for i in sorted(idxs)]
            print(f"  static/{group}: {len(idxs)} features → {names}")
    else:
        print("[WARN] feature_names not found — using empty masks (all features kept).")
        print("       Place feature_names.pkl or feature_names.txt in FedTFT_paper/ root.")
        feature_masks = {
            "seq":    {"cosinor": set(), "location": set(), "treatment": set()},
            "static": {"treatment": set()},
        }

    # ── Fit scaler on pooled train data ────────────────────────────
    hospitals = sorted(os.listdir(HOSPITAL_DIR))
    print(f"Hospitals: {hospitals}")

    seq_train, stat_train, tgt_train = load_all_train(hospitals)
    sc = fit_scaler(seq_train)
    print(f"Scaler fitted on {len(seq_train)} training samples")

    # ── Load global test ────────────────────────────────────────────
    global_test = {
        "static": load_memmap(os.path.join(GLOBAL_DIR, "static_data.npy"),   STATIC_SHAPE),
        "seq":    load_memmap(os.path.join(GLOBAL_DIR, "sequence_data.npy"), SEQ_SHAPE),
        "tgt":    load_memmap(os.path.join(GLOBAL_DIR, "targets.npy"),       TARGET_SHAPE),
    }
    print(f"Global test: N={len(global_test['tgt'])}")

    # ── Run variants ────────────────────────────────────────────────
    all_results = {}

    for variant in args.variants:
        zero_seq_cols, zero_stat_cols = get_variant_masks(variant, feature_masks)
        try:
            result = run_variant(
                variant, zero_seq_cols, zero_stat_cols, sc, hospitals, global_test, device,
                num_rounds=args.num_rounds, hidden_dim=args.hidden_dim)
            all_results[variant] = result
        except Exception as e:
            print(f"[ERROR] {variant} failed: {e}")
            all_results[variant] = None
        print(f"  [progress] Completed variants: {list(all_results.keys())}")

    # ── Summary table ───────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("FEATURE ABLATION SUMMARY")
    print(f"{'='*80}")
    cols = [f"AUC_{h}" for h in HORIZON_NAMES] + [f"F1_{h}" for h in HORIZON_NAMES] \
         + [f"Brier_{h}" for h in HORIZON_NAMES]
    print(f"{'Variant':<6} {'Description':<45} " + " ".join(f"{c:>10}" for c in cols))
    print("-" * 110)
    for v in args.variants:
        r = all_results.get(v)
        if r is None:
            print(f"{v:<6} {VARIANTS[v]:<45} " + "       N/A" * len(cols))
            continue
        vals = (  [r[h]["auroc"]  for h in HORIZON_NAMES]
                + [r[h]["f1"]     for h in HORIZON_NAMES]
                + [r[h]["brier"]  for h in HORIZON_NAMES])
        print(f"{v:<6} {VARIANTS[v]:<45} " + " ".join(f"{x:10.4f}" for x in vals))

    # ── Plots ───────────────────────────────────────────────────────
    for metric in ["auroc", "f1", "brier"]:
        plot_feature_ablation(all_results, metric, args.output)

    # ── Delta vs FA0 ────────────────────────────────────────────────
    if all_results.get("FA0"):
        print(f"\nDelta AUROC vs FA0 (all features):")
        fa0_aucs = [all_results["FA0"][h]["auroc"] for h in HORIZON_NAMES]
        for v in args.variants:
            if v == "FA0" or all_results.get(v) is None:
                continue
            v_aucs = [all_results[v][h]["auroc"] for h in HORIZON_NAMES]
            deltas = [v_aucs[i] - fa0_aucs[i] for i in range(len(HORIZON_NAMES))]
            delta_str = " | ".join(
                f"{h}:{d:+.4f}" for h, d in zip(HORIZON_NAMES, deltas))
            print(f"  {v} ({VARIANTS[v][:30]:<30}): {delta_str}")

    print("\nFeature ablation results (JSON save disabled).")
    print(all_results)


if __name__ == "__main__":
    main()
