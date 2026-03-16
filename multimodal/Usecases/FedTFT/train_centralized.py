"""
train_centralized.py — Centralized TFT training (privacy-upper-bound reference)

Pools all hospital train splits into one dataset and trains a single
TFTPredictor_FedTFT model with the same architecture and hyperparameters
as the federated R4 (FedTFT) run.  No federation, no proximal term.

Purpose:
  Provides the centralized upper-bound reference for the
  "Federated vs. Centralized Training" comparison in Section III-B
  of the manuscript.  Centralised pooling is not deployable (privacy),
  but it sets the discrimination ceiling that FedTFT approaches.

Architecture / hyperparameters (identical to R4 client):
  - Model   : TFTPredictor_FedTFT  (horizon-decoupled 3×64→1 heads)
  - Optimizer: AdamW, peak_lr=3e-5, OneCycleLR (pct_start=0.3, cos)
  - Loss     : BCEWithLogitsLoss with per-horizon pos_weight
  - Batch    : 32
  - Max epochs: 300  (= 50 FL rounds × 6 client epochs — same ceiling)
  - Early stop: patience=15 on pooled-val loss  (same as FL server)
  - Threshold : Bayesian optimisation on pooled val (Youden + min-spec 0.5)
  - Seeds    : 3  →  report mean ± SD AUROC

Data (identical splits as used in FL training):
  - Train: concatenation of all HospitalsData/<h>/static_train.npy + sequence_train.npy
  - Val  : concatenation of all HospitalsData/<h>/static_val.npy   + sequence_val.npy
  - Test : GlobalData/static_data.npy + sequence_data.npy  (held-out, never touched during training)

Usage:
  python train_centralized.py \\
      --data_root patient_level_split/last_npy_data \\
      --output    results/centralized/ \\
      --seeds     1 2 3

Run from the FedTFT_paper/ root directory.
"""

import os
import sys
import json
import copy
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from bayes_opt import BayesianOptimization
from torchmetrics.classification import (
    MultilabelAUROC, MultilabelF1Score, MultilabelRecall,
    MultilabelSpecificity,
)

# Model import — run from FedTFT_paper/ root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_fedtft_hdfp import TFTPredictor_FedTFT, TFTDataset

# ─────────────────────────────────────────────────────────────────────
# Constants (must match preprocessing_final.ipynb output)
# ─────────────────────────────────────────────────────────────────────
STATIC_SHAPE   = (14,)
SEQUENCE_SHAPE = (192, 25)
TARGETS_SHAPE  = (3,)
TARGET_NAMES   = ['dangerous_action_1h', 'dangerous_action_1d', 'dangerous_action_1w']

PEAK_LR    = 3e-5
BATCH_SIZE = 32
MAX_EPOCHS = 300          # ceiling; early stopping fires well before this
PATIENCE   = 15           # same as FL server


# ─────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────
def load_memmap(path, shape, dtype=np.float32):
    if not os.path.exists(path):
        return None
    size = os.path.getsize(path)
    per  = int(np.prod(shape)) * np.dtype(dtype).itemsize
    n    = size // per
    return np.memmap(path, dtype=dtype, mode="r", shape=(n,) + shape)


def pool_hospital_split(hospital_dir, split):
    """Concatenate one split across all hospitals. Returns (static, seq, targets)."""
    stat_list, seq_list, tgt_list = [], [], []
    for hosp in sorted(os.listdir(hospital_dir)):
        base = os.path.join(hospital_dir, hosp)
        s = load_memmap(os.path.join(base, f"static_{split}.npy"),   STATIC_SHAPE)
        q = load_memmap(os.path.join(base, f"sequence_{split}.npy"), SEQUENCE_SHAPE)
        t = load_memmap(os.path.join(base, f"targets_{split}.npy"),  TARGETS_SHAPE)
        if s is not None:
            stat_list.append(s); seq_list.append(q); tgt_list.append(t)
    if not stat_list:
        raise RuntimeError(f"No {split} data found in {hospital_dir}")
    return (np.concatenate(stat_list), np.concatenate(seq_list), np.concatenate(tgt_list))


# ─────────────────────────────────────────────────────────────────────
# Bayesian threshold tuning (identical to fl_client_fedtft.py)
# ─────────────────────────────────────────────────────────────────────
def tune_thresholds(model, val_loader, device, method="youden", min_specificity=0.5):
    model.eval()
    all_logits = {i: [] for i in range(3)}
    all_true   = {i: [] for i in range(3)}
    with torch.no_grad():
        for sx, seqx, y in val_loader:
            sx, seqx, y = sx.to(device), seqx.to(device), y.to(device)
            logits = model(sx, seqx).cpu().numpy()
            y_np   = y.cpu().numpy()
            for i in range(3):
                all_logits[i].extend(logits[:, i].tolist())
                all_true[i].extend(y_np[:, i].tolist())

    best_ts = []
    for i in range(3):
        y_true = np.array(all_true[i])
        y_prob = 1.0 / (1.0 + np.exp(-np.array(all_logits[i])))

        def objective(thresh, _yt=y_true, _yp=y_prob):
            thresh = np.clip(thresh, 0.1, 0.9)
            preds  = (_yp >= thresh).astype(int)
            tp = np.logical_and(preds==1, _yt==1).sum()
            tn = np.logical_and(preds==0, _yt==0).sum()
            fp = np.logical_and(preds==1, _yt==0).sum()
            fn = np.logical_and(preds==0, _yt==1).sum()
            sens = tp / (tp + fn + 1e-8)
            spec = tn / (tn + fp + 1e-8)
            f1   = f1_score(_yt, preds, zero_division=0)
            prec = precision_score(_yt, preds, zero_division=0)
            rec  = recall_score(_yt, preds, zero_division=0)
            if spec < min_specificity:
                return -1
            if method == "youden":
                return sens + spec - 1
            elif method == "min_spec_f1":
                return f1 - 0.5 * abs(prec - rec)
            return sens + spec - 1

        opt = BayesianOptimization(
            f=objective, pbounds={"thresh": (0.1, 0.9)},
            verbose=0, random_state=123 + i,
        )
        try:
            opt.maximize(init_points=5, n_iter=15)
            best_ts.append(float(opt.max["params"]["thresh"]))
        except Exception as e:
            print(f"  [WARN] Bayesian opt failed for horizon {i}: {e}")
            best_ts.append(0.5)
    return best_ts


# ─────────────────────────────────────────────────────────────────────
# Evaluation on test set
# ─────────────────────────────────────────────────────────────────────
def evaluate(model, test_loader, thresholds, device):
    model.eval()
    auc_m  = MultilabelAUROC( num_labels=3, average=None).to(device)
    f1_m   = MultilabelF1Score(num_labels=3, average=None).to(device)
    rec_m  = MultilabelRecall( num_labels=3, average=None).to(device)
    spec_m = MultilabelSpecificity(num_labels=3, average=None).to(device)

    all_probs, all_preds, all_targets = [], [], []
    thresh_t = torch.tensor(thresholds, device=device)

    with torch.no_grad():
        for sx, seqx, y in test_loader:
            sx, seqx, y = sx.to(device), seqx.to(device), y.to(device)
            probs  = torch.sigmoid(model(sx, seqx))
            preds  = (probs >= thresh_t).long()
            all_probs.append(probs)
            all_preds.append(preds)
            all_targets.append(y.long())

    all_probs   = torch.cat(all_probs)
    all_preds   = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    metrics = {}
    auroc_per = auc_m(all_probs, all_targets)
    f1_per    = f1_m(all_preds,  all_targets)
    rec_per   = rec_m(all_preds, all_targets)
    spec_per  = spec_m(all_preds, all_targets)

    macro_auroc = auroc_per.mean().item()
    print(f"  Test macro-AUROC: {macro_auroc:.4f}")
    for i, name in enumerate(TARGET_NAMES):
        print(f"    {name}: AUROC={auroc_per[i]:.4f}  F1={f1_per[i]:.4f}"
              f"  Recall={rec_per[i]:.4f}  Spec={spec_per[i]:.4f}")
        metrics[name] = {
            "auroc":       auroc_per[i].item(),
            "f1":          f1_per[i].item(),
            "recall":      rec_per[i].item(),
            "specificity": spec_per[i].item(),
        }
    metrics["macro_auroc"] = macro_auroc
    return metrics


# ─────────────────────────────────────────────────────────────────────
# Single-seed training run
# ─────────────────────────────────────────────────────────────────────
def train_one_seed(seed, train_loader, val_loader, test_loader,
                   tgt_train, device, output_dir):
    print(f"\n{'='*60}")
    print(f"Seed {seed}")
    print(f"{'='*60}")
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Model
    model = TFTPredictor_FedTFT(
        input_dim  = SEQUENCE_SHAPE[-1],
        static_dim = STATIC_SHAPE[-1],
        hidden_dim = 64,
    ).to(device)

    # pos_weight per horizon
    pos_counts = tgt_train.sum(axis=0)
    neg_counts = len(tgt_train) - pos_counts
    pos_weight = torch.tensor(neg_counts / (pos_counts + 1e-6),
                               dtype=torch.float32).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = AdamW(model.parameters(), lr=PEAK_LR / 25, weight_decay=1e-5)
    # OneCycleLR horizon: 2×PATIENCE+1 epochs so peak LR is reached well before
    # early stopping fires (avoids the LR still being in warm-up when training stops).
    onecycle_epochs = PATIENCE * 2 + 1   # 31 with PATIENCE=15
    scheduler = OneCycleLR(
        optimizer,
        max_lr          = PEAK_LR,
        steps_per_epoch = len(train_loader),
        epochs          = onecycle_epochs,
        pct_start       = 0.3,
        anneal_strategy = "cos",
    )

    best_val_loss = float("inf")
    no_improve    = 0
    best_state    = copy.deepcopy(model.state_dict())

    for epoch in range(1, MAX_EPOCHS + 1):
        # ── Train ─────────────────────────────────────────────────
        model.train()
        for sx, seqx, ty in train_loader:
            sx, seqx, ty = sx.to(device), seqx.to(device), ty.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(sx, seqx), ty)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # ── Val loss ───────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sx, seqx, ty in val_loader:
                sx, seqx, ty = sx.to(device), seqx.to(device), ty.to(device)
                val_loss += loss_fn(model(sx, seqx), ty).item() * sx.size(0)
        val_loss /= len(val_loader.dataset)

        lr = scheduler.get_last_lr()[0]
        print(f"  Epoch {epoch:3d}/{MAX_EPOCHS} | val_loss={val_loss:.4f} | lr={lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve    = 0
            best_state    = copy.deepcopy(model.state_dict())
            print(f"  ✅ New best val_loss={best_val_loss:.4f}")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping at epoch {epoch} (patience={PATIENCE})")
                break

    # ── Restore best, tune thresholds, evaluate ──────────────────────
    model.load_state_dict(best_state)
    print("\n  Tuning thresholds on pooled val set (Bayesian, Youden)...")
    thresholds = tune_thresholds(model, val_loader, device)
    print(f"  Tuned thresholds: {thresholds}")

    # Save checkpoint
    ckpt_path = os.path.join(output_dir, f"centralized_seed{seed}_best.pth")
    torch.save(model.state_dict(), ckpt_path)
    thresh_path = os.path.join(output_dir, f"centralized_seed{seed}_thresholds.json")
    with open(thresh_path, "w") as f:
        json.dump(thresholds, f)
    print(f"  Saved checkpoint → {ckpt_path}")

    # Evaluate on global test set
    print("\n  Evaluating on global test set...")
    metrics = evaluate(model, test_loader, thresholds, device)
    return metrics


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Centralized TFT training — privacy upper-bound reference")
    parser.add_argument("--data_root", default="patient_level_split/last_npy_data",
                        help="Root directory containing GlobalData/ and HospitalsData/")
    parser.add_argument("--output",    default="results/centralized/",
                        help="Directory to save checkpoints and results JSON")
    parser.add_argument("--seeds",     type=int, nargs="+", default=[1, 2, 3],
                        help="Random seeds for multi-seed evaluation (default: 1 2 3)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    hospital_dir = os.path.join(args.data_root, "HospitalsData")
    global_dir   = os.path.join(args.data_root, "GlobalData")

    # ── Pool train and val splits across all hospitals ────────────────
    print("\nLoading pooled train set...")
    stat_tr, seq_tr, tgt_tr = pool_hospital_split(hospital_dir, "train")
    print(f"  Train: {len(stat_tr)} samples across all hospitals")

    print("Loading pooled val set...")
    stat_val, seq_val, tgt_val = pool_hospital_split(hospital_dir, "val")
    print(f"  Val  : {len(stat_val)} samples across all hospitals")

    print("Loading global test set...")
    stat_te  = load_memmap(os.path.join(global_dir, "static_data.npy"),   STATIC_SHAPE)
    seq_te   = load_memmap(os.path.join(global_dir, "sequence_data.npy"), SEQUENCE_SHAPE)
    tgt_te   = load_memmap(os.path.join(global_dir, "targets.npy"),       TARGETS_SHAPE)
    print(f"  Test : {len(stat_te)} samples (global held-out)")

    # ── DataLoaders ───────────────────────────────────────────────────
    train_loader = DataLoader(
        TFTDataset(list(zip(stat_tr,  seq_tr,  tgt_tr))),
        batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(
        TFTDataset(list(zip(stat_val, seq_val, tgt_val))),
        batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(
        TFTDataset(list(zip(stat_te,  seq_te,  tgt_te))),
        batch_size=BATCH_SIZE, shuffle=False)

    # ── Multi-seed training ───────────────────────────────────────────
    all_metrics = {}
    for seed in args.seeds:
        metrics = train_one_seed(
            seed, train_loader, val_loader, test_loader,
            tgt_tr, device, args.output)
        all_metrics[f"seed{seed}"] = metrics

    # ── Aggregate across seeds: mean ± SD AUROC ───────────────────────
    print(f"\n{'='*60}")
    print("Multi-seed Summary (mean ± SD)")
    print(f"{'='*60}")

    summary = {}
    for name in TARGET_NAMES + ["macro_auroc"]:
        if name == "macro_auroc":
            vals = [all_metrics[f"seed{s}"]["macro_auroc"] for s in args.seeds]
        else:
            vals = [all_metrics[f"seed{s}"][name]["auroc"] for s in args.seeds]
        mean_v = float(np.mean(vals))
        std_v  = float(np.std(vals))
        summary[name] = {"auroc_mean": mean_v, "auroc_std": std_v, "values": vals}
        label = "Macro" if name == "macro_auroc" else name
        print(f"  {label}: AUROC = {mean_v:.4f} ± {std_v:.4f}")

    # F1 and recall summary
    print()
    for name in TARGET_NAMES:
        f1s   = [all_metrics[f"seed{s}"][name]["f1"]     for s in args.seeds]
        recs  = [all_metrics[f"seed{s}"][name]["recall"]  for s in args.seeds]
        print(f"  {name}: F1 = {np.mean(f1s):.4f} ± {np.std(f1s):.4f} | "
              f"Recall = {np.mean(recs):.4f} ± {np.std(recs):.4f}")

    # Save full results
    results = {"per_seed": all_metrics, "summary": summary}
    out_path = os.path.join(args.output, "centralized_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
