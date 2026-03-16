"""
significance_test.py — Statistical significance testing for FedTFT

Tests:
  1. Bootstrap confidence intervals (95% CI) for event-F1, event-recall, and Brier score
     (AUROC uses ±SD over 3 seeds, NOT bootstrap — omitted here)
  2. McNemar's test: FedTFT vs each baseline (paired binary predictions)
  3. Wilcoxon signed-rank: R3 shared-head baseline vs R4 FedTFT (probability differences)

Usage:
  python experiments/significance_test.py \
    --predictions results/fedtft_preds.npz \
    --baseline    results/timesnet_preds.npz \
    --fedtft_local results/fedtft_local_preds.npz \
    --output      results/significance/

Each .npz file must contain:
  - 'probs':   float32 array [N, 3] — predicted probabilities per horizon
  - 'preds':   int32   array [N, 3] — binary predictions (0/1) per horizon
  - 'targets': int32   array [N, 3] — ground truth labels per horizon
"""

import os
import sys
import argparse
import numpy as np
import json
from scipy.stats import ttest_rel, wilcoxon
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score

TARGET_NAMES = ['1h', '1d', '1w']
N_BOOTSTRAP  = 2000
ALPHA        = 0.05
SEED         = 42

rng = np.random.default_rng(SEED)


# ─────────────────────────────────────────────────────────────────
# Bootstrap CI
# ─────────────────────────────────────────────────────────────────
def bootstrap_metric(y_true, y_pred, y_prob, metric_fn, n=N_BOOTSTRAP):
    """
    Compute bootstrap CI for a scalar metric.
    metric_fn(y_true, y_pred, y_prob) → float
    """
    n_samples = len(y_true)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, n_samples, size=n_samples)
        try:
            scores.append(metric_fn(y_true[idx], y_pred[idx], y_prob[idx]))
        except Exception:
            pass  # skip degenerate bootstrap samples
    scores = np.array(scores)
    ci_lo = np.percentile(scores, 100 * ALPHA / 2)
    ci_hi = np.percentile(scores, 100 * (1 - ALPHA / 2))
    point = metric_fn(y_true, y_pred, y_prob)
    return point, ci_lo, ci_hi


def auroc_fn(yt, yp_bin, yp_prob):
    if yt.sum() == 0 or yt.sum() == len(yt):
        return float('nan')
    return roc_auc_score(yt, yp_prob)

def f1_fn(yt, yp_bin, yp_prob):
    return f1_score(yt, yp_bin, zero_division=0)

def recall_fn(yt, yp_bin, yp_prob):
    return recall_score(yt, yp_bin, zero_division=0)

def specificity_fn(yt, yp_bin, yp_prob):
    tn = ((yp_bin == 0) & (yt == 0)).sum()
    fp = ((yp_bin == 1) & (yt == 0)).sum()
    return tn / (tn + fp + 1e-8)

def brier_fn(yt, yp_bin, yp_prob):
    """Brier score = mean squared error of predicted probability vs true label."""
    return float(np.mean((yp_prob - yt.astype(np.float32)) ** 2))


# ─────────────────────────────────────────────────────────────────
# McNemar's test (paired predictions: correct vs incorrect)
# ─────────────────────────────────────────────────────────────────
def mcnemar_test(preds_a, preds_b, targets):
    """
    McNemar's test between model A and model B.
    Returns chi2, p-value for each horizon.
    """
    results = []
    for i in range(3):
        correct_a = (preds_a[:, i] == targets[:, i])
        correct_b = (preds_b[:, i] == targets[:, i])
        # Contingency table: [correct-both, A-only, B-only, neither]
        b = ((correct_a) & (~correct_b)).sum()   # A correct, B wrong
        c = ((~correct_a) & (correct_b)).sum()   # A wrong, B correct
        table = np.array([[0, b], [c, 0]])        # McNemar uses b and c
        try:
            res = mcnemar(table, exact=True if min(b, c) < 25 else False)
            results.append({'horizon': TARGET_NAMES[i],
                            'b': int(b), 'c': int(c),
                            'statistic': res.statistic, 'p_value': res.pvalue})
        except Exception as e:
            results.append({'horizon': TARGET_NAMES[i],
                            'b': int(b), 'c': int(c),
                            'statistic': None, 'p_value': None, 'error': str(e)})
    return results


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Statistical significance tests for FedTFT")
    parser.add_argument("--predictions", required=True,
                        help="Path to FedTFT predictions .npz")
    parser.add_argument("--baseline", default=None,
                        help="Path to baseline predictions .npz (for McNemar)")
    parser.add_argument("--fedtft_local", default=None,
                        help="Path to comparison model predictions .npz (typically R4 vs R3)")
    parser.add_argument("--output", default="results/significance/",
                        help="Output directory for results JSON")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load FedTFT predictions
    fedtft = np.load(args.predictions)
    probs_ft   = fedtft['probs'].astype(np.float32)    # [N, 3]
    preds_ft   = fedtft['preds'].astype(np.int32)      # [N, 3]
    targets    = fedtft['targets'].astype(np.int32)    # [N, 3]
    N = len(targets)
    print(f"Loaded FedTFT predictions: N={N} samples")

    results = {}

    # ── 1) Bootstrap CIs for FedTFT (event-F1, event-recall, Brier only) ─────────
    # NOTE: AUROC uses ±SD over 3 seeds (per manuscript Table III/IV) — NOT bootstrap.
    print("\n=== Bootstrap 95% CI (FedTFT) — F1, Recall, Brier ===")
    ci_results = {}
    for i, name in enumerate(TARGET_NAMES):
        yt = targets[:, i]
        yp_bin  = preds_ft[:, i]
        yp_prob = probs_ft[:, i]

        f1,    f1_lo,    f1_hi    = bootstrap_metric(yt, yp_bin, yp_prob, f1_fn)
        rec,   rec_lo,   rec_hi   = bootstrap_metric(yt, yp_bin, yp_prob, recall_fn)
        brier, brier_lo, brier_hi = bootstrap_metric(yt, yp_bin, yp_prob, brier_fn)

        ci_results[name] = {
            'f1':     {'point': f1,    'ci_95': [f1_lo,    f1_hi]},
            'recall': {'point': rec,   'ci_95': [rec_lo,   rec_hi]},
            'brier':  {'point': brier, 'ci_95': [brier_lo, brier_hi]},
        }
        print(f"  {name}: F1={f1:.3f} [{f1_lo:.3f}, {f1_hi:.3f}] | "
              f"Recall={rec:.3f} [{rec_lo:.3f}, {rec_hi:.3f}] | "
              f"Brier={brier:.4f} [{brier_lo:.4f}, {brier_hi:.4f}]")
    results['bootstrap_ci_fedtft'] = ci_results

    # ── 2) McNemar's test vs baseline ───────────────────────────────────
    if args.baseline:
        baseline = np.load(args.baseline)
        preds_bl = baseline['preds'].astype(np.int32)
        print("\n=== McNemar's Test: FedTFT vs Baseline ===")
        mc_results = mcnemar_test(preds_ft, preds_bl, targets)
        for r in mc_results:
            sig = "*" if (r['p_value'] is not None and r['p_value'] < ALPHA) else ""
            pval_str = f"{r['p_value']:.4f}" if r['p_value'] is not None else 'N/A'
            print(f"  {r['horizon']}: b={r['b']}, c={r['c']}, p={pval_str} {sig}")
        results['mcnemar_vs_baseline'] = mc_results

    # ── 3) Paired comparison: shared-head (R3) vs decoupled-head FedTFT (R4) ────────
    # Wilcoxon signed-rank on per-sample predicted probabilities.
    # NOTE: AUROC comparison uses ±SD over 3 seeds (Table III) — no bootstrap here.
    if args.fedtft_local:
        fedtft_local = np.load(args.fedtft_local)
        probs_hdfp = fedtft_local['probs'].astype(np.float32)
        preds_hdfp = fedtft_local['preds'].astype(np.int32)
        print("\n=== Shared-head (R3) vs Decoupled-head (R4) — Wilcoxon signed-rank ===")
        paired_results = {}
        for i, name in enumerate(TARGET_NAMES):
            try:
                _, p_wil = wilcoxon(probs_hdfp[:, i] - probs_ft[:, i])
            except Exception:
                p_wil = float('nan')
            sig = "*" if (not np.isnan(p_wil) and p_wil < ALPHA) else ""
            print(f"  {name}: p_wilcoxon={p_wil:.4f} {sig}")
            paired_results[name] = {
                'p_wilcoxon': float(p_wil),
                'significant': bool(not np.isnan(p_wil) and p_wil < ALPHA),
            }
        results['shared_vs_decoupled'] = paired_results

    print("\nResults (JSON save disabled):")
    print(results)


if __name__ == "__main__":
    main()
