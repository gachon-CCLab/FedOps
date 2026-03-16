"""
pvalue_ablation_fedtft.py — Statistical significance for FedTFT ablation table.

Uses the last 5 rounds of each ablation run as "folds" for paired testing.
Computes: paired t-test + Wilcoxon signed-rank between consecutive rows,
and vs FedTFT full (R4, our final model).

Usage:
    python experiments/ablation/pvalue_ablation_fedtft.py
    python experiments/ablation/pvalue_ablation_fedtft.py --seed 2
    python experiments/ablation/pvalue_ablation_fedtft.py --metric auroc_1h
    python experiments/ablation/pvalue_ablation_fedtft.py --last_n 10
    python experiments/ablation/pvalue_ablation_fedtft.py --all
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

RESULT_ROOT = Path("results") / "ablation"

# Ablation rows aligned to manuscript: R1–R4 (old R7 -> R4).
# Columns: proximal μ | FVWA | decoupled heads
ROWS = [
    ("R1 FedAvg",         "0.0",  "✗", "✗", "R1_fedavg"),
    ("R2 FedProx",        "1e-5", "✗", "✗", "R2_fedprox"),
    ("R3 +FVWA",          "1e-5", "✓", "✗", "R3_fvwa"),
    ("R4 FedTFT (ours)",  "1e-5", "✓", "✓", "R4_fedtft"),
]


def pvalue_stars(p):
    if p is None: return "N/A"
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def load_last_n(row_name, metric, last_n, seed):
    """Load last N rounds' metric from result.json, return as list."""
    path = RESULT_ROOT / row_name / f"seed{seed}" / "result.json"
    if not path.exists():
        # Legacy fallback (older layout without per-seed subdir)
        path = RESULT_ROOT / row_name / "result.json"
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text())
    except Exception:
        return None
    # Keys are "round1", "round2", ...
    rounds = sorted(d.keys(), key=lambda k: int(k.replace("round", "")))
    selected = rounds[-last_n:] if len(rounds) >= last_n else rounds
    values = []
    for r in selected:
        v = d[r].get(metric)
        if v is not None:
            values.append(float(v))
    return values if values else None


def _paired_test(a, b):
    """Paired t-test + Wilcoxon on equal-length tails. Returns (p_t, p_w)."""
    n = min(len(a), len(b))
    try:
        _, p_t = ttest_rel(a[-n:], b[-n:])
    except Exception:
        p_t = None
    try:
        _, p_w = wilcoxon(a[-n:], b[-n:])
    except Exception:
        p_w = 1.0
    return p_t, p_w


def run_table(metric, last_n, seed, verbose=True):
    if verbose:
        print(f"\n{'='*96}")
        print(f"FedTFT Ablation — metric={metric}, last_n={last_n} rounds as folds")
        print(f"{'='*96}")
        print(f"{'Row':<22} {'μ':^8} {'FVWA':^5} {'DH':^4} "
              f"{'Mean':>8} {'Std':>6}  {'vs prev':>24}  {'vs R4':>24}")
        print("-" * 96)

    data = []
    for label, mu, fvwa, dh, name in ROWS:
        vals = load_last_n(name, metric, last_n, seed)
        mean = float(np.mean(vals)) if vals else None
        std  = float(np.std(vals, ddof=1)) if vals and len(vals) > 1 else None
        data.append((label, mu, fvwa, dh, vals, mean, std))

    r4_vals = data[-1][4]   # R4 = FedTFT full (final row)

    table_out = []
    for i, (label, mu, fvwa, dh, vals, mean, std) in enumerate(data):
        mean_s = f"{mean:.4f}" if mean is not None else "N/A"
        std_s  = f"{std:.4f}"  if std  is not None else "N/A"

        # vs previous row
        vs_prev_s = ""
        if i > 0 and vals and data[i-1][4]:
            p_t, p_w = _paired_test(vals, data[i-1][4])
            if p_t is not None:
                vs_prev_s = f"p={p_t:.4f}{pvalue_stars(p_t)} W={p_w:.4f}"

        # vs R4 (FedTFT full)
        vs_r4_s = ""
        if i < len(data)-1 and vals and r4_vals:
            p_t, p_w = _paired_test(vals, r4_vals)
            if p_t is not None:
                vs_r4_s = f"p={p_t:.4f}{pvalue_stars(p_t)} W={p_w:.4f}"

        if verbose:
            print(f"{label:<22} {mu:^8} {fvwa:^5} {dh:^4} "
                  f"{mean_s:>8} {std_s:>6}  {vs_prev_s:>24}  {vs_r4_s:>24}")

        table_out.append({
            "row": label, "mean": mean, "std": std, "vals": vals,
        })

    if verbose:
        print()
        print("μ=FedProx proximal coefficient, FVWA=performance-weighted aggregation,")
        print("DH=decoupled horizon heads (FedTFT novelty).")
        print("Significance: *** p<0.001  ** p<0.01  * p<0.05  ns p≥0.05")
        print("paired t-test (df=last_n-1) + Wilcoxon W (non-parametric) shown")

    return table_out


def print_all_metrics(last_n, seed):
    metrics = [
        "overall_accuracy",
        "auroc_1h", "auroc_1d", "auroc_1w",
        "f1_1h", "f1_1d", "f1_1w",
        "spec_1h", "spec_1d", "spec_1w",
        "rec_1h", "rec_1d", "rec_1w",
    ]
    print("\n" + "="*100)
    print(f"FULL ABLATION SUMMARY (last {last_n} rounds as folds)")
    print("="*100)

    header = f"{'Row':<22}"
    for m in metrics:
        header += f"  {m:<12}"
    print(header)
    print("-" * (22 + 14 * len(metrics)))

    for label, _, _, _, name in ROWS:
        line = f"{label:<22}"
        for m in metrics:
            vals = load_last_n(name, m, last_n, seed)
            if vals:
                line += f"  {np.mean(vals):.4f}±{np.std(vals,ddof=1):.4f}"
            else:
                line += "  N/A         "
        print(line)

    # P-value matrix vs R4
    r4_name = ROWS[-1][4]
    print(f"\nP-VALUES vs R4 FedTFT (ours) (paired t-test, last {last_n} rounds)")
    print("-" * 80)
    for metric in ["auroc_1h", "auroc_1d", "auroc_1w", "f1_1h", "f1_1d", "f1_1w"]:
        ref_vals = load_last_n(r4_name, metric, last_n, seed)
        print(f"\n  {metric}:")
        for row in ROWS:
            label, name = row[0], row[4]
            if name == r4_name:
                continue
            vals = load_last_n(name, metric, last_n, seed)
            if vals and ref_vals:
                p_t, _ = _paired_test(vals, ref_vals)
                if p_t is not None:
                    print(f"    {label:<22} p={p_t:.4f} {pvalue_stars(p_t)}")
                else:
                    print(f"    {label:<22} err")
            else:
                print(f"    {label:<22} N/A (missing data)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric",  type=str, default="auroc_1h",
                        help="Metric to show in main table")
    parser.add_argument("--last_n",  type=int, default=5,
                        help="Use last N rounds as 'folds' for paired tests")
    parser.add_argument("--seed",    type=int, default=1,
                        help="Seed index to read (results/ablation/<run>/seedN/result.json)")
    parser.add_argument("--all",     action="store_true",
                        help="Print full multi-metric summary table")
    args = parser.parse_args()

    run_table(args.metric, args.last_n, args.seed, verbose=True)

    if args.all:
        print_all_metrics(args.last_n, args.seed)
