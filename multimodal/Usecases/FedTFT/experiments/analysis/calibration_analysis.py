"""
calibration_analysis.py — Model calibration: Brier score + reliability diagrams.

Outputs:
  - Brier score per horizon (lower = better calibration)
  - Bootstrap 95% CI for Brier score
  - Reliability diagram per horizon (predicted prob vs fraction positive)
  - Calibration summary JSON

Usage:
  python experiments/calibration_analysis.py \\
      --predictions results/fedtft_preds.npz \\
      --baseline    results/timesnet_preds.npz \\
      --output      results/significance/calibration/

Run from FedTFT_paper/ root directory.
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

HORIZON_NAMES = ["1h", "1d", "1w"]
N_BOOTSTRAP   = 2000
ALPHA         = 0.05
SEED          = 42

rng = np.random.default_rng(SEED)


# ──────────────────────────────────────────────────────────────────
# Brier score
# ──────────────────────────────────────────────────────────────────
def brier_score(y_true, y_prob):
    """Mean squared error between predicted probability and true label."""
    return float(np.mean((y_prob - y_true) ** 2))


def bootstrap_brier(y_true, y_prob, n=N_BOOTSTRAP):
    """95% bootstrap CI for Brier score."""
    n_samples = len(y_true)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, n_samples, size=n_samples)
        scores.append(brier_score(y_true[idx], y_prob[idx]))
    scores = np.array(scores)
    point  = brier_score(y_true, y_prob)
    ci_lo  = np.percentile(scores, 100 * ALPHA / 2)
    ci_hi  = np.percentile(scores, 100 * (1 - ALPHA / 2))
    return point, ci_lo, ci_hi


# ──────────────────────────────────────────────────────────────────
# Reliability diagrams
# ──────────────────────────────────────────────────────────────────
def plot_reliability_diagrams(models_data, outdir, n_bins=10):
    """
    models_data: dict of {model_label: (probs [N,3], targets [N,3])}
    Plots 3 panels (one per horizon), one curve per model.
    """
    colors = {"FedTFT": "#2196F3", "Baseline": "#FF9800"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, h in enumerate(HORIZON_NAMES):
        ax = axes[i]
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")

        for label, (probs, targets) in models_data.items():
            yt     = targets[:, i]
            yp     = probs[:, i]
            # Skip if no positives or no negatives
            if yt.sum() == 0 or yt.sum() == len(yt):
                continue
            try:
                frac_pos, mean_pred = calibration_curve(yt, yp, n_bins=n_bins,
                                                        strategy="uniform")
                color = colors.get(label, "#4CAF50")
                brier = brier_score(yt, yp)
                ax.plot(mean_pred, frac_pos, marker="o", lw=1.5, color=color,
                        label=f"{label} (Brier={brier:.3f})")
            except Exception as e:
                print(f"[WARN] Reliability curve failed for {label}/{h}: {e}")

        ax.set_xlabel("Mean predicted probability", fontsize=10)
        ax.set_ylabel("Fraction of positives", fontsize=10)
        ax.set_title(f"Reliability Diagram — {h} horizon", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.suptitle("Calibration Reliability Diagrams (FedTFT)", fontsize=13, y=1.02)
    plt.tight_layout()
    out = os.path.join(outdir, "reliability_diagrams.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Reliability diagram saved → {out}")


def plot_brier_comparison(brier_data, outdir):
    """
    brier_data: dict of {model_label: [brier_1h, brier_1d, brier_1w]}
    Bar chart comparing Brier scores across models and horizons.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(HORIZON_NAMES))
    n_models = len(brier_data)
    width = 0.7 / n_models
    colors = ["#2196F3", "#FF9800", "#9C27B0", "#4CAF50"]

    for j, (label, scores) in enumerate(brier_data.items()):
        ax.bar(x + j * width, scores, width, label=label, color=colors[j % len(colors)])

    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(HORIZON_NAMES)
    ax.set_ylabel("Brier Score (lower = better)")
    ax.set_title("Calibration: Brier Score Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out = os.path.join(outdir, "brier_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Brier comparison saved → {out}")


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Calibration analysis: Brier score + reliability diagrams")
    parser.add_argument("--predictions", required=True,
                        help="FedTFT predictions .npz (needs 'probs', 'preds', 'targets')")
    parser.add_argument("--baseline", default=None,
                        help="Baseline predictions .npz (optional)")
    parser.add_argument("--output", default="results/significance/calibration/")
    parser.add_argument("--n_bins", type=int, default=10,
                        help="Number of bins for reliability diagram")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # ── Load predictions ─────────────────────────────────────────
    ft = np.load(args.predictions)
    probs_ft   = ft["probs"].astype(np.float32)    # [N, 3]
    targets    = ft["targets"].astype(np.int32)    # [N, 3]
    N = len(targets)
    print(f"Loaded FedTFT: N={N} samples")

    models_data = {"FedTFT": (probs_ft, targets)}

    if args.baseline:
        bl = np.load(args.baseline)
        probs_bl = bl["probs"].astype(np.float32)
        models_data["Baseline"] = (probs_bl, targets)
        print(f"Loaded baseline predictions")

    results = {}

    # ── 1) Brier scores + bootstrap CI ───────────────────────────
    print("\n=== Brier Score (95% Bootstrap CI) ===")
    print(f"{'Model':<15} " + "  ".join(f"{'Brier_'+h:>18}" for h in HORIZON_NAMES))
    print("-" * 70)

    brier_data = {}
    for label, (probs, tgts) in models_data.items():
        row_scores = []
        ci_dict    = {}
        for i, h in enumerate(HORIZON_NAMES):
            yt = tgts[:, i].astype(np.float32)
            yp = probs[:, i]
            b, lo, hi = bootstrap_brier(yt, yp)
            row_scores.append(b)
            ci_dict[h] = {"brier": float(b), "ci_95": [float(lo), float(hi)]}
        results[label] = {"brier_ci": ci_dict}
        brier_data[label] = row_scores
        row_str = "  ".join(f"{row_scores[i]:6.4f} [{results[label]['brier_ci'][h]['ci_95'][0]:.4f}, "
                            f"{results[label]['brier_ci'][h]['ci_95'][1]:.4f}]"
                            for i, h in enumerate(HORIZON_NAMES))
        print(f"{label:<15} {row_str}")

    # ── 2) Null Brier (predict base rate) ────────────────────────
    print("\n--- Null model Brier (predict base rate) ---")
    null_scores = {}
    for i, h in enumerate(HORIZON_NAMES):
        yt = targets[:, i].astype(np.float32)
        base_rate = yt.mean()
        null_b = brier_score(yt, np.full_like(yt, base_rate))
        null_scores[h] = float(null_b)
        print(f"  {h}: base_rate={base_rate:.3f}, null_Brier={null_b:.4f}")
    results["null_brier"] = null_scores

    # ── 3) Brier skill score (BSS = 1 - Brier/Brier_ref) ─────────
    print("\n=== Brier Skill Score (BSS, higher = better) ===")
    bss_results = {}
    for label, (probs, tgts) in models_data.items():
        bss_dict = {}
        for i, h in enumerate(HORIZON_NAMES):
            yt   = tgts[:, i].astype(np.float32)
            yp   = probs[:, i]
            b    = brier_score(yt, yp)
            b_null = null_scores[h]
            bss  = 1 - b / (b_null + 1e-8)
            bss_dict[h] = float(bss)
        bss_results[label] = bss_dict
        bss_str = "  ".join(f"{bss_dict[h]:+.4f}" for h in HORIZON_NAMES)
        print(f"  {label:<15}: {bss_str}")
    results["brier_skill_score"] = bss_results

    # ── 4) Reliability diagrams ───────────────────────────────────
    print("\nGenerating reliability diagrams...")
    plot_reliability_diagrams(models_data, args.output, n_bins=args.n_bins)

    # ── 5) Brier comparison bar chart ─────────────────────────────
    if len(models_data) > 1:
        plot_brier_comparison(brier_data, args.output)

    # ── 6) Summary table (print) ──────────────────────────────────
    print("\n=== Calibration Summary Table ===")
    print(f"{'Model':<20} {'Brier 1h':>10} {'Brier 1d':>10} {'Brier 1w':>10} "
          f"{'BSS 1h':>10} {'BSS 1d':>10} {'BSS 1w':>10}")
    print("-" * 80)
    for label in models_data:
        briersrow = [results[label]["brier_ci"][h]["brier"] for h in HORIZON_NAMES]
        bssrow    = [results.get("brier_skill_score", {}).get(label, {}).get(h, float('nan'))
                     for h in HORIZON_NAMES]
        print(f"{label:<20} "
              + " ".join(f"{b:10.4f}" for b in briersrow) + "  "
              + " ".join(f"{b:+10.4f}" for b in bssrow))

    print("\nCalibration results (JSON save disabled):")
    print(results)


if __name__ == "__main__":
    main()
