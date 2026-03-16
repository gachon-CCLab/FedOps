"""
generate_figure4.py — Figure 4: Per-horizon SHAP feature importance (FedTFT).


Run from FedTFT_paper/ root:
  source activate akeel_env
  python experiments/analysis/generate_figure4.py
"""

import os, json, argparse, textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_JSON_PATH = os.path.join(
    ROOT, "results/shap/shap_rankings_all_hospitals.json"
)
DEFAULT_OUT_PNG   = os.path.join(ROOT, "Manuscript/Figure4.png")

RENAME = {
    "VE_tx_inj_time":          "Treatment timing",
    "VE_tx":                   "Treatment event marker",
    "HR_nunique":              "HR_nunique",
    "day_of_week":             "Day of week",
    "DIG_3class":              "Diagnosis (3-cls)",
    "DIG_4class":              "Diagnosis (4-cls)",
    "DIG_withPsychosis":       "With psychosis",
    "STEP_delta":              "Step change",
    "HR_median":               "HR median",
    "HR_mean":                 "HR mean",
    "HR_min":                  "HR min",
    "HR_max":                  "HR max",
    "HR_std":                  "HR std",
    "ENMO_mean":               "Activity (ENMO)",
    "ENMO_nunique":            "ENMO variability",
    "MESOR_HR_week":           "HR MESOR (wk)",
    "Amplitude_HR_week":       "HR amplitude (wk)",
    "Phase_hours_HR_week":     "HR phase (wk)",
    "MESOR_ENMO_week":         "ENMO MESOR (wk)",
    "Amplitude_ENMO_week":     "ENMO amplitude (wk)",
    "Phase_hours_ENMO_week":   "ENMO phase (wk)",
    "place_other":             "Location: other",
    "place_ward":              "Location: ward",
    "holidays":                "Holiday",
    "Daily_Entropy":           "Daily entropy",
    "SLEEP_delta":             "Sleep change",
    "DISTANCE_delta":          "Distance change",
    "CALORIES_delta":          "Calories change",
    "age":                     "Age",
    "sex":                     "Sex",
    "nonwearing":              "Non-wearing",
}

HORIZON_COLORS = {
    "1h": "#1565C0",   # blue
    "1d": "#2E7D32",   # green
    "1w": "#C62828",   # red
}
HORIZON_LABELS = {
    "1h": "1-hour horizon",
    "1d": "1-day horizon",
    "1w": "1-week horizon",
}

WHAT_TO_LEARN = None  # removed — discussed in manuscript text


def wrap_label(name, width=22):
    if len(name) <= width:
        return name
    return "\n".join(textwrap.wrap(name, width=width))


def horizon_specific_order(rankings):
    """
    Reorder features by horizon-specific lift:
      lift_h(f) = I_h(f) - mean_{o!=h} I_o(f)
    while keeping bar value as I_h(f).
    """
    horizons = ["1h", "1d", "1w"]
    features = list(rankings["1h"].keys())
    out = {}
    for h in horizons:
        others = [o for o in horizons if o != h]
        scored = []
        for f in features:
            ih = rankings[h][f]
            io = 0.5 * (rankings[others[0]][f] + rankings[others[1]][f])
            lift = ih - io
            scored.append((f, lift, ih))
        scored.sort(key=lambda t: t[1], reverse=True)
        out[h] = {f: float(ih) for f, _, ih in scored}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default=DEFAULT_JSON_PATH, help="Path to SHAP ranking JSON.")
    parser.add_argument("--output", default=DEFAULT_OUT_PNG, help="Output figure path.")
    parser.add_argument("--top_n", type=int, default=10, help="Number of features per horizon panel.")
    parser.add_argument(
        "--rank_mode",
        choices=["absolute", "horizon_specific"],
        default="absolute",
        help=(
            "absolute: top by mean |SHAP| per horizon; "
            "horizon_specific: reorder by per-horizon lift vs other horizons."
        ),
    )
    args = parser.parse_args()

    with open(args.json) as f:
        rankings = json.load(f)
    if args.rank_mode == "horizon_specific":
        rankings = horizon_specific_order(rankings)

    # Three per-horizon panels (one row per prediction horizon).
    fig, axes = plt.subplots(3, 1, figsize=(10.6, 14.6),
                             gridspec_kw={"hspace": 0.40})

    for ax, horizon in zip(axes, ["1h", "1d", "1w"]):
        items = sorted(rankings[horizon].items(), key=lambda kv: kv[1], reverse=True)[:args.top_n]
        names = [wrap_label(RENAME.get(k, k), width=22) for k, _ in items]
        values = [v for _, v in items]

        color = HORIZON_COLORS[horizon]
        y_pos = range(len(names))
        ax.barh(y_pos, values, height=0.56, color=color, alpha=0.82, linewidth=0)

        for i, v in enumerate(values):
            ax.text(v + max(values) * 0.03, i, f"{v:.4f}",
                    va="center", ha="left", fontsize=14, color=color, clip_on=False)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=18)
        ax.set_xlabel("Mean |SHAP value|", fontsize=16)
        ax.set_title(HORIZON_LABELS[horizon], fontsize=20, fontweight="bold", pad=12)
        ax.set_xlim(0, max(values) * 1.70)
        ax.margins(y=0.10)
        ax.invert_yaxis()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", length=0, labelsize=15)
        ax.grid(axis="x", lw=0.4, alpha=0.4)

    if args.rank_mode == "horizon_specific":
        title = "Per-Horizon Feature Importance — FedTFT (SHAP, horizon-specific order)"
    else:
        title = "Per-Horizon Feature Importance — FedTFT (SHAP)"
    fig.suptitle(title, fontsize=24, fontweight="bold", y=0.985)
    # Keep enough room for wrapped y-labels while avoiding right-shift.
    fig.subplots_adjust(left=0.40, right=0.98, top=0.91, bottom=0.06, hspace=0.40)

    # Key insight callout removed — discussed in manuscript text
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
