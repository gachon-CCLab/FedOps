"""
shap_analysis_fedtft.py — Faithful SHAP for FedTFT full model (R4).

Default behavior:
  - Loads the best FedTFT checkpoint (R4)
    (patient_level_split/last_npy_data/GlobalData/ablation_R4_fedtft_best.pth)
  - Uses full sequence input [N, 192, 25] (not last-timestep replication)
  - Uses SHAP GradientExplainer on model probability per horizon
  - Outputs ranking JSON + case-study JSON + metadata for traceability

IMPORTANT:
  - Case-study indices are pooled test-sample indices, not unique patient IDs.
  - This script does NOT modify manuscript text or tables.
"""

import os
import sys
import json
import argparse
import hashlib
from datetime import datetime

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add repo root to PYTHONPATH for model import
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

STATIC_SHAPE   = (14,)
SEQUENCE_SHAPE = (192, 25)
TARGETS_SHAPE  = (3,)
HOSPITAL_DIR   = "patient_level_split/last_npy_data/HospitalsData"

STATIC_FEATURE_NAMES = [
    "HR_nunique", "nonwearing", "VE_tx", "VE_tx_inj_time",
    "Restrictive_Intervention", "sex", "DIG_3class", "DIG_4class",
    "DIG_withPsychosis", "day_of_week", "holidays",
    "place_hallway", "place_other", "place_ward"
]
SEQ_FEATURE_NAMES = [
    "Daily_Entropy", "Normalized_Daily_Entropy", "Location_Variability",
    "ENMO_mean", "ENMO_std", "ENMO_min", "ENMO_max", "ENMO_median", "ENMO_nunique",
    "HR_mean", "HR_std", "HR_min", "HR_max", "HR_median",
    "DISTANCE_delta", "SLEEP_delta", "STEP_delta", "CALORIES_delta", "age",
    "MESOR_ENMO_week", "Amplitude_ENMO_week", "Phase_hours_ENMO_week",
    "MESOR_HR_week", "Amplitude_HR_week", "Phase_hours_HR_week"
]
HORIZON_NAMES = ["1h", "1d", "1w"]
FEATURE_NAMES = STATIC_FEATURE_NAMES + SEQ_FEATURE_NAMES


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_memmap_data(path, shape, dtype=np.float32):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    size = os.path.getsize(path)
    per  = np.prod(shape) * np.dtype(dtype).itemsize
    n    = size // per
    return np.memmap(path, dtype=dtype, mode="r", shape=(n,) + shape)


def load_all_hospitals(split="test"):
    all_s, all_q, all_t = [], [], []
    per_hospital_counts = {}
    for h in sorted(os.listdir(HOSPITAL_DIR)):
        base = os.path.join(HOSPITAL_DIR, h)
        try:
            s = load_memmap_data(os.path.join(base, f"static_{split}.npy"),   STATIC_SHAPE)
            q = load_memmap_data(os.path.join(base, f"sequence_{split}.npy"), SEQUENCE_SHAPE)
            t = load_memmap_data(os.path.join(base, f"targets_{split}.npy"),  TARGETS_SHAPE)
            all_s.append(s)
            all_q.append(q)
            all_t.append(t)
            per_hospital_counts[h] = int(len(s))
        except Exception as e:
            print(f"[WARN] Skip {h}: {e}")
    return (
        np.concatenate(all_s),
        np.concatenate(all_q),
        np.concatenate(all_t),
        per_hospital_counts,
    )


def load_model(checkpoint_path, hidden_dim, device, model_variant):
    """
    Load checkpoint strictly for a supported model variant.

    model_variant:
      - fedtft:      FedTFT full model with horizon-decoupled heads (model_fedtft_hdfp)
      - shared_head: shared-head FedTFT (model_fedtft, for R1–R3 ablation checkpoints)
    """
    if model_variant == "fedtft":
        from model_fedtft_hdfp import TFTPredictor_FedTFT
        model = TFTPredictor_FedTFT(
            input_dim=SEQUENCE_SHAPE[-1],
            static_dim=STATIC_SHAPE[-1],
            hidden_dim=hidden_dim,
        ).to(device)
    elif model_variant == "shared_head":
        from model_fedtft import TFTPredictor
        model = TFTPredictor(
            input_dim=SEQUENCE_SHAPE[-1],
            static_dim=STATIC_SHAPE[-1],
            hidden_dim=hidden_dim,
            output_dim=TARGETS_SHAPE[-1],
        ).to(device)
    else:
        raise ValueError(f"Unsupported model_variant: {model_variant}")

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"Loaded {model_variant} model checkpoint: {checkpoint_path}")
    return model


class HorizonProbabilityModel(torch.nn.Module):
    """Torch module returning p(y_h=1) for one horizon."""
    def __init__(self, model, horizon_idx):
        super().__init__()
        self.model = model
        self.horizon_idx = horizon_idx

    def forward(self, static_x, seq_x):
        logits = self.model(static_x, seq_x)  # [N, 3]
        return torch.sigmoid(logits[:, self.horizon_idx:self.horizon_idx + 1])  # [N,1]


def _squeeze_output_dim(arr):
    arr = np.asarray(arr)
    if arr.ndim >= 1 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr


def compute_shap_values_full_sequence(
    model,
    device,
    static_data,
    seq_data,
    horizon_idx,
    n_background=100,
    n_explain=200,
    n_shap_samples=100,
    seed=42,
):
    """
    Returns:
      shap_static: [Nexp, 14]
      shap_seq:    [Nexp, 192, 25]
      exp_idx:     [Nexp]
      bg_idx:      [Nbg]
    """
    import shap

    n_total = len(static_data)
    rng = np.random.default_rng(seed + horizon_idx)
    bg_idx  = rng.choice(n_total, size=min(n_background, n_total), replace=False)
    exp_idx = rng.choice(n_total, size=min(n_explain, n_total), replace=False)

    bg_static = torch.tensor(np.asarray(static_data[bg_idx]), dtype=torch.float32, device=device)
    bg_seq    = torch.tensor(np.asarray(seq_data[bg_idx]), dtype=torch.float32, device=device)
    ex_static = torch.tensor(np.asarray(static_data[exp_idx]), dtype=torch.float32, device=device)
    ex_seq    = torch.tensor(np.asarray(seq_data[exp_idx]), dtype=torch.float32, device=device)

    wrapped = HorizonProbabilityModel(model, horizon_idx)
    explainer = shap.GradientExplainer(wrapped, [bg_static, bg_seq])
    shap_vals = explainer.shap_values([ex_static, ex_seq], nsamples=n_shap_samples)
    if not isinstance(shap_vals, list) or len(shap_vals) != 2:
        raise RuntimeError("Unexpected SHAP return format for multi-input model.")

    shap_static = _squeeze_output_dim(shap_vals[0])  # [Nexp, 14]
    shap_seq    = _squeeze_output_dim(shap_vals[1])  # [Nexp, 192, 25]
    return shap_static, shap_seq, exp_idx, bg_idx


def rank_features(shap_static, shap_seq):
    """
    Aggregate sequence SHAP over time to keep 39-feature comparability:
      static feature importance = mean(|SHAP_static|)
      sequence feature importance = mean_t,mean_n(|SHAP_seq|)
    """
    imp_static = np.abs(shap_static).mean(axis=0)       # [14]
    imp_seq    = np.abs(shap_seq).mean(axis=(0, 1))     # [25]
    all_imp = np.concatenate([imp_static, imp_seq], axis=0)  # [39]
    ranked = sorted(zip(FEATURE_NAMES, all_imp), key=lambda x: -x[1])
    return {k: float(v) for k, v in ranked}


def plot_bar(rankings, horizon_name, label, outdir, top_n=15):
    top_items = list(rankings.items())[:top_n]
    names = [x[0] for x in top_items]
    vals  = [x[1] for x in top_items]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(list(reversed(names)), list(reversed(vals)), color="#2196F3")
    ax.set_xlabel("Mean |SHAP|")
    ax.set_title(f"Top-{top_n} Feature Importance [{label} — {horizon_name}]", fontsize=12)
    plt.tight_layout()
    out = os.path.join(outdir, f"shap_bar_{label}_{horizon_name}.png")
    plt.savefig(out, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def extract_case_study_full_sequence(
    model,
    device,
    static_data,
    seq_data,
    bg_idx,
    case_idx,
    horizon_idx,
    case_shap_samples=200,
):
    """
    Case-study output format (compatible with generate_figure5.py):
      {
        "features": [(name, shap), ... top5 ...],
        "fx": float(pred_prob),
        "efx": float(mean_bg_prob),
        "n_other": int,
        "other": float(sum_rest)
      }
    """
    import shap

    wrapped = HorizonProbabilityModel(model, horizon_idx)

    bg_static = torch.tensor(np.asarray(static_data[bg_idx]), dtype=torch.float32, device=device)
    bg_seq    = torch.tensor(np.asarray(seq_data[bg_idx]), dtype=torch.float32, device=device)

    x_static = torch.tensor(np.asarray(static_data[case_idx:case_idx + 1]), dtype=torch.float32, device=device)
    x_seq    = torch.tensor(np.asarray(seq_data[case_idx:case_idx + 1]), dtype=torch.float32, device=device)

    explainer = shap.GradientExplainer(wrapped, [bg_static, bg_seq])
    sv_static, sv_seq = explainer.shap_values([x_static, x_seq], nsamples=case_shap_samples)
    sv_static = _squeeze_output_dim(sv_static)[0]    # [14]
    sv_seq    = _squeeze_output_dim(sv_seq)[0]       # [192,25]

    # For a single interpretable contribution per sequence feature, sum over time.
    seq_feature_contrib = sv_seq.sum(axis=0)         # [25]
    contrib = np.concatenate([sv_static, seq_feature_contrib], axis=0)  # [39]

    pairs = sorted(zip(FEATURE_NAMES, contrib), key=lambda p: -abs(p[1]))
    top5  = [(name, float(val)) for name, val in pairs[:5]]
    other = float(sum(v for _, v in pairs[5:]))
    n_other = len(pairs) - 5

    with torch.no_grad():
        fx  = float(wrapped(x_static, x_seq).squeeze().item())
        efx = float(wrapped(bg_static, bg_seq).mean().item())

    return {
        "features": top5,
        "fx": fx,
        "efx": efx,
        "n_other": n_other,
        "other": other,
    }


def write_metadata(outdir, args, checkpoint_path, per_hospital_counts, n_total):
    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "method": "gradient_explainer_full_sequence",
        "model_variant": args.model_variant,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "n_total_test_samples": int(n_total),
        "per_hospital_test_samples": per_hospital_counts,
        "n_background": int(args.n_background),
        "n_explain": int(args.n_explain),
        "n_shap_samples": int(args.n_shap_samples),
        "case_shap_samples": int(args.case_shap_samples),
        "case_study_idx": [int(x) for x in args.case_study_idx],
        "notes": [
            "Case-study indices refer to pooled test-window indices, not unique patient IDs.",
            "No manuscript text/tables were modified by this script.",
        ],
    }
    meta_path = os.path.join(outdir, "shap_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved metadata: {meta_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_variant",
        choices=["fedtft", "shared_head"],
        default="fedtft",
        help="Model architecture of the checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        default="patient_level_split/last_npy_data/GlobalData/ablation_R4_fedtft_best.pth",
        help=(
            "Checkpoint path. "
            "For --model_variant fedtft, use ablation_R4_fedtft_best.pth. "
            "For --model_variant shared_head, use ablation_R3_fvwa_best.pth."
        ),
    )
    parser.add_argument("--n_background", type=int, default=100)
    parser.add_argument("--n_explain", type=int, default=200)
    parser.add_argument("--n_shap_samples", type=int, default=100)
    parser.add_argument("--case_shap_samples", type=int, default=200)
    parser.add_argument("--case_study_idx", type=int, nargs="*", default=[0, 5, 12])
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/shap/")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint_path = args.checkpoint
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(ROOT, checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = load_model(checkpoint_path, args.hidden_dim, device, args.model_variant)
    static_data, seq_data, _, per_hospital_counts = load_all_hospitals("test")
    print(f"Data: {len(static_data)} pooled test samples from {len(per_hospital_counts)} hospitals")
    print(f"Per-hospital counts: {per_hospital_counts}")
    print("NOTE: case_study_idx are pooled test-sample indices (window-level).")

    all_results = {}
    case_study_out = {int(idx): {} for idx in args.case_study_idx}

    for h_idx, horizon in enumerate(HORIZON_NAMES):
        print(f"\n--- Horizon: {horizon} ---")
        shap_static, shap_seq, exp_idx, bg_idx = compute_shap_values_full_sequence(
            model=model,
            device=device,
            static_data=static_data,
            seq_data=seq_data,
            horizon_idx=h_idx,
            n_background=args.n_background,
            n_explain=args.n_explain,
            n_shap_samples=args.n_shap_samples,
            seed=args.seed,
        )

        rankings = rank_features(shap_static, shap_seq)
        all_results[horizon] = rankings
        plot_bar(rankings, horizon, "all_hospitals", args.output)

        for s_idx in args.case_study_idx:
            if 0 <= s_idx < len(static_data):
                print(f"  Extracting case study sample index {s_idx}...")
                case_study_out[int(s_idx)][horizon] = extract_case_study_full_sequence(
                    model=model,
                    device=device,
                    static_data=static_data,
                    seq_data=seq_data,
                    bg_idx=bg_idx,
                    case_idx=int(s_idx),
                    horizon_idx=h_idx,
                    case_shap_samples=args.case_shap_samples,
                )
            else:
                print(f"  [WARN] case index out of range: {s_idx}")

    rankings_path = os.path.join(args.output, "shap_rankings.json")
    with open(rankings_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved SHAP rankings: {rankings_path}")

    case_path = os.path.join(args.output, "shap_case_studies.json")
    with open(case_path, "w") as f:
        json.dump(case_study_out, f, indent=2)
    print(f"Saved case study values: {case_path}")

    write_metadata(
        outdir=args.output,
        args=args,
        checkpoint_path=checkpoint_path,
        per_hospital_counts=per_hospital_counts,
        n_total=len(static_data),
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
