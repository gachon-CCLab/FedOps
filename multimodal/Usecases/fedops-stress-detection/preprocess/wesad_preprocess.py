"""
wesad_preprocess.py — WESAD → FedOps-Health memmap splits

One client per subject (15 subjects = 15 FL clients).
Task: binary stress classification (stress=1, non-stress=0).

Pipeline per subject:
  1. Load pickle file
  2. Downsample all signals to 4 Hz
  3. Build 60-second sliding windows (stride 30s)
  4. Extract 14 statistical features per window
  5. Stack 10 consecutive windows → (10, 14) sequence
  6. Binary label from majority vote within window (label=2 → stress)
  7. Compute 8 subject-level static features from baseline phase
  8. Chronological train/val/test split (70/15/15)
  9. Write memmap splits

Usage:
    python preprocess/wesad_preprocess.py \
        --data_dir /path/to/WESAD \
        --out_dir  subject_level_split/data

Expected input:  WESAD/{S2,S3,...,S17}/{SX}.pkl
"""

import os
import sys
import json
import pickle
import argparse
import glob
import numpy as np
from scipy.signal import resample_poly
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ── Constants ─────────────────────────────────────────────────────────────────
TARGET_FS     = 4          # Hz — common downsampled frequency
WINDOW_SEC    = 60         # seconds per feature window
STRIDE_SEC    = 30         # seconds stride
SEQ_WINDOWS   = 10         # lookback: 10 windows = 10 minutes
STATIC_DIM    = 8
FEATURE_DIM   = 14         # statistical features per window
TARGET_DIM    = 1

WINDOW_SAMP   = TARGET_FS * WINDOW_SEC   # 240 samples per window
STRIDE_SAMP   = TARGET_FS * STRIDE_SEC   # 120 samples stride

# Label mapping
STRESS_LABEL  = 2
EXCLUDE_LABEL = 0          # transient periods — excluded


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",  required=True, help="Root WESAD folder (contains S2/, S3/, ...)")
    p.add_argument("--out_dir",   default="subject_level_split/data")
    p.add_argument("--val_frac",  type=float, default=0.15)
    p.add_argument("--test_frac", type=float, default=0.15)
    return p.parse_args()


def load_subject(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    return data


def downsample(signal, src_fs, tgt_fs=TARGET_FS):
    """Resample signal from src_fs to tgt_fs using polyphase filtering."""
    from math import gcd
    g   = gcd(int(tgt_fs), int(src_fs))
    up  = int(tgt_fs) // g
    dn  = int(src_fs) // g
    if signal.ndim == 1:
        return resample_poly(signal, up, dn).astype(np.float32)
    return np.stack([resample_poly(signal[:, i], up, dn)
                     for i in range(signal.shape[1])], axis=1).astype(np.float32)


def extract_signals(data):
    """
    Downsample all signals to TARGET_FS and align lengths.
    Returns dict of arrays all at 4 Hz, and the label array at 4 Hz.
    """
    chest_fs  = 700
    wrist_acc_fs  = 32
    wrist_bvp_fs  = 64
    wrist_eda_fs  = 4
    wrist_temp_fs = 4

    chest   = data["signal"]["chest"]
    wrist   = data["signal"]["wrist"]
    label_700 = data["label"].squeeze()

    signals = {
        "chest_ecg":   downsample(chest["ECG"].squeeze(),  chest_fs),
        "chest_eda":   downsample(chest["EDA"].squeeze(),  chest_fs),
        "chest_resp":  downsample(chest["Resp"].squeeze(), chest_fs),
        "chest_temp":  downsample(chest["Temp"].squeeze(), chest_fs),
        "wrist_acc_x": downsample(wrist["ACC"][:, 0],     wrist_acc_fs),
        "wrist_acc_y": downsample(wrist["ACC"][:, 1],     wrist_acc_fs),
        "wrist_acc_z": downsample(wrist["ACC"][:, 2],     wrist_acc_fs),
        "wrist_bvp":   downsample(wrist["BVP"].squeeze(), wrist_bvp_fs),
        "wrist_eda":   downsample(wrist["EDA"].squeeze(), wrist_eda_fs),
        "wrist_temp":  downsample(wrist["TEMP"].squeeze(),wrist_temp_fs),
    }
    label_4hz = downsample(label_700.astype(np.float32), chest_fs)
    # Round labels back to integers after resampling
    label_4hz = np.round(label_4hz).astype(int)

    # Align all to shortest length
    min_len = min(len(v) for v in signals.values())
    min_len = min(min_len, len(label_4hz))
    signals = {k: v[:min_len] for k, v in signals.items()}
    label_4hz = label_4hz[:min_len]

    return signals, label_4hz


def window_features(signals, start, end):
    """
    Extract 14 statistical features from a window [start:end].

    Features:
      wrist_acc_magnitude: mean, std       (2)
      wrist_bvp:           mean, std       (2)
      wrist_eda:           mean, std       (2)
      wrist_temp:          mean            (1)
      chest_ecg:           std             (1)  — HRV proxy
      chest_eda:           mean, std       (2)
      chest_resp:          mean, std       (2)
      chest_temp:          mean            (1)
    Total: 14
    """
    def _s(key):
        return signals[key][start:end]

    acc_mag = np.sqrt(_s("wrist_acc_x")**2 + _s("wrist_acc_y")**2 + _s("wrist_acc_z")**2)

    feats = [
        acc_mag.mean(),           acc_mag.std(),
        _s("wrist_bvp").mean(),   _s("wrist_bvp").std(),
        _s("wrist_eda").mean(),   _s("wrist_eda").std(),
        _s("wrist_temp").mean(),
        _s("chest_ecg").std(),
        _s("chest_eda").mean(),   _s("chest_eda").std(),
        _s("chest_resp").mean(),  _s("chest_resp").std(),
        _s("chest_temp").mean(),
        # 14th: mean of abs chest EMG not available — use chest resp std diff
        float(np.diff(_s("chest_resp")).std()) if len(_s("chest_resp")) > 1 else 0.0,
    ]
    return np.array(feats, dtype=np.float32)


def build_static_features(signals, label_4hz):
    """
    8 subject-level static features computed from the baseline phase (label=1).
    Falls back to full signal if no baseline available.
    """
    baseline_mask = (label_4hz == 1)
    if baseline_mask.sum() < 10:
        baseline_mask = np.ones(len(label_4hz), dtype=bool)

    def _b(key):
        return signals[key][baseline_mask]

    acc_mag = np.sqrt(signals["wrist_acc_x"][baseline_mask]**2 +
                      signals["wrist_acc_y"][baseline_mask]**2 +
                      signals["wrist_acc_z"][baseline_mask]**2)

    stress_ratio = float((label_4hz == STRESS_LABEL).mean())

    return np.array([
        _b("wrist_eda").mean(),    # baseline EDA level
        _b("wrist_eda").std(),     # EDA variability
        _b("wrist_bvp").mean(),    # baseline BVP (HR proxy)
        _b("wrist_bvp").std(),
        _b("wrist_temp").mean(),   # baseline temperature
        _b("chest_resp").mean(),   # baseline respiration
        _b("chest_resp").std(),
        stress_ratio,              # overall stress prevalence
    ], dtype=np.float32)


def build_windows(signals, label_4hz):
    """
    Build (feature_windows, labels) sliding over the full signal.
    Each sample is SEQ_WINDOWS consecutive feature windows → (SEQ_WINDOWS, FEATURE_DIM).
    Label: 1 if majority of label in current (last) window is stress, else 0.
    Excludes windows that overlap with transient (label=0) regions.
    """
    n = min(len(v) for v in signals.values())
    n = min(n, len(label_4hz))

    # Build feature vector for every possible window position
    all_feat   = []
    all_labels = []

    pos = 0
    while pos + WINDOW_SAMP <= n:
        end = pos + WINDOW_SAMP
        win_labels = label_4hz[pos:end]

        # Skip windows with transient samples
        if (win_labels == EXCLUDE_LABEL).any():
            pos += STRIDE_SAMP
            continue

        majority = np.bincount(win_labels, minlength=5).argmax()
        binary   = 1 if majority == STRESS_LABEL else 0

        feat = window_features(signals, pos, end)
        all_feat.append(feat)
        all_labels.append(binary)
        pos += STRIDE_SAMP

    if len(all_feat) < SEQ_WINDOWS + 1:
        return (np.empty((0, SEQ_WINDOWS, FEATURE_DIM), dtype=np.float32),
                np.empty((0, TARGET_DIM), dtype=np.float32))

    feat_arr  = np.stack(all_feat)    # (W, FEATURE_DIM)
    label_arr = np.array(all_labels)  # (W,)

    seqs, targets = [], []
    for i in range(SEQ_WINDOWS, len(feat_arr)):
        seqs.append(feat_arr[i - SEQ_WINDOWS : i])   # (SEQ_WINDOWS, FEATURE_DIM)
        targets.append([label_arr[i]])                # (1,)

    return (np.stack(seqs).astype(np.float32),
            np.array(targets, dtype=np.float32))


def normalize_sequences(seqs_tr, seqs_val, seqs_te):
    N, T, F = seqs_tr.shape
    scaler  = StandardScaler()
    scaler.fit(seqs_tr.reshape(-1, F))
    def _apply(s):
        n, t, f = s.shape
        return scaler.transform(s.reshape(-1, f)).reshape(n, t, f).astype(np.float32)
    return _apply(seqs_tr), _apply(seqs_val), _apply(seqs_te)


def save_split(out_dir, prefix, static, seqs, tgts):
    os.makedirs(out_dir, exist_ok=True)
    for name, arr in [("static", static), ("sequence", seqs), ("targets", tgts)]:
        fp = np.memmap(os.path.join(out_dir, f"{name}_{prefix}.npy"),
                       dtype="float32", mode="w+", shape=arr.shape)
        fp[:] = arr[:]
        del fp


def process_subject(pkl_path, out_base, val_frac, test_frac):
    subject_id = os.path.basename(pkl_path).replace(".pkl", "")   # e.g. "S2"
    print(f"  {subject_id} ... ", end="", flush=True)

    data               = load_subject(pkl_path)
    signals, label_4hz = extract_signals(data)
    static_vec         = build_static_features(signals, label_4hz)
    seqs, targets      = build_windows(signals, label_4hz)

    if len(seqs) < 20:
        print(f"SKIP (windows={len(seqs)})")
        return None

    # Stratified split — keeps stress ratio balanced across train/val/test.
    # Chronological split fails for WESAD because the stress phase sits in the
    # middle of the protocol, leaving val/test with zero stress samples.
    labels_flat = targets[:, 0].astype(int)
    idx         = np.arange(len(seqs))

    # If only one class present, fall back to random split
    strat = labels_flat if len(np.unique(labels_flat)) > 1 else None
    idx_tr, idx_tmp = train_test_split(
        idx, test_size=val_frac + test_frac, random_state=42, stratify=strat)
    strat_tmp = labels_flat[idx_tmp] if strat is not None else None
    idx_val, idx_te = train_test_split(
        idx_tmp, test_size=test_frac / (val_frac + test_frac),
        random_state=42, stratify=strat_tmp)

    s_tr,  t_tr  = seqs[idx_tr],  targets[idx_tr]
    s_val, t_val = seqs[idx_val], targets[idx_val]
    s_te,  t_te  = seqs[idx_te],  targets[idx_te]

    s_tr, s_val, s_te = normalize_sequences(s_tr, s_val, s_te)

    subj_dir = os.path.join(out_base, "SubjectsData", subject_id)
    save_split(subj_dir, "train", np.tile(static_vec, (len(s_tr),  1)).astype(np.float32), s_tr,  t_tr)
    save_split(subj_dir, "val",   np.tile(static_vec, (len(s_val), 1)).astype(np.float32), s_val, t_val)
    save_split(subj_dir, "test",  np.tile(static_vec, (len(s_te),  1)).astype(np.float32), s_te,  t_te)

    n = len(seqs)
    stress_ratio = targets[:, 0].mean()
    print(f"OK  windows={n}  tr={len(s_tr)} val={len(s_val)} te={len(s_te)}  "
          f"stress_tr={t_tr[:,0].mean():.2f} val={t_val[:,0].mean():.2f} te={t_te[:,0].mean():.2f}")
    return subject_id, np.tile(static_vec, (len(s_te), 1)).astype(np.float32), s_te, t_te


def main():
    args     = parse_args()
    pkl_files = sorted(glob.glob(os.path.join(args.data_dir, "S*", "S*.pkl")))

    if not pkl_files:
        print(f"No .pkl files found under {args.data_dir}/S*/")
        sys.exit(1)

    print(f"Found {len(pkl_files)} subjects.")

    subject_mapping = {}
    global_st, global_sq, global_tg = [], [], []
    client_idx = 0

    for pkl_path in pkl_files:
        result = process_subject(pkl_path, args.out_dir, args.val_frac, args.test_frac)
        if result is None:
            continue
        subj_id, st, sq, tg = result
        subject_mapping[subj_id] = client_idx
        global_st.append(st); global_sq.append(sq); global_tg.append(tg)
        client_idx += 1

    if global_st:
        gd = os.path.join(args.out_dir, "GlobalData")
        os.makedirs(gd, exist_ok=True)
        for fname, arr in [("static_data.npy",   np.concatenate(global_st)),
                           ("sequence_data.npy",  np.concatenate(global_sq)),
                           ("targets.npy",         np.concatenate(global_tg))]:
            fp = np.memmap(os.path.join(gd, fname), dtype="float32",
                           mode="w+", shape=arr.shape)
            fp[:] = arr[:]
            del fp
        total = sum(len(x) for x in global_st)
        print(f"\nGlobalData: {total} test samples from {client_idx} subjects")

    mapping_path = os.path.join(os.path.dirname(__file__), "..", "subject_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump(subject_mapping, f, indent=2)
    print(f"subject_mapping.json → {client_idx} clients")


if __name__ == "__main__":
    main()
