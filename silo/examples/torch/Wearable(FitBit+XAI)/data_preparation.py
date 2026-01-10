# data_preparation.py
# -*- coding: utf-8 -*-

import os
import json
import logging
from datetime import datetime
from pathlib import Path
import hashlib
import urllib.request
import zipfile
import tarfile
import shutil

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GroupShuffleSplit, train_test_split

# ============================ Basic Configuration & Logging ============================
handlers_list = [logging.StreamHandler()]
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)8.8s] %(message)s",
    handlers=handlers_list,
)
logger = logging.getLogger(__name__)

# Unified dataset root directory (consistent between container and local)
DATASET_DIR = Path(os.environ.get("DATASET_DIR", "./dataset")).resolve()

# —— Archive download configuration —— (can specify ARCHIVE_URL; also compatible with previous DATASET_URL)
ARCHIVE_URL = os.environ.get("ARCHIVE_URL") or os.environ.get("DATASET_URL") or ""
ARCHIVE_NAME = os.environ.get("ARCHIVE_NAME", "archive.zip")
ARCHIVE_MD5 = os.environ.get("ARCHIVE_MD5", "")
ARCHIVE_TOP = os.environ.get("ARCHIVE_TOP", "archive")  # top-level directory name after extraction

# Directory glob patterns (can be overridden via environment variables)
FITBIT_EXPORT_GLOB = os.environ.get("FITBIT_EXPORT_GLOB", "mturkfitbit_export_*")
FITBIT_SUBDIR_GLOB = os.environ.get("FITBIT_SUBDIR_GLOB", "Fitabase Data */")  # if CSVs are under export root, set to ""

# Columns used for reading
FEATURES = ["Steps", "Calories", "AvgHeartRate", "StressLevel"]
LABEL = "SleepQuality"

# To remain consistent with original project: CSV base directory decided at runtime
FITBIT_BASE_DIR = None  # determined by _ensure_archive_ready() during runtime

# === NEW: KaggleHub download option and dataset configuration ===
try:
    import kagglehub  # pip install kagglehub
    _HAS_KAGGLEHUB = True
except Exception:
    _HAS_KAGGLEHUB = False

USE_KAGGLE = os.environ.get("USE_KAGGLE", "1")           # "1"=use kagglehub by default, "0"=disable
KAGGLE_DATASET = os.environ.get("KAGGLE_DATASET", "arashnic/fitbit")
# Map or copy the kagglehub-downloaded directory to DATASET_DIR/ARCHIVE_TOP (or user-specified alias)
KAGGLE_TARGET_TOP = os.environ.get("KAGGLE_TARGET_TOP", ARCHIVE_TOP)


# ============================ Utility Functions: Download & Extract ============================
def _md5(path: Path) -> str:
    """Compute MD5 hash of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_with_progress(url: str, dst: Path):
    """Download file from URL with a progress indicator."""
    dst.parent.mkdir(parents=True, exist_ok=True)

    def _report(blocknum, blocksize, totalsize):
        readsofar = blocknum * blocksize
        if totalsize > 0:
            percent = readsofar / totalsize * 100

    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, tmp.as_posix(), _report)
        print()
        tmp.replace(dst)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def _extract_archive(archive_path: Path, to_dir: Path):
    """Support extraction of .zip / .tar(.gz|.tgz) files."""
    to_dir.mkdir(parents=True, exist_ok=True)
    ap = archive_path.as_posix()
    if zipfile.is_zipfile(ap):
        with zipfile.ZipFile(ap, "r") as zf:
            zf.extractall(to_dir)
    elif ap.endswith(".tar.gz") or ap.endswith(".tgz"):
        with tarfile.open(ap, "r:gz") as tf:
            tf.extractall(to_dir)
    elif ap.endswith(".tar"):
        with tarfile.open(ap, "r:") as tf:
            tf.extractall(to_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")


def _auto_locate_fitbit_base(root: Path) -> Path:
    """
    Automatically locate the Fitbit base directory under root/ARCHIVE_TOP.
      Export directory: mturkfitbit_export_* (configurable via FITBIT_EXPORT_GLOB)
      Subdirectory: 'Fitabase Data */' (configurable via FITBIT_SUBDIR_GLOB; set empty to use export root)
    Returns the directory containing *_merged.csv files.
    """
    top = root / ARCHIVE_TOP
    export_dirs = sorted(top.glob(FITBIT_EXPORT_GLOB))
    if not export_dirs:
        raise FileNotFoundError(f"No export directory found: {top}/{FITBIT_EXPORT_GLOB}")
    export_dir = export_dirs[-1]  # use the latest one

    if FITBIT_SUBDIR_GLOB == "":
        csv_base = export_dir
    else:
        subdirs = list(export_dir.glob(FITBIT_SUBDIR_GLOB))
        if not subdirs:
            raise FileNotFoundError(f"No subdirectory found: {export_dir}/{FITBIT_SUBDIR_GLOB}")
        csv_base = subdirs[0]

    return csv_base.resolve()


# === NEW: KaggleHub Downloader ===
def _prepare_from_kagglehub(dataset_slug: str, target_top: str) -> Path:
    """
    Download dataset using kagglehub (usually already extracted),
    ensure DATASET_DIR/target_top exists (as a symlink or copy),
    and return the CSV base directory.
    """
    if not _HAS_KAGGLEHUB:
        raise RuntimeError("kagglehub not installed: run `pip install kagglehub` or set USE_KAGGLE=0")

    logger.info(f"[DATA] KaggleHub downloading: {dataset_slug}")
    src_path = Path(kagglehub.dataset_download(dataset_slug)).resolve()
    logger.info(f"[DATA] KaggleHub path: {src_path}")

    # Mount downloaded directory as DATASET_DIR/target_top (same as archive structure)
    top = DATASET_DIR / target_top
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    if top.exists():
        logger.info(f"[DATA] Reusing existing directory: {top}")
    else:
        try:
            if top.is_symlink() or top.exists():
                top.unlink()
            top.symlink_to(src_path, target_is_directory=True)
            logger.info(f"[DATA] Linked {src_path} -> {top}")
        except Exception as e:
            logger.warning(f"[DATA] symlink failed, copying instead: {e}")
            shutil.copytree(src_path, top)
            logger.info(f"[DATA] Copied {src_path} -> {top}")

    # Kaggle datasets are typically already extracted — auto-locate CSV base
    csv_base = _auto_locate_fitbit_base(DATASET_DIR)
    logger.info(f"[DATA] Kaggle ready. CSV base: {csv_base}")
    return csv_base


def _ensure_archive_ready() -> Path:
    """
    Ensure that DATASET_DIR contains the extracted ARCHIVE_TOP directory.
    - If it exists, use it directly;
    - Otherwise, try KaggleHub (if USE_KAGGLE=1 and kagglehub installed);
    - If that fails, download via ARCHIVE_URL and extract;
    - Return the CSV base directory (contains *_merged.csv).
    """
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    top = DATASET_DIR / ARCHIVE_TOP

    # If already exists → use it directly
    if top.exists():
        csv_base = _auto_locate_fitbit_base(DATASET_DIR)
        print(f"[DATA] Ready. CSV base: {csv_base}")
        return csv_base

    # Try KaggleHub first
    if USE_KAGGLE == "1":
        try:
            return _prepare_from_kagglehub(KAGGLE_DATASET, KAGGLE_TARGET_TOP)
        except Exception as e:
            logger.warning(f"[DATA] KaggleHub preparation failed, falling back to URL: {e}")

    # Fallback: URL download + extraction
    if not ARCHIVE_URL:
        raise FileNotFoundError(
            f"Data not ready: {top} missing, and ARCHIVE_URL/DATASET_URL not set (or Kaggle failed)."
        )
    downloads = DATASET_DIR / "archive_downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    zip_path = downloads / ARCHIVE_NAME

    if not zip_path.exists():
        print(f"[DATA] Downloading archive from: {ARCHIVE_URL}")
        _download_with_progress(ARCHIVE_URL, zip_path)
    else:
        print(f"[DATA] Using cached archive: {zip_path}")

    if ARCHIVE_MD5:
        m = _md5(zip_path)
        if m.lower() != ARCHIVE_MD5.lower():
            raise ValueError(f"MD5 mismatch: {m} != {ARCHIVE_MD5} ({zip_path})")
        print(f"[DATA] MD5 OK: {m}")

    print(f"[DATA] Extracting to: {DATASET_DIR}")
    _extract_archive(zip_path, DATASET_DIR)

    if not top.exists():
        raise FileNotFoundError(f"{top} not found after extraction. Check archive structure.")

    csv_base = _auto_locate_fitbit_base(DATASET_DIR)
    print(f"[DATA] Ready. CSV base: {csv_base}")
    return csv_base


# ============================ Data Reading & Merging ============================
def _binarize_np(y: np.ndarray) -> np.ndarray:
    """Convert numeric label arrays into binary format {0,1}."""
    u = set(np.unique(y).tolist())
    if u <= {0, 1}:
        return y.astype(int)
    if u <= {-1, 1}:
        return (y > 0).astype(int)
    if u <= {1, 2}:
        return (y >= 2).astype(int)
    return (y > 0).astype(int)


def _load_fitbit_raw() -> pd.DataFrame:
    """
    Read Fitabase CSVs, align (UserId, Hour) for steps/calories/heart rate,
    and merge with daily sleep labels.
    Output columns:
      ['UserId','Hour','Steps','Calories','AvgHeartRate','StressLevel','SleepQuality'] (sorted by time)
    """
    global FITBIT_BASE_DIR
    if FITBIT_BASE_DIR is None:
        FITBIT_BASE_DIR = _ensure_archive_ready().as_posix()

    # 1) Sleep (daily labels)
    sleep_fp = os.path.join(FITBIT_BASE_DIR, "sleepDay_merged.csv")
    sleep_df = pd.read_csv(sleep_fp, parse_dates=["SleepDay"])
    # Default: ≥360 minutes (6 hours) → high-quality sleep (1)
    sleep_df["SleepQuality"] = (sleep_df["TotalMinutesAsleep"] >= 360).astype(int)
    sleep_df["SleepDate"] = pd.to_datetime(sleep_df["SleepDay"].dt.date)

    # 2) Steps / Calories (hourly)
    steps_fp = os.path.join(FITBIT_BASE_DIR, "hourlySteps_merged.csv")
    cals_fp = os.path.join(FITBIT_BASE_DIR, "hourlyCalories_merged.csv")
    steps_df = pd.read_csv(steps_fp, parse_dates=["ActivityHour"])
    cals_df = pd.read_csv(cals_fp, parse_dates=["ActivityHour"])
    # Compatibility with older dataset versions
    if "StepTotal" in steps_df.columns and "Steps" not in steps_df.columns:
        steps_df = steps_df.rename(columns={"StepTotal": "Steps"})
    activity_df = pd.merge(steps_df, cals_df, on=["Id", "ActivityHour"], how="inner")

    # 3) Heart rate (seconds → hourly mean)
    hr_fp = os.path.join(FITBIT_BASE_DIR, "heartrate_seconds_merged.csv")
    hr_df = pd.read_csv(hr_fp, parse_dates=["Time"])
    hr_df["Hour"] = hr_df["Time"].dt.floor("H")
    hr_hourly = (
        hr_df.groupby(["Id", "Hour"])["Value"]
        .mean()
        .reset_index()
        .rename(columns={"Hour": "ActivityHour", "Value": "AvgHeartRate"})
    )

    # 4) Merge (Id, ActivityHour)
    merged = pd.merge(activity_df, hr_hourly, on=["Id", "ActivityHour"], how="inner")

    # 5) Merge with daily sleep labels
    merged["SleepDate"] = pd.to_datetime(merged["ActivityHour"].dt.date)
    final = pd.merge(
        merged,
        sleep_df[["Id", "SleepDate", "SleepQuality"]],
        on=["Id", "SleepDate"],
        how="inner",
    )

    # 6) Compute StressLevel = AvgHR - RestingHR (25th percentile of HR as baseline)
    resting = (
        final.groupby("Id")["AvgHeartRate"].quantile(0.25).rename("RestingHR").reset_index()
    )
    final = final.merge(resting, on="Id", how="left")
    final["StressLevel"] = final["AvgHeartRate"] - final["RestingHR"]
    final.drop(columns=["RestingHR"], inplace=True)

    # 7) Reorganize columns and sort chronologically
    final = final[
        ["Id", "ActivityHour", "Steps", "Calories", "AvgHeartRate", "StressLevel", "SleepQuality"]
    ]
    final = final.rename(columns={"Id": "UserId", "ActivityHour": "Hour"}).dropna()
    final = final.sort_values(["UserId", "Hour"]).reset_index(drop=True)
    return final
# ============================ Windowing and Dataset ============================
def create_sequences_by_user(
    final_df: pd.DataFrame,
    seq_length: int = 6,
    feature_cols=FEATURES,
    label_col=LABEL,
    restrict_hours=None,
):
    """
    Create sliding windows grouped by user:
      X[i] = features from time t .. t+T-1
      y[i] = SleepQuality at time t+T
    Returns:
      X: (N, T, F) float32 — sequence features
      y: (N, 1) float32 — binary labels (0/1)
      groups: (N,) — corresponding user IDs
    """
    df = final_df.copy()
    df["Hour"] = pd.to_datetime(df["Hour"])
    if restrict_hours is not None:
        # Optionally restrict to specific hours of the day
        hod = df["Hour"].dt.hour
        df = df[hod.isin(list(restrict_hours))]

    df = df.sort_values(["UserId", "Hour"])
    X, y, groups = [], [], []

    for uid, g in df.groupby("UserId", sort=False):
        f = g[feature_cols].to_numpy(dtype=np.float32)  # [M, F] features
        l = g[label_col].to_numpy()  # [M] labels
        if len(f) <= seq_length:
            continue
        for i in range(len(f) - seq_length):
            X.append(f[i : i + seq_length])  # [T, F]
            y.append(l[i + seq_length])      # label at next time step
            groups.append(uid)

    if len(X) == 0:
        # Return empty arrays if no valid sequences found
        return (
            np.empty((0, seq_length, len(feature_cols)), np.float32),
            np.empty((0, 1), np.float32),
            np.array([], dtype=object),
        )

    X = np.stack(X, axis=0)  # (N, T, F)
    y = _binarize_np(np.array(y)).reshape(-1, 1).astype(np.float32)  # (N,1)
    groups = np.array(groups)
    return X, y, groups


class SeqDataset(Dataset):
    """Sequence dataset: returns (X, y) where X:(T,F) float32 and y:(1,) float32"""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


# ============================ DataLoader Construction (User Split + Scaler Fit) ============================
def _build_loaders_windowed(
    final_df: pd.DataFrame,
    seq_length: int = 6,
    batch_size: int = 64,
    split_by_user: bool = True,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    restrict_hours=None,
    num_workers: int = 0,
    pin_memory: bool = True,
):
    """
    Main entry point:
      - Apply windowing;
      - Split dataset;
      - Fit MinMaxScaler using only the training set (to avoid leakage);
      - Return three DataLoaders (train/val/test).
    """
    X, y, groups = create_sequences_by_user(final_df, seq_length, FEATURES, LABEL, restrict_hours)

    N = len(X)
    if N == 0:
        raise ValueError("No sequence samples generated; check data range, seq_length, or restrict_hours.")

    rng = np.random.default_rng(seed)

    # ---- Dataset split: by user (recommended) or global stratified ----
    if split_by_user:
        gss = GroupShuffleSplit(n_splits=1, test_size=(1 - train_ratio), random_state=seed)
        train_idx, hold_idx = next(gss.split(np.arange(N), y.reshape(-1), groups))

        val_size = int(round(len(hold_idx) * val_ratio / max((1 - train_ratio), 1e-8)))
        hold_idx = rng.permutation(hold_idx)
        val_idx = hold_idx[:val_size]
        test_idx = hold_idx[val_size:]
    else:
        idx = np.arange(N)
        tr_idx, hold_idx = train_test_split(
            idx, test_size=(1 - train_ratio), stratify=y.reshape(-1), random_state=seed
        )
        val_size = int(round(len(hold_idx) * val_ratio / max((1 - train_ratio), 1e-8)))
        hold_idx = rng.permutation(hold_idx)
        val_idx = hold_idx[:val_size]
        test_idx = hold_idx[val_size:]
        train_idx = tr_idx

    # ---- Fit scaler using training set only (to prevent data leakage) ----
    F = X.shape[-1]
    scaler = MinMaxScaler()
    X_train_2d = X[train_idx].reshape(-1, F)
    scaler.fit(X_train_2d)

    def _apply_scale(X_in: np.ndarray) -> np.ndarray:
        sh = X_in.shape
        return scaler.transform(X_in.reshape(-1, sh[-1])).reshape(sh).astype(np.float32)

    X_train = _apply_scale(X[train_idx])
    y_train = y[train_idx]
    X_val = _apply_scale(X[val_idx])
    y_val = y[val_idx]
    X_test = _apply_scale(X[test_idx])
    y_test = y[test_idx]

    # ---- Construct DataLoaders ----
    ds_train = SeqDataset(X_train, y_train)
    ds_val = SeqDataset(X_val, y_val)
    ds_test = SeqDataset(X_test, y_test)

    dl_train = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
    )
    dl_val = DataLoader(
        ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )
    dl_test = DataLoader(
        ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )

    # ---- Log summary statistics ----
    pos_ratio = float(y_train.mean()) if len(y_train) else 0.0
    logger.info(
        f"[FITBIT_SLEEP] seq_len={seq_length}, feat_dim={F}, "
        f"train={len(ds_train)}, val={len(ds_val)}, test={len(ds_test)}, "
        f"train_pos%={pos_ratio:.3f}"
    )

    return dl_train, dl_val, dl_test


# ============================ Public Interface (Compatible with Original Project) ============================
def load_partition(
    dataset: str,
    validation_split: float,
    batch_size: int,
    seq_length: int = 6,
    test_split: float = 0.2,
    seed: int = 42,
    restrict_hours=None,
    num_workers: int = 0,
    pin_memory: bool = True,
):
    """
    Unified entry function (compatible with the original project API):
      - Prepare dataset via KaggleHub (preferred), archive.zip (HTTP), or pre-existing directory;
      - Read Fitabase CSV files and apply windowing with T=seq_length;
      - Split data by user into train/val/test sets;
      - Return (train_loader, val_loader, test_loader).
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f'FL_Task - {json.dumps({"dataset": dataset, "start_execution_time": now_str})}')

    final_df = _load_fitbit_raw()

    # Calculate data split ratios
    val_ratio = float(validation_split)
    test_ratio = float(test_split)
    train_ratio = 1.0 - val_ratio - test_ratio
    if train_ratio <= 0:
        raise ValueError(
            f"train_ratio <= 0 (validation_split={val_ratio}, test_split={test_ratio}), please adjust ratios."
        )

    return _build_loaders_windowed(
        final_df=final_df,
        seq_length=seq_length,
        batch_size=batch_size,
        split_by_user=True,  # Strongly recommended to split by user to avoid data leakage
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        restrict_hours=restrict_hours,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

