# data_preparation.py
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

# ============================ 基础配置 & 日志 ============================
handlers_list = [logging.StreamHandler()]
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)8.8s] %(message)s",
    handlers=handlers_list,
)
logger = logging.getLogger(__name__)

# 统一数据根目录（容器/本地一致）
DATASET_DIR = Path(os.environ.get("DATASET_DIR", "./dataset")).resolve()

# —— 压缩包下载配置 ——（可只给 ARCHIVE_URL；也兼容你之前的 DATASET_URL）
ARCHIVE_URL = os.environ.get("ARCHIVE_URL") or os.environ.get("DATASET_URL") or ""
ARCHIVE_NAME = os.environ.get("ARCHIVE_NAME", "archive.zip")
ARCHIVE_MD5 = os.environ.get("ARCHIVE_MD5", "")
ARCHIVE_TOP = os.environ.get("ARCHIVE_TOP", "archive")  # 解压后顶层目录名

# 目录通配（若你的结构不同，可用 env 覆盖）
FITBIT_EXPORT_GLOB = os.environ.get("FITBIT_EXPORT_GLOB", "mturkfitbit_export_*")
FITBIT_SUBDIR_GLOB = os.environ.get("FITBIT_SUBDIR_GLOB", "Fitabase Data */")  # 若 CSV 在 export 根，设为 "" 空串

# 读取用到的列
FEATURES = ["Steps", "Calories", "AvgHeartRate", "StressLevel"]
LABEL = "SleepQuality"

# 为了与原项目的调用保持一致：运行期确定的 CSV 根目录
FITBIT_BASE_DIR = None  # 运行时由 _ensure_archive_ready() 决定


# ============================ 工具函数：下载 & 解压 ============================
def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_with_progress(url: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)

    def _report(blocknum, blocksize, totalsize):
        readsofar = blocknum * blocksize
        if totalsize > 0:
            percent = readsofar / totalsize * 100
            print(
                f"\rDownloading {url}  {percent:5.1f}% ({readsofar/1e6:.1f}MB/{totalsize/1e6:.1f}MB)",
                end="",
                flush=True,
            )
        else:
            print(f"\rDownloading {url}  {readsofar/1e6:.1f}MB", end="", flush=True)

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
    """支持 .zip / .tar(.gz|.tgz)"""
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
    在 root/ARCHIVE_TOP 下自动定位：
      export 目录：mturkfitbit_export_*（可配 FITBIT_EXPORT_GLOB）
      子目录：'Fitabase Data */'（可配 FITBIT_SUBDIR_GLOB；设空串表示直接使用 export 根）
    返回包含 *_merged.csv 的目录路径。
    """
    top = root / ARCHIVE_TOP
    export_dirs = sorted(top.glob(FITBIT_EXPORT_GLOB))
    if not export_dirs:
        raise FileNotFoundError(f"未找到 export 目录：{top}/{FITBIT_EXPORT_GLOB}")
    export_dir = export_dirs[-1]  # 取最新

    if FITBIT_SUBDIR_GLOB == "":
        csv_base = export_dir
    else:
        subdirs = list(export_dir.glob(FITBIT_SUBDIR_GLOB))
        if not subdirs:
            raise FileNotFoundError(f"未找到子目录：{export_dir}/{FITBIT_SUBDIR_GLOB}")
        csv_base = subdirs[0]

    return csv_base.resolve()


def _ensure_archive_ready() -> Path:
    """
    确保 DATASET_DIR 下存在解压后的 ARCHIVE_TOP 目录。
    - 若已存在则直接用；
    - 否则下载 ARCHIVE_URL → DATASET_DIR/archive_downloads/ARCHIVE_NAME 并解压到 DATASET_DIR；
    - 返回 CSV 根目录（包含 *_merged.csv 的目录）。
    """
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    top = DATASET_DIR / ARCHIVE_TOP

    if not top.exists():
        if not ARCHIVE_URL:
            raise FileNotFoundError(
                f"数据未就绪：{top} 不存在，且未设置 ARCHIVE_URL/DATASET_URL 提供压缩包下载地址。"
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
                raise ValueError(f"MD5 不一致：{m} != {ARCHIVE_MD5}  ({zip_path})")
            print(f"[DATA] MD5 OK: {m}")

        print(f"[DATA] Extracting to: {DATASET_DIR}")
        _extract_archive(zip_path, DATASET_DIR)

        if not top.exists():
            raise FileNotFoundError(f"解压后仍未发现 {top}，请检查压缩包内部结构。")

    csv_base = _auto_locate_fitbit_base(DATASET_DIR)

    # 简单检查关键 CSV
    must_csv = [
        "sleepDay_merged.csv",
        "hourlySteps_merged.csv",
        "hourlyCalories_merged.csv",
        "heartrate_seconds_merged.csv",
    ]
    missing = [f for f in must_csv if not (csv_base / f).exists()]
    if missing:
        print(f"[WARN] 缺少文件：{missing}（若文件名/结构不同，可通过通配或在读取处调整）")

    print(f"[DATA] Ready. CSV base: {csv_base}")
    return csv_base


# ============================ 读取 & 融合 ============================
def _binarize_np(y: np.ndarray) -> np.ndarray:
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
    读取 Fitabase CSV，按 (UserId, Hour) 对齐步数/卡路里/心率，并与日级睡眠标签合并。
    产出列：['UserId','Hour','Steps','Calories','AvgHeartRate','StressLevel','SleepQuality']（按时间排序）
    """
    global FITBIT_BASE_DIR
    if FITBIT_BASE_DIR is None:
        FITBIT_BASE_DIR = _ensure_archive_ready().as_posix()

    # 1) 睡眠（日级标签）
    sleep_fp = os.path.join(FITBIT_BASE_DIR, "sleepDay_merged.csv")
    sleep_df = pd.read_csv(sleep_fp, parse_dates=["SleepDay"])
    # 6 小时=360 分钟及以上记为高质量睡眠(1)
    sleep_df["SleepQuality"] = (sleep_df["TotalMinutesAsleep"] >= 360).astype(int)
    sleep_df["SleepDate"] = pd.to_datetime(sleep_df["SleepDay"].dt.date)

    # 2) 步数/卡路里（小时级）
    steps_fp = os.path.join(FITBIT_BASE_DIR, "hourlySteps_merged.csv")
    cals_fp = os.path.join(FITBIT_BASE_DIR, "hourlyCalories_merged.csv")
    steps_df = pd.read_csv(steps_fp, parse_dates=["ActivityHour"])
    cals_df = pd.read_csv(cals_fp, parse_dates=["ActivityHour"])
    # 注意不同数据集版本中列名可能为 "StepTotal" 或 "Steps"；做个兜底
    if "StepTotal" in steps_df.columns and "Steps" not in steps_df.columns:
        steps_df = steps_df.rename(columns={"StepTotal": "Steps"})
    activity_df = pd.merge(steps_df, cals_df, on=["Id", "ActivityHour"], how="inner")

    # 3) 心率（秒级 → 小时均值）
    hr_fp = os.path.join(FITBIT_BASE_DIR, "heartrate_seconds_merged.csv")
    hr_df = pd.read_csv(hr_fp, parse_dates=["Time"])
    hr_df["Hour"] = hr_df["Time"].dt.floor("H")
    hr_hourly = (
        hr_df.groupby(["Id", "Hour"])["Value"]
        .mean()
        .reset_index()
        .rename(columns={"Hour": "ActivityHour", "Value": "AvgHeartRate"})
    )

    # 4) 合并：(Id, ActivityHour)
    merged = pd.merge(activity_df, hr_hourly, on=["Id", "ActivityHour"], how="inner")

    # 5) 用 ActivityHour 的日期与日级睡眠标签合并
    merged["SleepDate"] = pd.to_datetime(merged["ActivityHour"].dt.date)
    final = pd.merge(
        merged,
        sleep_df[["Id", "SleepDate", "SleepQuality"]],
        on=["Id", "SleepDate"],
        how="inner",
    )

    # 6) StressLevel = AvgHR - RestingHR（每人心率的 25 分位作为静息心率粗估）
    resting = (
        final.groupby("Id")["AvgHeartRate"].quantile(0.25).rename("RestingHR").reset_index()
    )
    final = final.merge(resting, on="Id", how="left")
    final["StressLevel"] = final["AvgHeartRate"] - final["RestingHR"]
    final.drop(columns=["RestingHR"], inplace=True)

    # 7) 整理列名与排序
    final = final[
        ["Id", "ActivityHour", "Steps", "Calories", "AvgHeartRate", "StressLevel", "SleepQuality"]
    ]
    final = final.rename(columns={"Id": "UserId", "ActivityHour": "Hour"}).dropna()
    final = final.sort_values(["UserId", "Hour"]).reset_index(drop=True)
    return final


# ============================ 窗口化与数据集 ============================
def create_sequences_by_user(
    final_df: pd.DataFrame,
    seq_length: int = 6,
    feature_cols=FEATURES,
    label_col=LABEL,
    restrict_hours=None,
):
    """
    按用户分组做滑窗：X[i]= t..t+T-1 的特征，y[i]= t+T 时刻的 SleepQuality。
    返回：X:(N,T,F) float32, y:(N,1) float32(0/1), groups:(N,) 记录 userId
    """
    df = final_df.copy()
    df["Hour"] = pd.to_datetime(df["Hour"])
    if restrict_hours is not None:
        hod = df["Hour"].dt.hour
        df = df[hod.isin(list(restrict_hours))]

    df = df.sort_values(["UserId", "Hour"])
    X, y, groups = [], [], []

    for uid, g in df.groupby("UserId", sort=False):
        f = g[feature_cols].to_numpy(dtype=np.float32)  # [M, F]
        l = g[label_col].to_numpy()  # [M]
        if len(f) <= seq_length:
            continue
        for i in range(len(f) - seq_length):
            X.append(f[i : i + seq_length])  # [T, F]
            y.append(l[i + seq_length])  # 下一时刻标签
            groups.append(uid)

    if len(X) == 0:
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
    """窗口序列数据集：返回 (X, y)，其中 X:(T,F) float32, y:(1,) float32"""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


# ============================ Loader 构建（按用户切分，训练集拟合缩放器） ============================
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
    主入口：窗口化 → 分割 → 仅用训练集拟合 MinMaxScaler → 产出 DataLoader（三分）。
    """
    X, y, groups = create_sequences_by_user(final_df, seq_length, FEATURES, LABEL, restrict_hours)

    N = len(X)
    if N == 0:
        raise ValueError("没有生成任何序列样本；请检查数据范围/seq_length/restrict_hours。")

    rng = np.random.default_rng(seed)

    # ---- 划分：按用户（推荐）或全局分层 ----
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

    # ---- 仅用 train 拟合缩放器（防泄漏） ----
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

    # ---- DataLoader ----
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

    # ---- 统计信息 ----
    pos_ratio = float(y_train.mean()) if len(y_train) else 0.0
    logger.info(
        f"[FITBIT_SLEEP] seq_len={seq_length}, feat_dim={F}, "
        f"train={len(ds_train)}, val={len(ds_val)}, test={len(ds_test)}, "
        f"train_pos%={pos_ratio:.3f}"
    )

    return dl_train, dl_val, dl_test


# ============================ 对外主接口（与原项目保持一致） ============================
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
    统一入口（兼容你现有调用签名）：
      - 从 archive.zip（HTTP）或已存在目录准备数据；
      - 读取 Fitabase CSV，做 T=seq_length 窗口化，按用户分组切分 train/val/test；
      - 返回 (train_loader, val_loader, test_loader)。
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f'FL_Task - {json.dumps({"dataset": dataset, "start_execution_time": now_str})}')

    final_df = _load_fitbit_raw()

    # 计算各子集比例
    val_ratio = float(validation_split)
    test_ratio = float(test_split)
    train_ratio = 1.0 - val_ratio - test_ratio
    if train_ratio <= 0:
        raise ValueError(
            f"train_ratio <= 0（validation_split={val_ratio}, test_split={test_ratio}），请调整比例。"
        )

    return _build_loaders_windowed(
        final_df=final_df,
        seq_length=seq_length,
        batch_size=batch_size,
        split_by_user=True,  # 强烈建议按用户切分，避免泄漏
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        restrict_hours=restrict_hours,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
