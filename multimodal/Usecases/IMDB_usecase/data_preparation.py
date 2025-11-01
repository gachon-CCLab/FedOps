import os
import json
import logging
from typing import Tuple, Optional

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from transformers import BertTokenizer

# ------------ logging ------------
handlers_list = [logging.StreamHandler()]
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)8.8s] %(message)s", handlers=handlers_list
)
logger = logging.getLogger(__name__)

# ------------ Paths (project-wide defaults) ------------
BASE_DATASET_DIR = os.path.abspath("./dataset")                  # client_0/, client_1/, ...
IMG_DIR = os.path.abspath("./all_in_one_dataset/mmimdb_posters") # project-wide posters

# Server-side (may be nested after unzip)
SERVER_DATA_DIR = os.path.abspath("./server_data")
SERVER_CSV_PATH_DEFAULT = os.path.join(SERVER_DATA_DIR, "server_test.csv")
SERVER_ZIP_PATH = os.path.join(SERVER_DATA_DIR, "server_data.zip")

# Optional: auto-download tiny server eval set from Google Drive
try:
    import gdown, zipfile
except Exception:
    gdown = None
    zipfile = None

GDRIVE_FILE_ID = "1FdLbq-cvREJ99KjemQC6XzPmcOawzuQq"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

# ------------ helpers ------------
def _server_root() -> str:
    """
    Normalize server_data root. If the zip contains a top-level 'server_data/',
    you'll get SERVER_DATA_DIR/server_data/* — walk down until nesting stops.
    """
    root = SERVER_DATA_DIR
    while os.path.isdir(os.path.join(root, "server_data")):
        root = os.path.join(root, "server_data")
    return root

def _resolve_labels_json() -> str:
    """
    Works for BOTH client & server:
      0) LABELS_JSON_PATH env var (absolute or relative to cwd)
      1) ./all_in_one_dataset/labels.json
      2) <normalized server root>/labels.json
      3) first labels.json found recursively under <normalized server root>
    Returns a path (may not exist; caller checks).
    """
    env_path = os.environ.get("LABELS_JSON_PATH")
    if env_path:
        cand = env_path if os.path.isabs(env_path) else os.path.abspath(env_path)
        if os.path.exists(cand):
            return cand

    main_path = os.path.abspath("./all_in_one_dataset/labels.json")
    if os.path.exists(main_path):
        return main_path

    root = _server_root()
    root_candidate = os.path.join(root, "labels.json")
    if os.path.exists(root_candidate):
        return root_candidate

    if os.path.isdir(root):
        for r, _, files in os.walk(root):
            if "labels.json" in files:
                return os.path.join(r, "labels.json")

    # last tried (for a clear error message)
    return root_candidate

def _resolve_server_img_dir() -> str:
    """
    Accept either <server_root>/mmimdb_posters or <server_root>/img.
    """
    root = _server_root()
    for name in ("mmimdb_posters", "img"):
        cand = os.path.join(root, name)
        if os.path.isdir(cand):
            return cand

    # as a last resort, search recursively
    if os.path.isdir(root):
        for r, dirs, _ in os.walk(root):
            base = os.path.basename(r)
            if base in ("mmimdb_posters", "img"):
                return r

    raise FileNotFoundError(
        "No images folder found under server_data/ (expected 'mmimdb_posters' or 'img')."
    )

# ============================================================
# Dataset (MM-IMDb, multilabel → multi-hot)
# ============================================================
class MMIMDbDataset(Dataset):
    """
    Expects DataFrame with columns: ['img_name', 'text', 'labels'] where 'labels' is pipe-separated.
    Returns: dict with input_ids, attention_mask, image (3x224x224), label (float multi-hot).
    """
    def __init__(self, df: pd.DataFrame, img_dir: str, max_len: int = 128, bert_name: str = "bert-base-uncased"):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.max_len = int(max_len)

        # Tokenizer and image transform
        self.tokenizer = BertTokenizer.from_pretrained(bert_name)
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

        # Labels (robust for server & client)
        labels_json_path = _resolve_labels_json()
        if not os.path.exists(labels_json_path):
            raise FileNotFoundError(
                "labels.json not found. Place it at one of:"
                f" - {os.path.abspath('./all_in_one_dataset/labels.json')}"
                f" - {_server_root()}/labels.json (or nested under server_data/)"
                " - or set env LABELS_JSON_PATH=/abs/path/to/labels.json"
                f"Last tried: {labels_json_path}"
            )
        with open(labels_json_path, "r") as f:
            self.label_list = json.load(f)
        self.label_to_idx = {lab: i for i, lab in enumerate(self.label_list)}
        self.num_labels = len(self.label_list)
        logger.info(f"Using labels.json from: {labels_json_path} (C={self.num_labels})")

        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")

        logger.info(f"🗂️ MMIMDbDataset: {len(self.df)} samples | img_dir={self.img_dir} | num_labels={self.num_labels}")

    def __len__(self):
        return len(self.df)

    def _encode_labels(self, label_str: str) -> torch.Tensor:
        y = torch.zeros(self.num_labels, dtype=torch.float32)
        if isinstance(label_str, str) and label_str.strip():
            for t in label_str.split("|"):
                t = t.strip()
                if t in self.label_to_idx:
                    y[self.label_to_idx[t]] = 1.0
        return y

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # Text → BERT tokens
        text = str(row["text"])
        tok = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        input_ids = tok["input_ids"].squeeze(0)
        attention_mask = tok["attention_mask"].squeeze(0)

        # Image
        img_name = os.path.basename(str(row["img_name"]))
        img_path = os.path.join(self.img_dir, img_name)
        try:
            img = Image.open(img_path).convert("RGB")
            image = self.transform(img)
        except Exception:
            image = torch.zeros((3, 224, 224), dtype=torch.float32)

        # Labels → multi-hot
        label = self._encode_labels(str(row["labels"]))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image": image,
            "label": label,
        }

# ============================================================
# Optional wrapper to enforce per-client modality masking
# ============================================================
class ModalityMaskingDataset(Dataset):
    def __init__(self, base_ds: Dataset, modality_json_path: str):
        self.base = base_ds
        try:
            with open(modality_json_path, "r") as f:
                m = json.load(f)
        except Exception:
            m = {"use_text": 1, "use_image": 1}
        self.use_text = bool(m.get("use_text", 1))
        self.use_image = bool(m.get("use_image", 1))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        if not self.use_text:
            sample["input_ids"] = torch.zeros_like(sample["input_ids"])
            sample["attention_mask"] = torch.zeros_like(sample["attention_mask"])
        if not self.use_image:
            sample["image"] = torch.zeros_like(sample["image"])
        return sample

# ============================================================
# Client loaders (per-client CSVs)
# ============================================================
def _read_client_csvs(client_id: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    client_dir = os.path.join(BASE_DATASET_DIR, f"client_{client_id}")
    train_csv = os.path.join(client_dir, "train.csv")
    val_csv   = os.path.join(client_dir, "val.csv")
    test_csv  = os.path.join(client_dir, "test.csv")

    if not all(os.path.exists(p) for p in [train_csv, val_csv, test_csv]):
        raise FileNotFoundError(
            f"Missing CSV for client {client_id} in {client_dir} (need train.csv, val.csv, test.csv)"
        )

    train_df = pd.read_csv(train_csv)
    val_df   = pd.read_csv(val_csv)
    test_df  = pd.read_csv(test_csv)
    return train_df, val_df, test_df

def load_partition(
    dataset: str,
    validation_split: float,
    batch_size: int,
    *,
    client_id: Optional[int] = None,
    max_len: int = 128,
    num_workers: int = 0,
):
    """
    Federated client partition for MM-IMDb.
    - Uses per-client CSVs in ./dataset/client_{id}/
    - Uses posters in ./all_in_one_dataset/mmimdb_posters
    """
    if client_id is None:
        raise ValueError("client_id is required for MM-IMDb federated split (set in conf/config.yaml)")

    train_df, val_df, test_df = _read_client_csvs(client_id)
    train_ds = MMIMDbDataset(train_df, IMG_DIR, max_len=max_len)
    val_ds   = MMIMDbDataset(val_df,   IMG_DIR, max_len=max_len)
    test_ds  = MMIMDbDataset(test_df,  IMG_DIR, max_len=max_len)

    modality_path = os.path.join(BASE_DATASET_DIR, f"client_{client_id}", "modality.json")
    if os.path.exists(modality_path):
        train_ds = ModalityMaskingDataset(train_ds, modality_path)
        val_ds   = ModalityMaskingDataset(val_ds,   modality_path)
        test_ds  = ModalityMaskingDataset(test_ds,  modality_path)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    fl_task = {"dataset": "MM_IMDB", "client_id": client_id}
    logger.info(f"FL_Task - {json.dumps(fl_task)}")

    return train_loader, val_loader, test_loader

# ============================================================
# Server validation loader (central eval on server)
# ============================================================
def _download_server_validation_zip_if_needed():
    """Download and unzip a small server validation set if missing (optional)."""
    root = _server_root()
    imgs_ok = (
        os.path.isdir(os.path.join(root, "mmimdb_posters")) or
        os.path.isdir(os.path.join(root, "img"))
    )
    if os.path.exists(os.path.join(root, "server_test.csv")) and imgs_ok:
        return  # present (even if nested)

    if gdown is None or zipfile is None:
        raise RuntimeError(
            "server_test.csv/img missing and gdown/zipfile not available. "
            "Either mount server_data/ or: pip install gdown"
        )

    os.makedirs(SERVER_DATA_DIR, exist_ok=True)
    logger.info("⬇️ Downloading server validation zip from Google Drive...")
    gdown.download(GDRIVE_URL, SERVER_ZIP_PATH, quiet=False)

    logger.info("📦 Unzipping server validation data...")
    with zipfile.ZipFile(SERVER_ZIP_PATH, "r") as zf:
        zf.extractall(SERVER_DATA_DIR)  # fine if it creates server_data/server_data/*

def _find_server_csv_and_img():
    root = _server_root()

    # CSV
    server_csv_path = None
    for r, _, files in os.walk(root):
        if "server_test.csv" in files:
            server_csv_path = os.path.join(r, "server_test.csv")
            break
    if server_csv_path is None:
        raise FileNotFoundError("server_test.csv not found under server_data/")

    # IMG dir
    server_img_dir = _resolve_server_img_dir()
    return server_csv_path, server_img_dir

def gl_model_torch_validation(batch_size: int = 16, max_len: int = 128, limit: int = 100):
    """
    Used by BOTH server_main.py and client_main.py:
      Priority A: ./dataset/server/server_test.csv + project posters IMG_DIR
      Priority B: <normalized server root>/server_test.csv + <normalized images>
    """
    # A) Fast path
    fast_csv = os.path.abspath("./dataset/server/server_test.csv")
    if os.path.exists(fast_csv):
        df = pd.read_csv(fast_csv)
        if limit and len(df) > limit:
            logger.info(f"⚠️ Limiting server validation from {len(df)} → {limit} samples")
            df = df.sample(n=limit, random_state=42).reset_index(drop=True)
        ds = MMIMDbDataset(df, IMG_DIR, max_len=max_len)
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # B) Fallback (mounted or downloadable server_data/)
    _download_server_validation_zip_if_needed()
    server_csv_path, server_img_dir = _find_server_csv_and_img()

    df = pd.read_csv(server_csv_path)
    if limit and len(df) > limit:
        logger.info(f"⚠️ Limiting server validation from {len(df)} → {limit} samples")
        df = df.sample(n=limit, random_state=42).reset_index(drop=True)

    ds = MMIMDbDataset(df, server_img_dir, max_len=max_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
