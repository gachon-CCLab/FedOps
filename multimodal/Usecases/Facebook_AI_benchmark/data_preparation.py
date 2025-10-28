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

# Optional: light dependencies to pull the server validation set if missing
try:
    import gdown, zipfile
except Exception:
    gdown = None
    zipfile = None

# ------------ logging ------------
handlers_list = [logging.StreamHandler()]
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)8.8s] %(message)s", handlers=handlers_list
)
logger = logging.getLogger(__name__)

# ------------ Paths ------------
# Client-side layout (you already have)
BASE_DATASET_DIR = os.path.abspath("/dataset")          # contains client_1/, client_2/, ...
IMG_DIR = os.path.abspath("/all_in_one_dataset/img")    # all images live here

# Server-side validation layout (downloadable)
SERVER_DATA_DIR = os.path.abspath("./server_data")
SERVER_IMG_DIR = os.path.join(SERVER_DATA_DIR, "img")
SERVER_CSV_PATH_DEFAULT = os.path.join(SERVER_DATA_DIR, "server_test.csv")
SERVER_ZIP_PATH = os.path.join(SERVER_DATA_DIR, "server_data.zip")

# Google Drive (zip includes server_test.csv and an img/ folder)
GDRIVE_FILE_ID = "1PEjxFajumhAopGlLxpirXmH5jJ_FQ3AW"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"


# ============================================================
# Dataset
# ============================================================
class HatefulMemesDataset(Dataset):
    """
    Expects a DataFrame with columns: ["img_name", "text", "label"].
    Images are loaded from IMG_DIR/SERVER_IMG_DIR.
    Returns a dict: input_ids, attention_mask, image (3x224x224), label.
    """
    def __init__(self, df: pd.DataFrame, img_dir: str, max_len: int = 128, bert_name: str = "bert-base-uncased"):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.max_len = int(max_len)

        self.tokenizer = BertTokenizer.from_pretrained(bert_name)
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                # You can add normalization if you’d like:
                # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                #                      std=[0.229, 0.224, 0.225]),
            ]
        )

        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")

        logger.info(f"🗂️ HatefulMemesDataset: {len(self.df)} samples | img_dir={self.img_dir}")

    def __len__(self):
        return len(self.df)

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
        except Exception as e:
            logger.warning(f"⚠️ Failed to load image: {img_path} ({e}); using zeros")
            image = torch.zeros((3, 224, 224), dtype=torch.float32)

        label = int(row["label"])
        label = torch.tensor(label, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image": image,
            "label": label,
        }


# ============================================================
# Client loaders
# ============================================================
def _read_client_csvs(client_id: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    client_dir = os.path.join(BASE_DATASET_DIR, f"client_{client_id}")
    train_csv = os.path.join(client_dir, "train.csv")
    val_csv   = os.path.join(client_dir, "val.csv")
    test_csv  = os.path.join(client_dir, "test.csv")

    if not all(os.path.exists(p) for p in [train_csv, val_csv, test_csv]):
        raise FileNotFoundError(
            f"Missing CSV for client {client_id} in {client_dir} "
            f"(need train.csv, val.csv, test.csv)"
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
    Match your client_main call signature but add client_id as a kwarg.
    - dataset / validation_split are unused (kept for API compatibility).
    - client_id must be provided via cfg (see config.yaml).
    """
    if client_id is None:
        raise ValueError("client_id is required for HatefulMemes federated split (set in conf/config.yaml)")

    # Read CSVs for this client and build datasets
    train_df, val_df, test_df = _read_client_csvs(client_id)
    train_ds = HatefulMemesDataset(train_df, IMG_DIR, max_len=max_len)
    val_ds   = HatefulMemesDataset(val_df,   IMG_DIR, max_len=max_len)
    test_ds  = HatefulMemesDataset(test_df,  IMG_DIR, max_len=max_len)

    # DataLoaders (keep num_workers=0 for pod stability unless you tune it)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Log like before
    fl_task = {"dataset": "HATEFUL_MEMES", "client_id": client_id}
    logger.info(f"FL_Task - {json.dumps(fl_task)}")

    return train_loader, val_loader, test_loader


# ============================================================
# Server validation loader (central eval on server)
# ============================================================
def _download_server_validation_zip_if_needed():
    if os.path.exists(SERVER_CSV_PATH_DEFAULT) and os.path.isdir(SERVER_IMG_DIR):
        return  # everything present

    if gdown is None or zipfile is None:
        raise RuntimeError(
            "server_test.csv/img missing and gdown/zipfile not available. "
            "Install extras or pre-mount server_data/ with CSV and images."
        )

    os.makedirs(SERVER_DATA_DIR, exist_ok=True)
    logger.info("⬇️ Downloading server validation zip from Google Drive...")
    gdown.download(GDRIVE_URL, SERVER_ZIP_PATH, quiet=False)

    logger.info("📦 Unzipping server validation data...")
    with zipfile.ZipFile(SERVER_ZIP_PATH, "r") as zf:
        zf.extractall(SERVER_DATA_DIR)


def _find_server_csv_and_img():
    # CSV
    server_csv_path = None
    for root, _, files in os.walk(SERVER_DATA_DIR):
        if "server_test.csv" in files:
            server_csv_path = os.path.join(root, "server_test.csv")
            break
    if server_csv_path is None:
        raise FileNotFoundError("server_test.csv not found under server_data/")

    # IMG dir
    server_img_dir = None
    for root, dirs, _ in os.walk(SERVER_DATA_DIR):
        if os.path.basename(root) == "img":
            server_img_dir = root
            break
    if server_img_dir is None:
        raise FileNotFoundError("img/ folder for server validation not found under server_data/")

    return server_csv_path, server_img_dir


def gl_model_torch_validation(batch_size: int = 16, max_len: int = 128, limit: int = 20):
    """
    Build a DataLoader the server uses to validate global parameters.
    Will download a small validation split if not present.
    """
    if not (os.path.exists(SERVER_CSV_PATH_DEFAULT) and os.path.isdir(SERVER_IMG_DIR)):
        _download_server_validation_zip_if_needed()

    server_csv_path, server_img_dir = _find_server_csv_and_img()
    df = pd.read_csv(server_csv_path)

    # Subsample to keep eval time manageable per Optuna trial
    if limit and len(df) > limit:
        logger.info(f"⚠️ Limiting server validation from {len(df)} → {limit} samples")
        df = df.sample(n=limit, random_state=42).reset_index(drop=True)

    ds = HatefulMemesDataset(df, server_img_dir, max_len=max_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
