import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torchvision import transforms
from PIL import Image
import gdown, zipfile

# ✅ NEW (server dataset location)
SERVER_DATA_DIR = os.path.abspath("./server_data")
SERVER_IMG_DIR = os.path.join(SERVER_DATA_DIR, "img")
SERVER_CSV_PATH_DEFAULT = os.path.join(SERVER_DATA_DIR, "server_test.csv")
SERVER_ZIP_PATH = os.path.join(SERVER_DATA_DIR, "server_data.zip")

# 🔑 Google Drive link for auto-download
GDRIVE_FILE_ID = "1PEjxFajumhAopGlLxpirXmH5jJ_FQ3AW"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

# ✅ Keep client paths unchanged
BASE_DATASET_DIR = os.path.abspath("../../dataset")
IMG_DIR = os.path.abspath("../../all_in_one_dataset/img")


def load_partition_for_client(client_id: int):
    """Load train/val/test CSVs for a given client."""
    client_dir = os.path.join(BASE_DATASET_DIR, f"client_{client_id}")
    train_df = pd.read_csv(os.path.join(client_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(client_dir, "val.csv"))
    test_df = pd.read_csv(os.path.join(client_dir, "test.csv"))
    return train_df, val_df, test_df


def load_server_test_data(limit: int = 200):
    """
    Load server_test.csv (download & unzip if missing).
    Optionally limit number of samples to reduce CPU usage.
    """
    server_csv_path = SERVER_CSV_PATH_DEFAULT

    if not os.path.exists(server_csv_path):
        print(f"⚠️ server_test.csv not found at {server_csv_path}")
        print("⬇️ Downloading from Google Drive...")

        os.makedirs(SERVER_DATA_DIR, exist_ok=True)

        # Download zip
        gdown.download(GDRIVE_URL, SERVER_ZIP_PATH, quiet=False)

        # Unzip contents into server_data/
        with zipfile.ZipFile(SERVER_ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(SERVER_DATA_DIR)

        # 🔎 Search recursively for server_test.csv
        found_csv = None
        for root, dirs, files in os.walk(SERVER_DATA_DIR):
            if "server_test.csv" in files:
                found_csv = os.path.join(root, "server_test.csv")
                break

        if not found_csv:
            raise FileNotFoundError(
                f"❌ server_test.csv still not found after download/unzip in {SERVER_DATA_DIR}"
            )

        print(f"✅ Found server_test.csv at {found_csv}")
        server_csv_path = found_csv  # update to discovered path

    df = pd.read_csv(server_csv_path)

    # ✅ Subsample to avoid CPU overload
    if limit and len(df) > limit:
        print(f"⚠️ Limiting server_test.csv from {len(df)} → {limit} samples")
        df = df.sample(n=limit, random_state=42).reset_index(drop=True)

    return df


def gl_model_torch_validation(batch_size: int = 32, max_len: int = 128):
    """Return DataLoader for global server validation (server_test.csv)."""
    df = load_server_test_data(limit=200)
    dataset = HatefulMemesDataset(df, max_len=max_len, use_server_data=True)
    # ✅ Use num_workers=0 to avoid CPU spike in Kubernetes
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


class HatefulMemesDataset(Dataset):
    def __init__(self, df, max_len=128, use_server_data=False):
        print(f"🗂️ Creating HatefulMemesDataset with {len(df)} samples...")
        self.df = df.reset_index(drop=True)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        )
        self.max_len = max_len

        # ✅ Use server images if required
        if use_server_data:
            img_dir = SERVER_IMG_DIR
            if not os.path.exists(img_dir):
                # 🔎 Search recursively for "img" folder
                found_img = None
                for root, dirs, files in os.walk(SERVER_DATA_DIR):
                    if os.path.basename(root) == "img":
                        found_img = root
                        break
                if not found_img:
                    raise FileNotFoundError(
                        f"❌ Server images directory not found in {SERVER_DATA_DIR}"
                    )
                print(f"✅ Found img folder at {found_img}")
                img_dir = found_img

            self.cache_dir = img_dir
        else:
            if not os.path.exists(IMG_DIR):
                raise FileNotFoundError(
                    f"❌ Local image directory not found: {IMG_DIR}\n"
                    f"Make sure you ran `git lfs pull` or have local data."
                )
            self.cache_dir = IMG_DIR

        print(f"📁 Using image directory: {self.cache_dir}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Tokenize text
        tok = self.tokenizer(
            row["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        input_ids = tok["input_ids"].squeeze(0)
        attention_mask = tok["attention_mask"].squeeze(0)

        # Load image
        img_name = os.path.basename(row["img_name"])
        img_path = os.path.join(self.cache_dir, img_name)

        try:
            img = Image.open(img_path).convert("RGB")
            image = self.transform(img)
        except Exception as e:
            print(f"⚠️ Failed to load image at {img_path}: {e}")
            image = torch.zeros((3, 224, 224))

        label = torch.tensor(row["label"], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image": image,
            "label": label,
        }
