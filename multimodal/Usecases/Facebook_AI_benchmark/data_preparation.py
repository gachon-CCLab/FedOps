# data_preparation.py

import os
import torch
import __main__
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from datasets import load_dataset, concatenate_datasets
from transformers import BertTokenizer
from torchvision import transforms
from PIL import Image
from huggingface_hub import snapshot_download

# ─── AUTO-DETECT ROLE & CLIENT ───────────────────────────────────────────────
_main_file = getattr(__main__, "__file__", "")
if "server_main.py" in os.path.basename(_main_file):
    ROLE = "server"
else:
    ROLE = "client"

CLIENT_ID = None
if ROLE == "client":
    CLIENT_ID = int(0) #int(os.getenv("HM_CLIENT_ID", "0"))

MAX_SAMPLES = int(2000) #int(os.getenv("MAX_SAMPLES", "0"))  # 0 = no cap

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def _slice_then_split(full_ds, start, end, val_frac, test_frac, seed):
    total  = len(full_ds)
    s      = start or 0
    e      = min(end or total, total)
    sliced = full_ds.select(range(s, e))
    # train/test
    tvt      = sliced.train_test_split(test_size=test_frac, seed=seed)
    test_ds  = tvt["test"]
    trainval = tvt["train"]
    # train/val
    rel_val  = val_frac / (1 - test_frac)
    tv       = trainval.train_test_split(test_size=rel_val, seed=seed)
    return tv["train"], tv["test"], test_ds

# ─── EMPTY DATASET FOR SERVER train/val ───────────────────────────────────────
class _EmptyDataset(Dataset):
    def __len__(self): return 0
    def __getitem__(self, idx): raise IndexError

# ─── HatefulMemesDataset ──────────────────────────────────────────────────────
class HatefulMemesDataset(Dataset):
    def __init__(self, split_name: str, ds, max_length: int = 128):
        """
        split_name: "train"|"validation"|"test" (for logging)
        ds: a 🤗 Dataset object already sliced to exactly what we need
        """
        self.ds        = ds
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.max_len   = max_length

        # ─── download only this slice’s images ───────────────────────────────
        arrow_path   = ds.cache_files[0]["filename"]
        cache_dir    = os.path.dirname(arrow_path)
        imgs         = ds["img"]  # e.g. ["img/00001.png", ...]
        snapshot_dir = snapshot_download(
            repo_id="neuralcatcher/hateful_memes",
            repo_type="dataset",
            cache_dir=cache_dir,
            allow_patterns=imgs,   # only these files
            max_workers=1,         # serialized to avoid rate-limits
        )
        self.img_dir = os.path.join(snapshot_dir, "img")
        print(f"✅ [{ROLE.upper()}] {split_name}: {len(ds)} examples → images in {self.img_dir}")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        item = self.ds[i]
        # — text →
        tok = self.tokenizer(
            item["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        input_ids      = tok["input_ids"].squeeze(0)
        attention_mask = tok["attention_mask"].squeeze(0)
        # — image →
        img_name = os.path.basename(item["img"])
        img_path = os.path.join(self.img_dir, img_name)
        try:
            img   = Image.open(img_path).convert("RGB")
            image = self.transform(img)
        except:
            image = torch.zeros((3, 224, 224))
        # — label →
        label = torch.tensor(item["label"], dtype=torch.long)
        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "image":          image,
            "label":          label,
        }

# ─── DATALOADER FACTORY ────────────────────────────────────────────────────────
def load_partition(batch_size: int = 8):
    # 1) load each HF split then concatenate
    ds_train = load_dataset("neuralcatcher/hateful_memes", split="train")
    ds_val   = load_dataset("neuralcatcher/hateful_memes", split="validation")
    ds_test  = load_dataset("neuralcatcher/hateful_memes", split="test")
    full     = concatenate_datasets([ds_train, ds_val, ds_test])

    # 2) cap to MAX_SAMPLES if set
    total = len(full)
    if MAX_SAMPLES > 0 and MAX_SAMPLES < total:
        full  = full.select(range(MAX_SAMPLES))
        total = MAX_SAMPLES
    half = total // 2

    if ROLE == "client":
        # client: slice its half, then train/val/test split
        start, end         = (0, half) if CLIENT_ID == 0 else (half, None)
        train_ds, val_ds, test_ds = _slice_then_split(
            full, start, end, val_frac=0.1, test_frac=0.1, seed=42
        )
        tr = HatefulMemesDataset("train",      train_ds)
        va = HatefulMemesDataset("validation", val_ds)
        te = HatefulMemesDataset("test",       test_ds)
        return (
            DataLoader(tr, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True),
            DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True),
            DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True),
        )

    else:
        # server: build a global test loader from both halves’ test splits
        print("🧪 SERVER building global test loader")
        _, _, t1 = _slice_then_split(full, 0,    half, 0.1, 0.1, 42)
        _, _, t2 = _slice_then_split(full, half, None, 0.1, 0.1, 42)

        ds1 = HatefulMemesDataset("test", t1); ds1.ds = t1
        ds2 = HatefulMemesDataset("test", t2); ds2.ds = t2
        global_test = ConcatDataset([ds1, ds2])

        # return empty train/val + a real test loader
        empty_loader = DataLoader(_EmptyDataset(), batch_size=batch_size)
        test_loader  = DataLoader(global_test, batch_size=batch_size,
                                  shuffle=False, num_workers=0, pin_memory=True)
        return empty_loader, empty_loader, test_loader

def gl_model_torch_validation(batch_size: int):
    _, val_loader, _ = load_partition(batch_size=batch_size)
    return val_loader
