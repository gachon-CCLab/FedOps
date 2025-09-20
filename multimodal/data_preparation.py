# data_preparation.py

import os
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import BertTokenizer
from torchvision import transforms
from PIL import Image

class HatefulMemesDataset(Dataset):
    def __init__(self, split="train", max_length=128):
        print(f"📥 Loading Hateful Memes split: {split}")
        self.dataset = load_dataset("neuralcatcher/hateful_memes", split=split)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.max_length = max_length

        # Locate where HuggingFace cached the images
        arrow_path = self.dataset.cache_files[0]["filename"]
        cache_folder = os.path.dirname(arrow_path)
        self.base_img_dir = os.path.join(cache_folder, "img")
        print(f"✅ Loaded {len(self.dataset)} samples from {split} split")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # Process text
        text = item["text"]
        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)

        # Process image
        rel_path = item["img"]  # e.g. "img/42953.png"
        img_filename = os.path.basename(rel_path)
        img_path = os.path.join(self.base_img_dir, img_filename)
        try:
            img = Image.open(img_path).convert("RGB")
            image_tensor = self.transform(img)
        except Exception as e:
            print(f"❌ Could not load image at {img_path}: {e}")
            image_tensor = torch.zeros((3, 224, 224))

        label = torch.tensor(item["label"], dtype=torch.long)

        return {
            "input_ids":     input_ids,
            "attention_mask":attention_mask,
            "image":         image_tensor,
            "label":         label
        }

def load_partition(batch_size=8):
    print("🧪 Preparing data loaders for Hateful Memes...")
    train_data = HatefulMemesDataset(split="train")
    val_data   = HatefulMemesDataset(split="validation")
    test_data  = HatefulMemesDataset(split="test")

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size,
        num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size,
        num_workers=0, pin_memory=True
    )
    print("✅ Data loaders ready.")
    return train_loader, val_loader, test_loader

def gl_model_torch_validation(batch_size):
    _, val_loader, _ = load_partition(batch_size=batch_size)
    return val_loader
