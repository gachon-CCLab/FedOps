# client_main.py
import os
import sys
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import data_preparation as dp
import models
from fedmap import FedMAPClientTask  # our wrapper over FedOps FLClientTask

# ── FedProx & Loss hyperparams
FEDPROX_MU = 0.05
FOCAL_GAMMA = 2.0

@hydra.main(config_path="./conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Seeds
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)

    print(OmegaConf.to_yaml(cfg))

    # Expect client index as env or argv (like MNIST manager triggers)
    if len(sys.argv) >= 2 and sys.argv[1].isdigit():
        client_idx = int(sys.argv[1])
    else:
        client_idx = int(os.environ.get("CLIENT_IDX", "1"))
    print(f"📥 [Client {client_idx}] Loading local data partition...")

    train_df, val_df, test_df = dp.load_partition_for_client(client_idx)
    train_loader = DataLoader(dp.HatefulMemesDataset(train_df), batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(dp.HatefulMemesDataset(val_df),   batch_size=cfg.batch_size, shuffle=False)
    test_loader  = DataLoader(dp.HatefulMemesDataset(test_df),  batch_size=cfg.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # Model via Hydra (like MNIST)
    model = instantiate(cfg.model).to(device)
    model_name = type(model).__name__

    # Class weights (imbalance)
    num_hateful = (train_df["label"] == 1).sum()
    num_nonhateful = (train_df["label"] == 0).sum()
    cw = torch.tensor([1.0 / max(1, num_nonhateful), 1.0 / max(1, num_hateful)],
                      dtype=torch.float32, device=device)
    cw = cw / cw.sum()

    # Train/Test fns (FedProx + Focal)
    train_fn = models.train_torch(mu=FEDPROX_MU, class_weights=cw, focal_gamma=FOCAL_GAMMA)
    test_fn  = models.test_torch()

    # Modality flags
    client_dir = os.path.join("dataset", f"client_{client_idx}")
    modality_path = os.path.join(client_dir, "modality.json")
    if os.path.exists(modality_path):
        with open(modality_path, "r") as f:
            modality_flags = json.load(f)
    else:
        modality_flags = {"use_text": 1, "use_image": 1}

    # Registration dictionary expected by FedOps FLClientTask
    registration = {
        "train_loader": train_loader,
        "val_loader":   val_loader,
        "test_loader":  test_loader,
        "model":        model,
        "model_name":   model_name,
        "train_torch":  train_fn,
        "test_torch":   test_fn,
        # FedMAP-specific config (passed to our FedMAPClient under the hood by FedOps adapter)
        "modality_flags": modality_flags,
        "local_lr":      cfg.learning_rate,
        "local_weight_decay": 1e-4,
        "local_epochs":  cfg.num_epochs,
    }

    # Launch FedOps-wrapped client (no change to FedOps core)
    fl_client = FedMAPClientTask(cfg, registration, metadata_fn=None)
    fl_client.start()

if __name__ == "__main__":
    main()
