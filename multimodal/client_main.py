# client_main.py

import random
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

import data_preparation
import models
from fedops.client import client_utils
from fedops.client.app import FLClientTask

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # ── GPU & Seeds ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔌 Using device: {device}")
    print("🔧 Step 1: Setting random seeds")
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # ── Configuration ──
    print("📋 Step 2: Loaded configuration:")
    print(OmegaConf.to_yaml(cfg))

    # ── Data loaders ──
    print("🧪 Step 3: Preparing data loaders for Hateful Memes")
    train_loader, val_loader, test_loader = data_preparation.load_partition(
        batch_size=cfg.batch_size
    )
    print(f"✅ Data loaders ready: "
          f"train={len(train_loader.dataset)}, "
          f"val={len(val_loader.dataset)}, "
          f"test={len(test_loader.dataset)}")

    # ── Make val_loader available to train_torch ──
    print("🔗 Step 4: Injecting validation loader into config")
    cfg.val_loader = val_loader

    # ── Model & FL setup ──
    print("🧠 Step 5: Instantiating fusion model")
    model = instantiate(cfg.model).to(device)
    model_name = type(model).__name__
    print(f"✅ Model instantiated: {model_name}")

    print("🛠️ Step 6: Preparing training and testing functions")
    train_fn = models.train_torch()
    test_fn  = models.test_torch()

    # ── Load local checkpoint if present ──
    print("📁 Step 7: Checking for existing local model checkpoint")
    local_list = client_utils.local_model_directory(cfg.task_id)
    if local_list:
        print("📦 Loading existing local checkpoint")
        model = client_utils.download_local_model(
            cfg.model_type,
            cfg.task_id,
            listdir=local_list,
            model=model
        )
    else:
        print("⏭️ No local checkpoint found, starting fresh")

    # ── Register with FL client ──
    print("📝 Step 8: Registering model and data with FL client")
    registration = {
        "train_loader": train_loader,
        "val_loader":   val_loader,
        "test_loader":  test_loader,
        "model":        model,
        "model_name":   model_name,
        "train_torch":  train_fn,
        "test_torch":   test_fn,
    }

    # ── Start FL ──
    print("🚀 Step 9: Launching FL client task")
    fl_client = FLClientTask(cfg, registration)
    fl_client.start()
    print("🏁 Training started!")

if __name__ == "__main__":
    main()
