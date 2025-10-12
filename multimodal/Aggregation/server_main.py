# server_main.py
import os
import zipfile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

# One-time lazy install for gdown (same as your original)
try:
    import gdown
except ImportError:
    import subprocess, sys
    print("📦 Installing gdown (first-time only)...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
    import gdown

import data_preparation as dp
import models
from fedmap import FedMAPServer   # wrapper over FedOps FLServer

def ensure_server_data(gdrive_zip_url: str, local_zip: str, server_dir: str):
    if os.path.exists(server_dir) and os.path.isdir(server_dir):
        print(f"✅ Found existing {server_dir}/, skipping download.")
        return server_dir
    print("⚠️ server_data/ not found — downloading zip from Google Drive...")
    gdown.download(gdrive_zip_url, local_zip, quiet=False)
    print("🔓 Unzipping downloaded server_data.zip ...")
    with zipfile.ZipFile(local_zip, "r") as zip_ref:
        zip_ref.extractall(".")
    print("✅ Unzip completed.")
    if not os.path.isdir(server_dir):
        raise FileNotFoundError(f"{server_dir}/ directory not found after unzip.")
    return server_dir

@hydra.main(config_path="./conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # --- Prepare server-side test set ---
    server_dir = ensure_server_data(cfg.server_data.gdrive_zip_url,
                                    cfg.server_data.local_zip,
                                    cfg.server_data.dir)
    test_df = dp.load_server_test_data()
    test_loader = DataLoader(dp.HatefulMemesDataset(test_df, use_server_data=True),
                             batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # --- Global model via Hydra ---
    model = instantiate(cfg.model).to(device)
    model_name = type(model).__name__
    gl_test_torch = models.test_torch()

    @torch.no_grad()
    def compute_f1_macro_current_model(m):
        m.eval()
        total = 0
        correct = 0
        ce = nn.CrossEntropyLoss()
        total_loss = 0.0
        tp0 = fp0 = fn0 = 0
        tp1 = fp1 = fn1 = 0
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            image = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = m(input_ids=input_ids, attention_mask=attention_mask, image=image)
            loss = ce(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            tp1 += ((preds == 1) & (labels == 1)).sum().item()
            fp1 += ((preds == 1) & (labels == 0)).sum().item()
            fn1 += ((preds == 0) & (labels == 1)).sum().item()
            tp0 += ((preds == 0) & (labels == 0)).sum().item()
            fp0 += ((preds == 0) & (labels == 1)).sum().item()
            fn0 += ((preds == 1) & (labels == 0)).sum().item()

        def f1(tp, fp, fn):
            precision = tp / (tp + fp + 1e-12)
            recall    = tp / (tp + fn + 1e-12)
            if precision + recall == 0:
                return 0.0
            return 2 * precision * recall / (precision + recall + 1e-12)

        f1_0 = f1(tp0, fp0, fn0)
        f1_1 = f1(tp1, fp1, fn1)
        f1_macro = 0.5 * (f1_0 + f1_1)
        avg_loss = total_loss / max(total, 1)
        acc = correct / max(total, 1)
        return avg_loss, acc, f1_macro

    # Global evaluate_fn that Flower Strategy will call
    from flwr.common import parameters_to_ndarrays
    def global_evaluate(_server_round, parameters, _config):
        print("🔍 Running global evaluation...")
        ndarrays = parameters_to_ndarrays(parameters)
        state_dict = dict(zip(model.state_dict().keys(), [torch.tensor(v) for v in ndarrays]))
        model.load_state_dict(state_dict, strict=True)
        loss_f1, acc_f1, f1_macro = compute_f1_macro_current_model(model)
        print(f"🌍 Global Test Loss: {loss_f1:.4f}, Acc: {acc_f1:.4f}, F1(macro): {f1_macro:.4f}")
        return loss_f1, {"test_accuracy": acc_f1, "test_f1_macro": f1_macro}

    # Instantiate your custom strategy via Hydra (cfg.server.strategy)
    strategy = instantiate(cfg.server.strategy, evaluate_fn=global_evaluate)

    # Start FedOps-wrapped server that uses your custom strategy
    from fedmap import FedMAPServer
    fl_server = FedMAPServer(
        cfg=cfg, model=model, model_name=model_name, model_type=cfg.model_type,
        gl_val_loader=test_loader, test_torch=gl_test_torch,
        strategy=strategy
    )
    fl_server.start()

if __name__ == "__main__":
    main()
