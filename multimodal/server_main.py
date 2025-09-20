# server_main.py

import hydra
from omegaconf import DictConfig
from fedops.server.app import FLServer
import models
import data_preparation
import torch.nn as nn
from Aggregation.FedMAP import Map2FedAvgStrategy

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # ── Step 1: Instantiate the global fusion model ──
    print("🧠 Step 1: Instantiating global fusion model")
    model = hydra.utils.instantiate(cfg.model)
    model_name = type(model).__name__
    print(f"✅ Model instantiated: {model_name}")

    # ── Step 2: Prepare validation loader for meta-updates ──
    print("📦 Step 2: Loading validation data for meta-update")
    dev_loader = data_preparation.gl_model_torch_validation(batch_size=cfg.batch_size)
    print(f"✅ Validation loader ready: {len(dev_loader.dataset)} samples")

    # ── Step 3: Wrap test function ──
    criterion = nn.CrossEntropyLoss()
    gl_test = models.test_torch()
    def test_wrapper(m, loader, _):
        return gl_test(m, loader, criterion)

    # ── Step 4: Configure Map²-FedAvg strategy ──
    print("⚙️ Step 4: Configuring Map²-FedAvg strategy")
    strategy = Map2FedAvgStrategy(
        dev_loader=dev_loader,
        mlp_hidden=cfg.server.strategy.mlp_hidden,
        meta_lr=cfg.server.strategy.meta_lr,
        fraction_fit=cfg.server.strategy.fraction_fit,
        fraction_evaluate=cfg.server.strategy.fraction_evaluate,
        min_fit_clients=cfg.server.strategy.min_fit_clients,
        min_evaluate_clients=cfg.server.strategy.min_evaluate_clients,
        min_available_clients=cfg.server.strategy.min_available_clients,
    )
    print("✅ Map²-FedAvg strategy configured")

    # ── Step 5: Launch Flower server ──
    print("🚀 Step 5: Launching Flower server with custom strategy")
    fl_server = FLServer(
        cfg=cfg,
        model=model,
        model_name=model_name,
        model_type=cfg.model_type,
        gl_val_loader=dev_loader,
        test_torch=test_wrapper,
        strategy=strategy,
    )
    fl_server.start()
    print("🏁 Flower server started and awaiting clients...")

if __name__ == "__main__":
    main()
