"""pytorch-example: A Flower / PyTorch app."""

import os
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

from fedops.server.app import FLServer
import models
import data_preparation
import torch
import logging


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Entry point for launching the Flower (FedOps) federated learning server.
    This script:
      1. Initializes the global model.
      2. Loads the global validation dataset.
      3. Prepares the evaluation function.
      4. Starts the federated learning server.
    """
    # --- Logging setup ---
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("server_main")

    # 1) Initialize the global model
    model = instantiate(cfg.model)          # cfg.model._target_ -> models.YourModel
    model_type = cfg.model_type             # Typically 'torch'
    model_name = type(model).__name__
    log.info(f"[SERVER] Initialize model: {model_name} ({model_type})")

    # 2) Build the global validation dataset
    seq_len        = getattr(cfg.dataset, "seq_len", 6)
    test_split     = getattr(cfg.dataset, "test_split", 0.20)
    restrict_hours = getattr(cfg.dataset, "restrict_hours", None)
    num_workers    = getattr(cfg, "num_workers", 0)
    pin_memory     = torch.cuda.is_available()

    log.info("[SERVER] Loading global validation set...")
    _, _, gl_val_loader = data_preparation.load_partition(
        dataset=cfg.dataset.name,
        validation_split=cfg.dataset.validation_split,
        batch_size=cfg.batch_size,
        seq_length=seq_len,
        test_split=test_split,
        seed=getattr(cfg, "random_seed", 42),
        restrict_hours=restrict_hours,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    log.info("[SERVER] Global validation set loading completed ✅")

    # 3) Evaluation function (shared with clients)
    gl_test_torch = models.test_torch()

    # 4) Launch the federated learning server
    fl_server = FLServer(
        cfg=cfg,
        model=model,
        model_name=model_name,
        model_type=model_type,
        gl_val_loader=gl_val_loader,
        test_torch=gl_test_torch,
    )

    log.info("[SERVER] Starting the federated server...")
    fl_server.start()


if __name__ == "__main__":
    main()


