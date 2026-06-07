"""
FedOps-WESAD FL Server entry point.

Usage:
    python server_main.py

Starts a Flower FL server with:
  - FedAvg aggregation strategy
  - Server-side evaluation on pooled GlobalData (all subjects' test splits)
  - Per-round .pth snapshots saved locally
"""

import logging
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import hydra
from omegaconf import DictConfig, OmegaConf

from fedops.server.app import FLServer

import data_preparation
import models
from model_wesad import StressTFT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)8.8s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@hydra.main(config_path="./conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)

    logger.info("FedOps-WESAD Server starting ...")
    logger.info(OmegaConf.to_yaml(cfg))

    model = StressTFT(
        input_dim=14,
        static_dim=8,
        hidden_dim=int(getattr(cfg.model, "hidden_dim", 64)),
        dropout=float(getattr(cfg.model, "dropout",    0.1)),
        num_heads=int(getattr(cfg.model, "num_heads",  4)),
    )

    try:
        gl_val_loader = data_preparation.load_global_data(cfg)
        logger.info("GlobalData loaded — %d samples", len(gl_val_loader.dataset))
        test_fn = models.test_torch
    except FileNotFoundError as e:
        logger.warning("GlobalData not found (%s) — server-side eval disabled.", e)
        gl_val_loader = None
        # FedOps calls test_torch even on init; return safe defaults instead of None
        def test_fn(model, loader, cfg):
            return 0.0, 0.0, None

    fl_server = FLServer(
        cfg=cfg,
        model=model,
        model_name=str(cfg.model.name),
        model_type="Pytorch",
        gl_val_loader=gl_val_loader,
        test_torch=test_fn,
    )

    fl_server.start()


if __name__ == "__main__":
    main()
