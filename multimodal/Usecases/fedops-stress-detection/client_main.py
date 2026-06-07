"""
FedOps-WESAD FL Client entry point.

Usage:
    FEDOPS_PARTITION_ID=0  python client_main.py   # subject S2
    FEDOPS_PARTITION_ID=1  python client_main.py   # subject S3
    ...
    FEDOPS_PARTITION_ID=14 python client_main.py   # subject S17

Environment variables:
    FEDOPS_PARTITION_ID   : subject index from subject_mapping.json (default 0)
    TASK_ID               : FedOps task identifier (for manager API)
"""

import logging
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import hydra
from omegaconf import DictConfig, OmegaConf

from fedops.client.app import FLClientTask

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

    partition_id = int(os.environ.get("FEDOPS_PARTITION_ID", 0))
    logger.info("FedOps-WESAD Client — partition_id=%d", partition_id)
    logger.info(OmegaConf.to_yaml(cfg))

    train_loader, val_loader, test_loader = data_preparation.load_subject_data(cfg)
    logger.info(
        "Data loaded — train=%d  val=%d  test=%d",
        len(train_loader.dataset),
        len(val_loader.dataset),
        len(test_loader.dataset),
    )

    model = StressTFT(
        input_dim=14,
        static_dim=8,
        hidden_dim=int(getattr(cfg.model, "hidden_dim", 64)),
        dropout=float(getattr(cfg.model, "dropout",    0.1)),
        num_heads=int(getattr(cfg.model, "num_heads",  4)),
    )

    fl_task = {
        "model":        model,
        "model_name":   str(cfg.model.name),
        "train_loader": train_loader,
        "val_loader":   val_loader,
        "test_loader":  test_loader,
        "train_torch":  models.train_torch,
        "test_torch":   models.test_torch,
    }

    os.makedirs(f"local_model/{cfg.task_id}", exist_ok=True)

    fl_client = FLClientTask(cfg, fl_task)
    fl_client.start()


if __name__ == "__main__":
    main()
