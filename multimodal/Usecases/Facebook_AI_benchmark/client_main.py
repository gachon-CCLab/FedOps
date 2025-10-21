import random
import hydra
from hydra.utils import instantiate
import numpy as np
import torch
import data_preparation
import models

from fedops.client import client_utils
from fedops.client.app import FLClientTask
import logging
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="./conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # set log format
    handlers_list = [logging.StreamHandler()]
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)8.8s] %(message)s",
                        handlers=handlers_list)
    logger = logging.getLogger(__name__)

    # seeds
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)

    print(OmegaConf.to_yaml(cfg))

    
    # Data
    train_loader, val_loader, test_loader = data_preparation.load_partition(
        dataset=cfg.dataset.name,
        validation_split=cfg.dataset.validation_split,
        batch_size=cfg.batch_size,
        client_id=getattr(cfg, "client_id", None),   # ← NEW
    )

    logger.info('data loaded')

    # Model
    model = instantiate(cfg.model)
    model_type = cfg.model_type
    model_name = type(model).__name__
    train_torch = models.train_torch()
    test_torch = models.test_torch()

    # Resume local model if present
    task_id = cfg.task_id
    local_list = client_utils.local_model_directory(task_id)
    if local_list:
        logger.info('Latest Local Model download')
        model = client_utils.download_local_model(model_type=model_type, task_id=task_id, listdir=local_list, model=model)

    registration = {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "model": model,
        "model_name": model_name,
        "train_torch": train_torch,
        "test_torch": test_torch,
    }

    # --- Ensure flclient_patch.py is importable even after Hydra chdir ---
    import os, sys
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    if BASE_DIR not in sys.path:
        sys.path.insert(0, BASE_DIR)

    # --- Monkey-patch FedOps client class BEFORE creating FLClientTask ---
    from fedops.client import client_fl
    from flclient_patch import MyFLClient
    client_fl.FLClient = MyFLClient

    # Start original task (now using your subclass under the hood)
    fl_client = FLClientTask(cfg, registration)
    fl_client.start()

if __name__ == "__main__":
    main()
