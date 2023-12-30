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
    
    # Set random_seed
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    
    print(OmegaConf.to_yaml(cfg))
    
    """
    Client data load function
    After setting model method in data_preparation.py, call the model method.
    """
    train_loader, val_loader, test_loader= data_preparation.load_partition(dataset=cfg.dataset.name, 
                                                                        validation_split=cfg.dataset.validation_split, 
                                                                        batch_size=cfg.batch_size) 
    
    logger.info('data loaded')

    """
    Client local model build function
    Set init local model
    After setting model method in models.py, call the model method.
    """
    # torch model
    model = instantiate(cfg.model)
    model_type = cfg.model_type     # Check tensorflow or torch model
    model_name = type(model).__name__
    train_torch = models.train_torch() # set torch train
    test_torch = models.test_torch() # set torch test

    # Local model directory for saving local models
    task_id = cfg.task_id  # FL task ID
    local_list = client_utils.local_model_directory(task_id)

    # If you have local model, download latest local model 
    if local_list:
        logger.info('Latest Local Model download')
        # If you use torch model, you should input model variable in model parameter
        model = client_utils.download_local_model(model_type=model_type, task_id=task_id, listdir=local_list, model=model)  
    
    # Don't change "registration"
    registration = {
        "train_loader" : train_loader,
        "val_loader" : val_loader,
        "test_loader" : test_loader,
        "model" : model,
        "model_name" : model_name,
        "train_torch" : train_torch,
        "test_torch" : test_torch
    }
    
    
    fl_client = FLClientTask(cfg, registration)
    fl_client.start()


if __name__ == "__main__":
    main()
    

