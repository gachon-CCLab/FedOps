import random
import hydra
from hydra.core.hydra_config import HydraConfig
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
    
    # 랜덤 시드 설정
    random.seed(cfg.random_seed)  # cfg.random_seed는 설정된 시드 값입니다.
    np.random.seed(cfg.random_seed) # 데이터 파티션 부분에서 np.random 사용
    torch.manual_seed(cfg.random_seed) # Model Init Seed
    
    print(OmegaConf.to_yaml(cfg))
    
    """
   Client data load function
   Split partition => apply each client dataset(Options)
   After setting data method in client_data.py, call the data method.
   Keep these variables.
   """
    train_loader, val_loader, test_loader, y_label_counter = data_preparation.load_partition(dataset=cfg.dataset.name, 
                                                                        validation_split=cfg.dataset.validation_split, 
                                                                        label_count=cfg.model.output_size,
                                                                        batch_size=cfg.batch_size) # Pytorch version
    # (x_train, y_train), (x_test, y_test), y_label_counter = client_data.load_partition(dataset, FL_client_num, label_count) # Tensorflow version

    logger.info('data loaded')

    """
    #     Client local model build function
    #     Set init local model
    #     After setting model method in client_model.py, call the model method.
    #     """
    # torch model
    model = instantiate(cfg.model)
    model_type = cfg.model_type     # Check tensorflow or torch model
    criterion, optimizer = models.set_model_hyperparameter(model,cfg.learning_rate)
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
        
    registration = {
        "train_loader" : train_loader,
        "val_loader" : val_loader,
        "test_loader" : test_loader,
        "y_label_counter" : y_label_counter,
        "criterion" : criterion,
        "optimizer" : optimizer,
        "model" : model,
        "model_name" : model_name,
        "train_torch" : train_torch,
        "test_torch" : test_torch
    } # torch version
    
    
    fl_client = FLClientTask(cfg, registration)
    fl_client.start()


if __name__ == "__main__":
    main()
    

