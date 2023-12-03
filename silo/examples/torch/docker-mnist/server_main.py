import hydra
from omegaconf import DictConfig

from fedops.server.app import FLServer
import models
import data_preparation
from hydra.utils import instantiate



@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    
    """
    Build initial global model based on dataset name.
    Set the initial global model you created in fl_model.py to match the dataset name.
    """
    # Build init global model using torch
    model = instantiate(cfg.model)
    model_type = cfg.model_type # Check tensorflow or torch model
    criterion, optimizer = models.set_model_hyperparameter(model, cfg.learning_rate)
    model_name = type(model).__name__
    gl_test_torch = models.test_torch() # set torch test
    
    # model, model_name = cfg.model # Build init global model using tensorflow
    
    

    # Load validation data for evaluating global model
    gl_val_loader = data_preparation.gl_model_torch_validation(batch_size=cfg.batch_size) # torch
    # x_val, y_val = fl_data.gl_model_tensorflow_validation() # tensorflow
    
    # Start fl server
    fl_server = FLServer(cfg=cfg, model=model, model_name=model_name, model_type=model_type,criterion=criterion, 
                             optimizer=optimizer, gl_val_loader=gl_val_loader, test_torch=gl_test_torch) # torch
    # fl_server = app.FLServer(config=config, model=model, model_name=model_name, x_val=x_val, y_val=y_val) # tensorflow
    fl_server.start()
    

if __name__ == "__main__":
    main()

