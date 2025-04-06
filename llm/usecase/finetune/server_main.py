import hydra
from omegaconf import DictConfig

from fedops.server.app import FLServer
import models
import data_preparation
from hydra.utils import instantiate
from transformers import AutoModelForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    
    """
    Set the initial global model you created in models.py.
    """
    # Build init global model using transformers

    model = None
    model_type = cfg.model_type # Check tensorflow or torch model
    model_name = cfg.model.name
    
    # Start fl server
    fl_server = FLServer(cfg=cfg, model=model, model_name=model_name, model_type=model_type)
    fl_server.start()
    

if __name__ == "__main__":
    main()

