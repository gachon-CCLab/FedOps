"""Runs DNN federated learning for NEDIS dataset."""

from typing import Dict, Union
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from data_preparation import load_datasets
import random
import numpy as np
import utils
from utils import save_results_as_pickle
from hydra.core.hydra_config import HydraConfig
import models
# import sys
# sys.path.append('/home/ccl/Desktop/FedOps/src/python')
from fedops.simulation.app import FLSimulation

FitConfig = Dict[str, Union[bool, float]]

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run DNN federated learning on NEDIS.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    
    random.seed(cfg.random_seed)  # cfg.random_seed는 설정된 시드 값입니다.
    np.random.seed(cfg.random_seed) # 데이터 파티션 부분에서 np.random 사용
    torch.manual_seed(cfg.random_seed) # Model Init Seed
    
    # print config structured as YAML
    print(OmegaConf.to_yaml(cfg))
    
    # partition dataset and get dataloaders
    trainloaders, valloaders, testloader = load_datasets(
        num_clients=cfg.num_clients,
        val_ratio=cfg.dataset.validation_split,
        batch_size=cfg.batch_size,
    )
    
    model = cfg.model
    client_train = models.client_train()
    client_test = models.client_test()
    server_test = models.server_test()
    
    fl_simulation = FLSimulation(cfg, trainloaders, valloaders, testloader, 
                                 client_train, client_test, server_test, model)
    history = fl_simulation.start()
    
    # Experiment completed. Now we save the results and
    # generate plots using the `history`
    print("................")
    print(history)

    # Hydra automatically creates an output directory
    # Let's retrieve it and save some results there
    save_path = HydraConfig.get().runtime.output_dir

    # save results as a Python pickle using a file_path
    # the directory created by Hydra for each run
    save_results_as_pickle(history, file_path=save_path, extra_results={})

    # plot results and include them in the readme
    # strategy_name = strategy.__class__.__name__
    file_suffix: str = (
        f"{cfg.strategy.metric}"
        f"_{cfg.strategy.standard}"
        f"_S={cfg.random_seed}"
        f"_C={cfg.num_clients}"
        f"_E={cfg.num_epochs}"
        f"_R={cfg.num_rounds}"
    )

    utils.plot_metric_from_history(
        history,
        save_path,
        (file_suffix),
    )


if __name__ == "__main__":
    main()
