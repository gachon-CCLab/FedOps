# server_main.py

import hydra
from omegaconf import DictConfig
from fedops.server.app import FLServer
import models
import data_preparation
import torch.nn as nn
from aggregation.FEDMAP import Map2FedAvgStrategy

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # 1) Instantiate the base fusion model
    model = hydra.utils.instantiate(cfg.model)
    model_name = type(model).__name__

    # 2) Prepare the global dev‐set loader for meta‐updates
    gl_val_loader = data_preparation.gl_model_torch_validation(batch_size=cfg.batch_size)

    # 3) Wrap the test function to fit Flower’s expected signature
    criterion = nn.CrossEntropyLoss()
    def test_wrapper(m, loader, _):
        return models.test_torch()(m, loader, criterion)

    # 4) Build the Map²-FedAvg strategy
    strategy = Map2FedAvgStrategy(
        dev_loader=gl_val_loader,
        mlp_hidden=cfg.server.strategy.mlp_hidden,
        meta_lr=cfg.server.strategy.meta_lr,
        fraction_fit=cfg.server.strategy.fraction_fit,
        fraction_evaluate=cfg.server.strategy.fraction_evaluate,
        min_fit_clients=cfg.server.strategy.min_fit_clients,
        min_evaluate_clients=cfg.server.strategy.min_evaluate_clients,
        min_available_clients=cfg.server.strategy.min_available_clients,
    )

    # 5) Launch the Flower server with custom strategy
    fl_server = FLServer(
        cfg=cfg,
        model=model,
        model_name=model_name,
        model_type=cfg.model_type,
        gl_val_loader=gl_val_loader,
        test_torch=test_wrapper,
        strategy=strategy,
    )
    fl_server.start()


if __name__ == "__main__":
    main()
