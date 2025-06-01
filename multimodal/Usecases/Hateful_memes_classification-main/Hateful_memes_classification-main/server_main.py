import hydra
from omegaconf import DictConfig
from fedops.server.app import FLServer
import models
import data_preparation
from hydra.utils import instantiate
import torch.nn as nn


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    model = instantiate(cfg.model)
    model_type = cfg.model_type
    model_name = type(model).__name__

    criterion = nn.CrossEntropyLoss()
    gl_test_torch = models.test_torch()

    # Wrapper to match expected signature (model, loader, cfg)
    def test_wrapper(model, test_loader, _cfg):
        return gl_test_torch(model, test_loader, criterion)

    gl_val_loader = data_preparation.gl_model_torch_validation(batch_size=cfg.batch_size)

    fl_server = FLServer(
        cfg=cfg,
        model=model,
        model_name=model_name,
        model_type=model_type,
        gl_val_loader=gl_val_loader,
        test_torch=test_wrapper  # pass wrapped function
    )

    fl_server.start()


if __name__ == "__main__":
    main()
