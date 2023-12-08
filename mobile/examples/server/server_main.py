from logging import WARNING

import hydra
from omegaconf import DictConfig
from fedops.server import mobile_app




@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Start fl server
    fl_server = mobile_app.FLMobileServer(cfg=cfg) 
    fl_server.start()
    

if __name__ == "__main__":
    main()