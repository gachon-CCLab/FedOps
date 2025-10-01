"""pytorch-example: A Flower / PyTorch app."""

import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

from fedops.server.app import FLServer
import models
import data_preparation


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # 1) 构建初始化全局模型（SleepSeqClassifier 等）
    model = instantiate(cfg.model)              # cfg.model._target_ -> models.SleepSeqClassifier
    model_type = cfg.model_type                 # 一般为 'torch'
    model_name = type(model).__name__

    # 2) 构建 Fitbit 数据的 DataLoader（三分）并取 test 作为全局评测集
    #    可选参数：seq_length / restrict_hours 可从配置读取（不存在就给默认值）
    seq_len = getattr(cfg.dataset, "seq_len", 6)
    restrict_hours = getattr(cfg.dataset, "restrict_hours", None)

    _, _, gl_val_loader = data_preparation.load_partition(
        dataset=cfg.dataset.name,                  # 仅用于日志
        validation_split=cfg.dataset.validation_split,
        batch_size=cfg.batch_size,
        seq_length=seq_len,
        # restrict_hours=restrict_hours,           # 如需仅夜间，可在配置里开启
    )

    # 3) 评测函数使用你在 models.py 里定义的 test 闭包（BCE + micro 指标）
    gl_test_torch = models.test_torch()

    # 4) 启动联邦服务端
    fl_server = FLServer(
        cfg=cfg,
        model=model,
        model_name=model_name,
        model_type=model_type,
        gl_val_loader=gl_val_loader,
        test_torch=gl_test_torch,
    )
    fl_server.start()


if __name__ == "__main__":
    main()

