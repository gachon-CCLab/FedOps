"""pytorch-example: A Flower / PyTorch app."""

import os
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

from fedops.server.app import FLServer
import models
import data_preparation

# --- 新增：空数据集/Loader & 哑评测 ---
from torch.utils.data import Dataset, DataLoader

class _EmptyDataset(Dataset):
    def __len__(self): return 0
    def __getitem__(self, idx): raise IndexError

def _make_empty_loader():
    return DataLoader(_EmptyDataset(), batch_size=1, shuffle=False)

def _dummy_test_fn():
    """与 models.test_torch() 同型的占位评测函数。"""
    def _inner(model, loader, device):
        loss = 0.0
        num_examples = int(getattr(getattr(loader, "dataset", []), "__len__", lambda: 0)())
        metrics = {"acc": 0.0}
        return loss, num_examples, metrics
    return _inner


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # --- 读跳过开关：优先配置，其次环境变量 ---
    # 可在 config 里加 server.skip_data / server.skip_eval，或用环境变量 SKIP_DATA / SKIP_EVAL
    skip_data = bool(
        getattr(getattr(cfg, "server", {}), "skip_data", False)
        or os.environ.get("SKIP_DATA", "0") not in ("0", "", "false", "False")
    )
    skip_eval = bool(
        getattr(getattr(cfg, "server", {}), "skip_eval", False)
        or os.environ.get("SKIP_EVAL", "0") not in ("0", "", "false", "False")
    )

    # 1) 构建初始化全局模型（SleepSeqClassifier 等）
    model = instantiate(cfg.model)              # cfg.model._target_ -> models.SleepSeqClassifier
    model_type = cfg.model_type                 # 一般为 'torch'
    model_name = type(model).__name__

    # 2) （可选）构建 Fitbit 数据的 DataLoader；若跳过则给空 Loader

    gl_val_loader = _make_empty_loader()
    '''else:
        seq_len = getattr(cfg.dataset, "seq_len", 6)
        # restrict_hours = getattr(cfg.dataset, "restrict_hours", None)
        _, _, gl_val_loader = data_preparation.load_partition(
            dataset=cfg.dataset.name,                  # 仅用于日志
            validation_split=cfg.dataset.validation_split,
            batch_size=cfg.batch_size,
            seq_length=seq_len,
            # restrict_hours=restrict_hours,
        )'''

    # 3) 评测函数：若跳过评测则使用哑评测
    gl_test_torch = _dummy_test_fn()

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


