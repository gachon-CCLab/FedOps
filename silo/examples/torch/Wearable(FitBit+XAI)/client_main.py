import random
import hydra
from hydra.utils import instantiate
import numpy as np
import torch
import logging
import os
import data_preparation
import models
import xai_utils  
from fedops.client import client_utils
from fedops.client.app import FLClientTask
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
import seaborn as sns


def plot_heatmap(arr, title, save_path):
    arr = np.array(arr)

    T, F = arr.shape

    if F == 4:
        xticks = ["Steps", "Calories", "HR", "Stress"]
    else:
        xticks = [f"F{i+1}" for i in range(F)]

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        arr,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        cbar=True,
        xticklabels=xticks,
        yticklabels=[f"H{i+1}" for i in range(T)],
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()



def run_xai_after_training(model, test_loader):


    model.eval()


    xb, yb = next(iter(test_loader))
    x = xb[:1] 
    gradcam_raw = xai_utils.grad_cam_lstm(model, x.clone())   # (6,1)
    lime        = xai_utils.lime_ts(model, x.squeeze(0).numpy(), num_samples=200)  # (6,4)
    ig          = xai_utils.ig_ts(model, x.clone())           # (6,4)

    

    
    out_dir = "/home/shi/d_folder/fedops/Wearable(FitBit)/outputs/xai_result"
    os.makedirs(out_dir, exist_ok=True)

    
    '''np.save(f"{out_dir}/grad_cam.npy", gradcam)
    np.save(f"{out_dir}/lime.npy", lime)
    np.save(f"{out_dir}/ig.npy", ig)'''

    
    #plot_heatmap(gradcam, "Grad-CAM Attribution (6×4)", f"{out_dir}/grad_cam.png")
    plot_heatmap(lime,    "LIME",    f"{out_dir}/lime.png")
    plot_heatmap(ig,      "IG",      f"{out_dir}/ig.png")

    #print(f"  - {out_dir}/grad_cam.png")
    print(f"  - {out_dir}/lime.png")
    print(f"  - {out_dir}/ig.png")






# ---------------------------------------------------------
# 🔧 MAIN CLIENT (No seq_len dependency)
# ---------------------------------------------------------
@hydra.main(config_path="./conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:

    # ----- Logging -----
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)8.8s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)

    # ----- Random seed -----
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)

    print(OmegaConf.to_yaml(cfg))

    FIXED_SEQ_LEN = 6   #time
    train_loader, val_loader, test_loader = data_preparation.load_partition(
        dataset=cfg.dataset.name,
        validation_split=cfg.dataset.validation_split,
        batch_size=cfg.batch_size,
        seq_length=FIXED_SEQ_LEN,               
        test_split=getattr(cfg.dataset, "test_split", 0.2),
        seed=cfg.random_seed,
        restrict_hours=getattr(cfg.dataset, "restrict_hours", None),
        num_workers=getattr(cfg, "num_workers", 0),
        pin_memory=torch.cuda.is_available(),
    )

    logger.info("Dataset Loaded.")


    model = instantiate(cfg.model)
    model_type = cfg.model_type
    model_name = type(model).__name__

    train_torch = models.train_torch()
    test_torch = models.test_torch()

    # Load local model if exists
    task_id = cfg.task_id
    local_list = client_utils.local_model_directory(task_id)

    if local_list:
        logger.info("Restoring previous local model...")
        model = client_utils.download_local_model(
            model_type=model_type,
            task_id=task_id,
            listdir=local_list,
            model=model,
        )

    registration = {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "model": model,
        "model_name": model_name,
        "train_torch": train_torch,
        "test_torch": test_torch,
    }
     
    fl_client = FLClientTask(cfg, registration)
    fl_client.start()
    run_xai_after_training(model, test_loader) 

    


if __name__ == "__main__":
    main()
