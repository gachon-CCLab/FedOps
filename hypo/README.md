# FedOps hypo

This guide provides step-by-step instructions on how to implement FedOps clustering + optuna, a federated learning lifecycle management operations framework.

This use case will work just fine without modifying anything.

## Baseline

```
- Baseline
    - client_main.py
    - client_mananger_main.py
    - server_main.py
    - models.py
    - data_preparation.py
    - requirements.txt (for server)
    - conf
        - config.yaml

```

## Step

1. **Start by cloning the FedOps**

```
git clone <https://github.com/gachon-CCLab/FedOps.git> \\
&& mv FedOps/hypo/usecase . \\
&& rm -rf FedOps

```

1. **Customize the FedOps Baseline code.**
- config.yaml

```yaml
# Common

#conf/config.yaml
random_seed: 42

learning_rate: 0.001 # Input model's learning rate

# clustering/HPO Options 
hyperparams: 
- [0.001, 128]
- [0.005, 64]
- [0.01, 32]

model_type: 'Pytorch' # This value should be maintained
model:
  _target_: models.MNISTClassifier # Input your custom model
  output_size: 10 # Input your model's output size (only classification)

dataset:
    name: 'MNIST' # Input your data name
    validation_split: 0.2 # Ratio of dividing train data by validation

# client
task_id: '' # Input your Task Name that you register in FedOps Website

wandb: 
  use: true # Whether to use wandb
  key: '38f6cf3c2c37660d42ebfe8ab434b72d34be3b31' # Input your wandb api key
  account: 'rirakang@gachon.ac.kr' # Input your wandb account
  project: '${dataset.name}_${task_id}'

# server
num_epochs: 1 # number of local epochs
batch_size: 128
num_rounds: 2 # Number of rounds to perform
clients_per_round: 3 # Number of clients participating in the round

server:
  strategy:
	  # clustering/HPO Options : _target_
    _target_: fedops.server.strategy_cluster_optuna.ClusterOptunaFedAvg
    fraction_fit: 0.00001
    fraction_evaluate: 0.000001
    min_fit_clients: ${clients_per_round}
    min_available_clients: ${clients_per_round}
    min_evaluate_clients: ${clients_per_round}

    # clustering/HPO Options 
    warmup_rounds: 1        # Number of warmup rounds before clustering
    recluster_every: 1      # Re-cluster frequency (in rounds)
    eps: 0.2                # DBSCAN epsilon parameter
    min_samples: 2          # DBSCAN min_samples parameter
    objective: "maximize_f1"        # Other options: "maximize_acc" / "minimize_loss"
    search_lr_log: [-5.0, -2.0]     # Search space for log10(lr), e.g. 1e-5 to 1e-2
    search_bs_exp: [3, 7]           # Search space for batch size as 2^exp (8~128)
    search_local_epochs: [1, 3]     # Range of local epochs

```

- **hyperparams**: The initial set of hyperparameter candidates. Specified in the form [learning rate, batch size].
- **ClusterOptunaFedAvg**: A FedAvg variant that combines clustering with Optuna-based hyperparameter search.
- **warmup_rounds**: The number of rounds of warm-up training to run before starting clustering.
- **recluster_every**: Defines how many rounds should pass before re-running clustering.
- **eps**: The distance threshold for DBSCAN (controls how close clients must be to be grouped in the same cluster).
- **min_samples**: The minimum number of samples for DBSCAN (the minimum number of clients required to form a valid cluster).
- **objective**: The optimization target for HPO. Can be set to maximize F1 score, maximize accuracy, or minimize loss.
- **search_lr_log**: Defines the search range for the learning rate in log scale. For example, [-5, -2] corresponds to 1e-5 to 1e-2.
- **search_bs_exp**: Defines the search range for batch size in exponential scale. For example, [3, 7] corresponds to 2³–2⁷ = 8–128.
- **search_local_epochs**: The search range for the number of local epochs each client should use.

- data_preparation.py

```yaml
# data_preparation.py

import os
import json
import logging
from collections import Counter
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchvision import datasets, transforms

# Non-IID partition utility (use exactly this import path as requested)
from fedops.utils.fedco.datasetting import build_parts  # ← keep as-is

# Configure logging
handlers_list = [logging.StreamHandler()]
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=handlers_list)
logger = logging.getLogger(__name__)

"""
Create your data loader for training/testing local & global models.
Return variables must be (train_loader, val_loader, test_loader) for normal operation.
"""

# === Environment variable mapping ===
# FEDOPS_PARTITION_CODE: "0"(iid) | "1"(dirichlet) | "2"(label_skew) | "3"(qty_skew)
#   - if "1": FEDOPS_DIRICHLET_ALPHA (default 0.3)
#   - if "2": FEDOPS_LABELS_PER_CLIENT (default 2)
#   - if "3": FEDOPS_QTY_BETA (default 0.5)
#
# Common:
#   FEDOPS_NUM_CLIENTS (default 1)
#   FEDOPS_CLIENT_ID   (default 0)
#   FEDOPS_SEED        (default 42)
#
# Example:
#   export FEDOPS_PARTITION_CODE=1
#   export FEDOPS_DIRICHLET_ALPHA=0.3
#   export FEDOPS_NUM_CLIENTS=3
#   export FEDOPS_CLIENT_ID=0
def _resolve_mode_from_env() -> str:
    code = os.getenv("FEDOPS_PARTITION_CODE", "0").strip()
    if code == "0":
        return "iid"
    elif code == "1":
        alpha = os.getenv("FEDOPS_DIRICHLET_ALPHA", "0.3").strip()
        return f"dirichlet:{alpha}"
    elif code == "2":
        n_labels = os.getenv("FEDOPS_LABELS_PER_CLIENT", "2").strip()
        return f"label_skew:{n_labels}"
    elif code == "3":
        beta = os.getenv("FEDOPS_QTY_BETA", "0.5").strip()
        return f"qty_skew:beta{beta}"
    else:
        logger.warning(f"[partition] Unknown FEDOPS_PARTITION_CODE={code}, fallback to iid")
        return "iid"

# MNIST
def load_partition(dataset, validation_split, batch_size):
    """
    Build per-client partitioned loaders.
    Returns: train_loader, val_loader, test_loader
    """
    # Basic task logging
    now = datetime.now()
    now_str = now.strftime('%Y-%m-%d %H:%M:%S')
    fl_task = {"dataset": dataset, "start_execution_time": now_str}
    fl_task_json = json.dumps(fl_task)
    logging.info(f'FL_Task - {fl_task_json}')

    # Read Non-IID settings from environment variables
    num_clients = int(os.getenv("FEDOPS_NUM_CLIENTS", "1"))
    client_id   = int(os.getenv("FEDOPS_CLIENT_ID", "0"))
    seed        = int(os.getenv("FEDOPS_SEED", "42"))
    mode_str    = _resolve_mode_from_env()

    logging.info(f"[partition] mode={mode_str}, num_clients={num_clients}, client_id={client_id}, seed={seed}")

    # MNIST preprocessing (grayscale normalization)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load full MNIST training split (download if needed)
    full_dataset = datasets.MNIST(root='./dataset/mnist', train=True, download=True, transform=transform)

    # Build Non-IID index lists per client, then select only this client's subset
    targets_np = full_dataset.targets.numpy() if torch.is_tensor(full_dataset.targets) else full_dataset.targets
    parts = build_parts(targets_np, num_clients=num_clients, mode_str=mode_str, seed=seed)

    if not (0 <= client_id < num_clients):
        raise ValueError(f"CLIENT_ID must be 0..{num_clients-1}, got {client_id}")

    client_indices = parts[client_id]
    if len(client_indices) == 0:
        logger.warning(f"[partition] client {client_id} has 0 samples (mode={mode_str})")

    subset_for_client = Subset(full_dataset, client_indices)

    # Keep original behavior: split the client subset again into train/val/test
    test_split = 0.2
    total_len = len(subset_for_client)
    train_size = int((1 - validation_split - test_split) * total_len)
    validation_size = int(validation_split * total_len)
    test_size = total_len - train_size - validation_size

    if train_size <= 0:
        raise ValueError(
            f"[partition] Not enough samples after partition: total={total_len}, "
            f"val={validation_size}, test={test_size}"
        )

    train_dataset, val_dataset, test_dataset = random_split(
        subset_for_client,
        [train_size, validation_size, test_size],
        generator=torch.Generator().manual_seed(seed + client_id)
    )

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size) if validation_size > 0 else DataLoader([])
    test_loader = DataLoader(test_dataset, batch_size=batch_size) if test_size > 0 else DataLoader([])

    # Simple label histogram for sanity check
    def _count_labels(ds):
        if len(ds) == 0:
            return {}
        labels = []
        for i in range(len(ds)):
            _, y = ds[i]
            y = int(y.item()) if torch.is_tensor(y) else int(y)
            labels.append(y)
        return dict(Counter(labels))

    logging.info(f"[partition] train_size={len(train_dataset)}, val_size={len(val_dataset)}, test_size={len(test_dataset)}")
    logging.info(f"[partition] train_label_hist={_count_labels(train_dataset)}")

    return train_loader, val_loader, test_loader

def gl_model_torch_validation(batch_size):
    """
    Build a loader for centralized/global validation (server-side).
    Uses MNIST test split.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    val_dataset = datasets.MNIST(root='./dataset/mnist', train=False, download=True, transform=transform)
    gl_val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return gl_val_loader
```

Pick the Non-IID mode via environment variables—no code changes required.

### Variables

- `FEDOPS_PARTITION_CODE`
    - `"0"` → IID (default)
    - `"1"` → Dirichlet (use `FEDOPS_DIRICHLET_ALPHA`, default `0.3`)
    - `"2"` → Label-skew (use `FEDOPS_LABELS_PER_CLIENT`, default `2`)
    - `"3"` → Quantity-skew (use `FEDOPS_QTY_BETA`, default `0.5`)
- `FEDOPS_NUM_CLIENTS` — total clients (default `1`)
- `FEDOPS_CLIENT_ID` — this client’s id (default `0`)
- `FEDOPS_SEED` — RNG seed (default `42`)

### Examples

**IID (even split)**

```bash
export FEDOPS_PARTITION_CODE=0
export FEDOPS_NUM_CLIENTS=3
export FEDOPS_CLIENT_ID=0
```

**Dirichlet Non-IID (α = 0.3)**

```bash
export FEDOPS_PARTITION_CODE=1
export FEDOPS_DIRICHLET_ALPHA=0.3
export FEDOPS_NUM_CLIENTS=3
export FEDOPS_CLIENT_ID=1
```

**Label-skew (2 labels per client)**

```bash
export FEDOPS_PARTITION_CODE=2
export FEDOPS_LABELS_PER_CLIENT=2
export FEDOPS_NUM_CLIENTS=5
export FEDOPS_CLIENT_ID=3
```

**Quantity-skew (β = 0.5)**

```bash
export FEDOPS_PARTITION_CODE=3
export FEDOPS_QTY_BETA=0.5
export FEDOPS_NUM_CLIENTS=4
export FEDOPS_CLIENT_ID=2
```

This keeps your pipeline intact, adds clean Non-IID control via env vars, and relies only on `build_parts` from your existing `fedops.utils.fedco.datasetting`.

- Customize the FedOps hypo code to align with your FL task.
    - config.yaml
        - [model.name](http://model.name/): Select from the models of Hugging Face.
            - The current model is `DeepSeek-R1-Distill-Qwen-1.5B`, which requires about 27 GB of GPU memory.
        - [dataset.name](http://dataset.name/): Select from the datasets of HuggingFace.
            - The current dataset is `medical_meadow_medical_flashcards` and is set for medical fine-tuning.
        - task_id: Your taskname that you register in FedOps Website.
        - finetune: Modify LoRA hyperparameter settings.
        - num_epochs: Number of local learnings per round for a client.
        - num_rounds: Number of rounds.
        - clients_per_round: Number of clients participating per round.
    - If you want to change the dataset.
        - You need to modify the files below.
            - data_preparation.py
            - client_main.py
1. **Make it your own GitHub repository**
2. **Login to the FedOps Website**
- Create a Task.
    - Title: It must be the same as the task_id specified in config.
    - Client Type: Silo
    - Description: Your own task description
    - Server Git Repository Address: Repository address created in step 4
1. **Start federated learning**
- Client
    - start `client_main.py` & `client_manager_main.py`
- Task window of FedOps Website
    - Select your clients who are online and press `FL START`.
- Then you can see ui like this


![FedOps hypo example result](../docs/images/fedops_hypo_result_example)

