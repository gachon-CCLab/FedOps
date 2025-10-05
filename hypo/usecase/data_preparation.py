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
    code = os.getenv("FEDOPS_PARTITION_CODE", "1").strip()
    if code == "0":
        return "iid"
    elif code == "1":
        alpha = os.getenv("FEDOPS_DIRICHLET_ALPHA", "0.05").strip()
        return f"dirichlet:{alpha}"
    elif code == "2":
        n_labels = os.getenv("FEDOPS_LABELS_PER_CLIENT", "3").strip()
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
    num_clients = int(os.getenv("FEDOPS_NUM_CLIENTS", "3"))
    client_id   = int(os.getenv("FEDOPS_CLIENT_ID", "2"))
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
