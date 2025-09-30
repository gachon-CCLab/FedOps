# data_preparation.py

import json
import logging
from collections import Counter
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms


# set log format
handlers_list = [logging.StreamHandler()]

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=handlers_list)

logger = logging.getLogger(__name__)


"""
Create your data loader for training/testing local & global model.
Keep the value of the return variable for normal operation.
"""
# Pytorch version

# MNIST
def load_partition(dataset, validation_split, batch_size):
    """
    The variables train_loader, val_loader, and test_loader must be returned fixedly.
    """
    now = datetime.now()
    now_str = now.strftime('%Y-%m-%d %H:%M:%S')
    fl_task = {"dataset": dataset, "start_execution_time": now_str}
    fl_task_json = json.dumps(fl_task)
    logging.info(f'FL_Task - {fl_task_json}')

    # MNIST Data Preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Adjusted for grayscale
    ])

    # Download MNIST Dataset
    full_dataset = datasets.MNIST(root='./dataset/mnist', train=True, download=True, transform=transform)

    # Splitting the full dataset into train, validation, and test sets
    test_split = 0.2
    train_size = int((1 - validation_split - test_split) * len(full_dataset))
    validation_size = int(validation_split * len(full_dataset))
    test_size = len(full_dataset) - train_size - validation_size
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, validation_size, test_size])

    # DataLoader for training, validation, and test
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader

def gl_model_torch_validation(batch_size):
    """
    Setting up a dataset to evaluate a global model on the server
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Adjusted for grayscale
    ])

    # Load the test set of MNIST Dataset
    val_dataset = datasets.MNIST(root='./dataset/mnist', train=False, download=True, transform=transform)

    # DataLoader for validation
    gl_val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return gl_val_loader


""" If you would like to recreate a noniid situation, please remove the comment below   





import os, json, logging, math, errno
from datetime import datetime
from collections import defaultdict, Counter
from typing import List, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


 
# set log format
handlers_list = [logging.StreamHandler()]
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=handlers_list)
logger = logging.getLogger(__name__)


NUM_CLIENTS   = int(os.getenv("FEDOPS_NUM_CLIENTS", "3"))
CLIENT_ID     = int(os.getenv("FEDOPS_CLIENT_ID", "2"))  # 0~NUM_CLIENTS-1
PARTITION_MODE= os.getenv("FEDOPS_PARTITION", "dirichlet:0.1")  # 예: "dirichlet:0.1", "label_skew:2", "qty_skew:beta0.5"
SEED          = int(os.getenv("FEDOPS_SEED", "42"))
PART_DIR      = os.getenv("FEDOPS_PARTITION_DIR", "./partitions")
DATA_ROOT     = os.getenv("FEDOPS_DATA_ROOT", "./dataset/mnist")

os.makedirs(PART_DIR, exist_ok=True)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)

# Non-IID partition utility

def _targets_numpy(train_ds) -> np.ndarray:
    # torchvision MNIST는 targets(Tensor) 보유
    t = train_ds.targets
    return t.numpy() if torch.is_tensor(t) else np.asarray(t)

def partition_dirichlet(targets: np.ndarray, num_clients: int, alpha: float, min_per_client: int = 5) -> List[List[int]]:
   
  # Sampling a ratio with Dirichlet (alpha) for each class → Indexing each class by that ratio.
    # alpha↓ → deflection↑
   
    n_classes = int(targets.max() + 1)
    idx_by_class = [np.where(targets == c)[0] for c in range(n_classes)]
    for arr in idx_by_class:
        np.random.shuffle(arr)

    client_indices = [[] for _ in range(num_clients)]
    for c, idxs in enumerate(idx_by_class):
        if len(idxs) == 0:
            continue
        # 비율 샘플링
        proportions = np.random.dirichlet(alpha=[alpha] * num_clients)
        # 실제 개수로 변환
        counts = (proportions * len(idxs)).astype(int)

        # 라운딩 보정으로 합 맞추기
        while counts.sum() < len(idxs):
            counts[np.argmax(proportions)] += 1
        while counts.sum() > len(idxs):
            counts[np.argmax(counts)] -= 1

        start = 0
        for k in range(num_clients):
            take = counts[k]
            if take > 0:
                client_indices[k].extend(idxs[start:start+take].tolist())
                start += take

    # 최소 샘플 보장(필요 시 간단 보정)
    for k in range(num_clients):
        if len(client_indices[k]) < min_per_client:
            logger.warning(f"[Dirichlet] client {k} has only {len(client_indices[k])} samples.")

    # 셔플
    for k in range(num_clients):
        np.random.shuffle(client_indices[k])
    return client_indices

def partition_label_skew(targets: np.ndarray, num_clients: int, n_labels_per_client: int = 2) -> List[List[int]]:
   
    # 각 클라이언트가 소수의 라벨만 갖도록(병리적 분할).
    
    n_classes = int(targets.max() + 1)
    idx_by_class = [np.where(targets == c)[0] for c in range(n_classes)]
    for arr in idx_by_class:
        np.random.shuffle(arr)

    client_indices = [[] for _ in range(num_clients)]
    # 라벨 배정(라운드 로빈)
    label_assign = []
    perm = np.random.permutation(n_classes)
    # 각 클라에 n_labels_per_client개씩 라벨 할당
    for k in range(num_clients):
        start = (k * n_labels_per_client) % n_classes
        chosen = [perm[(start + i) % n_classes] for i in range(n_labels_per_client)]
        label_assign.append(set(chosen))

    # 각 라벨 샘플을 해당 라벨을 가진 클라들에게 분배
    for c in range(n_classes):
        candidate_clients = [k for k in range(num_clients) if c in label_assign[k]]
        if not candidate_clients:  # 안전망
            candidate_clients = [np.random.randint(0, num_clients)]
        splits = np.array_split(idx_by_class[c], len(candidate_clients))
        for k, chunk in zip(candidate_clients, splits):
            client_indices[k].extend(chunk.tolist())

    for k in range(num_clients):
        np.random.shuffle(client_indices[k])
    return client_indices

def partition_quantity_skew(targets: np.ndarray, num_clients: int, beta: float = 0.5) -> List[List[int]]:
    #
    각 클라이언트가 데이터 개수 자체가 다르게 되도록(수량 편향). beta↓ → 편차↑
    
    all_idxs = np.arange(len(targets))
    np.random.shuffle(all_idxs)
    # 각 클라 비율 ~ Dirichlet(beta,...)
    props = np.random.dirichlet(alpha=[beta] * num_clients)
    counts = (props * len(all_idxs)).astype(int)
    while counts.sum() < len(all_idxs):
        counts[np.argmax(props)] += 1
    while counts.sum() > len(all_idxs):
        counts[np.argmax(counts)] -= 1

    client_indices, start = [], 0
    for k in range(num_clients):
        take = counts[k]
        client_indices.append(all_idxs[start:start+take].tolist())
        start += take
    return client_indices

def parse_partition_mode(s: str) -> Tuple[str, Dict]:
    #"dirichlet:0.1" -> ("dirichlet", {"alpha":0.1})
    #"label_skew:2" -> ("label_skew", {"n_labels":2})
    #"qty_skew:beta0.5" -> ("qty_skew", {"beta":0.5})
    
    s = s.strip().lower()
    if s.startswith("dirichlet:"):
        alpha = float(s.split(":")[1])
        return "dirichlet", {"alpha": alpha}
    if s.startswith("label_skew:"):
        n = int(s.split(":")[1])
        return "label_skew", {"n_labels": n}
    if s.startswith("qty_skew:beta"):
        beta = float(s.split("beta")[1])
        return "qty_skew", {"beta": beta}
    # 기본값
    return "dirichlet", {"alpha": 0.1}

def _partition_file_name(num_clients: int, mode: str, params: Dict, seed: int) -> str:
    tag = ""
    if mode == "dirichlet":
        tag = f"dirichlet_alpha{params['alpha']}"
    elif mode == "label_skew":
        tag = f"labelskew_n{params['n_labels']}"
    elif mode == "qty_skew":
        tag = f"qtyskew_beta{params['beta']}"
    else:
        tag = "unknown"
    return os.path.join(PART_DIR, f"mnist_{tag}_clients{num_clients}_seed{seed}.json")

def build_or_load_partitions(train_ds, num_clients: int, mode: str, params: Dict, seed: int) -> List[List[int]]:
    set_seed(seed)
    path = _partition_file_name(num_clients, mode, params, seed)
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
        return [list(map(int, idxs)) for idxs in data["partitions"]]

    targets = _targets_numpy(train_ds)

    if mode == "dirichlet":
        parts = partition_dirichlet(targets, num_clients, alpha=float(params["alpha"]))
    elif mode == "label_skew":
        parts = partition_label_skew(targets, num_clients, n_labels_per_client=int(params["n_labels"]))
    elif mode == "qty_skew":
        parts = partition_quantity_skew(targets, num_clients, beta=float(params["beta"]))
    else:
        raise ValueError(f"Unknown partition mode: {mode}")

    # 저장
    payload = {
        "dataset": "MNIST",
        "num_clients": num_clients,
        "mode": mode,
        "params": params,
        "seed": seed,
        "partitions": parts,
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, path)
    logger.info(f"Saved partition to {path}")
    return parts


# Pytorch version
def load_partition(dataset: str, validation_split: float, batch_size: int):
  
    now_str = datetime.now()
    now_str = now.strftime('%Y-%m-%d %H:%M:%S')
    logging.info(json.dumps({"dataset": dataset, "start_execution_time": now_str,
                             "client_id": CLIENT_ID, "num_clients": NUM_CLIENTS,
                             "partition": PARTITION_MODE, "seed": SEED}))


    # MNIST Data Preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),   # Adjusted for grayscale
    ])

    # 2) 전체 train split(=원래 MNIST train)을 다운로드
    full_train = datasets.MNIST(root=DATA_ROOT, train=True, download=True, transform=transform)

    # 3) non-IID 파티션 로드/생성
    mode, params = parse_partition_mode(PARTITION_MODE)
    parts = build_or_load_partitions(full_train, NUM_CLIENTS, mode, params, SEED)

    if CLIENT_ID < 0 or CLIENT_ID >= NUM_CLIENTS:
        raise ValueError(f"CLIENT_ID must be 0..{NUM_CLIENTS-1}, got {CLIENT_ID}")

    my_indices = parts[CLIENT_ID]
    if len(my_indices) == 0:
        logger.warning(f"Client {CLIENT_ID} received 0 samples!")

    client_full = Subset(full_train, my_indices)

    # 4) 클라이언트 내부에서만 train/val 분리 (validation_split 비율)
    val_size = int(len(client_full) * validation_split)
    train_size = len(client_full) - val_size
    if val_size > 0:
        train_ds, val_ds = random_split(client_full, [train_size, val_size],
                                        generator=torch.Generator().manual_seed(SEED + CLIENT_ID))
    else:
        train_ds, val_ds = client_full, Subset(full_train, [])  # 빈 검증

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    # 5) 테스트셋(중앙 공통 평가용) → 필요 시 클라에서도 갖고 있게 함
    test_ds = datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def gl_model_torch_validation(batch_size: int):
    # Common test set loader used to validate a global model on a server (central).
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    val_dataset = datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=transform)
    gl_val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return gl_val_loader

"""
