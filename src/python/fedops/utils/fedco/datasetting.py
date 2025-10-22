# fedops/utils/fedco/datasetting.py

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np


# -------- Public API --------
def build_parts(
    targets: np.ndarray,
    num_clients: int,
    mode_str: str,
    seed: int = 42,
) -> List[List[int]]:
    """
    Build client index lists for Non-IID partitioning.

    Args:
        targets: 1D numpy array of integer class labels (length = N samples).
        num_clients: number of clients to split into (>=1).
        mode_str: one of
            - "iid"
            - "dirichlet:<alpha>"       e.g., "dirichlet:0.3"
            - "label_skew:<k>"          e.g., "label_skew:2"
            - "qty_skew:beta<b>"        e.g., "qty_skew:beta0.5"
        seed: RNG seed for reproducibility.

    Returns:
        List of length `num_clients`, where each item is a list of sample indices
        assigned to that client.
    """
    if num_clients < 1:
        raise ValueError("num_clients must be >= 1")
    if targets.ndim != 1:
        raise ValueError("targets must be a 1D array of integer labels")

    mode, params = _parse_mode(mode_str)

    if num_clients == 1 or mode == "iid":
        rng = np.random.default_rng(seed)
        idxs = np.arange(len(targets))
        rng.shuffle(idxs)
        splits = np.array_split(idxs, num_clients)
        return [s.tolist() for s in splits]

    if mode == "dirichlet":
        alpha = float(params["alpha"])
        return _partition_dirichlet(targets, num_clients, alpha, seed)

    if mode == "label_skew":
        n_labels = int(params["n_labels"])
        return _partition_label_skew(targets, num_clients, n_labels, seed)

    if mode == "qty_skew":
        beta = float(params["beta"])
        return _partition_quantity_skew(targets, num_clients, beta, seed)

    raise ValueError(f"Unsupported mode: {mode_str}")


# -------- Mode parsing --------
def _parse_mode(s: str) -> Tuple[str, Dict]:
    """
    Parse mode string into (mode, params) dict.
    """
    if not s:
        return "iid", {}

    s = s.strip().lower()
    if s == "iid":
        return "iid", {}

    if s.startswith("dirichlet:"):
        try:
            alpha = float(s.split(":", 1)[1])
        except Exception as e:
            raise ValueError(f"Invalid dirichlet spec: {s}") from e
        return "dirichlet", {"alpha": alpha}

    if s.startswith("label_skew:"):
        try:
            n = int(s.split(":", 1)[1])
        except Exception as e:
            raise ValueError(f"Invalid label_skew spec: {s}") from e
        return "label_skew", {"n_labels": n}

    if s.startswith("qty_skew:beta"):
        try:
            beta = float(s.split("beta", 1)[1])
        except Exception as e:
            raise ValueError(f"Invalid qty_skew spec: {s}") from e
        return "qty_skew", {"beta": beta}

    # Fallback
    return "iid", {}


# -------- Partition strategies --------
def _partition_dirichlet(
    targets: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
) -> List[List[int]]:
    """
    Class-wise Dirichlet sampling over clients; lower alpha => more skew.
    """
    if alpha <= 0:
        raise ValueError("alpha must be > 0 for dirichlet")

    rng = np.random.default_rng(seed)
    n_classes = int(targets.max()) + 1
    idx_by_class = [np.where(targets == c)[0] for c in range(n_classes)]
    for arr in idx_by_class:
        rng.shuffle(arr)

    parts: List[List[int]] = [[] for _ in range(num_clients)]

    for idxs in idx_by_class:
        if len(idxs) == 0:
            continue
        p = rng.dirichlet([alpha] * num_clients)
        counts = (p * len(idxs)).astype(int)

        # Adjust rounding so sum(counts) == len(idxs)
        while counts.sum() < len(idxs):
            counts[int(np.argmax(p))] += 1
        while counts.sum() > len(idxs):
            counts[int(np.argmax(counts))] -= 1

        start = 0
        for k in range(num_clients):
            take = int(counts[k])
            if take > 0:
                parts[k].extend(idxs[start : start + take].tolist())
                start += take

    for k in range(num_clients):
        rng.shuffle(parts[k])
    return parts


def _partition_label_skew(
    targets: np.ndarray,
    num_clients: int,
    n_labels: int,
    seed: int,
) -> List[List[int]]:
    """
    Each client gets samples from only n_labels distinct classes (hard label skew).
    """
    if n_labels < 1:
        raise ValueError("n_labels must be >= 1")

    rng = np.random.default_rng(seed)
    n_classes = int(targets.max()) + 1
    idx_by_class = [np.where(targets == c)[0] for c in range(n_classes)]
    for arr in idx_by_class:
        rng.shuffle(arr)

    parts: List[List[int]] = [[] for _ in range(num_clients)]

    # Round-robin class assignment (with wrap-around)
    perm = rng.permutation(n_classes)
    assigned = []
    for k in range(num_clients):
        start = (k * n_labels) % n_classes
        assigned.append(set(perm[start : start + n_labels]))

    # Distribute each class to the clients that were assigned that class
    for c in range(n_classes):
        candidates = [k for k in range(num_clients) if c in assigned[k]]
        if not candidates:
            candidates = [int(rng.integers(0, num_clients))]
        splits = np.array_split(idx_by_class[c], len(candidates))
        for k, chunk in zip(candidates, splits):
            parts[k].extend(chunk.tolist())

    for k in range(num_clients):
        rng.shuffle(parts[k])
    return parts


def _partition_quantity_skew(
    targets: np.ndarray,
    num_clients: int,
    beta: float,
    seed: int,
) -> List[List[int]]:
    """
    Vary only the number of samples per client (no class preference).
    Lower beta => more variance in quantities.
    """
    if beta <= 0:
        raise ValueError("beta must be > 0 for qty_skew")

    rng = np.random.default_rng(seed)
    idxs = np.arange(len(targets))
    rng.shuffle(idxs)

    p = rng.dirichlet([beta] * num_clients)
    counts = (p * len(idxs)).astype(int)

    # Adjust rounding
    while counts.sum() < len(idxs):
        counts[int(np.argmax(p))] += 1
    while counts.sum() > len(idxs):
        counts[int(np.argmax(counts))] -= 1

    parts: List[List[int]] = []
    start = 0
    for k in range(num_clients):
        take = int(counts[k])
        parts.append(idxs[start : start + take].tolist())
        start += take

    return parts
