"""Load WESAD memmap splits for a given FL partition (subject)."""

import json
import os

import numpy as np
from torch.utils.data import DataLoader

from model_wesad import WESADDataset

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _data_root(cfg):
    data_dir = str(cfg.dataset.data_dir)
    if os.path.isabs(data_dir):
        return data_dir
    return os.path.normpath(os.path.join(_SCRIPT_DIR, data_dir))


def _subject_ids_for_partition(cfg):
    """
    Return the list of subject IDs assigned to this partition.

    Subjects are sorted by their client index, then split evenly across
    FEDOPS_NUM_PARTITIONS partitions.  FEDOPS_PARTITION_ID selects one slice.

    Example (15 subjects, 2 partitions):
      partition 0 → subjects 0-7  (8 subjects, ~700 train samples)
      partition 1 → subjects 8-14 (7 subjects, ~580 train samples)
    """
    partition_id   = int(os.environ.get("FEDOPS_PARTITION_ID",   0))
    num_partitions = int(os.environ.get("FEDOPS_NUM_PARTITIONS", getattr(cfg.dataset, "num_partitions", 1)))

    mapping_path = os.path.normpath(
        os.path.join(_SCRIPT_DIR, "../fedops-health-wesad/subject_mapping.json")
    )
    with open(mapping_path) as f:
        mapping = json.load(f)      # {"S2": 0, "S3": 1, ...}

    # sorted list of subject_ids by their index
    sorted_subjects = [k for k, _ in sorted(mapping.items(), key=lambda x: x[1])]
    n = len(sorted_subjects)

    # split indices evenly across partitions
    indices = list(range(partition_id, n, num_partitions))
    subjects = [sorted_subjects[i] for i in indices]

    if not subjects:
        raise ValueError(
            f"FEDOPS_PARTITION_ID={partition_id} out of range for "
            f"{n} subjects and {num_partitions} partitions."
        )
    return subjects


def _load_split(cfg, subject_id, split):
    subj_dir = os.path.join(_data_root(cfg), "SubjectsData", subject_id)
    static = np.memmap(os.path.join(subj_dir, f"static_{split}.npy"),
                       dtype="float32", mode="r").reshape(-1, 8).copy()
    seqs   = np.memmap(os.path.join(subj_dir, f"sequence_{split}.npy"),
                       dtype="float32", mode="r").reshape(-1, 10, 14).copy()
    tgts   = np.memmap(os.path.join(subj_dir, f"targets_{split}.npy"),
                       dtype="float32", mode="r").reshape(-1, 1).copy()
    return static, seqs, tgts


def load_subject_data(cfg):
    """
    Returns (train_loader, val_loader, test_loader) for this partition.

    Each partition aggregates multiple subjects so that val/test sets are
    large enough to produce meaningful metrics.

    Controlled by:
      FEDOPS_PARTITION_ID     — which partition (default 0)
      FEDOPS_NUM_PARTITIONS   — total partitions (default cfg.dataset.num_partitions or 1)
    """
    subjects = _subject_ids_for_partition(cfg)
    print(f"[data_preparation] partition={os.environ.get('FEDOPS_PARTITION_ID',0)} "
          f"subjects={subjects}")

    def _loader(split, shuffle):
        statics, seqs_list, tgts_list = [], [], []
        for sid in subjects:
            st, sq, tg = _load_split(cfg, sid, split)
            statics.append(st); seqs_list.append(sq); tgts_list.append(tg)
        static = np.concatenate(statics, axis=0)
        seqs   = np.concatenate(seqs_list, axis=0)
        tgts   = np.concatenate(tgts_list, axis=0)
        ds = WESADDataset(static, seqs, tgts)
        return DataLoader(ds, batch_size=int(cfg.batch_size),
                          shuffle=shuffle, num_workers=0, pin_memory=True)

    return (
        _loader("train", shuffle=True),
        _loader("val",   shuffle=False),
        _loader("test",  shuffle=False),
    )


def load_global_data(cfg):
    """Load GlobalData pooled test set for server-side evaluation."""
    gd = os.path.join(_data_root(cfg), "GlobalData")

    static = np.memmap(os.path.join(gd, "static_data.npy"),
                       dtype="float32", mode="r").reshape(-1, 8).copy()
    seqs   = np.memmap(os.path.join(gd, "sequence_data.npy"),
                       dtype="float32", mode="r").reshape(-1, 10, 14).copy()
    tgts   = np.memmap(os.path.join(gd, "targets.npy"),
                       dtype="float32", mode="r").reshape(-1, 1).copy()

    ds = WESADDataset(static, seqs, tgts)
    return DataLoader(ds, batch_size=int(cfg.batch_size),
                      shuffle=False, num_workers=0, pin_memory=True)
