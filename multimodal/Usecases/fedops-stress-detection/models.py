"""
FedOps Pytorch callbacks for StressTFT.

Signatures required by FedOps:
  train_torch(model, train_loader, epochs, cfg) -> trained_model
  test_torch(model, loader, cfg)                -> (loss, accuracy, metrics_dict)
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_torch(model, train_loader, epochs, cfg):
    model = model.to(DEVICE)
    model.train()

    lr    = float(getattr(cfg.train, "learning_rate", 1e-3))
    wd    = float(getattr(cfg.train, "weight_decay",  1e-4))
    pos_w = float(getattr(cfg.train, "pos_weight",    3.0))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_w, device=DEVICE))

    for _ in range(epochs):
        for static_x, seq_x, targets in train_loader:
            static_x = static_x.to(DEVICE)
            seq_x    = seq_x.to(DEVICE)
            targets  = targets.to(DEVICE)

            optimizer.zero_grad()
            logits = model(static_x, seq_x)
            loss   = criterion(logits, targets)
            loss.backward()
            optimizer.step()

    return model


def test_torch(model, loader, cfg):
    model = model.to(DEVICE)
    model.eval()

    pos_w = float(getattr(cfg.train, "pos_weight", 3.0))
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_w, device=DEVICE))

    total_loss  = 0.0
    all_logits  = []
    all_targets = []

    with torch.no_grad():
        for static_x, seq_x, targets in loader:
            static_x = static_x.to(DEVICE)
            seq_x    = seq_x.to(DEVICE)
            targets  = targets.to(DEVICE)

            logits = model(static_x, seq_x)
            loss   = criterion(logits, targets)
            total_loss += loss.item() * len(targets)

            all_logits.extend(logits.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

    all_logits  = np.array(all_logits,  dtype=np.float32)
    all_targets = np.array(all_targets, dtype=np.float32)

    avg_loss = total_loss / max(len(all_targets), 1)
    preds    = (all_logits > 0.0).astype(np.float32)
    accuracy = float((preds == all_targets).mean())

    try:
        auroc = float(roc_auc_score(all_targets, all_logits))
    except Exception:
        auroc = 0.5

    tp = float(((preds == 1) & (all_targets == 1)).sum())
    fp = float(((preds == 1) & (all_targets == 0)).sum())
    fn = float(((preds == 0) & (all_targets == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return avg_loss, accuracy, {"auroc": auroc, "f1": f1}
