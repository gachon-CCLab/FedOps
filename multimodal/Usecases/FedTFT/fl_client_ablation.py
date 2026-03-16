"""
fl_client_ablation.py — Parametrized client for ablation studies.

Usage:
  python fl_client_ablation.py <hospital_index> [--use_cv] [--use_bayes_thresh]

Toggles:
  --use_cv            5-fold CV rotation of training data per FL round
  --use_bayes_thresh  Bayesian threshold tuning (otherwise fixed 0.5)
  --proximal_mu       FedProx mu (0.0 = FedAvg)
"""

import os
import sys
import json
import copy
import argparse
import torch
import flwr as fl
import numpy as np
from torch.utils.data import DataLoader, Subset
from model_fedtft import TFTDataset, TFTPredictor
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.model_selection import KFold
from torchmetrics.classification import MultilabelAUROC

# ── Args ──────────────────────────────────────────────────────────
# sys.argv layout: script hospital_index [--use_cv] [--use_bayes_thresh] ...
# hospital_index must be first positional arg
_idx = sys.argv[1] if len(sys.argv) > 1 else None
_rest = sys.argv[2:] if len(sys.argv) > 2 else []

parser = argparse.ArgumentParser()
parser.add_argument("hospital_index", type=int)
parser.add_argument("--use_cv",           action="store_true", default=False)
parser.add_argument("--use_bayes_thresh", action="store_true", default=False)
parser.add_argument("--proximal_mu",      type=float, default=0.0)
parser.add_argument("--port",             type=int, default=8089,
                    help="Flower server port (match server's --port)")
args = parser.parse_args()

hospital_index = args.hospital_index

def load_memmap_data(path, shape, dtype=np.float32):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    size = os.path.getsize(path)
    per  = np.prod(shape) * np.dtype(dtype).itemsize
    n    = size // per
    return np.memmap(path, dtype=dtype, mode="r", shape=(n,) + shape)

def tune_thresholds_bayes(model, val_loader, device):
    from bayes_opt import BayesianOptimization
    from sklearn.metrics import f1_score
    model.eval()
    all_logits = {i: [] for i in range(3)}
    all_true   = {i: [] for i in range(3)}
    with torch.no_grad():
        for sx, seqx, y in val_loader:
            sx, seqx, y = sx.to(device), seqx.to(device), y.to(device)
            logits = model(sx, seqx).cpu().numpy()
            y_np   = y.cpu().numpy()
            for i in range(3):
                all_logits[i].extend(logits[:, i])
                all_true[i].extend(y_np[:, i])
    best_ts = []
    for i in range(3):
        y_true = np.array(all_true[i])
        y_prob = 1 / (1 + np.exp(-np.array(all_logits[i])))
        def objective(thresh, _yt=y_true, _yp=y_prob):
            thresh = np.clip(thresh, 0.1, 0.9)
            preds = (_yp >= thresh).astype(int)
            tp = np.logical_and(preds==1, _yt==1).sum()
            tn = np.logical_and(preds==0, _yt==0).sum()
            fp = np.logical_and(preds==1, _yt==0).sum()
            fn = np.logical_and(preds==0, _yt==1).sum()
            sens = tp / (tp + fn + 1e-8); spec = tn / (tn + fp + 1e-8)
            if spec < 0.5: return -1
            return sens + spec - 1
        opt = BayesianOptimization(f=objective, pbounds={"thresh": (0.1, 0.9)},
                                   verbose=0, random_state=123+i)
        try:
            opt.maximize(init_points=5, n_iter=15)
            best_ts.append(float(opt.max['params']['thresh']))
        except Exception:
            best_ts.append(0.5)
    return best_ts

# ── Hospital setup ────────────────────────────────────────────────
with open("hospital_mapping.json", encoding="utf-8") as f:
    hospital_mapping = json.load(f)
hospital_name = next((n for n, i in hospital_mapping.items() if i == hospital_index), None)
if hospital_name is None:
    print(f"Invalid hospital index: {hospital_index}"); sys.exit(1)
print(f"[Ablation client] Hospital: {hospital_name}")

base_dir       = os.path.join("patient_level_split", "last_npy_data", "HospitalsData", hospital_name)
STATIC_SHAPE   = (14,)
SEQUENCE_SHAPE = (192, 25)
TARGETS_SHAPE  = (3,)

static_train = load_memmap_data(os.path.join(base_dir, "static_train.npy"), STATIC_SHAPE)
seq_train    = load_memmap_data(os.path.join(base_dir, "sequence_train.npy"), SEQUENCE_SHAPE)
tgt_train    = load_memmap_data(os.path.join(base_dir, "targets_train.npy"), TARGETS_SHAPE)
static_val   = load_memmap_data(os.path.join(base_dir, "static_val.npy"), STATIC_SHAPE)
seq_val      = load_memmap_data(os.path.join(base_dir, "sequence_val.npy"), SEQUENCE_SHAPE)
tgt_val      = load_memmap_data(os.path.join(base_dir, "targets_val.npy"), TARGETS_SHAPE)
static_test  = load_memmap_data(os.path.join(base_dir, "static_test.npy"), STATIC_SHAPE)
seq_test     = load_memmap_data(os.path.join(base_dir, "sequence_test.npy"), SEQUENCE_SHAPE)
tgt_test     = load_memmap_data(os.path.join(base_dir, "targets_test.npy"), TARGETS_SHAPE)

_BASE_TRAIN_DS  = TFTDataset(list(zip(static_train, seq_train, tgt_train)))
_ORIG_VAL_DS    = TFTDataset(list(zip(static_val, seq_val, tgt_val)))
test_loader     = DataLoader(TFTDataset(list(zip(static_test, seq_test, tgt_test))),
                             batch_size=32, shuffle=False)

# 5-fold CV setup
_cv_kf      = KFold(n_splits=5, shuffle=True, random_state=42)
_cv_indices = list(_cv_kf.split(np.arange(len(_BASE_TRAIN_DS))))
_cv_round   = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = TFTPredictor(
    input_dim=SEQUENCE_SHAPE[-1], static_dim=STATIC_SHAPE[-1],
    hidden_dim=64, output_dim=3).to(device)

# ── Flower Client ─────────────────────────────────────────────────
class AblationClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [v.cpu().numpy() for v in model.state_dict().values()]

    def set_parameters(self, parameters, config):
        sd = {k: torch.tensor(v)
              for k, v in zip(model.state_dict().keys(), parameters)}
        model.load_state_dict(sd, strict=True)

    def fit(self, parameters, config):
        global _cv_round
        self.set_parameters(parameters, config)

        # Select training data
        if args.use_cv:
            fold_id = _cv_round % 5
            tr_idx, _ = _cv_indices[fold_id]
            tr_ds = Subset(_BASE_TRAIN_DS, tr_idx.tolist())
            pc = tgt_train[tr_idx].sum(axis=0)
            nc = len(tr_idx) - pc
        else:
            tr_ds = _BASE_TRAIN_DS
            tr_idx = np.arange(len(_BASE_TRAIN_DS))
            pc = tgt_train.sum(axis=0)
            nc = len(tgt_train) - pc

        pw = torch.tensor(nc / (pc + 1e-6), dtype=torch.float32).to(device)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pw)

        tr_loader = DataLoader(tr_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(_ORIG_VAL_DS, batch_size=32, shuffle=False)

        # FedProx snapshot
        mu = args.proximal_mu or float(config.get("proximal_mu", 0.0))
        global_params = [p.clone().detach() for p in model.parameters()]

        optimizer = AdamW(model.parameters(), lr=3e-5/25, weight_decay=1e-5)
        scheduler = OneCycleLR(optimizer, max_lr=3e-5,
                               steps_per_epoch=len(tr_loader), epochs=6,
                               pct_start=0.3, anneal_strategy="cos")

        best_loss, no_imp = float("inf"), 0
        best_state = copy.deepcopy(model.state_dict())

        for epoch in range(1, 7):
            model.train()
            for sx, seqx, ty in tr_loader:
                sx, seqx, ty = sx.to(device), seqx.to(device), ty.to(device)
                optimizer.zero_grad()
                logits = model(sx, seqx)
                loss   = loss_fn(logits, ty)
                if mu > 0:
                    prox = sum((p - gp).norm()**2
                               for p, gp in zip(model.parameters(), global_params))
                    loss = loss + (mu / 2.0) * prox
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for sx, seqx, ty in val_loader:
                    sx, seqx, ty = sx.to(device), seqx.to(device), ty.to(device)
                    val_loss += loss_fn(model(sx, seqx), ty).item() * sx.size(0)
            val_loss /= len(val_loader.dataset)
            print(f"  [{hospital_name}] E{epoch} val={val_loss:.4f}")
            if val_loss < best_loss:
                best_loss, no_imp = val_loss, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                no_imp += 1
                if no_imp >= 2:
                    break

        model.load_state_dict(best_state)

        # Threshold
        if args.use_bayes_thresh:
            thresh = tune_thresholds_bayes(model, val_loader, device)
        else:
            thresh = [0.5, 0.5, 0.5]

        # Val AUROC (used by server FVWA weighting)
        model.eval()
        auc_m = MultilabelAUROC(num_labels=3, average="macro").to(device)
        all_probs, all_targets = [], []
        with torch.no_grad():
            for sx, seqx, ty in val_loader:
                sx, seqx, ty = sx.to(device), seqx.to(device), ty.to(device)
                probs = torch.sigmoid(model(sx, seqx))
                all_probs.append(probs)
                all_targets.append(ty.long())
        val_auroc = auc_m(torch.cat(all_probs), torch.cat(all_targets)).item()
        print(f"  [{hospital_name}] Val AUROC: {val_auroc:.4f}")

        if args.use_cv:
            _cv_round = (_cv_round + 1) % 5

        return self.get_parameters(config), len(tr_ds), {
            "val_auroc":   val_auroc,
            "threshold_0": thresh[0],
            "threshold_1": thresh[1],
            "threshold_2": thresh[2],
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters, config)
        model.eval()
        thresh = config.get("thresholds", [0.5]*3)
        thresh_t = torch.tensor(thresh, device=device)
        total_loss = 0.0
        tp=[0]*3; tn=[0]*3; fp=[0]*3; fn=[0]*3
        with torch.no_grad():
            for sx, seqx, ty in test_loader:
                sx, seqx, ty = sx.to(device), seqx.to(device), ty.to(device)
                logits = model(sx, seqx)
                total_loss += torch.nn.BCEWithLogitsLoss()(logits, ty).item() * sx.size(0)
                probs = torch.sigmoid(logits)
                preds = (probs >= thresh_t).long()
                y     = ty.long()
                for i in range(3):
                    tp[i] += int(((preds[:,i]==1)&(y[:,i]==1)).sum())
                    tn[i] += int(((preds[:,i]==0)&(y[:,i]==0)).sum())
                    fp[i] += int(((preds[:,i]==1)&(y[:,i]==0)).sum())
                    fn[i] += int(((preds[:,i]==0)&(y[:,i]==1)).sum())
        metrics = {}
        for i in range(3):
            tot  = tp[i]+tn[i]+fp[i]+fn[i]+1e-8
            metrics.update({
                f"accuracy_{i}":  (tp[i]+tn[i])/tot,
                f"recall_{i}":    tp[i]/(tp[i]+fn[i]+1e-8),
                f"precision_{i}": tp[i]/(tp[i]+fp[i]+1e-8),
                f"f1_{i}":        2*(tp[i]/(tp[i]+fp[i]+1e-8))*(tp[i]/(tp[i]+fn[i]+1e-8)) /
                                  ((tp[i]/(tp[i]+fp[i]+1e-8))+(tp[i]/(tp[i]+fn[i]+1e-8))+1e-8),
            })
        return float(total_loss/len(test_loader.dataset)), len(test_loader.dataset), metrics


fl.client.start_numpy_client(server_address=f"localhost:{args.port}", client=AblationClient())
