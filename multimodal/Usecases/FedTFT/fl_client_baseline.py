import os
import sys
import json
import copy
import argparse
import importlib
import torch
import flwr as fl
import numpy as np
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "baselines"))

_MODEL_MAP = {
    "itransformer": ("iTransformer", "ITransformerClassifier"),
    "patchtst":     ("PatchTST",     "PatchTSTClassifier"),
    "fedformer":    ("FEDformer",    "FedformerClassifier"),
    "dlinear":      ("DLinear",      "DLinearClassifier"),
}
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.classification import MultilabelAUROC
from bayes_opt import BayesianOptimization

# -------------------------------
# Helper: Load memmap data
# -------------------------------
def load_memmap_data(file_path, sample_shape, dtype=np.float32):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No such file: {file_path}")
    size = os.path.getsize(file_path)
    per  = np.prod(sample_shape) * np.dtype(dtype).itemsize
    n    = size // per
    return np.memmap(file_path, dtype=dtype, mode="r", shape=(n,) + sample_shape)

# -------------------------------
# Threshold‐tuning helper
# -------------------------------
from sklearn.metrics import f1_score, precision_score, recall_score

def tune_thresholds(model, val_loader, method="youden", min_specificity=0.5):
    """
    Find per-target threshold using Bayesian optimization, optionally maximizing F1 
    and penalizing for imbalance between precision and recall.
    """
    device = next(model.parameters()).device
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
        y_prob = 1.0 / (1.0 + np.exp(-np.array(all_logits[i])))

        def objective(thresh, _y_true=y_true, _y_prob=y_prob, _min_spec=min_specificity):
            thresh = np.clip(thresh, 0.1, 0.9)
            preds = (_y_prob >= thresh).astype(int)
            tp = np.logical_and(preds==1, _y_true==1).sum()
            tn = np.logical_and(preds==0, _y_true==0).sum()
            fp = np.logical_and(preds==1, _y_true==0).sum()
            fn = np.logical_and(preds==0, _y_true==1).sum()
            sens = tp / (tp + fn + 1e-8)
            spec = tn / (tn + fp + 1e-8)
            f1   = f1_score(_y_true, preds, zero_division=0)
            prec = precision_score(_y_true, preds, zero_division=0)
            rec  = recall_score(_y_true, preds, zero_division=0)
            if spec < _min_spec:
                return -1
            if method == "youden":
                score = sens + spec - 1
            elif method == "min_spec_f1":
                balance_penalty = abs(prec - rec)
                score = f1 - 0.5 * balance_penalty
            elif method == "harmonic":
                score = 2 * (sens * spec) / (sens + spec + 1e-8)
            else:
                raise ValueError(f"Unknown method {method}")
            return score

        optimizer = BayesianOptimization(
            f=objective,
            pbounds={"thresh": (0.1, 0.9)},
            verbose=0,
            random_state=123+i,
        )
        try:
            optimizer.maximize(init_points=5, n_iter=15)
            best_t = optimizer.max['params']['thresh']
        except Exception as e:
            print(f"[WARN] Bayesian optimization failed for target {i}, fallback to 0.5 ({e})")
            best_t = 0.5

        best_ts.append(float(best_t))
    return best_ts

# -------------------------------
# Parse args & mapping
# -------------------------------
_parser = argparse.ArgumentParser()
_parser.add_argument("hospital_index", type=int,
                     help="Hospital index (0=동국대, 1=서울대병원, 2=용인관리자)")
_parser.add_argument("--model_name", default="fedformer",
                     choices=list(_MODEL_MAP.keys()),
                     help="Which baseline model to run (must match server's --model_name)")
_parser.add_argument("--port",       type=int, default=8089,
                     help="Flower server port (match server's --port)")
_args = _parser.parse_args()

_mod_name, _cls_name = _MODEL_MAP[_args.model_name]
_mod   = importlib.import_module(_mod_name)
Model  = getattr(_mod, _cls_name)
TFTDataset = getattr(_mod, "TFTDataset")

with open("hospital_mapping.json", "r", encoding="utf-8") as f:
    hospital_mapping = json.load(f)

hospital_index = _args.hospital_index
hospital_name  = next((n for n, i in hospital_mapping.items() if i == hospital_index), None)
if hospital_name is None:
    print(f"Invalid hospital index: {hospital_index}")
    sys.exit(1)
print(f"Starting client for {hospital_name} | model={_args.model_name}")

# -------------------------------
# Paths & shapes
# -------------------------------
base_dir       = os.path.join("patient_level_split", "last_npy_data", "HospitalsData", hospital_name)
STATIC_SHAPE   = (14,)
SEQUENCE_SHAPE = (192, 25)
TARGETS_SHAPE  = (3,)

# -------------------------------
# Load splits
# -------------------------------
static_train = load_memmap_data(os.path.join(base_dir, "static_train.npy"), STATIC_SHAPE)
seq_train    = load_memmap_data(os.path.join(base_dir, "sequence_train.npy"), SEQUENCE_SHAPE)
tgt_train    = load_memmap_data(os.path.join(base_dir, "targets_train.npy"),  TARGETS_SHAPE)

static_val   = load_memmap_data(os.path.join(base_dir, "static_val.npy"),    STATIC_SHAPE)
seq_val      = load_memmap_data(os.path.join(base_dir, "sequence_val.npy"),   SEQUENCE_SHAPE)
tgt_val      = load_memmap_data(os.path.join(base_dir, "targets_val.npy"),    TARGETS_SHAPE)

static_test  = load_memmap_data(os.path.join(base_dir, "static_test.npy"),   STATIC_SHAPE)
seq_test     = load_memmap_data(os.path.join(base_dir, "sequence_test.npy"), SEQUENCE_SHAPE)
tgt_test     = load_memmap_data(os.path.join(base_dir, "targets_test.npy"),  TARGETS_SHAPE)

# -------------------------------
# Build DataLoaders
# -------------------------------
train_loader = DataLoader(TFTDataset(list(zip(static_train, seq_train, tgt_train))), batch_size=32, shuffle=True)
val_loader   = DataLoader(TFTDataset(list(zip(static_val,   seq_val,   tgt_val))),   batch_size=32, shuffle=False)
test_loader  = DataLoader(TFTDataset(list(zip(static_test,  seq_test,  tgt_test))),  batch_size=32, shuffle=False)

# -------------------------------
# Device, model, loss
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODEL_INIT_KWARGS = {
    "itransformer":  dict(seq_len=192, n_vars=25, static_dim=14, out_dim=3,
                          d_model=128, n_heads=4, e_layers=4, d_ff=512, dropout=0.1,
                          use_norm=True),
    "patchtst":      dict(seq_len=192, n_vars=25, static_dim=14, out_dim=3,
                          d_model=128, n_heads=4, e_layers=4, d_ff=512, dropout=0.1),
    "fedformer":     dict(seq_len=192, n_vars=25, static_dim=14, out_dim=3,
                          d_model=128, n_heads=8, e_layers=4, d_ff=512, dropout=0.1,
                          version="Fourier"),
    "dlinear":       dict(seq_len=192, n_vars=25, static_dim=14, out_dim=3,
                          kernel_size=25, individual=True, hidden_dim=256, dropout=0.1),
}
model = Model(**_MODEL_INIT_KWARGS[_args.model_name]).to(device)
# pos‐weight for class imbalance
pos_counts = tgt_train.sum(axis=0)
neg_counts = len(tgt_train) - pos_counts
pos_weight = torch.tensor(neg_counts / (pos_counts + 1e-6), dtype=torch.float32).to(device)
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# -------------------------------
# One‐Cycle optimizer & scheduler
# -------------------------------
max_epochs = 6
peak_lr    = 3e-5
from bayes_opt import BayesianOptimization

# -------------------------------
# Flower NumPyClient
# -------------------------------
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [v.cpu().numpy() for v in model.state_dict().values()]

    def set_parameters(self, parameters, config):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict  = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        # 1) Load global parameters
        self.set_parameters(parameters, config)
        # Re-create optimizer/scheduler fresh for every FL round!
        optimizer  = AdamW(model.parameters(), lr=peak_lr/25, weight_decay=1e-5)
        scheduler  = OneCycleLR(
            optimizer,
            max_lr       = peak_lr,
            steps_per_epoch = len(train_loader),
            epochs       = max_epochs,
            pct_start    = 0.3,
            anneal_strategy="cos",
        )
        # 2) FedProx snapshot
        mu            = float(config.get("proximal_mu", 1e-5))
        global_params = [p.clone().detach() for p in model.parameters()]

        # 3) Train with OneCycleLR + early stopping
        best_loss, no_imp = float("inf"), 0
        patience          = 2
        best_state        = copy.deepcopy(model.state_dict())

        for epoch in range(1, max_epochs + 1):
            model.train()
            for sx, seqx, ty in train_loader:
                sx, seqx, ty = sx.to(device), seqx.to(device), ty.to(device)
                optimizer.zero_grad()

                logits = model(sx, seqx)
                loss   = loss_fn(logits, ty)

                # FedProx penalty
                if mu > 0:
                    prox = sum((p - gp).norm()**2 for p, gp in zip(model.parameters(), global_params))
                    loss = loss + (mu / 2.0) * prox

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()  # ← exactly one scheduler.step() per batch

            # validation (no scheduler.step() here)
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for sx, seqx, ty in val_loader:
                    sx, seqx, ty = sx.to(device), seqx.to(device), ty.to(device)
                    val_loss += loss_fn(model(sx, seqx), ty).item() * sx.size(0)
            val_loss /= len(val_loader.dataset)

            lr = scheduler.get_last_lr()[0]
            print(f"[Client:{hospital_name}] Epoch {epoch} | Val loss={val_loss:.4f} | LR={lr:.1e}")

            if val_loss < best_loss:
                best_loss, no_imp = val_loss, 0
                best_state        = copy.deepcopy(model.state_dict())
            else:
                no_imp += 1
                if no_imp >= patience:
                    print("  → Early stopping")
                    break

        # restore best weights
        model.load_state_dict(best_state)

        # 4) Dynamic threshold tuning on local val set
        best_thresh = tune_thresholds(model, val_loader, method="youden")
        print(f"  → Tuned thresholds: {best_thresh}")
        # Save tuned thresholds to disk (per-hospital)
        thresh_path = os.path.join(base_dir, "best_thresholds.json")
        with open(thresh_path, "w") as f:
            json.dump(best_thresh, f)
        print(f"  → Saved best thresholds to {thresh_path}")

        # 5) Compute train accuracy with those thresholds
        model.eval()
        correct, total = 0, 0
        thresh_arr = torch.tensor(best_thresh, device=device)
        with torch.no_grad():
            for sx, seqx, ty in train_loader:
                sx, seqx, ty = sx.to(device), seqx.to(device), ty.to(device)
                probs = torch.sigmoid(model(sx, seqx))
                preds = (probs >= thresh_arr).long()
                correct += (preds.cpu().numpy() == ty.cpu().numpy()).sum()
                total   += np.prod(ty.shape)
        train_acc = correct / total

        return self.get_parameters(config), len(train_loader.dataset), {
            "accuracy":    train_acc,
            "threshold_0": best_thresh[0],
            "threshold_1": best_thresh[1],
            "threshold_2": best_thresh[2],
        }

    def evaluate(self, parameters, config):
        # standard test‐set eval
        self.set_parameters(parameters, config)
        model.eval()
        thresholds = config.get("thresholds", [0.5]*3)
        thresh_arr = torch.tensor(thresholds, device=device)

        total_loss = 0.0
        tp = [0]*3; tn=[0]*3; fp=[0]*3; fn=[0]*3

        with torch.no_grad():
            for sx, seqx, ty in test_loader:
                sx, seqx, ty = sx.to(device), seqx.to(device), ty.to(device)
                logits = model(sx, seqx)
                total_loss += loss_fn(logits, ty).item() * sx.size(0)

                probs = torch.sigmoid(logits)
                preds = (probs >= thresh_arr).long()
                y     = ty.long()
                for i in range(3):
                    tp[i] += int(((preds[:,i]==1)&(y[:,i]==1)).sum())
                    tn[i] += int(((preds[:,i]==0)&(y[:,i]==0)).sum())
                    fp[i] += int(((preds[:,i]==1)&(y[:,i]==0)).sum())
                    fn[i] += int(((preds[:,i]==0)&(y[:,i]==1)).sum())

        metrics = {}
        for i in range(3):
            tot = tp[i]+tn[i]+fp[i]+fn[i]+1e-8
            acc  = (tp[i]+tn[i]) / tot
            rec  = tp[i] / (tp[i]+fn[i]+1e-8)
            prec = tp[i] / (tp[i]+fp[i]+1e-8)
            f1   = 2*prec*rec/(prec+rec+1e-8)
            metrics.update({
                f"accuracy_{i}":  acc,
                f"recall_{i}":    rec,
                f"precision_{i}": prec,
                f"f1_{i}":        f1,
            })

        avg_loss = total_loss / len(test_loader.dataset)
        return float(avg_loss), len(test_loader.dataset), metrics

# -------------------------------
# ADD-ONLY PATCH: 5-fold CV (train-only CV; keep original val_loader)
# -------------------------------
from sklearn.model_selection import KFold
from torch.utils.data import Subset

# Keep handles to the ORIGINAL datasets/loaders once
_BASE_TRAIN_DS = train_loader.dataset        # TFTDataset built from *_train.npy
_ORIG_VAL_LOADER = val_loader                # <-- keep your original memmap val

_CV_K = 5
_cv_kf = KFold(n_splits=_CV_K, shuffle=True, random_state=42)
_cv_indices = list(_cv_kf.split(np.arange(len(_BASE_TRAIN_DS))))
_cv_round = 0  # rotates each federated round

# Keep a handle to the original fit so we can wrap it
_OriginalFit = FlowerClient.fit

def _make_fold_train_loader(fold_id):
    tr_idx, va_idx = _cv_indices[fold_id]
    tr_ds = Subset(_BASE_TRAIN_DS, tr_idx.tolist())
    tr_loader = DataLoader(tr_ds, batch_size=32, shuffle=True)
    return tr_loader, tr_idx

def _recompute_pos_weight(tr_idx):
    # Recompute pos_weight on the CURRENT FOLD'S training indices
    pos_counts = tgt_train[tr_idx].sum(axis=0)
    neg_counts = len(tr_idx) - pos_counts
    pw = torch.tensor(neg_counts / (pos_counts + 1e-6), dtype=torch.float32).to(device)
    return torch.nn.BCEWithLogitsLoss(pos_weight=pw)

def _cv_wrapped_fit(self, parameters, config):
    global train_loader, val_loader, loss_fn, _cv_round
    # Select fold for this round
    fold_id = _cv_round % _CV_K
    tr_loader, tr_idx = _make_fold_train_loader(fold_id)

    # Swap ONLY the train loader to this fold
    train_loader = tr_loader
    # Keep the original validation loader from *_val.npy
    val_loader   = _ORIG_VAL_LOADER

    # Update loss to use per-fold class weights (based on fold's train indices)
    loss_fn = _recompute_pos_weight(tr_idx)

    print(f"[Client:{hospital_name}] Using 5-fold CV (train only) | Fold {fold_id+1}/{_CV_K} "
          f"(train={len(tr_idx)}, val={len(val_loader.dataset)})")

    # Delegate to the original training routine (unchanged)
    result = _OriginalFit(self, parameters, config)

    # Advance to the next fold for the next FL round
    _cv_round = (_cv_round + 1) % _CV_K
    return result

# Monkey-patch the class with our CV-aware fit
FlowerClient.fit = _cv_wrapped_fit
# -------------------------------  END PATCH  ---------------------------------

# Start Flower client
fl.client.start_numpy_client(server_address=f"localhost:{_args.port}", client=FlowerClient())
