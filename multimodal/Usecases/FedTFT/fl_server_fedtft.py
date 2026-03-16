"""
fl_server_fedtft.py — FedTFT FL Server

Uses TFTPredictor_FedTFT with 3 horizon-decoupled federated heads.

Aggregation: FVWA (N_k × val_auroc_k weighted) — replaces uniform FedAvg.
Early stopping: on global val loss improvement — same as original.
All parameters are fully federated each round.
"""

import os
import time
import argparse
import torch
import flwr as fl
import numpy as np
from torch.utils.data import DataLoader
from model_fedtft_hdfp import TFTPredictor_FedTFT, TFTDataset
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from fvwa import fvwa_aggregate
from torchmetrics.classification import (
    MultilabelAccuracy, MultilabelF1Score, MultilabelPrecision,
    MultilabelRecall, MultilabelAUROC, MultilabelSpecificity,
)
from bayes_opt import BayesianOptimization
from sklearn.metrics import f1_score, precision_score, recall_score

# -------------------------------------------------------------------
# Args
# -------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data_root",         default="patient_level_split/last_npy_data")
parser.add_argument("--seq_len",           type=int, default=192)
parser.add_argument("--result_tag",        default="R4_fedtft")
parser.add_argument("--port",              type=int, default=8089)
parser.add_argument("--seed",              type=int, default=1)
parser.add_argument("--resume_checkpoint", default=None,
                    help="Path to checkpoint .pth to resume from")
parser.add_argument("--resume_round",      type=int, default=0,
                    help="Round offset: checkpoints saved as round(resume_round+rnd)")
parser.add_argument("--num_rounds",        type=int, default=50,
                    help="Total number of FL rounds to run")
args = parser.parse_args()

# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------
target_names   = ['dangerous_action_1h', 'dangerous_action_1d', 'dangerous_action_1w']
num_targets    = len(target_names)
GLOBAL_DIR     = os.path.join(args.data_root, "GlobalData")
HOSPITAL_DIR   = os.path.join(args.data_root, "HospitalsData")
STATIC_SHAPE   = (14,)
SEQUENCE_SHAPE = (args.seq_len, 25)
TARGETS_SHAPE  = (3,)

def load_memmap_data(file_path, sample_shape, dtype=np.float32):
    if not os.path.exists(file_path):
        return None
    size = os.path.getsize(file_path)
    per  = np.prod(sample_shape) * np.dtype(dtype).itemsize
    n    = size // per
    return np.memmap(file_path, dtype=dtype, mode="r", shape=(n,) + sample_shape)

# -------------------------------------------------------------------
# Load global test set
# -------------------------------------------------------------------
static_test  = load_memmap_data(os.path.join(GLOBAL_DIR, "static_data.npy"),   STATIC_SHAPE)
seq_test     = load_memmap_data(os.path.join(GLOBAL_DIR, "sequence_data.npy"), SEQUENCE_SHAPE)
tgt_test     = load_memmap_data(os.path.join(GLOBAL_DIR, "targets.npy"),       TARGETS_SHAPE)
global_test_loader = DataLoader(
    TFTDataset(list(zip(static_test, seq_test, tgt_test))), batch_size=32)

# -------------------------------------------------------------------
# Load pooled validation set (all hospitals) — identical to original
# -------------------------------------------------------------------
val_static_list, val_seq_list, val_tgt_list = [], [], []
for hospital in os.listdir(HOSPITAL_DIR):
    base = os.path.join(HOSPITAL_DIR, hospital)
    s = load_memmap_data(os.path.join(base, "static_val.npy"),   STATIC_SHAPE)
    q = load_memmap_data(os.path.join(base, "sequence_val.npy"), SEQUENCE_SHAPE)
    t = load_memmap_data(os.path.join(base, "targets_val.npy"),  TARGETS_SHAPE)
    if s is not None:
        val_static_list.append(s); val_seq_list.append(q); val_tgt_list.append(t)

val_static = np.concatenate(val_static_list, axis=0)
val_seq    = np.concatenate(val_seq_list,    axis=0)
val_tgt    = np.concatenate(val_tgt_list,    axis=0)
global_val_loader = DataLoader(
    TFTDataset(list(zip(val_static, val_seq, val_tgt))), batch_size=32)

# -------------------------------------------------------------------
# Model & loss
# -------------------------------------------------------------------
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model   = TFTPredictor_FedTFT(
    input_dim  = SEQUENCE_SHAPE[-1],
    static_dim = STATIC_SHAPE[-1],
    hidden_dim = 64,
).to(device)
loss_fn = torch.nn.BCEWithLogitsLoss()

def save_global_model(model, filename="fedtft_backbone_best.pth"):
    path = os.path.join(GLOBAL_DIR, filename)
    os.makedirs(GLOBAL_DIR, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"✅ Saved model: {path}")

# -------------------------------------------------------------------
# Threshold tuning on pooled val — identical to original
# -------------------------------------------------------------------
def tune_global_thresholds(model, val_loader, method="youden", min_specificity=0.5):
    device = next(model.parameters()).device
    model.eval()

    all_logits = {i: [] for i in range(num_targets)}
    all_true   = {i: [] for i in range(num_targets)}
    with torch.no_grad():
        for sx, seqx, y in val_loader:
            sx, seqx, y = sx.to(device), seqx.to(device), y.to(device)
            logits = model(sx, seqx).cpu().numpy()
            y_np   = y.cpu().numpy()
            for i in range(num_targets):
                all_logits[i].extend(logits[:, i])
                all_true[i].extend(y_np[:, i])

    best_ts = []
    for i in range(num_targets):
        y_true = np.array(all_true[i])
        y_prob = 1.0 / (1.0 + np.exp(-np.array(all_logits[i])))

        def objective(thresh, _yt=y_true, _yp=y_prob):
            thresh = np.clip(thresh, 0.1, 0.9)
            preds = (_yp >= thresh).astype(int)
            tp = np.logical_and(preds==1, _yt==1).sum()
            tn = np.logical_and(preds==0, _yt==0).sum()
            fp = np.logical_and(preds==1, _yt==0).sum()
            fn = np.logical_and(preds==0, _yt==1).sum()
            sens = tp / (tp + fn + 1e-8)
            spec = tn / (tn + fp + 1e-8)
            if spec < min_specificity:
                return -1
            return sens + spec - 1

        opt = BayesianOptimization(
            f=objective, pbounds={"thresh": (0.1, 0.9)},
            verbose=0, random_state=123+i,
        )
        try:
            opt.maximize(init_points=5, n_iter=15)
            best_ts.append(float(opt.max['params']['thresh']))
        except Exception as e:
            print(f"[WARN] Bayesian opt failed for target {i}: {e}")
            best_ts.append(0.5)
    return best_ts

# -------------------------------------------------------------------
# Global evaluation on test set — identical to original
# -------------------------------------------------------------------
def global_evaluate(parameters, thresholds):
    ndarrays = parameters if isinstance(parameters, list) \
               else parameters_to_ndarrays(parameters)
    state = {k: torch.from_numpy(v) for k, v in zip(model.state_dict().keys(), ndarrays)}
    model.load_state_dict(state)
    model.eval()

    acc_m  = MultilabelAccuracy(   num_labels=num_targets, average=None).to(device)
    f1_m   = MultilabelF1Score(    num_labels=num_targets, average=None).to(device)
    prec_m = MultilabelPrecision(  num_labels=num_targets, average=None).to(device)
    rec_m  = MultilabelRecall(     num_labels=num_targets, average=None).to(device)
    auc_m  = MultilabelAUROC(      num_labels=num_targets, average=None).to(device)
    spec_m = MultilabelSpecificity(num_labels=num_targets, average=None).to(device)

    all_preds, all_probs, all_targets = [], [], []
    total_loss = 0.0
    thresh_t = torch.tensor(thresholds, device=device)

    with torch.no_grad():
        for sx, seqx, y in global_test_loader:
            sx, seqx, y = sx.to(device), seqx.to(device), y.to(device)
            logits = model(sx, seqx)
            total_loss += loss_fn(logits, y).item() * sx.size(0)
            probs = torch.sigmoid(logits)
            preds = (probs >= thresh_t).long()
            all_preds.append(preds); all_probs.append(probs); all_targets.append(y.long())

    all_preds   = torch.cat(all_preds)
    all_probs   = torch.cat(all_probs)
    all_targets = torch.cat(all_targets)
    avg_loss    = total_loss / len(global_test_loader.dataset)
    overall_acc = (all_preds == all_targets).float().mean()

    print(f"🌍 Global Eval → Loss {avg_loss:.4f}, Acc {overall_acc:.4f}")
    for i, name in enumerate(target_names):
        auc = auc_m(all_probs, all_targets)[i]
        f1  = f1_m(all_preds, all_targets)[i]
        print(f"  • {name}: AUC={auc:.4f}, F1={f1:.4f}")

    metrics = {"overall_accuracy": overall_acc.item()}
    for i, name in enumerate(target_names):
        metrics.update({
            f"accuracy_{name}":    acc_m(all_preds,  all_targets)[i].item(),
            f"f1_{name}":          f1_m(all_preds,   all_targets)[i].item(),
            f"precision_{name}":   prec_m(all_preds, all_targets)[i].item(),
            f"recall_{name}":      rec_m(all_preds,  all_targets)[i].item(),
            f"auroc_{name}":       auc_m(all_probs,  all_targets)[i].item(),
            f"specificity_{name}": spec_m(all_preds, all_targets)[i].item(),
        })
    return avg_loss, metrics

# -------------------------------------------------------------------
# Strategy: standard FedProx aggregation — identical to original
# -------------------------------------------------------------------
_seed_dir = f"seed{args.seed}"
class FedTFTStrategy(fl.server.strategy.FedProx):
    def __init__(self, *, proximal_mu: float, patience: int, round_offset: int = 0, **kwargs):
        super().__init__(proximal_mu=proximal_mu, **kwargs)
        self.best_loss     = float("inf")
        self.no_improve    = 0
        self.patience      = patience
        self.round_offset  = round_offset
        self.loss_history  = []

    def aggregate_fit(self, rnd, results, failures):
        true_rnd = rnd + self.round_offset  # offset so checkpoints/logs show true round
        if not results:
            return super().aggregate_fit(rnd, results, failures)
        rnd = true_rnd

        # FVWA: w_k = N_k × auroc_k  (see fvwa.py for full description)
        val_aurocs = [fit_res.metrics.get("val_auroc", None) for _, fit_res in results]
        aggregated, norm_w, used_fvwa = fvwa_aggregate(results)
        if used_fvwa:
            print(f"  [FVWA] val AUROCs: {[f'{v:.4f}' for v in val_aurocs]}")
        else:
            print(f"  [FVWA] val_auroc missing — falling back to sample-count weights")
        agg_params = ndarrays_to_parameters(aggregated)
        nd = aggregated

        # Tune thresholds on pooled val, evaluate on global test
        state = {k: torch.from_numpy(v) for k, v in zip(model.state_dict().keys(), nd)}
        model.load_state_dict(state)
        thresholds = tune_global_thresholds(model, global_val_loader)
        print(f"--- Round {rnd} tuned thresholds: {thresholds}")
        loss, metrics = global_evaluate(nd, thresholds)

        # Save every round
        save_global_model(model, filename=f"fedtft_backbone_round{rnd}.pth")

        # Early stopping on val loss — identical to original
        self.loss_history.append(loss)
        if loss < self.best_loss:
            self.best_loss  = loss
            self.no_improve = 0
            save_global_model(model, filename="fedtft_backbone_best.pth")
            save_global_model(model, filename="ablation_R4_fedtft_best.pth")
            print(f"✅ New best (loss={loss:.4f}) at round {rnd}")
        else:
            self.no_improve += 1
            if self.no_improve >= self.patience:
                print(f"!!! Early stopping at round {rnd} (no val loss improvement)")

        return agg_params, {}


# -------------------------------------------------------------------
# Load initial parameters (from checkpoint if resuming)
# -------------------------------------------------------------------
initial_params = None
if args.resume_checkpoint:
    print(f"[Resume] Loading checkpoint: {args.resume_checkpoint} (round offset={args.resume_round})")
    ckpt = torch.load(args.resume_checkpoint, map_location=device)
    model.load_state_dict(ckpt)
    initial_params = ndarrays_to_parameters(
        [v.cpu().numpy() for v in model.state_dict().values()]
    )

# -------------------------------------------------------------------
# Launch server
# -------------------------------------------------------------------
strategy = FedTFTStrategy(
    proximal_mu           = 1e-5,
    patience              = 15,
    round_offset          = args.resume_round,
    fraction_fit          = 1.0,
    min_fit_clients       = 3,
    fraction_evaluate     = 0.0,
    min_available_clients = 3,
    initial_parameters    = initial_params,
)

remaining_rounds = args.num_rounds - args.resume_round
start = time.time()
fl.server.start_server(
    server_address=f"localhost:{args.port}",
    config=fl.server.ServerConfig(num_rounds=remaining_rounds),
    strategy=strategy,
)
print(f"\nFinished {len(strategy.loss_history)} rounds in {time.time()-start:.1f}s")
print("Loss history:", strategy.loss_history)
