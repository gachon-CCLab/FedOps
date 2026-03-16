"""
fl_server_ablation.py — Parametrized server for ablation studies (R1–R4, Row A).

Toggles:
  --use_fvwa       Performance-weighted aggregation (N_k × AUROC_k) — R3, R4, Row B
  --proximal_mu    FedProx mu: 0.0 = FedAvg (R1, Row A), 1e-5 = FedProx (R2–R4, Row B)
  --ablation_name  Label for checkpoint naming (e.g. R1_fedavg, R2_fedprox, R3_fvwa, R4_fedtft, A_fedavg_decoupled)

Note: μ is FIXED per run — no adaptive mu. Grid search selected μ=1e-5 for R2–R4.
"""

import os
import time
import argparse
import torch
import flwr as fl
import numpy as np
from torch.utils.data import DataLoader
from model_fedtft import TFTPredictor, TFTDataset
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from fvwa import fvwa_aggregate
from torchmetrics.classification import (
    MultilabelAccuracy, MultilabelF1Score, MultilabelPrecision,
    MultilabelRecall, MultilabelAUROC, MultilabelSpecificity,
)
from bayes_opt import BayesianOptimization

# ── Args ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--use_fvwa",        action="store_true", default=False)
parser.add_argument("--proximal_mu",     type=float, default=0.0,
                    help="0.0 = FedAvg, >0 = FedProx")
parser.add_argument("--num_rounds",      type=int, default=50)
parser.add_argument("--patience",        type=int, default=15)
parser.add_argument("--ablation_name",   type=str, default="fedavg",
                    help="Label for checkpoint naming (e.g. fedavg, fedprox, fvwa, full)")
parser.add_argument("--port",            type=int, default=8089,
                    help="Flower server port (use different ports for parallel runs)")
parser.add_argument("--seed",            type=int, default=1,
                    help="Seed index (1/2/3)")
args = parser.parse_args()

# ── Constants ─────────────────────────────────────────────────────
target_names   = ['dangerous_action_1h', 'dangerous_action_1d', 'dangerous_action_1w']
num_targets    = len(target_names)
GLOBAL_DIR     = "patient_level_split/last_npy_data/GlobalData"
HOSPITAL_DIR   = "patient_level_split/last_npy_data/HospitalsData"
STATIC_SHAPE   = (14,)
SEQUENCE_SHAPE = (192, 25)
TARGETS_SHAPE  = (3,)

def load_memmap_data(path, shape, dtype=np.float32):
    if not os.path.exists(path):
        return None
    size = os.path.getsize(path)
    per  = np.prod(shape) * np.dtype(dtype).itemsize
    n    = size // per
    return np.memmap(path, dtype=dtype, mode="r", shape=(n,) + shape)

# ── Test data ─────────────────────────────────────────────────────
static_test  = load_memmap_data(os.path.join(GLOBAL_DIR, "static_data.npy"), STATIC_SHAPE)
seq_test     = load_memmap_data(os.path.join(GLOBAL_DIR, "sequence_data.npy"), SEQUENCE_SHAPE)
tgt_test     = load_memmap_data(os.path.join(GLOBAL_DIR, "targets.npy"), TARGETS_SHAPE)
global_test_loader = DataLoader(
    TFTDataset(list(zip(static_test, seq_test, tgt_test))), batch_size=32)

# ── Val data (pooled) ─────────────────────────────────────────────
val_s, val_q, val_t = [], [], []
for h in os.listdir(HOSPITAL_DIR):
    base = os.path.join(HOSPITAL_DIR, h)
    s = load_memmap_data(os.path.join(base, "static_val.npy"), STATIC_SHAPE)
    q = load_memmap_data(os.path.join(base, "sequence_val.npy"), SEQUENCE_SHAPE)
    t = load_memmap_data(os.path.join(base, "targets_val.npy"), TARGETS_SHAPE)
    if s is not None:
        val_s.append(s); val_q.append(q); val_t.append(t)
global_val_loader = DataLoader(
    TFTDataset(list(zip(
        np.concatenate(val_s), np.concatenate(val_q), np.concatenate(val_t)))),
    batch_size=32)

# ── Model ─────────────────────────────────────────────────────────
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model   = TFTPredictor(
    input_dim=SEQUENCE_SHAPE[-1], static_dim=STATIC_SHAPE[-1],
    hidden_dim=64, output_dim=num_targets).to(device)
loss_fn = torch.nn.BCEWithLogitsLoss()


def tune_thresholds(model, val_loader, min_spec=0.5):
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
        y_prob = 1 / (1 + np.exp(-np.array(all_logits[i])))
        def objective(thresh, _yt=y_true, _yp=y_prob):
            thresh = np.clip(thresh, 0.1, 0.9)
            preds = (_yp >= thresh).astype(int)
            tp = np.logical_and(preds==1, _yt==1).sum()
            tn = np.logical_and(preds==0, _yt==0).sum()
            fp = np.logical_and(preds==1, _yt==0).sum()
            fn = np.logical_and(preds==0, _yt==1).sum()
            sens = tp / (tp + fn + 1e-8); spec = tn / (tn + fp + 1e-8)
            if spec < min_spec: return -1
            return sens + spec - 1
        opt = BayesianOptimization(f=objective, pbounds={"thresh": (0.1, 0.9)},
                                   verbose=0, random_state=123+i)
        try:
            opt.maximize(init_points=5, n_iter=15)
            best_ts.append(float(opt.max['params']['thresh']))
        except Exception:
            best_ts.append(0.5)
    return best_ts


def global_evaluate(ndarrays, thresholds):
    all_keys = list(model.state_dict().keys())
    if len(ndarrays) < len(all_keys):
        backbone_keys = [k for k in all_keys if not k.startswith("grn2")]
        current_sd = model.state_dict()
        current_sd.update({k: torch.from_numpy(v) for k, v in zip(backbone_keys, ndarrays)})
        model.load_state_dict(current_sd)
    else:
        sd = {k: torch.from_numpy(v) for k, v in zip(all_keys, ndarrays)}
        model.load_state_dict(sd)
    model.eval()

    auc_m  = MultilabelAUROC(num_labels=num_targets, average=None).to(device)
    f1_m   = MultilabelF1Score(num_labels=num_targets, average=None).to(device)
    spec_m = MultilabelSpecificity(num_labels=num_targets, average=None).to(device)
    rec_m  = MultilabelRecall(num_labels=num_targets, average=None).to(device)
    prec_m = MultilabelPrecision(num_labels=num_targets, average=None).to(device)
    acc_m  = MultilabelAccuracy(num_labels=num_targets, average=None).to(device)

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
            all_preds.append(preds); all_probs.append(probs)
            all_targets.append(y.long())

    all_preds   = torch.cat(all_preds)
    all_probs   = torch.cat(all_probs)
    all_targets = torch.cat(all_targets)
    avg_loss    = total_loss / len(global_test_loader.dataset)
    overall_acc = (all_preds == all_targets).float().mean().item()

    auc  = auc_m(all_probs, all_targets)
    f1   = f1_m(all_preds, all_targets)
    spec = spec_m(all_preds, all_targets)
    rec  = rec_m(all_preds, all_targets)
    prec = prec_m(all_preds, all_targets)
    acc  = acc_m(all_preds, all_targets)

    short = ['1h', '1d', '1w']
    metrics = {"loss": avg_loss, "overall_accuracy": overall_acc}
    for i, s in enumerate(short):
        metrics.update({
            f"auroc_{s}": auc[i].item(),  f"f1_{s}": f1[i].item(),
            f"spec_{s}": spec[i].item(),  f"rec_{s}": rec[i].item(),
            f"prec_{s}": prec[i].item(),  f"acc_{s}": acc[i].item(),
        })
    print(f"  Loss={avg_loss:.4f} Acc={overall_acc:.4f} | "
          + " | ".join(f"AUC_{s}={auc[i].item():.4f} F1_{s}={f1[i].item():.4f}"
                       for i, s in enumerate(short)))
    return avg_loss, metrics


# ── Strategy ──────────────────────────────────────────────────────
class AblationStrategy(fl.server.strategy.FedProx if args.proximal_mu > 0
                       else fl.server.strategy.FedAvg):

    def __init__(self):
        kwargs = dict(
            fraction_fit=1.0, min_fit_clients=3,
            fraction_evaluate=0.0, min_available_clients=3,
        )
        if args.proximal_mu > 0:
            kwargs["proximal_mu"] = args.proximal_mu
        super().__init__(**kwargs)
        self.best_loss    = float("inf")
        self.no_improve   = 0
        self.loss_history = []
        self.mu = args.proximal_mu

    def aggregate_fit(self, rnd, results, failures):
        if not results:
            return super().aggregate_fit(rnd, results, failures)

        if args.use_fvwa:
            # FVWA: w_k = N_k × auroc_k  (see fvwa.py for full description)
            aggregated, norm_w, _ = fvwa_aggregate(results)
            agg_params = ndarrays_to_parameters(aggregated)
        else:
            agg_res = super().aggregate_fit(rnd, results, failures)
            agg_params = agg_res[0] if isinstance(agg_res, tuple) else agg_res
            aggregated = parameters_to_ndarrays(agg_params)

        # Tune thresholds & evaluate
        nd = aggregated
        all_keys = list(model.state_dict().keys())
        if len(nd) < len(all_keys):
            # FedPer/FedRep: clients sent backbone-only params (no grn2 head)
            backbone_keys = [k for k in all_keys if not k.startswith("grn2")]
            current_sd = model.state_dict()
            current_sd.update({k: torch.from_numpy(v) for k, v in zip(backbone_keys, nd)})
            model.load_state_dict(current_sd)
        else:
            sd = {k: torch.from_numpy(v) for k, v in zip(all_keys, nd)}
            model.load_state_dict(sd)
        thresholds = tune_thresholds(model, global_val_loader)
        print(f"Round {rnd} [{args.ablation_name}] thresholds={thresholds}")
        loss, metrics = global_evaluate(nd, thresholds)

        # Early stopping & save
        self.loss_history.append(loss)
        if loss < self.best_loss:
            self.best_loss  = loss
            self.no_improve = 0
            save_path = os.path.join(GLOBAL_DIR, f"ablation_{args.ablation_name}_best.pth")
            os.makedirs(GLOBAL_DIR, exist_ok=True)
            torch.save(model.state_dict(), save_path)
        else:
            self.no_improve += 1
            if self.no_improve >= args.patience:
                print(f"Early stopping at round {rnd}")

        return agg_params, metrics


strategy = AblationStrategy()

print(f"\n=== Ablation: {args.ablation_name} | "
      f"FVWA={args.use_fvwa} | mu={args.proximal_mu} ===\n")

start = time.time()
fl.server.start_server(
    server_address=f"localhost:{args.port}",
    config=fl.server.ServerConfig(num_rounds=args.num_rounds),
    strategy=strategy,
)
print(f"\nDone in {time.time()-start:.1f}s.")
