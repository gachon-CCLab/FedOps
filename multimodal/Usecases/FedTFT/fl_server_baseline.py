import os
import sys
import time
import argparse
import importlib
import torch
import flwr as fl
import numpy as np
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "baselines"))
from flwr.common import parameters_to_ndarrays

# -------------------------------------------------------------------
# Args (must come before model import so --model_name is available)
# -------------------------------------------------------------------
_parser = argparse.ArgumentParser()
_parser.add_argument("--model_name", default="fedformer",
                     choices=["itransformer","patchtst","fedformer","dlinear"],
                     help="Which baseline model to run")
_parser.add_argument("--port",       type=int, default=8089,
                     help="Flower server port (use different ports for parallel runs)")
_parser.add_argument("--seed",       type=int, default=1,
                     help="Seed index (1/2/3)")
_args = _parser.parse_args()

_MODEL_MAP = {
    "itransformer": ("iTransformer", "ITransformerClassifier"),
    "patchtst":     ("PatchTST",     "PatchTSTClassifier"),
    "fedformer":    ("FEDformer",    "FedformerClassifier"),
    "dlinear":      ("DLinear",      "DLinearClassifier"),
}
_mod_name, _cls_name = _MODEL_MAP[_args.model_name]
_mod   = importlib.import_module(_mod_name)
Model  = getattr(_mod, _cls_name)
TFTDataset = getattr(_mod, "TFTDataset")

print(f"[Baseline] Model={_args.model_name}, Port={_args.port}")
from torchmetrics.classification import (
    MultilabelAccuracy, MultilabelF1Score, MultilabelPrecision,
    MultilabelRecall, MultilabelAUROC, MultilabelSpecificity,
)
from sklearn.metrics import f1_score
from bayes_opt import BayesianOptimization
#from multimodal_intermediate import TFTDataset, TFTPredictor


# -------------------------------------------------------------------
# Global constants / helper to load memmap data
# -------------------------------------------------------------------
target_names   = ['dangerous_action_1h', 'dangerous_action_1d', 'dangerous_action_1w']
num_targets    = len(target_names)
GLOBAL_DIR     = "patient_level_split/last_npy_data/GlobalData"
HOSPITAL_DIR   = "patient_level_split/last_npy_data/HospitalsData"
STATIC_SHAPE   = (14,)
SEQUENCE_SHAPE = (192, 25)
TARGETS_SHAPE  = (3,)

def load_memmap_data(file_path, sample_shape, dtype=np.float32):
    if not os.path.exists(file_path):
        return None
    size = os.path.getsize(file_path)
    per  = np.prod(sample_shape) * np.dtype(dtype).itemsize
    n    = size // per
    return np.memmap(file_path, dtype=dtype, mode="r", shape=(n,) + sample_shape)

# -------------------------------------------------------------------
# 1) Load *test* data for final global evaluation
# -------------------------------------------------------------------
static_test  = load_memmap_data(os.path.join(GLOBAL_DIR, "static_data.npy"),   STATIC_SHAPE)
seq_test     = load_memmap_data(os.path.join(GLOBAL_DIR, "sequence_data.npy"), SEQUENCE_SHAPE)
tgt_test     = load_memmap_data(os.path.join(GLOBAL_DIR, "targets.npy"),       TARGETS_SHAPE)
global_test_dataset = TFTDataset(list(zip(static_test, seq_test, tgt_test)))
global_test_loader  = DataLoader(global_test_dataset, batch_size=32)

# -------------------------------------------------------------------
# 2) Load & *concatenate* all hospitals’ validation splits
# -------------------------------------------------------------------
val_static_list, val_seq_list, val_tgt_list = [], [], []
for hospital in os.listdir(HOSPITAL_DIR):
    base = os.path.join(HOSPITAL_DIR, hospital)
    s = load_memmap_data(os.path.join(base, "static_val.npy"), STATIC_SHAPE)
    q = load_memmap_data(os.path.join(base, "sequence_val.npy"), SEQUENCE_SHAPE)
    t = load_memmap_data(os.path.join(base, "targets_val.npy"), TARGETS_SHAPE)
    if s is not None:
        val_static_list.append(s)
        val_seq_list.append(q)
        val_tgt_list.append(t)

# stack them into one big validation set
val_static = np.concatenate(val_static_list, axis=0)
val_seq    = np.concatenate(val_seq_list,    axis=0)
val_tgt    = np.concatenate(val_tgt_list,    axis=0)
global_val_dataset = TFTDataset(list(zip(val_static, val_seq, val_tgt)))
global_val_loader  = DataLoader(global_val_dataset, batch_size=32)


# -------------------------------------------------------------------
# Model & loss  (per-model kwargs to avoid unexpected-argument errors)
# -------------------------------------------------------------------
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
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(**_MODEL_INIT_KWARGS[_args.model_name]).to(device)
loss_fn = torch.nn.BCEWithLogitsLoss()


# -------------------------------------------------------------------
# Save helper
# -------------------------------------------------------------------
def save_global_model(model, filename="global_model_best_baseline.pth"):
    path = os.path.join(GLOBAL_DIR, filename)
    os.makedirs(GLOBAL_DIR, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"✅ Saved global model: {path}")


from bayes_opt import BayesianOptimization
from sklearn.metrics import f1_score
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
def tune_global_thresholds(model, val_loader, method="youden", min_specificity=0.5, min_recall=0.0):
    """
    Tune global thresholds by aggregating per-hospital optimal thresholds.
    For each hospital, finds the best threshold for each target with constraints,
    then averages the thresholds across hospitals.
    """
    device = next(model.parameters()).device
    model.eval()
    num_targets = model.output_dim if hasattr(model, "output_dim") else 3

    # --- Helper: get all hospital val loaders ---
    def get_all_hospital_val_loaders():
        loaders = []
        for hospital in os.listdir(HOSPITAL_DIR):
            base = os.path.join(HOSPITAL_DIR, hospital)
            s = load_memmap_data(os.path.join(base, "static_val.npy"), STATIC_SHAPE)
            q = load_memmap_data(os.path.join(base, "sequence_val.npy"), SEQUENCE_SHAPE)
            t = load_memmap_data(os.path.join(base, "targets_val.npy"), TARGETS_SHAPE)
            if (s is not None) and (len(s) > 0):
                ds = TFTDataset(list(zip(s, q, t)))
                loaders.append(DataLoader(ds, batch_size=32, shuffle=False))
        return loaders

    hospital_thresholds = []
    for loader in get_all_hospital_val_loaders():
        # Collect logits and trues for this hospital
        all_logits = {i: [] for i in range(num_targets)}
        all_true   = {i: [] for i in range(num_targets)}
        with torch.no_grad():
            for sx, seqx, y in loader:
                sx, seqx, y = sx.to(device), seqx.to(device), y.to(device)
                logits = model(sx, seqx).cpu().numpy()
                y_np   = y.cpu().numpy()
                for i in range(num_targets):
                    all_logits[i].extend(logits[:, i])
                    all_true[i].extend(y_np[:, i])
        hosp_best = []
        for i in range(num_targets):
            y_true = np.array(all_true[i])
            y_prob = 1.0 / (1.0 + np.exp(-np.array(all_logits[i])))

            

            def objective(thresh):
                thresh = np.clip(thresh, 0.1, 0.9)
                preds = (y_prob >= thresh).astype(int)
                tp = np.logical_and(preds == 1, y_true == 1).sum()
                tn = np.logical_and(preds == 0, y_true == 0).sum()
                fp = np.logical_and(preds == 1, y_true == 0).sum()
                fn = np.logical_and(preds == 0, y_true == 1).sum()
                sens = tp / (tp + fn + 1e-8)
                spec = tn / (tn + fp + 1e-8)
                f1   = f1_score(y_true, preds, zero_division=0)
                prec = precision_score(y_true, preds, zero_division=0)
                rec  = recall_score(y_true, preds, zero_division=0)
                # Enforce min_specificity for all methods
                if spec < min_specificity:
                    return -1
                if method == "min_spec_f1":
                    # F1 minus penalty for imbalance
                    balance_penalty = abs(prec - rec)
                    return f1 - 0.5 * balance_penalty
                elif method == "youden":
                    return sens + spec - 1
                elif method == "harmonic":
                    return 2 * (sens * spec) / (sens + spec + 1e-8)
                else:
                    raise ValueError(f"Unknown method {method}")


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
                print(f"[WARN] Bayesian optimization failed for hospital target {i}, fallback to 0.5 ({e})")
                best_t = 0.5
            hosp_best.append(float(best_t))
        hospital_thresholds.append(hosp_best)

    # Aggregate across hospitals (mean per target)
    hospital_thresholds = np.array(hospital_thresholds)
    global_best = hospital_thresholds.mean(axis=0)
    return [float(t) for t in global_best]



# -------------------------------------------------------------------
# 4) Server‐side eval on *test* set, given thresholds
# -------------------------------------------------------------------
def global_evaluate(parameters, thresholds):
    # unpack parameters
    ndarrays = parameters if isinstance(parameters, list) \
               else parameters_to_ndarrays(parameters)
    # load into model
    state = {k: torch.from_numpy(v) for k,v in zip(model.state_dict().keys(), ndarrays)}
    model.load_state_dict(state)
    model.eval()

    # metrics trackers
    acc_m  = MultilabelAccuracy(num_labels=num_targets, average=None).to(device)
    f1_m   = MultilabelF1Score(   num_labels=num_targets, average=None).to(device)
    prec_m = MultilabelPrecision( num_labels=num_targets, average=None).to(device)
    rec_m  = MultilabelRecall(    num_labels=num_targets, average=None).to(device)
    auc_m  = MultilabelAUROC(     num_labels=num_targets, average=None).to(device)
    spec_m = MultilabelSpecificity(num_labels=num_targets, average=None).to(device)

    all_preds, all_probs, all_targets = [], [], []
    total_loss = 0.0
    thresh_tensor = torch.tensor(thresholds, device=device)

    with torch.no_grad():
        for sx, seqx, y in global_test_loader:
            sx, seqx, y = sx.to(device), seqx.to(device), y.to(device)
            logits = model(sx, seqx)
            total_loss += loss_fn(logits, y).item() * sx.size(0)

            probs = torch.sigmoid(logits)
            preds = (probs >= thresh_tensor).long()

            all_preds.append(preds)
            all_probs.append(probs)
            all_targets.append(y.long())

    all_preds   = torch.cat(all_preds, dim=0)
    all_probs   = torch.cat(all_probs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    overall_acc    = (all_preds == all_targets).float().mean()
    acc_per_target = acc_m(all_preds,   all_targets)
    f1_per_target  = f1_m(all_preds,   all_targets)
    prec_per_target= prec_m(all_preds,   all_targets)
    rec_per_target = rec_m(all_preds,   all_targets)
    auc_per_target = auc_m(all_probs,  all_targets)
    spec_per_target= spec_m(all_preds, all_targets)
    avg_loss       = total_loss / len(global_test_loader.dataset)

    print(f"🌍 Global Eval → Loss {avg_loss:.4f}, Overall Acc {overall_acc:.4f}")
    for i, name in enumerate(target_names):
        print(
            f"  • {name}: "
            f"Acc={acc_per_target[i]:.4f}, F1={f1_per_target[i]:.4f}, "
            f"Prec={prec_per_target[i]:.4f}, Rec={rec_per_target[i]:.4f}, "
            f"AUC={auc_per_target[i]:.4f}, Spec={spec_per_target[i]:.4f}"
        )

    # pack metrics for Flower
    metrics = {"overall_accuracy": overall_acc.item()}
    for i, name in enumerate(target_names):
        metrics.update({
            f"accuracy_{name}":    acc_per_target[i].item(),
            f"f1_{name}":          f1_per_target[i].item(),
            f"precision_{name}":   prec_per_target[i].item(),
            f"recall_{name}":      rec_per_target[i].item(),
            f"auroc_{name}":       auc_per_target[i].item(),
            f"specificity_{name}": spec_per_target[i].item(),
        })
    return avg_loss, metrics


# -------------------------------------------------------------------
# 5) Custom FedProx with in‐round global tuning & eval
# -------------------------------------------------------------------
class FedProxWithGlobalEval(fl.server.strategy.FedProx):
    def __init__(self, *, proximal_mu: float, patience: int, **kwargs):
        super().__init__(proximal_mu=proximal_mu, **kwargs)
        self.best_loss    = float("inf")
        self.no_improve   = 0
        self.patience     = patience
        self.loss_history = []

    def aggregate_fit(self, rnd, results, failures):
        # standard aggregation
        agg_res = super().aggregate_fit(rnd, results, failures)
        if agg_res is None:
            return None

        # unpack parameters
        agg_params = agg_res[0] if isinstance(agg_res, tuple) else agg_res

        # load into model for tuning
        nd = agg_params if isinstance(agg_params, list) \
             else parameters_to_ndarrays(agg_params)
        state = {k: torch.from_numpy(v) for k,v in zip(model.state_dict().keys(), nd)}
        model.load_state_dict(state)

        # 1) tune thresholds on pooled validation
        tuned_thresholds = tune_global_thresholds(model, global_val_loader)
        print(f"--- Round {rnd} tuned global thresholds: {tuned_thresholds}")

        # 2) evaluate on test set
        loss, metrics = global_evaluate(agg_params, thresholds=tuned_thresholds)

        # 3) checkpoint & early stop
        self.loss_history.append(loss)
        if loss < self.best_loss:
            self.best_loss  = loss
            self.no_improve = 0
            save_global_model(model)
        else:
            self.no_improve += 1
            if self.no_improve >= self.patience:
                print(f"!!! Early stopping at round {rnd}")
                self.should_stop = True

        return agg_res


# -------------------------------------------------------------------
# 6) Launch Flower server
# -------------------------------------------------------------------
strategy = FedProxWithGlobalEval(
    proximal_mu       = 1e-5,
    patience          = 15,
    fraction_fit      = 1.0,
    min_fit_clients   = 3,
    fraction_evaluate = 0.0,  # we do server-side test eval only
    min_available_clients = 3,
)

start = time.time()
fl.server.start_server(
    server_address=f"0.0.0.0:{_args.port}",
    config=fl.server.ServerConfig(num_rounds=50),
    strategy=strategy,
)
end = time.time()

print(f"\nFinished {len(strategy.loss_history)} rounds in {end - start:.1f}s")
print("Loss history:", strategy.loss_history)
