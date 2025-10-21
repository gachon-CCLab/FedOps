# fedmap/strategy.py
import os
from typing import List, Dict, Tuple, Optional, Callable, Any

import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
import optuna

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AggregatorMLP(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=16):
        """
        Expects 10 raw meta-features per client (from the client metrics):
        [ck, pk, m_img, m_txt, n_k, diversity, modality_div, val_loss, grad_norm, imbalance_ratio]
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


# -------------------- Utilities (NaN/Inf-safe) -------------------- #

def _safe_nan_to_num(t: torch.Tensor) -> torch.Tensor:
    """Replace NaN/+Inf/-Inf with zeros (fallback if torch.nan_to_num not available)."""
    try:
        return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    except AttributeError:
        t = t.clone()
        t[~torch.isfinite(t)] = 0
        return t


def _safe_softmax(logits: torch.Tensor) -> torch.Tensor:
    logits = _safe_nan_to_num(logits)
    if logits.numel() == 0:
        return logits
    if not torch.isfinite(logits).all():
        logits = torch.zeros_like(logits)
    probs = torch.softmax(logits, dim=0)
    probs = _safe_nan_to_num(probs)
    if not torch.isfinite(probs).all() or probs.sum().item() == 0.0:
        probs = torch.ones_like(probs) / max(probs.numel(), 1)
    return probs


def _nanmean_std_cols(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    NaN-safe per-column mean/std for 2D tensor [N, K].
    Returns: mean [K], std [K] (std >= 1 to avoid divide-by-zero).
    """
    assert x.dim() == 2, "Expected 2D tensor [N,K]"
    mask = torch.isfinite(x)
    count = mask.sum(dim=0).clamp(min=1)                 # [K]
    x0 = torch.where(mask, x, torch.zeros_like(x))
    mean = x0.sum(dim=0) / count                         # [K]

    # Variance with masked entries
    sq = (x - mean) ** 2
    sq = torch.where(mask, sq, torch.zeros_like(sq))
    var = sq.sum(dim=0) / count                          # [K]
    std = var.sqrt()
    std = torch.where(std > 0, std, torch.ones_like(std))  # enforce >= 1
    return mean, std


def _nanmean_std_1d(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    NaN-safe mean/std for 1D tensor.
    Returns: scalar mean (), scalar std () with std >= 1e-8.
    """
    assert x.dim() == 1, "Expected 1D tensor"
    mask = torch.isfinite(x)
    count = mask.sum().clamp(min=1)
    x0 = torch.where(mask, x, torch.zeros_like(x))
    mean = x0.sum() / count

    sq = torch.where(mask, (x - mean) ** 2, torch.zeros_like(x))
    var = sq.sum() / count
    std = var.sqrt()
    if not torch.isfinite(std) or std.item() <= 0:
        std = torch.tensor(1.0, device=x.device, dtype=x.dtype)
    return mean, std


def _z_clip_1d(x: torch.Tensor, clip: float) -> torch.Tensor:
    """z-score a 1D tensor (NaN-safe), then clip to [-clip, clip]."""
    m, s = _nanmean_std_1d(x)
    z = (x - m) / (s + 1e-8)
    z = torch.clamp(z, -clip, clip)
    return _safe_nan_to_num(z)


class ModalityAwareAggregation(fl.server.strategy.Strategy):
    """
    Custom aggregator (teacher-student + Optuna) with safety guards.
    """

    def __init__(
        self,
        *,
        # injected by server
        initial_parameters: Optional[fl.common.Parameters] = None,
        evaluate_fn: Optional[
            Callable[[int, fl.common.NDArrays, Dict[str, fl.common.Scalar]], Tuple[float, Dict[str, fl.common.Scalar]]]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, fl.common.Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, fl.common.Scalar]]] = None,

        # ours
        aggregator_path: str = "aggregator_mlp.pth",
        input_dim: int = 10,
        hidden_dim: int = 16,
        aggregator_lr: float = 1e-3,
        entropy_coeff: float = 0.01,
        n_trials_per_round: int = 6,
        perf_mix_lambda: float = 0.7,
        z_clip: float = 3.0,

        **_: Any,
    ):
        self.initial_parameters = initial_parameters
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn

        self.aggregator_path = aggregator_path
        self.entropy_coeff = float(entropy_coeff)
        self.n_trials_per_round = int(n_trials_per_round)
        self.perf_mix_lambda = float(perf_mix_lambda)
        self.z_clip = float(z_clip)

        self.aggregator = AggregatorMLP(input_dim=input_dim, hidden_dim=hidden_dim).to(_DEVICE)
        if os.path.exists(self.aggregator_path):
            self.aggregator.load_state_dict(torch.load(self.aggregator_path, map_location=_DEVICE))
            print(f"✅ Loaded saved aggregator weights from {self.aggregator_path}")
        else:
            print("⚠️ No saved aggregator found, starting fresh.")

        self.optimizer = optim.Adam(self.aggregator.parameters(), lr=float(aggregator_lr))
        self.loss_fn = nn.KLDivLoss(reduction="batchmean")
        self.study = None

        # last-best alphas (logging continuity / fallback)
        self.alpha_perf = 0.45
        self.alpha_size = 0.10
        self.alpha_div  = 0.15
        self.alpha_mod  = 0.10
        self.alpha_vl   = 0.10
        self.alpha_gn   = 0.05
        self.alpha_bal  = 0.05

    # ---------------- Flower Strategy API ----------------

    def initialize_parameters(self, client_manager) -> Optional[fl.common.Parameters]:
        return self.initial_parameters

    def configure_fit(self, server_round, parameters, client_manager):
        cfg = self.on_fit_config_fn(server_round) if self.on_fit_config_fn is not None else {}
        return [(client, fl.common.FitIns(parameters, cfg)) for client in client_manager.all().values()]

    def configure_evaluate(self, server_round, parameters, client_manager):
        cfg = self.on_evaluate_config_fn(server_round) if self.on_evaluate_config_fn is not None else {}
        return [(client, fl.common.EvaluateIns(parameters, cfg)) for client in client_manager.all().values()]

    def aggregate_fit(self, server_round, results, failures):
        print(f"🔗 Aggregating round {server_round} with {len(results)} client updates...")
        if not results:
            # No results → keep previous params (Flower will handle), return empty metrics
            return None, {}

        # -------- collect client params + meta --------
        weights_per_client: List[List[torch.Tensor]] = []
        meta_features: List[List[float]] = []

        pk_values, ck_values = [], []
        diversity_scores, modality_scores = [], []
        nks = []
        val_losses, grad_norms, imb_ratios = [], [], []

        for _, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            weights_per_client.append([torch.tensor(arr, dtype=torch.float32, device="cpu") for arr in ndarrays])

            # Extract metrics with defaults
            ck = float(fit_res.metrics.get("c_k", 0.0) or 0.0)
            pk = float(fit_res.metrics.get("p_k", 0.0) or 0.0)
            m_img = int(fit_res.metrics.get("m_img", 1) or 1)
            m_txt = int(fit_res.metrics.get("m_txt", 1) or 1)
            n_k = int(getattr(fit_res, "num_examples", 0) or fit_res.metrics.get("n_k", 0) or 0)
            diversity = float(fit_res.metrics.get("diversity", 0.0) or 0.0)
            modality_div = float(1.0 if (m_img == 1 and m_txt == 1) else 0.5)

            val_loss = float(fit_res.metrics.get("val_loss", 0.0) or 0.0)
            grad_norm = float(fit_res.metrics.get("grad_norm", 0.0) or 0.0)
            imbalance_ratio = float(fit_res.metrics.get("imbalance_ratio", 0.0) or 0.0)

            meta_features.append([ck, pk, m_img, m_txt, n_k, diversity, modality_div, val_loss, grad_norm, imbalance_ratio])

            pk_values.append(pk)
            ck_values.append(max(ck, 0.0))
            diversity_scores.append(diversity)
            modality_scores.append(modality_div)
            nks.append(float(n_k))
            val_losses.append(val_loss)
            grad_norms.append(grad_norm)
            imb_ratios.append(imbalance_ratio)

        # tensors (device)
        def T(x): return torch.tensor(x, dtype=torch.float32, device=_DEVICE)

        meta = T(meta_features)
        meta = _safe_nan_to_num(meta)

        pk_values     = _safe_nan_to_num(T(pk_values))
        ck_values     = _safe_nan_to_num(T(ck_values))
        diversity_s   = _safe_nan_to_num(T(diversity_scores))
        modality_s    = _safe_nan_to_num(T(modality_scores))
        nks           = _safe_nan_to_num(T(nks))
        val_losses    = _safe_nan_to_num(T(val_losses))
        grad_norms    = _safe_nan_to_num(T(grad_norms))
        imb_ratios    = _safe_nan_to_num(T(imb_ratios))

        # student input (z-score per column) -- NaN-safe, no torch.nan*
        meta_mean, meta_std = _nanmean_std_cols(meta)      # [K], [K]
        meta_norm = (meta - meta_mean) / (meta_std + 1e-8)
        meta_norm = torch.clamp(_safe_nan_to_num(meta_norm), -self.z_clip, self.z_clip)

        # -------- teacher features --------
        perf_mix = self.perf_mix_lambda * pk_values + (1.0 - self.perf_mix_lambda) * ck_values
        perf_feat = _z_clip_1d(perf_mix, self.z_clip)
        size_feat = _z_clip_1d(torch.log(torch.clamp(nks, min=1.0)), self.z_clip)
        div_feat  = _z_clip_1d(diversity_s, self.z_clip)
        mod_feat  = _z_clip_1d(modality_s, self.z_clip)
        vl_inv    = -_z_clip_1d(val_losses, self.z_clip)
        gn_inv    = -_z_clip_1d(grad_norms, self.z_clip)
        bal_feat  = _z_clip_1d(1.0 - 2.0 * torch.abs(imb_ratios - 0.5), self.z_clip)

        teacher_feats = torch.stack([perf_feat, size_feat, div_feat, mod_feat, vl_inv, gn_inv, bal_feat], dim=1)
        teacher_feats = _safe_nan_to_num(teacher_feats)

        # helper: attention-based aggregation → returns NDArrays (list of numpy arrays)
        def aggregate_with_attention(attn_vec: torch.Tensor):
            attn_vec = _safe_nan_to_num(attn_vec)
            if attn_vec.numel() == 0:
                return None
            if not torch.isfinite(attn_vec).all() or attn_vec.sum().item() == 0.0:
                attn_vec = torch.ones_like(attn_vec) / attn_vec.numel()
            attn_cpu = attn_vec.detach().cpu().numpy()
            new_params = []
            for i in range(len(weights_per_client[0])):
                agg = torch.zeros_like(weights_per_client[0][i], dtype=torch.float32, device="cpu")
                for j in range(len(weights_per_client)):
                    w = float(attn_cpu[j])
                    if w != w:  # NaN check
                        w = 1.0 / len(weights_per_client)
                    agg += w * weights_per_client[j][i]
                new_params.append(agg.numpy())
            return new_params  # NDArrays

        # student attention (pre-Optuna, for logging)
        with torch.no_grad():
            logits = self.aggregator(meta_norm).view(-1)
            logits = _safe_nan_to_num(logits)
            attn_student = _safe_softmax(logits)
        print("👩‍🎓 Student pre-Optuna:", attn_student.detach().cpu().numpy())

        assert self.evaluate_fn is not None, "evaluate_fn is required (server supplies it)."

        # -------- Optuna over alphas --------
        best = {"score": -1.0, "attn_teacher": attn_student, "nds": None, "alphas": None}

        def objective(trial):
            a = torch.tensor([
                trial.suggest_float("alpha_perf", 0.0, 1.0),
                trial.suggest_float("alpha_size", 0.0, 1.0),
                trial.suggest_float("alpha_div",  0.0, 1.0),
                trial.suggest_float("alpha_mod",  0.0, 1.0),
                trial.suggest_float("alpha_vl",   0.0, 1.0),
                trial.suggest_float("alpha_gn",   0.0, 1.0),
                trial.suggest_float("alpha_bal",  0.0, 1.0),
            ], dtype=torch.float32, device=_DEVICE)

            if not torch.isfinite(a).all() or a.sum().item() <= 0:
                raise optuna.TrialPruned()
            a = a / a.sum()

            logits_teacher = _safe_nan_to_num(teacher_feats @ a)
            attn_teacher = _safe_softmax(logits_teacher)

            # aggregate (NDArrays) and evaluate using server-side eval_fn
            nds = aggregate_with_attention(attn_teacher)
            if nds is None:
                raise optuna.TrialPruned()

            loss, metrics = self.evaluate_fn(server_round, nds, {})
            # Choose available metric
            score = float(metrics.get("test_f1_macro", metrics.get("accuracy", 0.0)) or 0.0)

            nonlocal best
            if score > best["score"]:
                best = {"score": score, "attn_teacher": attn_teacher.detach(), "nds": nds, "alphas": a.detach()}
            return score

        self.study = optuna.create_study(direction="maximize")
        self.study.optimize(objective, n_trials=self.n_trials_per_round)

        # -------- distill student to teacher --------
        if best["alphas"] is not None:
            a = best["alphas"].detach().cpu().numpy().tolist()
            (self.alpha_perf, self.alpha_size, self.alpha_div, self.alpha_mod,
             self.alpha_vl, self.alpha_gn, self.alpha_bal) = a

        attn_teacher = best["attn_teacher"]
        logits_student = _safe_nan_to_num(self.aggregator(meta_norm).view(-1))
        log_probs_student = torch.log_softmax(logits_student, dim=0)
        log_probs_student = _safe_nan_to_num(log_probs_student)
        probs_teacher = _safe_nan_to_num(attn_teacher.to(_DEVICE))

        kl = self.loss_fn(log_probs_student, probs_teacher)
        entropy = -(torch.softmax(logits_student, dim=0) * log_probs_student).sum()
        loss_distill = kl - self.entropy_coeff * entropy

        self.optimizer.zero_grad()
        loss_distill.backward()
        self.optimizer.step()
        torch.save(self.aggregator.state_dict(), self.aggregator_path)

        with torch.no_grad():
            new_attn = _safe_softmax(self.aggregator(meta_norm).view(-1))
            print("👩‍🎓 Student post-Optuna:", new_attn.detach().cpu().numpy())
            print("🧑‍🏫 Teacher (best):   ", _safe_nan_to_num(attn_teacher).detach().cpu().numpy())
            print(f"🌟 Best score this round: {best['score']:.4f}")

        # Convert best NDArrays -> Parameters
        nds_final = best["nds"]
        if nds_final is None:
            # Fallback to uniform averaging (or student attention if valid)
            n = len(weights_per_client)
            fallback_attn = new_attn if new_attn.numel() == n else torch.ones(n, device=_DEVICE) / n
            nds_final = aggregate_with_attention(fallback_attn)

        final_params = ndarrays_to_parameters(nds_final) if nds_final is not None else None
        return final_params, {}

    def aggregate_evaluate(self, server_round, results, failures):
        # results: List[Tuple[ClientProxy, EvaluateRes]]
        if not results:
            return 0.0, {}

        total_examples = 0.0
        weighted_loss = 0.0
        acc_num = 0.0

        for _, ev in results:
            n = float(getattr(ev, "num_examples", 0) or 0)
            loss_val = float(getattr(ev, "loss", 0.0) or 0.0)
            if not (loss_val == loss_val):  # NaN check
                loss_val = 0.0
            total_examples += (n if n > 0 else 1.0)
            weighted_loss += loss_val * (n if n > 0 else 1.0)
            if getattr(ev, "metrics", None) and "accuracy" in ev.metrics:
                acc = float(ev.metrics["accuracy"] or 0.0)
                acc_num += acc * (n if n > 0 else 1.0)

        weighted_loss /= max(total_examples, 1.0)
        metrics = {}
        if total_examples > 0 and acc_num > 0:
            metrics["accuracy"] = acc_num / total_examples
        return float(weighted_loss), metrics

    def evaluate(self, server_round: int, parameters: fl.common.Parameters):
         
        
        if self.evaluate_fn is None:
            return None  # Flower will skip
        
        nds = parameters_to_ndarrays(parameters)
        loss, metrics = self.evaluate_fn(server_round, nds, {})
        safe_metrics = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in metrics.items()}
        # Guard loss in case eval returned NaN/Inf
        loss = float(loss)
        if not (loss == loss):  # NaN
            loss = 0.0
        return float(loss), safe_metrics
