# fedmap/strategy.py
import os
import logging
from typing import List, Dict, Tuple, Optional, Callable, Any

import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
import optuna

# ---------- logging ----------
LOGGER = logging.getLogger(__name__)
# Tip: set level and format in your server entrypoint, e.g.:
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
# )

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


# --------- small helpers to read env knobs (optional) --------- #
def _get_int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except Exception:
        return default

def _get_bool_env(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "y")


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
            LOGGER.info("Aggregator: loaded saved weights from '%s'.", self.aggregator_path)
        else:
            LOGGER.info("Aggregator: no saved weights found at '%s' (starting fresh).", self.aggregator_path)

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

        # --- student gating knobs (can be overridden by env) ---
        self.use_student = _get_bool_env("FEDMAP_USE_STUDENT", True)
        self.student_warmup = _get_int_env("FEDMAP_STUDENT_WARMUP", 5)      # rounds before student can dominate
        self.student_win_needed = _get_int_env("FEDMAP_STUDENT_WIN_NEEDED", 3)  # consecutive wins needed
        self._student_streak = 0  # internal counter

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
        LOGGER.info("Aggregation: round %d — received %d client update(s).", server_round, len(results))
        if not results:
            # No results → keep previous params (Flower will handle), return empty metrics
            LOGGER.warning("Aggregation: round %d — no client results, keeping previous global parameters.", server_round)
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
            ck = float(fit_res.metrics.get("c_k", 0.0) or 0.0)                 # client consistency/quality
            pk = float(fit_res.metrics.get("p_k", 0.0) or 0.0)                 # client performance (e.g., F1/Acc)
            m_img = int(fit_res.metrics.get("m_img", 1) or 1)                  # has image modality?
            m_txt = int(fit_res.metrics.get("m_txt", 1) or 1)                  # has text modality?
            n_k = int(getattr(fit_res, "num_examples", 0) or fit_res.metrics.get("n_k", 0) or 0)
            diversity = float(fit_res.metrics.get("diversity", 0.0) or 0.0)    # data diversity proxy
            modality_div = float(1.0 if (m_img == 1 and m_txt == 1) else 0.5)  # multi-modality bonus

            val_loss = float(fit_res.metrics.get("val_loss", 0.0) or 0.0)      # client val loss
            grad_norm = float(fit_res.metrics.get("grad_norm", 0.0) or 0.0)    # gradient norm proxy
            imbalance_ratio = float(fit_res.metrics.get("imbalance_ratio", 0.0) or 0.0)  # class balance (≈0.5 is good)

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
        vl_inv    = -_z_clip_1d(val_losses, self.z_clip)            # lower loss → higher score
        gn_inv    = -_z_clip_1d(grad_norms, self.z_clip)            # smaller grad norm → higher score (stability)
        bal_feat  = _z_clip_1d(1.0 - 2.0 * torch.abs(imb_ratios - 0.5), self.z_clip)  # closer to 0.5 is better

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
        LOGGER.info(
            "Student attention (before hyperparameter search): per-client weights = %s",
            attn_student.detach().cpu().numpy(),
        )

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
            LOGGER.info(
                "Student attention (after distillation): %s",
                new_attn.detach().cpu().numpy(),
            )
            LOGGER.info(
                "Teacher attention (best found this round): %s",
                _safe_nan_to_num(attn_teacher).detach().cpu().numpy(),
            )
            LOGGER.info("Best server metric this round (higher is better): %.4f", best["score"])

        # --- evaluate student attention directly (single eval) ---
        def _metric_score_from_nds(nds):
            loss, metrics = self.evaluate_fn(server_round, nds, {})
            return float(metrics.get("test_f1_macro", metrics.get("accuracy", 0.0)) or 0.0)

        score_T = float(best["score"])
        nds_S = aggregate_with_attention(new_attn)
        score_S = _metric_score_from_nds(nds_S) if nds_S is not None else -1.0
        LOGGER.info(
            "Server evaluation — Teacher vs Student: teacher=%.6f, student=%.6f",
            score_T, score_S
        )

        # --------- Decide student vs teacher (KL-bound + probe + blend + streak gate) ---------
        # Target max drop on server metric (e.g., macro-F1) compared to teacher
        delta = 0.002  # tighten/loosen as desired

        # Compute KL(teacher || student)
        def _kl_t_s(p, q):
            p = _safe_nan_to_num(p.to(_DEVICE))
            q = _safe_nan_to_num(q.to(_DEVICE))
            p = torch.clamp(p, 1e-12, 1.0)
            q = torch.clamp(q, 1e-12, 1.0)
            return (p * (p.log() - q.log())).sum()

        kl_ts = _kl_t_s(_safe_nan_to_num(attn_teacher), _safe_nan_to_num(new_attn)).item()

        # Small Lipschitz probe toward student to estimate local L
        alpha = 0.1
        with torch.no_grad():
            w_alpha = _safe_nan_to_num((1 - alpha) * attn_teacher + alpha * new_attn)
            nds_alpha = aggregate_with_attention(w_alpha)
        score_alpha = _metric_score_from_nds(nds_alpha) if nds_alpha is not None else score_T

        l1_step = torch.sum(torch.abs(w_alpha - attn_teacher)).item()
        Lhat = abs(score_alpha - score_T) / max(l1_step, 1e-8)

        import math
        tau = 0.5 * (delta / max(Lhat, 1e-8)) ** 2  # Pinsker + Lipschitz bound

        denom = math.sqrt(max(2.0 * kl_ts, 1e-16))
        eta = 0.2  # larger → switch faster to student
        beta = min(1.0, eta / denom) if denom > 0 else 1.0

        # --- streak gate: student must be consistently good ---
        student_ok = (score_S >= score_T - delta) and (kl_ts <= tau)

        if student_ok:
            self._student_streak += 1
        else:
            self._student_streak = 0

        LOGGER.info(
            "Student acceptance gate: %s (streak %d/%d, warmup=%d rounds)",
            "PASSED" if student_ok else "FAILED",
            self._student_streak, self.student_win_needed, self.student_warmup
        )

        if not self.use_student:
            # hard-disable student: stick to teacher
            w_final = attn_teacher
            decision = "teacher-only (student disabled via config)"
        elif server_round < self.student_warmup:
            # warmup: allow only cautious blend toward student
            beta_guarded = min(beta, 0.25)
            w_final = _safe_nan_to_num((1 - beta_guarded) * attn_teacher + beta_guarded * new_attn)
            decision = "teacher-dominant (warmup period)"
        elif self._student_streak >= self.student_win_needed:
            # student has repeatedly matched/beaten teacher with small KL → let it dominate
            beta_student = max(0.6, min(1.0, beta))  # strong lean to student
            w_final = _safe_nan_to_num((1 - (1 - beta_student)) * new_attn + (1 - beta_student) * attn_teacher)
            decision = "student-dominant (gate passed)"
        else:
            # default: trust-region blend using KL/Lipschitz logic
            if kl_ts <= tau:
                w_final = _safe_nan_to_num((1 - beta) * attn_teacher + beta * new_attn)
                decision = "blend (KL small: cautious move toward student)"
            else:
                beta_guarded = min(beta, 0.25)
                w_final = _safe_nan_to_num((1 - beta_guarded) * attn_teacher + beta_guarded * new_attn)
                decision = "teacher-dominant (KL large: trust-region guarded)"

        LOGGER.info(
            "Trust-region diagnostics: KL(T||S)=%.6f, L̂=%.6f, τ=%.6f, β=%.3f → decision: %s",
            kl_ts, Lhat, tau, beta, decision
        )

        nds_blend = aggregate_with_attention(w_final)

        # Prefer blend → then teacher → then student/uniform fallback
        if nds_blend is not None:
            nds_final = nds_blend
        elif best["nds"] is not None:
            nds_final = best["nds"]
        else:
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
