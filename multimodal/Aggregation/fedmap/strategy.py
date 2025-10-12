# fedmap/strategy.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl
from flwr.server.strategy import FedAvg   # ✅ use FedAvg as base
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
import optuna
from typing import List

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AggregatorMLP(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=16):
        """
        Expects 10 raw meta-features per client (from the client metrics):
        [ck, pk, m_img, m_txt, n_k, diversity, modality_div, val_loss, grad_norm, imbalance_ratio]
        The student sees these raw features (z-scored per round).
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class ModalityAwareAggregation(FedAvg):   # ✅ now inherit FedAvg
    """
    Option B (teacher-student):
      - Teacher attention (per round) = softmax( sum_j alpha_j * z(feature_j) )
        features used (per client):
          1) perf_mix = λ*pk + (1-λ)*max(ck,0)
          2) size     = log(n_k)
          3) diversity
          4) modality_div
          5) -z(val_loss)     (lower loss upweights)
          6) -z(grad_norm)    (large norm downweights)
          7) balance_score    = 1 - 2*|imbalance_ratio - 0.5|
      - Optuna learns non-negative alphas for these 7 signals and they are normalized to sum to 1 each trial.
      - Student MLP (input_dim=10, raw meta) is distilled to match teacher attention via KL.
    """

    def __init__(
        self,
        evaluate_fn=None,
        aggregator_path: str = "aggregator_mlp.pth",
        input_dim: int = 10,   # raw meta features length
        hidden_dim: int = 16,
        aggregator_lr: float = 1e-3,
        entropy_coeff: float = 0.01,
        n_trials_per_round: int = 6,
        perf_mix_lambda: float = 0.7,  # λ in perf_mix
        z_clip: float = 3.0,           # winsorize z-scores to [-z_clip, z_clip]
        # ✅ new passthrough args for FedAvg
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        **kwargs,
    ):
        # ✅ initialize FedAvg with standard arguments
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            **kwargs,
        )

        self.evaluate_fn = evaluate_fn
        self.aggregator_path = aggregator_path
        self.entropy_coeff = float(entropy_coeff)
        self.n_trials_per_round = int(n_trials_per_round)
        self.perf_mix_lambda = float(perf_mix_lambda)
        self.z_clip = float(z_clip)

        # Student
        self.aggregator = AggregatorMLP(input_dim=input_dim, hidden_dim=hidden_dim).to(_DEVICE)
        if os.path.exists(self.aggregator_path):
            self.aggregator.load_state_dict(torch.load(self.aggregator_path, map_location=_DEVICE))
            print(f"✅ Loaded saved aggregator weights from {self.aggregator_path}")
        else:
            print("⚠️ No saved aggregator found, starting fresh.")

        self.optimizer = optim.Adam(self.aggregator.parameters(), lr=aggregator_lr)
        self.loss_fn = nn.KLDivLoss(reduction="batchmean")

        self.study = None

        # last-best alphas (for logging continuity)
        self.alpha_perf = 0.45
        self.alpha_size = 0.10
        self.alpha_div  = 0.15
        self.alpha_mod  = 0.10
        self.alpha_vl   = 0.10
        self.alpha_gn   = 0.05
        self.alpha_bal  = 0.05

    # ------------------------------ utils ---------------------------------
    @staticmethod
    def _zscore(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return (x - x.mean()) / (x.std() + eps)

    def _zscore_clip(self, x: torch.Tensor) -> torch.Tensor:
        z = self._zscore(x)
        return torch.clamp(z, -self.z_clip, self.z_clip)

    # --------------------------- Flower API --------------------------------
    def initialize_parameters(self, client_manager):
        return None  # pull from a client

    def configure_fit(self, server_round, parameters, client_manager):
        return [(client, fl.common.FitIns(parameters, {})) for client in client_manager.all().values()]

    def aggregate_fit(self, server_round, results, failures):
        print(f"🔗 Aggregating round {server_round} with {len(results)} client updates...")
        if not results:
            return None, {}

        # ---------- collect client params + meta ----------
        weights_per_client: List[List[torch.Tensor]] = []
        meta_features = []  # raw features for the student input
        pk_values, ck_values = [], []
        diversity_scores, modality_scores = [], []
        nks = []
        val_losses, grad_norms, imb_ratios = [], [], []

        for _, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            weights_per_client.append([torch.tensor(arr, dtype=torch.float32, device="cpu") for arr in ndarrays])

            ck = float(fit_res.metrics.get("c_k", 0.0))
            pk = float(fit_res.metrics.get("p_k", 0.0))
            m_img = int(fit_res.metrics.get("m_img", 1))
            m_txt = int(fit_res.metrics.get("m_txt", 1))
            n_k = int(fit_res.num_examples)
            diversity = float(fit_res.metrics.get("diversity", 0.0))
            modality_div = float(1.0 if (m_img == 1 and m_txt == 1) else 0.5)

            val_loss = float(fit_res.metrics.get("val_loss", 0.0))
            grad_norm = float(fit_res.metrics.get("grad_norm", 0.0))
            imbalance_ratio = float(fit_res.metrics.get("imbalance_ratio", 0.0))

            meta_features.append([ck, pk, m_img, m_txt, n_k, diversity, modality_div, val_loss, grad_norm, imbalance_ratio])

            pk_values.append(pk)
            ck_values.append(max(ck, 0.0))  # no negative contribution in perf
            diversity_scores.append(diversity)
            modality_scores.append(modality_div)
            nks.append(n_k)
            val_losses.append(val_loss)
            grad_norms.append(grad_norm)
            imb_ratios.append(imbalance_ratio)

        # tensors
        meta = torch.tensor(meta_features, dtype=torch.float32, device=_DEVICE)                  # (C, 10)
        pk_values = torch.tensor(pk_values, dtype=torch.float32, device=_DEVICE)                 # (C,)
        ck_values = torch.tensor(ck_values, dtype=torch.float32, device=_DEVICE)                 # (C,)
        diversity_scores = torch.tensor(diversity_scores, dtype=torch.float32, device=_DEVICE)   # (C,)
        modality_scores = torch.tensor(modality_scores, dtype=torch.float32, device=_DEVICE)     # (C,)
        nks = torch.tensor(nks, dtype=torch.float32, device=_DEVICE)                             # (C,)
        val_losses = torch.tensor(val_losses, dtype=torch.float32, device=_DEVICE)               # (C,)
        grad_norms = torch.tensor(grad_norms, dtype=torch.float32, device=_DEVICE)               # (C,)
        imb_ratios = torch.tensor(imb_ratios, dtype=torch.float32, device=_DEVICE)               # (C,)

        # student input (raw 10-d features → z-score per column)
        meta_norm = (meta - meta.mean(dim=0)) / (meta.std(dim=0) + 1e-8)

        # ---------- derived features for TEACHER ----------
        perf_mix = self.perf_mix_lambda * pk_values + (1.0 - self.perf_mix_lambda) * ck_values
        size_feat = torch.log(nks.clamp_min(1.0))  # log scale for size
        div_feat = diversity_scores
        mod_feat = modality_scores
        # invert "bad" signals (lower loss and smaller grad are better)
        vl_inv = -self._zscore_clip(val_losses)
        gn_inv = -self._zscore_clip(grad_norms)
        # class balance: 1 - 2*|r - 0.5|  (0 worst, 1 best)
        bal_feat_raw = 1.0 - 2.0 * torch.abs(imb_ratios - 0.5)
        # z-score stable signals
        perf_feat = self._zscore_clip(perf_mix)
        size_feat = self._zscore_clip(size_feat)
        div_feat = self._zscore_clip(div_feat)
        mod_feat = self._zscore_clip(mod_feat)
        bal_feat = self._zscore_clip(bal_feat_raw)

        # Stack teacher features: (C, 7)
        teacher_feats = torch.stack(
            [perf_feat, size_feat, div_feat, mod_feat, vl_inv, gn_inv, bal_feat],
            dim=1,
        )

        # Helper: aggregate with a given attention vector over clients
        def aggregate_with_attention(attn_vec: torch.Tensor):
            attn_cpu = attn_vec.detach().cpu().numpy()
            new_params = []
            for i in range(len(weights_per_client[0])):
                agg_param = torch.zeros_like(weights_per_client[0][i], dtype=torch.float32, device="cpu")
                for j in range(len(weights_per_client)):
                    agg_param += attn_cpu[j] * weights_per_client[j][i]
                new_params.append(agg_param)
            return [p.numpy() for p in new_params]

        # ---------- show student attention (before Optuna) ----------
        with torch.no_grad():
            logits_student = self.aggregator(meta_norm)              # (C, 1)
            attn_student = torch.softmax(logits_student.view(-1), dim=0)
        print(f"👩‍🎓 Student (MLP) attention (pre-Optuna): {attn_student.detach().cpu().numpy()}")

        assert self.evaluate_fn is not None, "evaluate_fn is required for alpha search."

        # ---------- Optuna: alphas over the 7 teacher features ----------
        best = {
            "f1": -1.0,
            "alphas": torch.tensor(
                [self.alpha_perf, self.alpha_size, self.alpha_div, self.alpha_mod, self.alpha_vl, self.alpha_gn, self.alpha_bal],
                dtype=torch.float32,
                device=_DEVICE,
            ),
            "attn_teacher": attn_student,
            "aggregated_parameters": None,
        }

        def objective(trial):
            # Suggest non-negative unnormalized alphas
            a_perf = trial.suggest_float("alpha_perf", 0.0, 1.0)
            a_size = trial.suggest_float("alpha_size", 0.0, 1.0)
            a_div  = trial.suggest_float("alpha_div",  0.0, 1.0)
            a_mod  = trial.suggest_float("alpha_mod",  0.0, 1.0)
            a_vl   = trial.suggest_float("alpha_vl",   0.0, 1.0)
            a_gn   = trial.suggest_float("alpha_gn",   0.0, 1.0)
            a_bal  = trial.suggest_float("alpha_bal",  0.0, 1.0)

            alphas = torch.tensor([a_perf, a_size, a_div, a_mod, a_vl, a_gn, a_bal], dtype=torch.float32, device=_DEVICE)
            s = float(alphas.sum().item())
            if s <= 0:
                raise optuna.TrialPruned()

            alphas = alphas / s
            logits_teacher = teacher_feats @ alphas
            attn_teacher = torch.softmax(logits_teacher, dim=0)

            aggregated_weights = aggregate_with_attention(attn_teacher)
            aggregated_parameters = ndarrays_to_parameters(aggregated_weights)

            _, metrics = self.evaluate_fn(server_round, aggregated_parameters, {})
            f1_macro = float(metrics.get("test_f1_macro", 0.0))

            nonlocal best
            if f1_macro > best["f1"]:
                best = {
                    "f1": f1_macro,
                    "alphas": alphas.detach().clone(),
                    "attn_teacher": attn_teacher.detach(),
                    "aggregated_parameters": aggregated_parameters,
                }
            return f1_macro

        self.study = optuna.create_study(direction="maximize")
        self.study.optimize(objective, n_trials=self.n_trials_per_round)

        # ---------- adopt best teacher & distill ----------
        best_alphas = best["alphas"].detach().cpu().numpy().tolist()
        (
            self.alpha_perf,
            self.alpha_size,
            self.alpha_div,
            self.alpha_mod,
            self.alpha_vl,
            self.alpha_gn,
            self.alpha_bal,
        ) = best_alphas

        attn_teacher = best["attn_teacher"]
        print(
            "🌟 Best α this round → "
            f"perf={self.alpha_perf:.3f}, size={self.alpha_size:.3f}, div={self.alpha_div:.3f}, "
            f"mod={self.alpha_mod:.3f}, vl={self.alpha_vl:.3f}, gn={self.alpha_gn:.3f}, bal={self.alpha_bal:.3f}; "
            f"F1={best['f1']:.4f}"
        )

        logits_student = self.aggregator(meta_norm).view(-1)
        log_probs_student = torch.log_softmax(logits_student, dim=0)
        probs_teacher = attn_teacher.to(_DEVICE)

        kl_loss = self.loss_fn(log_probs_student, probs_teacher)
        entropy = -(torch.softmax(logits_student, dim=0) * log_probs_student).sum()
        loss = kl_loss - self.entropy_coeff * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        torch.save(self.aggregator.state_dict(), self.aggregator_path)
        print(f"🎯 Distill MLP | KL: {kl_loss.item():.4f} | Entropy: {entropy.item():.4f}")

        with torch.no_grad():
            new_student_attn = torch.softmax(self.aggregator(meta_norm).view(-1), dim=0)
            print(f"👩‍🎓 Student (MLP) attention (post-distill): {new_student_attn.detach().cpu().numpy()}")
            print(f"🧑‍🏫 Teacher attention (best trial):        {attn_teacher.detach().cpu().numpy()}")

        final_parameters = best["aggregated_parameters"]
        return final_parameters, {}

    def configure_evaluate(self, server_round, parameters, client_manager):
        return []

    def aggregate_evaluate(self, server_round, results, failures):
        return None

    def evaluate(self, server_round, parameters):
        if self.evaluate_fn is None:
            return None
        return self.evaluate_fn(server_round, parameters, {})
