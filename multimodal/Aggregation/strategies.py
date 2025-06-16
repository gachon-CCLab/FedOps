import math
import copy
import torch
import torch.nn as nn
from flwr.server.strategy import FedAvg


class Map2FedAvgStrategy(FedAvg):
    """
    Map²-FedAvg: server‐side strategy that
    1) embeds each client summary via an MLP,
    2) computes per-modality attention weights,
    3) aggregates branch deltas accordingly,
    4) meta-updates the MLP & query vectors on a dev set.
    """

    def __init__(
        self,
        dev_loader,
        mlp_hidden: int,
        meta_lr: float,
        fraction_fit: float,
        fraction_evaluate: float,
        min_fit_clients: int,
        min_evaluate_clients: int,
        min_available_clients: int,
        **kwargs
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            **kwargs
        )
        # 1) Summary MLP: 5 → mlp_hidden → mlp_hidden//2
        self.embed = nn.Sequential(
            nn.Linear(5, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.ReLU(),
        )
        d = mlp_hidden // 2

        # 2) One query vector per branch
        self.queries = nn.ParameterDict({
            b: nn.Parameter(torch.randn(d))
            for b in ("img", "txt", "ts", "fusion")
        })

        self.dev_loader = dev_loader
        self.meta_lr = meta_lr
        self._model_order = None  # will capture state_dict key order once

    def aggregate_fit(self, rnd, results, failures):
        if not results:
            return None, {}

        # Capture model key order on first round
        if self._model_order is None:
            params0 = self.initial_parameters()
            temp_model = self._model_fn()
            temp_model.load_state_dict(self.parameters_to_weights(params0))
            self._model_order = list(temp_model.state_dict().keys())

        # 1) Unpack summaries & full-model deltas
        summaries = {}
        deltas = {}
        global_params = dict(zip(
            self._model_order,
            self.parameters_to_weights(self.initial_parameters())
        ))

        for _, fit in results:
            # metrics must include keys: perf, contrib, cov_img, cov_txt, cov_ts
            m = fit.metrics or {}
            x = torch.tensor([
                m.get("perf", 0.0),
                m.get("contrib", 0.0),
                m.get("cov_img", 0.0),
                m.get("cov_txt", 0.0),
                m.get("cov_ts", 0.0),
            ])
            summaries[fit.client_name] = x

            # compute delta_state: local_state - global_state
            weights = self.parameters_to_weights(fit.parameters)
            local_sd = dict(zip(self._model_order, weights))
            delta_sd = {
                k: local_sd[k] - global_params[k]
                for k in self._model_order
            }
            deltas[fit.client_name] = delta_sd

        # 2) Embed each summary
        embeddings = {
            k: self.embed(v)
            for k, v in summaries.items()
        }
        d = next(iter(embeddings.values())).shape[0]

        # 3) Aggregate per-branch
        new_state = {}
        branches = [
            ("img", 2),
            ("txt", 3),
            ("ts", 4),
            ("fusion", 4),  # fusion uses same mask as ts slot index
        ]

        for branch, mask_idx in branches:
            # select clients with coverage=1
            valid = [k for k, x in summaries.items() if x[mask_idx] > 0.5]
            if not valid:
                # no update: carry forward global params
                for name, val in global_params.items():
                    if name.startswith(branch):
                        new_state[name] = val.clone()
                continue

            # compute attention alphas
            q = self.queries[branch]
            alphas = torch.tensor([
                math.exp((embeddings[k] @ q).item() / math.sqrt(d))
                for k in valid
            ])
            weights = alphas / alphas.sum()

            # weighted aggregate each parameter in this branch
            for name, base in global_params.items():
                if not name.startswith(branch):
                    continue
                agg = torch.zeros_like(base)
                for i, k in enumerate(valid):
                    agg += weights[i] * (base + deltas[k][name])
                new_state[name] = agg

        # 4) Meta-update MLP & queries on dev set
        self._meta_update(new_state)

        # 5) Pack new_state into Parameters
        new_weights = [new_state[k] for k in self._model_order]
        return self.weights_to_parameters(new_weights), {}

    def _meta_update(self, state_dict):
        # Load global state into a fresh model and eval on dev set
        model = self._model_fn()
        sd = model.state_dict()
        for k in sd:
            sd[k] = state_dict[k]
        model.load_state_dict(sd)
        model.eval()

        # Compute dev loss
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        with torch.no_grad():
            for batch in self.dev_loader:
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    image=batch["image"]
                )
                total_loss += criterion(out, batch["label"]).item()

        # Backprop through MLP & queries
        opt = torch.optim.SGD(
            list(self.embed.parameters()) +
            list(self.queries.parameters()),
            lr=self.meta_lr
        )
        opt.zero_grad()
        # turn scalar loss into a tensor for grad
        torch.tensor(total_loss, requires_grad=True).backward()
        opt.step()
