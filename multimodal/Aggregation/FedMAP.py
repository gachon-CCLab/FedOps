# aggregation/map2_fedavg.py

import math
import torch
import torch.nn as nn
from flwr.server.strategy import FedAvg


class Map2FedAvgStrategy(FedAvg):
    """
    Map²-FedAvgStrategy for text+image only:
    1) Embed each client summary [perf, contrib, cov_img, cov_txt] via a small MLP.
    2) Compute per-modality attention weights for 'img', 'txt', and 'fusion'.
    3) Aggregate each branch’s deltas with those weights.
    4) Meta-update the MLP & query vectors on the dev set.
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
            **kwargs,
        )
        # 4 → mlp_hidden → mlp_hidden//2 embedding MLP
        self.embed = nn.Sequential(
            nn.Linear(4, mlp_hidden),  # input: [perf, contrib, cov_img, cov_txt]
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.ReLU(),
        )
        d = mlp_hidden // 2

        # One learnable query vector per branch
        self.queries = nn.ParameterDict({
            "img":    nn.Parameter(torch.randn(d)),
            "txt":    nn.Parameter(torch.randn(d)),
            "fusion": nn.Parameter(torch.randn(d)),
        })

        self.dev_loader = dev_loader
        self.meta_lr    = meta_lr
        self._model_order = None

    def aggregate_fit(self, rnd, results, failures):
        if not results:
            return None, {}

        # Capture parameter order on first call
        if self._model_order is None:
            params0     = self.initial_parameters()
            temp_model  = self._model_fn()
            temp_model.load_state_dict(self.parameters_to_weights(params0))
            self._model_order = list(temp_model.state_dict().keys())

        # 1) Unpack summaries & compute per-client deltas
        summaries, deltas = {}, {}
        global_params = dict(zip(
            self._model_order,
            self.parameters_to_weights(self.initial_parameters()),
        ))

        for _, fit in results:
            m = fit.metrics or {}
            # Build 4-d summary: perf, contrib, cov_img, cov_txt
            x = torch.tensor([
                m.get("perf", 0.0),
                m.get("contrib", 0.0),
                m.get("cov_img", 0.0),
                m.get("cov_txt", 0.0),
            ])
            summaries[fit.client_name] = x

            # Compute full-model delta = local_state - global_state
            local_weights = self.parameters_to_weights(fit.parameters)
            local_sd = dict(zip(self._model_order, local_weights))
            deltas[fit.client_name] = {
                k: local_sd[k] - global_params[k]
                for k in self._model_order
            }

        # 2) Embed all summaries
        embeddings = {k: self.embed(v) for k, v in summaries.items()}
        d = next(iter(embeddings.values())).shape[0]

        # 3) Branch‐wise aggregation
        new_state = {}
        # Define branches with the index of their coverage flag in summary
        branches = [
            ("img",    2),  # coverage at x[2]
            ("txt",    3),  # coverage at x[3]
            ("fusion", None),  # fusion uses any client that has img or txt
        ]

        for branch, idx in branches:
            # Select clients valid for this branch
            if branch == "fusion":
                valid = [
                    k for k, x in summaries.items()
                    if x[2] > 0.5 or x[3] > 0.5
                ]
            else:
                valid = [
                    k for k, x in summaries.items()
                    if x[idx] > 0.5
                ]

            if not valid:
                # No updates: carry forward global params for this branch
                for name, base in global_params.items():
                    if name.startswith(branch):
                        new_state[name] = base.clone()
                continue

            # Compute unnormalized attention scores
            q = self.queries[branch]
            alphas = torch.tensor([
                math.exp((embeddings[k] @ q).item() / math.sqrt(d))
                for k in valid
            ])
            weights = alphas / alphas.sum()

            # Weighted aggregate each parameter in this branch
            for name, base in global_params.items():
                if not name.startswith(branch):
                    continue
                agg = torch.zeros_like(base)
                for i, k in enumerate(valid):
                    agg += weights[i] * (base + deltas[k][name])
                new_state[name] = agg

        # 4) Meta‐update the MLP and queries on the dev set
        self._meta_update(new_state)

        # 5) Pack and return new global parameters
        new_weights = [new_state[k] for k in self._model_order]
        return self.weights_to_parameters(new_weights), {}

    def _meta_update(self, state_dict):
        # Load the new global state into a fresh model
        model = self._model_fn()
        sd = model.state_dict()
        for k in sd:
            sd[k] = state_dict[k]
        model.load_state_dict(sd)
        model.eval()

        # Compute dev-set loss
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        with torch.no_grad():
            for batch in self.dev_loader:
                out = model(**batch)
                total_loss += criterion(out, batch["label"]).item()

        # Backprop through MLP & query vectors
        optimizer = torch.optim.SGD(
            list(self.embed.parameters()) + list(self.queries.parameters()),
            lr=self.meta_lr
        )
        optimizer.zero_grad()
        torch.tensor(total_loss, requires_grad=True).backward()
        optimizer.step()
