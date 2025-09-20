# aggregation/map2_fedavg.py

import math
import torch
import torch.nn as nn
from flwr.server.strategy import FedAvg

class Map2FedAvgStrategy(FedAvg):
    """
    Map²-FedAvgStrategy for text+image only:
      1) Embed each client summary [perf, contrib, cov_img, cov_txt] via an MLP.
      2) Compute per-branch attention weights for 'img', 'txt', and 'fusion'.
      3) Aggregate branch deltas accordingly.
      4) Meta-update the MLP & query vectors on a dev set.
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
        # 4 → mlp_hidden → mlp_hidden//2
        self.embed = nn.Sequential(
            nn.Linear(4, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.ReLU(),
        )
        d = mlp_hidden // 2

        # One learnable query per branch
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

        # Capture param key order once
        if self._model_order is None:
            params0    = self.initial_parameters()
            tmp_model  = self._model_fn()
            tmp_model.load_state_dict(self.parameters_to_weights(params0))
            self._model_order = list(tmp_model.state_dict().keys())

        # 1) Unpack summaries & compute client deltas
        summaries, deltas = {}, {}
        global_params = dict(zip(
            self._model_order,
            self.parameters_to_weights(self.initial_parameters()),
        ))

        for _, fit in results:
            m = fit.metrics or {}
            x = torch.tensor([
                m.get("perf", 0.0),
                m.get("contrib", 0.0),
                m.get("cov_img", 0.0),
                m.get("cov_txt", 0.0),
            ])
            summaries[fit.client_name] = x

            local_weights = self.parameters_to_weights(fit.parameters)
            local_sd = dict(zip(self._model_order, local_weights))
            deltas[fit.client_name] = {
                k: local_sd[k] - global_params[k] for k in self._model_order
            }

        # 2) Embed summaries
        embeddings = {k: self.embed(v) for k, v in summaries.items()}
        d = next(iter(embeddings.values())).shape[0]

        # 3) Branch-wise aggregation
        new_state = {}
        branches = [("img", 2), ("txt", 3), ("fusion", None)]
        for branch, idx in branches:
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
                # carry forward global params for this branch
                for name, base in global_params.items():
                    if name.startswith(branch):
                        new_state[name] = base.clone()
                continue

            # compute attention scores
            q = self.queries[branch]
            alphas = torch.tensor([
                math.exp((embeddings[k] @ q).item() / math.sqrt(d))
                for k in valid
            ])
            weights = alphas / alphas.sum()

            # weighted aggregate each param in branch
            for name, base in global_params.items():
                if not name.startswith(branch):
                    continue
                agg = torch.zeros_like(base)
                for i, k in enumerate(valid):
                    agg += weights[i] * (base + deltas[k][name])
                new_state[name] = agg

        # 4) Meta-update on dev set
        self._meta_update(new_state)

        # 5) Pack new global parameters
        new_weights = [new_state[k] for k in self._model_order]
        return self.weights_to_parameters(new_weights), {}

    def _meta_update(self, state_dict):
        # Load new global state into fresh model
        model = self._model_fn()
        sd = model.state_dict()
        for k in sd:
            sd[k] = state_dict[k]
        model.load_state_dict(sd)
        model.eval()

        # Compute dev-loss
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        with torch.no_grad():
            for batch in self.dev_loader:
                out = model(**batch)
                total_loss += criterion(out, batch["label"]).item()

        # Backprop through embed & queries
        opt = torch.optim.SGD(
            list(self.embed.parameters()) + list(self.queries.parameters()),
            lr=self.meta_lr
        )
        opt.zero_grad()
        torch.tensor(total_loss, requires_grad=True).backward()
        opt.step()
