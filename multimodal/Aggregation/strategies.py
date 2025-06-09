# aggregation/strategies.py
import torch, torch.nn as nn
from flwr.server.strategy import Strategy
from flwr.common import Parameters

class Map2FedAvgStrategy(Strategy):
    def __init__(
        self,
        dev_loader,
        mlp_hidden: int,
        meta_lr: float,
        # plus all the usual FedAvg args: fraction_fit, min_fit_clients, etc.
        **kwargs
    ):
        super().__init__(**kwargs)
        # 1) build your tiny 5→mlp_hidden→(mlp_hidden//2) MLP
        self.embed = nn.Sequential(
            nn.Linear(5, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.ReLU(),
        )
        d = mlp_hidden // 2
        # 2) one query vector per branch
        self.queries = {
            b: nn.Parameter(torch.randn(d))
            for b in ("img", "txt", "aud", "fusion")
        }
        # 3) store dev‐set loader & meta‐lr
        self.dev_loader = dev_loader
        self.meta_lr    = meta_lr

    def aggregate_fit(self, rnd, results, failures):
        # 1) unpack each client’s (cid, FitRes):
        #    - extract their delta weights
        #    - extract their 5-dim summary from FitRes.metrics
        # 2) embed all summaries via self.embed → e_k
        # 3) for each branch b:
        #     • select only clients with coverage=1
        #     • compute α_{b,k}=exp((q^b·e_k)/√d)
        #     • normalize to w_{b,k}
        #     • do W_new[b] = Σ_k w_{b,k}·(W_old[b] + ΔW_k[b])
        # 4) (optional) meta-update self.embed & self.queries on self.dev_loader
        # 5) reassemble W_new → a flat state_dict → Parameters and return
        ...
