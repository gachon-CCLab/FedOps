# server/strategy_cluster_optuna.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import logging, json
import numpy as np
import optuna
from sklearn.cluster import DBSCAN
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import Parameters, FitIns

# 실험중이니 이후에 방법이 바뀔수도있음 일단 버전 1 클러스터링 + optuna 

logger = logging.getLogger(__name__)

# ----- 유틸: metrics에서 점수 뽑기 -----
def _score_from_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    # client.fit()가 반환한 metrics에서 사용할 값 추출
    # 없으면 0.0으로 대체
    return {
        "val_loss": float(metrics.get("val_loss", metrics.get("loss", 0.0))),
        "val_accuracy": float(metrics.get("val_accuracy", metrics.get("accuracy", 0.0))),
        "val_f1_score": float(metrics.get("val_f1_score", metrics.get("f1_score", 0.0))),
    }


# ----- DBSCAN 클러스터링 -----
## 나중에 추가로 eps를 동적으로 작업하는 내용 추가할 예정 지금은 일단 고정
class _DBSCANClusterer:
    def __init__(self, eps: float = 0.35, min_samples: int = 1):
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, profiles: Dict[str, Dict[str, float]]) -> Dict[str, int]:
        if not profiles:
            return {}
        cids, feats = [], []
        for cid, p in profiles.items():
            vloss = float(p.get("val_loss", 0.0))
            vacc = float(p.get("val_accuracy", 0.0))
            vf1 = float(p.get("val_f1_score", 0.0))
            n = float(p.get("num_examples", 0.0))
            feats.append(np.array([vloss, vacc, vf1, np.log1p(n)], dtype=np.float32))
            cids.append(cid)
        X = np.stack(feats) if feats else np.zeros((0, 4), dtype=np.float32)
        if len(X) == 0:
            return {cid: 0 for cid in profiles.keys()}
        labels = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit_predict(X)
        mapped, noise_id = {}, 0
        for cid, lab in zip(cids, labels):
            if lab == -1:
                mapped[cid] = 10_000 + noise_id
                noise_id += 1
            else:
                mapped[cid] = int(lab)
        return mapped


# ----- Optuna HPO -----
class _OptunaHPO:
    _OBJECTIVES = {
        "maximize_f1": ("val_f1_score", "maximize"),
        "maximize_acc": ("val_accuracy", "maximize"),
        "minimize_loss": ("val_loss", "minimize"),
    }

    def __init__(
        self,
        objective: str = "maximize_f1",
        search_lr_log: Tuple[float, float] = (-4.5, -2.0),  # lr = 10**x
        search_bs_exp: Tuple[int, int] = (4, 7),            # bs = 2**x
        search_local_epochs: Tuple[int, int] = (1, 3),
        seed_points: Optional[List[Tuple[float, int, int]]] = None,
    ):
        key, direction = self._OBJECTIVES.get(objective, ("val_f1_score", "maximize"))
        self.objective_key = key
        self.direction = direction
        self.search_lr_log = search_lr_log
        self.search_bs_exp = search_bs_exp
        self.search_local_epochs = search_local_epochs
        self.seed_points = seed_points or []
        self.studies: Dict[int, optuna.Study] = {}

    def _study(self, cluster_id: int) -> optuna.Study:
        if cluster_id not in self.studies:
            self.studies[cluster_id] = optuna.create_study(direction=self.direction)
        return self.studies[cluster_id]

    def ask(self, cluster_id: int) -> Tuple[Dict, int]:
        study = self._study(cluster_id)
        # warm start
        if len(study.trials) < len(self.seed_points):
            lr, bs, ep = self.seed_points[len(study.trials)]
            t = study.ask()
            return {
                "hp_learning_rate": float(lr),
                "hp_batch_size": int(bs),
                "hp_local_epochs": int(ep),
            }, t.number

        t = study.ask()
        lr_log = t.suggest_float("lr_log", self.search_lr_log[0], self.search_lr_log[1])
        bs_exp = t.suggest_int("bs_exp", self.search_bs_exp[0], self.search_bs_exp[1])
        ep = t.suggest_int("local_epochs", self.search_local_epochs[0], self.search_local_epochs[1])

        hp = {
            "hp_learning_rate": float(10.0**lr_log),
            "hp_batch_size": int(2**bs_exp),
            "hp_local_epochs": int(ep),
        }
        return hp, t.number

    def tell(self, cluster_id: int, trial_number: int, score: float) -> None:
        study = self._study(cluster_id)
        # trial 번호가 사라졌을 수도 있으니 보수적으로 처리
        has = [tr for tr in study.trials if tr.number == trial_number]
        if not has:
            tr = study.ask()
            trial_number = tr.number
        study.tell(trial_number, float(score))



class ClusterOptunaFedAvg(FedAvg):
    def __init__(
        self,
        objective: str = "maximize_f1",
        eps: float = 0.3,
        min_samples: int = 1,
        search_lr_log: Tuple[float, float] = (-4.5, -2.0),
        search_bs_exp: Tuple[int, int] = (4, 7),
        search_local_epochs: Tuple[int, int] = (1, 3),
        seed_points: Optional[List[Tuple[float, int, int]]] = None,
        warmup_rounds: int = 0,     # 이미 추가한 부분
        recluster_every: int = 1,   # 내부에 저장
        **fedavg_kwargs,
    ):
        super().__init__(**fedavg_kwargs)
        self.objective = objective
        self.clusterer = _DBSCANClusterer(eps=eps, min_samples=min_samples)
        self.hpo = _OptunaHPO(
            objective=objective,
            search_lr_log=search_lr_log,
            search_bs_exp=search_bs_exp,
            search_local_epochs=search_local_epochs,
            seed_points=seed_points,
        )
        self.warmup_rounds = warmup_rounds
        self.recluster_every = recluster_every
        self.client_profiles: Dict[str, Dict[str, float]] = {}
        self.assignments: Dict[str, int] = {}
        self.last_trial_by_cluster: Dict[int, int] = {}


    def _log_cluster_summary(self, server_round: int) -> None:
        """현재 self.assignments를 요약해서 로그로 남긴다."""
        if not self.assignments:
            logger.info(f"[CLUSTER][r{server_round}] no assignments yet")
            return
        # cluster_id -> [cid1, cid2, ...]
        groups = {}
        for cid, gid in self.assignments.items():
            groups.setdefault(gid, []).append(cid)

        summary = {
            "round": server_round,
            "num_clusters": len(groups),
            "clusters": [
                {
                    "cluster_id": int(gid),
                    "size": len(cids),
                    "sample_cids": cids[:8],  # 너무 길어지지 않게 일단 앞 8개만
                }
                for gid, cids in sorted(groups.items(), key=lambda kv: kv[0])
            ],
        }
        logger.info("[CLUSTER] %s", json.dumps(summary, ensure_ascii=False))

    
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        base = super().configure_fit(server_round, parameters, client_manager)

        if (server_round == 1) or (self.recluster_every <= 1) or (server_round % self.recluster_every == 0):
            self.assignments = self.clusterer.fit(self.client_profiles) if self.client_profiles else {}
            self._log_cluster_summary(server_round)

        picked: Dict[int, Dict] = {}
        out: List[Tuple[ClientProxy, FitIns]] = []
        for client, fitins in base:
            cid = client.cid
            cluster_id = self.assignments.get(cid, 0)

            if cluster_id not in picked:
                hp, trial_num = self.hpo.ask(cluster_id)
                picked[cluster_id] = {"hp": hp, "trial": trial_num}
                self.last_trial_by_cluster[cluster_id] = trial_num

            hp = picked[cluster_id]["hp"]
            cfg = dict(fitins.config)
            cfg.update(hp)

            # ★ 여기 추가: 각 클라이언트에 자신의 cluster_id 전달
            cfg["cluster_id"] = int(cluster_id)

            out.append((client, FitIns(parameters=fitins.parameters, config=cfg)))
        return out


    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ):
        agg_params, agg_metrics = super().aggregate_fit(server_round, results, failures)

        # 클러스터별 점수 평균
        cluster_scores: Dict[int, List[float]] = {}
        for client, fitres in results:
            cid = client.cid
            cluster_id = self.assignments.get(cid, 0)
            m = fitres.metrics or {}
            s = _score_from_metrics(m)
            if self.objective == "minimize_loss":
                score = s["val_loss"]
            elif self.objective == "maximize_acc":
                score = s["val_accuracy"]
            else:
                score = s["val_f1_score"]
            cluster_scores.setdefault(cluster_id, []).append(float(score))

        # HPO.tell
        for clus_id, arr in cluster_scores.items():
            trial_num = self.last_trial_by_cluster.get(clus_id, None)
            if trial_num is None:
                continue
            avg_score = float(sum(arr) / max(1, len(arr)))
            self.hpo.tell(clus_id, trial_num, avg_score)

        # 다음 라운드용 client_profiles 갱신
        for client, fitres in results:
            cid = client.cid
            m = fitres.metrics or {}
            s = _score_from_metrics(m)
            self.client_profiles[cid] = {
                "val_loss": s["val_loss"],
                "val_accuracy": s["val_accuracy"],
                "val_f1_score": s["val_f1_score"],
                "num_examples": float(fitres.num_examples),
            }

        return agg_params, agg_metrics
