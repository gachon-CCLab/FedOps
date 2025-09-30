# server/best_keeper.py
import os, json, torch
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters

# 목적:
# - 각 라운드 평가 직후 "최고 성능"을 낸 전역 파라미터 스냅샷을 디스크에 보관
# - 마지막 라운드가 최고가 아닐 수 있으므로 학습 종료 후 최고 파라미터로 최종 산출물 덮어쓰기
# - 재현성, 중단 복구, 사후 평가 등에 활용

class BestKeeper:
    def __init__(self, save_dir="./gl_best", metric_key="accuracy"):
        self.best = None  # {"metric": float, "round": int, "params": Parameters}
        self.metric_key = metric_key
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def update(self, server_round, parameters, metrics: dict):
        m = float(metrics.get(self.metric_key, -1.0))
        if (self.best is None) or (m > self.best["metric"]):
            self.best = {"metric": m, "round": server_round, "params": parameters}
            # ndarrays 형태로 디스크 저장
            torch.save(
                parameters_to_ndarrays(parameters),
                os.path.join(self.save_dir, "best_params.pt"),
            )
            with open(os.path.join(self.save_dir, "best_meta.json"), "w") as f:
                json.dump({"round": server_round, "metric": m}, f)

    def load_params(self):
        path = os.path.join(self.save_dir, "best_params.pt")
        if not os.path.exists(path):
            return None
        nds = torch.load(path, map_location="cpu")
        return ndarrays_to_parameters(nds)
