# server/best_keeper.py
import os, json, torch
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters

# 목적: 연합학습중 가장 좋은 전역 가중치를 잃지 않도록 매 라운드의 평가 지표를 모니터링하여 최고 성능 모델을 즉시 저장/복원하기 위함.
#       - 마지막 라운드가 항상 최고 성능이 아닐 수 있어 안전하게 최고 성능을 보존
#       - 실험 재현/중단 후 재개/테스트셋 최종 평가 등에 바로 활용
#       - 메타(라운드, 점수)도 함께 저장해 결과 추적 용이

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
            # 파일로도 저장(원하면 npy로 저장 가능)
            torch.save(parameters_to_ndarrays(parameters), os.path.join(self.save_dir, "best_params.pt"))
            with open(os.path.join(self.save_dir, "best_meta.json"), "w") as f:
                json.dump({"round": server_round, "metric": m}, f)

    def load_params(self):
        import numpy as np
        path = os.path.join(self.save_dir, "best_params.pt")
        if not os.path.exists(path):
            return None
        nds = torch.load(path, map_location="cpu")
        return ndarrays_to_parameters(nds)
