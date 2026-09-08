"""Run inside a prepared FL Server, before starting a Campaign.

python -m fedops.server.validation
Only status/counts are emitted; data and model predictions are not returned.
"""
import inspect
import json
import math
import io
from contextlib import redirect_stdout

from .evaluation import evaluation_policy, prepare_validation_loader


def check():
    from omegaconf import OmegaConf
    from federated_task.config import load_config
    from federated_task.federated_learning import server_main
    config = OmegaConf.create(load_config())
    policy = evaluation_policy(config)
    if "enabled" not in policy:
        return {"status": "release_default", "source": "release_default"}
    if "prepare_validation_loader" not in inspect.getsource(server_main.main):
        raise ValueError("This Release needs the optional-validation Baseline before changing evaluation mode")
    if policy["enabled"] is False:
        if float(config.server.strategy.get("fraction_evaluate", 1)) <= 0:
            raise ValueError("Client evaluation requires fraction_evaluate > 0")
        if str(config.model_type) != "Pytorch":
            raise ValueError("This Release does not support client-aggregated evaluation")
        return {"status": "ready", "source": "client_aggregated"}
    from federated_task.local_training.data_preparation import gl_model_torch_validation
    from federated_task.local_training.model import build_model
    from federated_task.runtime.model_release import MODEL_PATH, load_released_model, test_torch
    loader = prepare_validation_loader(config, gl_model_torch_validation)
    model = load_released_model() if MODEL_PATH.is_file() else build_model(dict(config.model))
    loss, primary, _ = test_torch(1)(model, loader, config)
    if not math.isfinite(float(loss)) or not math.isfinite(float(primary)):
        raise ValueError("Validation test returned non-finite metrics")
    return {"status": "ready", "source": "server_validation", "batches": len(loader),
            "samples": len(loader.dataset), "checkedBatches": 1}


def main():
    try:
        # Task code may emit local-training progress; expose only check status.
        with redirect_stdout(io.StringIO()):
            result = {"success": True, **check()}
    except Exception as exc:
        result = {"success": False, "status": "error", "error": str(exc)}
    print("FEDOPS_VALIDATION_RESULT=" + json.dumps(result))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
