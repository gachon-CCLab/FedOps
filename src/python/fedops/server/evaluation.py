"""Explicit server validation policy and sample-weighted client evaluation.

No synthetic fallback is permitted here. Missing settings retain legacy mode.
"""
import json
import math
import os
from numbers import Integral, Real
from pathlib import Path


def evaluation_policy(config):
    raw = config.get("server_evaluation", {}) or {}
    policy = dict(raw)
    campaign = json.loads(os.environ.get("FEDOPS_CAMPAIGN_CONFIG", "{}"))
    override = campaign.get("serverEvaluation")
    if override is not None:
        if not isinstance(override, dict) or type(override.get("enabled")) is not bool:
            raise ValueError("serverEvaluation.enabled must be boolean")
        policy["enabled"] = override["enabled"]
        relative = override.get("dataPath", "")
        if policy["enabled"]:
            validate_relative_path(relative)
            policy["data_root"] = str(Path("/app/data/server-validation") / relative)
            policy["allowed_root"] = "/app/data/server-validation"
    if "enabled" in policy and type(policy["enabled"]) is not bool:
        raise ValueError("server_evaluation.enabled must be boolean")
    return policy


def validate_relative_path(value):
    if (not isinstance(value, str) or not value.strip() or value != value.strip()
            or value.startswith("/") or "\\" in value
            or any(part in ("", ".", "..") for part in value.split("/"))
            or any(ord(c) < 32 for c in value)):
        raise ValueError("Validation data path must be a relative directory without traversal")
    return value


def validation_directory(policy):
    value = (policy.get("data_root") if policy.get("allowed_root") else None) or os.environ.get("FEDOPS_SERVER_DATA_DIR") or policy.get("data_root")
    if not value:
        raise ValueError("Server Validation requires a connected server data directory")
    path = Path(value).expanduser().resolve(strict=True)
    root = policy.get("allowed_root") or os.environ.get("FEDOPS_SERVER_VALIDATION_ROOT")
    if root:
        if Path(root).is_symlink():
            raise ValueError("Validation area must not be a symlink")
        allowed = Path(root).resolve(strict=True)
        if path == allowed or allowed not in path.parents:
            raise ValueError("Validation directory must stay inside the Task validation area")
    if not path.is_dir() or not os.access(path, os.R_OK | os.X_OK):
        raise ValueError("Validation directory is not readable")
    if not any(path.iterdir()):
        raise ValueError("Validation directory is empty")
    # A symlink in the data tree must not escape the selected directory.
    for item in path.rglob("*"):
        if item.is_symlink():
            target = item.resolve(strict=True)
            if target != path and path not in target.parents:
                raise ValueError("Validation data symlink escapes the connected directory")
    return path


def prepare_validation_loader(config, loader_factory):
    policy = evaluation_policy(config)
    if policy.get("enabled") is False:
        return None
    if "enabled" not in policy:
        # Compatibility only: old Releases used their own dataset/root policy.
        root = os.environ.get("FEDOPS_SERVER_DATA_DIR", str(config.dataset.root))
        download = bool(config.dataset.download)
    else:
        root = str(validation_directory(policy))
        download = False
    loader = loader_factory(batch_size=int(config.batch_size), data_root=root, download=download)
    if loader is None or len(loader) == 0:
        raise ValueError("Server Validation loader contains no evaluation batches")
    return loader


def _finite(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def valid_client_results(results):
    valid = []
    seen = set()
    rejected = 0
    for client, result in results:
        identity = getattr(client, "cid", id(client))
        n = result.num_examples
        if (identity in seen or not isinstance(n, Integral) or isinstance(n, bool)
                or n <= 0 or not _finite(result.loss)):
            rejected += 1
            continue
        seen.add(identity)
        valid.append((client, result))
    return valid, rejected


def aggregate_client_evaluations(results, failures=0):
    """Results belong to a single Flower evaluate RPC round, not local fit metrics."""
    pairs, rejected = valid_client_results(results)
    valid = [result for _, result in pairs]
    count = sum(r.num_examples for r in valid)
    accuracy_results = [r for r in valid if _finite(r.metrics.get("accuracy"))
                        and 0 <= r.metrics["accuracy"] <= 1]
    accuracy_count = sum(r.num_examples for r in accuracy_results)
    return {
        "gl_loss": float(math.fsum(float(r.loss) * (r.num_examples / count) for r in valid)) if count else None,
        "gl_accuracy": (float(math.fsum(float(r.metrics["accuracy"]) * (r.num_examples / accuracy_count)
                                      for r in accuracy_results)) if accuracy_count else None),
        "evaluation_source": "client_aggregated",
        "evaluation_status": ("not_evaluated" if not count else
                              "partial" if failures or rejected else "evaluated"),
        "evaluation_clients": len(valid),
        "evaluation_samples": count,
        "accuracy_samples": accuracy_count,
        "evaluation_failures": failures + rejected,
    }
