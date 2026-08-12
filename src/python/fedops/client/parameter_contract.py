"""Authoritative model-parameter contract used by FedOps clients and tooling.

Federated Task readiness must describe exactly the tensors that the running
FedOps client sends. Keep framework-specific selection and application here,
not in individual Task projects.
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from typing import Any, Sequence

import numpy as np
import torch


def _pytorch_items(model):
    # Preserve the existing FedOps 1.2 transport rule. BatchNorm state remains
    # local and is not included in the Flower parameter payload.
    return [(name, value) for name, value in model.state_dict().items() if "bn" not in name]


def get_parameters(model, model_type: str) -> list[np.ndarray]:
    """Return the exact ndarray payload sent by a FedOps client."""
    if model_type == "Pytorch":
        return [value.detach().cpu().numpy() for _, value in _pytorch_items(model)]
    if model_type == "Huggingface":
        from peft import get_peft_model_state_dict

        return [value.detach().cpu().numpy() for value in get_peft_model_state_dict(model).values()]
    if model_type == "Tensorflow":
        raise NotImplementedError("Tensorflow client-side parameter initialization is not implemented.")
    raise ValueError(f"Unsupported FedOps model_type: {model_type}")


def set_parameters(model, model_type: str, parameters: Sequence[np.ndarray]) -> None:
    """Apply one FedOps parameter payload using the existing runtime semantics."""
    if model_type == "Pytorch":
        keys = [name for name, _ in _pytorch_items(model)]
        state_dict = OrderedDict(
            (name, torch.as_tensor(value))
            for name, value in zip(keys, parameters)
        )
        model.load_state_dict(state_dict, strict=False)
        return
    if model_type == "Huggingface":
        from peft import get_peft_model_state_dict, set_peft_model_state_dict

        keys = get_peft_model_state_dict(model).keys()
        state_dict = OrderedDict(
            (name, torch.as_tensor(value))
            for name, value in zip(keys, parameters)
        )
        set_peft_model_state_dict(model, state_dict)
        return
    if model_type == "Tensorflow":
        raise NotImplementedError("Tensorflow client-side parameter initialization is not implemented.")
    raise ValueError(f"Unsupported FedOps model_type: {model_type}")


def describe_parameters(model, model_type: str) -> list[dict[str, Any]]:
    """Describe the payload without exposing parameter values."""
    if model_type == "Pytorch":
        items = _pytorch_items(model)
    elif model_type == "Huggingface":
        from peft import get_peft_model_state_dict

        items = list(get_peft_model_state_dict(model).items())
    elif model_type == "Tensorflow":
        raise NotImplementedError("Tensorflow client-side parameter initialization is not implemented.")
    else:
        raise ValueError(f"Unsupported FedOps model_type: {model_type}")
    return [
        {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype).removeprefix("torch."),
            "elements": int(value.numel()),
        }
        for name, value in items
    ]


def parameter_signature(model, model_type: str) -> dict[str, Any]:
    """Return a value-free fingerprint for Registry compatibility checks."""
    tensors = describe_parameters(model, model_type)
    encoded = json.dumps(tensors, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schemaVersion": 1,
        "algorithm": "sha256",
        "fingerprint": hashlib.sha256(encoded).hexdigest(),
        "tensorCount": len(tensors),
        "elementCount": sum(item["elements"] for item in tensors),
        "tensors": tensors,
    }


def verify_parameter_round_trip(model, model_factory, model_type: str) -> dict[str, Any]:
    """Exercise the same get/set functions used by the running FedOps client."""
    values = [value.copy() for value in get_parameters(model, model_type)]
    restored = model_factory()
    set_parameters(restored, model_type, values)
    restored_values = get_parameters(restored, model_type)
    if len(values) != len(restored_values) or not all(
        left.shape == right.shape
        and left.dtype == right.dtype
        and np.array_equal(left, right)
        for left, right in zip(values, restored_values)
    ):
        raise ValueError("FedOps parameter round-trip changed model values")
    return {
        "ok": True,
        "signature": parameter_signature(model, model_type),
        "payloadBytes": sum(value.nbytes for value in values),
    }
