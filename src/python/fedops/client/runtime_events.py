"""Structured local events for FedOps participant runtimes.

The event sink is optional. Legacy clients that do not set ``FEDOPS_EVENT_FILE``
keep their existing behaviour, while Agent Studio can render the current user's
client lifecycle without parsing human-readable logs.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import threading
from typing import Any


logger = logging.getLogger(__name__)
_WRITE_LOCK = threading.Lock()


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_value(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def emit_runtime_event(
    stage: str,
    *,
    task_id: str | None = None,
    round_number: int | None = None,
    progress: float | None = None,
    message: str | None = None,
    metrics: dict[str, Any] | None = None,
    **details: Any,
) -> dict[str, Any] | None:
    """Append one best-effort JSONL event when an event sink is configured."""

    destination = os.environ.get("FEDOPS_EVENT_FILE", "").strip()
    if not destination:
        return None
    event: dict[str, Any] = {
        "schemaVersion": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "taskId": task_id or os.environ.get("FEDOPS_TASK_ID") or None,
        "releaseId": os.environ.get("FEDOPS_RELEASE_ID") or None,
        "clientInstanceId": os.environ.get("FEDOPS_CLIENT_INSTANCE_ID") or None,
        "stage": stage,
    }
    if round_number is not None:
        event["round"] = int(round_number)
    if progress is not None:
        event["progress"] = float(progress)
    if message:
        event["message"] = str(message)
    if metrics:
        event["metrics"] = _json_value(metrics)
    event.update({key: _json_value(value) for key, value in details.items()})
    event = {key: value for key, value in event.items() if value is not None}

    path = Path(destination).expanduser()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
        with _WRITE_LOCK, path.open("a", encoding="utf-8") as output:
            output.write(line)
            output.flush()
    except OSError as error:
        logger.warning("Unable to write FedOps runtime event: %s", error)
        return None
    return event


__all__ = ["emit_runtime_event"]
