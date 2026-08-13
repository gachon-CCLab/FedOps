import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fedops.client.client_api import ClientMangerAPI, ClientServerAPI
from fedops.client.runtime_events import emit_runtime_event


class ClientRuntimeConfigurationTest(unittest.TestCase):
    def test_environment_overrides_runtime_endpoints(self):
        values = {
            "FEDOPS_CLIENT_MANAGER_URL": "http://127.0.0.1:18004/",
            "FEDOPS_SERVER_MANAGER_URL": "https://manager.example.test/",
            "FEDOPS_CLIENT_PERFORMANCE_URL": "https://metrics.example.test/",
            "FEDOPS_AGGREGATION_SERVER": "aggregator.example.test:24443",
        }
        with patch.dict(os.environ, values, clear=False):
            manager = ClientMangerAPI()
            server = ClientServerAPI("task-1")
            aggregation_server = server.get_port()
        self.assertEqual(manager.client_manager_addr, "http://127.0.0.1:18004")
        self.assertEqual(server.server_manager_url, "https://manager.example.test")
        self.assertEqual(server.client_performance_url, "https://metrics.example.test")
        self.assertEqual(aggregation_server, "aggregator.example.test:24443")

    def test_legacy_defaults_remain_available(self):
        names = [
            "FEDOPS_CLIENT_MANAGER_URL",
            "FEDOPS_SERVER_MANAGER_URL",
            "FEDOPS_CLIENT_PERFORMANCE_URL",
            "FEDOPS_AGGREGATION_SERVER",
        ]
        with patch.dict(os.environ, {name: "" for name in names}, clear=False):
            server = ClientServerAPI("task-1")
        self.assertEqual(server.server_manager_url, "http://ccl.gachon.ac.kr:40019")
        self.assertEqual(server.client_performance_url, "http://ccl.gachon.ac.kr:40015")

    def test_structured_event_is_optional_and_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            with patch.dict(os.environ, {
                "FEDOPS_EVENT_FILE": str(path),
                "FEDOPS_TASK_ID": "task-1",
                "FEDOPS_RELEASE_ID": "release-1",
                "FEDOPS_CLIENT_INSTANCE_ID": "client-opaque",
            }, clear=False):
                event = emit_runtime_event(
                    "training",
                    round_number=2,
                    progress=25,
                    metrics={"loss": 0.5},
                )
            stored = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(stored, event)
            self.assertEqual(stored["clientInstanceId"], "client-opaque")
            self.assertEqual(stored["metrics"]["loss"], 0.5)


if __name__ == "__main__":
    unittest.main()
