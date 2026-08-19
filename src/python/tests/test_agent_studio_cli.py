import io
from contextlib import redirect_stderr
from pathlib import Path
import tempfile
import unittest

from fedops import agent_studio_host, agent_studio_runner
from fedops.cli import build_parser


class AgentStudioCliTest(unittest.TestCase):
    def test_cli_registers_agent_studio(self):
        parser = build_parser()

        studio = parser.parse_args(["run", "agent-studio", "--dry-run"])

        self.assertIs(studio.func, agent_studio_runner.run_agent_studio)
        self.assertEqual(studio.image, "gachonccl/fedops-agent-studio:latest")
        self.assertEqual(studio.port, 24368)

    def test_legacy_launcher_target_is_removed(self):
        parser = build_parser()

        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["run", "fedops-launcher"])

    def test_agent_studio_stop_mode_is_available(self):
        args = build_parser().parse_args(["stop", "agent-studio"])

        self.assertIs(args.func, agent_studio_runner.stop_agent_studio)

    def test_run_agent_studio_does_not_accept_the_old_stop_option(self):
        parser = build_parser()

        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["run", "agent-studio", "--stop"])

    def test_cpu_command_mounts_workspace_agent_ports_and_host_token(self):
        command = agent_studio_runner.build_container_command(
            "docker",
            image="gachonccl/fedops-agent-studio:latest",
            container_name="fedops-agent-studio",
            workspace=Path("/tmp/fedops-workspace"),
            studio_port=24368,
            token_file=Path("/tmp/host-token"),
            bridge_port=5602,
            nvidia=False,
        )

        self.assertIn("0.0.0.0:24368:24368", command)
        self.assertIn("0.0.0.0:24400-24499:24400-24499", command)
        self.assertIn("/tmp/fedops-workspace:/workspace", command)
        self.assertIn(
            "type=volume,source=fedops-agent-studio-uv,target=/var/cache/fedops-uv",
            command,
        )
        self.assertIn("UV_CACHE_DIR=/var/cache/fedops-uv/cache", command)
        self.assertIn("/tmp/host-token:/run/secrets/agent-studio-host-token:ro", command)
        self.assertNotIn("--gpus", command)

    def test_local_only_bind_address_remains_available(self):
        command = agent_studio_runner.build_container_command(
            "docker",
            image="image",
            container_name="studio",
            workspace=Path("/tmp/workspace"),
            studio_port=24368,
            token_file=None,
            bridge_port=5602,
            nvidia=False,
            bind_address="127.0.0.1",
        )

        self.assertIn("127.0.0.1:24368:24368", command)
        self.assertIn("127.0.0.1:24400-24499:24400-24499", command)

    def test_nvidia_command_exposes_all_gpus(self):
        command = agent_studio_runner.build_container_command(
            "docker",
            image="image",
            container_name="studio",
            workspace=Path("/tmp/workspace"),
            studio_port=24368,
            token_file=None,
            bridge_port=5602,
            nvidia=True,
        )

        self.assertEqual(command[command.index("--gpus") + 1], "all")

    def test_port_validation_rejects_agent_serving_overlap(self):
        args = agent_studio_runner.build_parser().parse_args(["--port", "24400"])

        with self.assertRaises(agent_studio_runner.AgentStudioError):
            agent_studio_runner._validate_args(args)

    def test_host_token_and_hardware_contract_use_no_third_party_packages(self):
        with tempfile.TemporaryDirectory() as directory:
            token_path = Path(directory) / "token"
            first = agent_studio_host.ensure_token(token_path)
            second = agent_studio_host.ensure_token(token_path)

        hardware = agent_studio_host.collect_hardware()
        self.assertEqual(first, second)
        self.assertEqual(hardware["source"], "host")
        self.assertIn("platform", hardware)
        self.assertIn("cpu", hardware)
        self.assertIn("memory", hardware)
        self.assertIn("gpu", hardware)

    def test_ascii_banner_is_part_of_the_fedops_package(self):
        assets = Path(agent_studio_runner.__file__).parent / "assets"
        logo = assets / "fedops_logo_ascii.txt"
        wordmark = assets / "fedops-agent-studio-ascii.txt"

        self.assertTrue(logo.read_text(encoding="utf-8").strip())
        self.assertIn("agent studio", wordmark.read_text(encoding="utf-8").casefold())


if __name__ == "__main__":
    unittest.main()
