import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import MagicMock, patch

from fedops import agent_studio_host, agent_studio_runner
from fedops.cli import build_parser


class AgentStudioCliTest(unittest.TestCase):
    def test_healthy_current_bridge_is_reused_without_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with (
                patch.object(
                    agent_studio_runner,
                    "_bridge_payload",
                    return_value={"workspace": str(root), "protocolVersion": 2},
                ),
                patch.object(agent_studio_runner, "_run") as run,
            ):
                token, port = agent_studio_runner._prepare_host_bridge(
                    root, False, 5602, runtime_dir=root
                )
            self.assertEqual((token, port), (root / "host-token", 5602))
            run.assert_not_called()

    def test_older_bridge_is_restarted_after_package_upgrade(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "host.pid").write_text("1234")
            with (
                patch.object(
                    agent_studio_runner,
                    "_bridge_payload",
                    side_effect=[
                        {"workspace": str(root)},
                        {"workspace": str(root), "protocolVersion": 2},
                    ],
                ),
                patch.object(agent_studio_runner, "_port_available", return_value=True),
                patch.object(
                    agent_studio_runner,
                    "_run",
                    return_value=subprocess.CompletedProcess([], 0, "", ""),
                ) as run,
            ):
                token, port = agent_studio_runner._prepare_host_bridge(
                    root, False, 5602, runtime_dir=root
                )
            self.assertEqual((token, port), (root / "host-token", 5602))
            self.assertEqual(run.call_count, 2)
            self.assertIn("--stop", run.call_args_list[0].args[0])
            self.assertIn("--daemon", run.call_args_list[1].args[0])

    def test_each_os_uses_its_native_folder_opener(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory).resolve()
            for platform in ("darwin", "win32", "linux"):
                with (
                    self.subTest(platform=platform),
                    patch.object(agent_studio_host.sys, "platform", platform),
                    patch.object(
                        agent_studio_host.os, "startfile", create=True
                    ) as startfile,
                    patch.object(
                        agent_studio_host.shutil,
                        "which",
                        return_value="/usr/bin/xdg-open",
                    ),
                    patch.object(agent_studio_host.subprocess, "Popen") as popen,
                ):
                    agent_studio_host.open_directory(target)
                    if platform == "win32":
                        startfile.assert_called_once_with(str(target))
                        popen.assert_not_called()
                    else:
                        executable = (
                            "open" if platform == "darwin" else "/usr/bin/xdg-open"
                        )
                        self.assertEqual(
                            popen.call_args.args[0], [executable, str(target)]
                        )
                        startfile.assert_not_called()

    def test_repair_host_does_not_pull_or_restart_studio(self):
        args = build_parser().parse_args(["run", "agent-studio", "--repair-host"])
        with (
            patch.object(agent_studio_runner, "print_banner"),
            patch.object(Path, "mkdir") as mkdir,
            patch.object(agent_studio_runner, "_docker_cli", return_value="docker"),
            patch.object(agent_studio_runner, "_ensure_daemon"),
            patch.object(
                agent_studio_runner, "_repair_host_bridge", return_value=0
            ) as repair,
            patch.object(agent_studio_runner, "_prepare_image") as pull,
            patch.object(agent_studio_runner, "_replace_container") as restart,
        ):
            self.assertEqual(agent_studio_runner.run_agent_studio(args), 0)
        repair.assert_called_once_with("docker", "fedops-agent-studio", False)
        pull.assert_not_called()
        restart.assert_not_called()
        mkdir.assert_not_called()

    def test_repair_preserves_the_running_containers_workspace_and_fallback_port(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            token = root / "host-token"
            token.write_text("test-token")
            container = {
                "State": {"Running": True},
                "Config": {
                    "Env": [
                        "STUDIO_HOST_WORKSPACE_DIR=" + str(root),
                        "STUDIO_FOLDER_OPENER_URL=http://host.docker.internal:5611",
                        "STUDIO_FOLDER_OPENER_TOKEN_FILE=/run/secrets/agent-studio-host-token",
                    ]
                },
                "Mounts": [
                    {
                        "Type": "bind",
                        "Destination": "/workspace",
                        "Source": "/host_mnt/some-vm-path",
                    },
                    {
                        "Type": "bind",
                        "Destination": "/run/secrets/agent-studio-host-token",
                        "Source": "/host_mnt/token",
                    },
                ],
            }
            result = subprocess.CompletedProcess([], 0, json.dumps([container]), "")
            with (
                patch.object(agent_studio_runner, "_run", return_value=result),
                patch.object(
                    agent_studio_runner, "_runtime_directory", return_value=root
                ),
                patch.object(
                    agent_studio_runner,
                    "_prepare_host_bridge",
                    return_value=(token, 5611),
                ) as prepare,
                patch.object(
                    agent_studio_runner, "_verify_container_bridge", return_value=True
                ),
            ):
                self.assertEqual(
                    agent_studio_runner._repair_host_bridge("docker", "studio", False),
                    0,
                )
            prepare.assert_called_once_with(
                root,
                False,
                5611,
                runtime_dir=root,
                token_file=token,
                allow_fallback=False,
            )

    def test_repair_cannot_fall_back_to_a_port_not_mounted_in_container(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch.object(agent_studio_runner, "_bridge_payload", return_value=None),
                patch.object(
                    agent_studio_runner, "_port_available", return_value=False
                ) as available,
                patch.object(agent_studio_runner, "_run") as run,
            ):
                token, port = agent_studio_runner._prepare_host_bridge(
                    root, False, 5611, runtime_dir=root, allow_fallback=False
                )
            self.assertIsNone(token)
            self.assertEqual(port, 5611)
            available.assert_called_once_with(5611)
            run.assert_not_called()

    def test_slow_hardware_and_system_proxy_do_not_block_host_health(self):
        release = threading.Event()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            token = root / "token"
            token.write_text("test-token")
            with patch.object(
                agent_studio_host,
                "collect_hardware",
                side_effect=lambda: release.wait(10),
            ):
                server = agent_studio_host.HostBridgeServer(
                    ("127.0.0.1", 0), root, "test-token"
                )
                worker = threading.Thread(target=server.serve_forever, daemon=True)
                worker.start()
                try:
                    with patch(
                        "urllib.request.getproxies",
                        return_value={"http": "http://127.0.0.1:1"},
                    ):
                        payload = agent_studio_runner._bridge_payload(
                            token, server.server_port
                        )
                    self.assertEqual(payload["workspace"], str(root.resolve()))
                    self.assertEqual(payload["protocolVersion"], 2)
                finally:
                    release.set()
                    server.shutdown()
                    server.server_close()
                    worker.join(2)

    def test_windows_daemon_is_detached_from_console_and_stdin(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = MagicMock(pid=1234)
            process.poll.return_value = None
            with (
                patch.object(agent_studio_host.sys, "platform", "win32"),
                patch.object(
                    subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200, create=True
                ),
                patch.object(subprocess, "DETACHED_PROCESS", 0x8, create=True),
                patch.object(
                    agent_studio_host.subprocess, "Popen", return_value=process
                ) as popen,
                patch.object(agent_studio_host, "_stop"),
                patch.object(agent_studio_host.time, "sleep"),
            ):
                result = agent_studio_host.main(
                    [
                        "--workspace",
                        str(root),
                        "--token-file",
                        str(root / "token"),
                        "--pid-file",
                        str(root / "host.pid"),
                        "--log-file",
                        str(root / "host.log"),
                        "--daemon",
                    ]
                )
            self.assertEqual(result, 0)
            self.assertEqual(popen.call_args.kwargs["creationflags"], 0x208)
            self.assertEqual(popen.call_args.kwargs["stdin"], subprocess.DEVNULL)
            self.assertTrue(popen.call_args.kwargs["close_fds"])

    def test_windows_stop_does_not_use_posix_zero_signal(self):
        with tempfile.TemporaryDirectory() as directory:
            pid = Path(directory) / "host.pid"
            pid.write_text("1234")
            with (
                patch.object(agent_studio_host.sys, "platform", "win32"),
                patch.object(agent_studio_host.os, "kill") as kill,
            ):
                agent_studio_host._stop(pid)
            self.assertEqual(kill.call_count, 1)
            self.assertNotEqual(kill.call_args.args[1], 0)
            self.assertFalse(pid.exists())

    def test_daemon_survives_launcher_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with agent_studio_runner.socket.socket() as reservation:
                reservation.bind(("127.0.0.1", 0))
                port = reservation.getsockname()[1]
            token, pid = root / "token", root / "host.pid"
            command = [
                sys.executable,
                "-m",
                "fedops.agent_studio_host",
                "--workspace",
                str(root),
                "--token-file",
                str(token),
                "--pid-file",
                str(pid),
                "--log-file",
                str(root / "host.log"),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--daemon",
            ]
            try:
                result = subprocess.run(
                    command, capture_output=True, text=True, timeout=5, check=False
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                for _ in range(20):
                    payload = agent_studio_runner._bridge_payload(token, port)
                    if payload:
                        break
                    agent_studio_runner.time.sleep(0.1)
                self.assertEqual(payload["status"], "ok")
                child_pid = int(pid.read_text())
                if sys.platform != "win32":
                    self.assertEqual(os.getsid(child_pid), child_pid)
            finally:
                agent_studio_host._stop(pid)

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
        self.assertIn("{}:/workspace".format(Path("/tmp/fedops-workspace")), command)
        self.assertIn(
            "type=volume,source=fedops-agent-studio-uv,target=/var/cache/fedops-uv",
            command,
        )
        self.assertIn("UV_CACHE_DIR=/var/cache/fedops-uv/cache", command)
        self.assertIn(
            "{}:/run/secrets/agent-studio-host-token:ro".format(
                Path("/tmp/host-token")
            ),
            command,
        )
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
