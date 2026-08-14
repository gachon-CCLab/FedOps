"""Cross-platform Docker runner for ``fedops run agent-studio``."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple


DEFAULT_IMAGE = "gachonccl/fedops-agent-studio:latest"
DEFAULT_CONTAINER = "fedops-agent-studio"
DEFAULT_STUDIO_PORT = 24368
AGENT_PORT_START = 24400
AGENT_PORT_END = 24499
DEFAULT_BRIDGE_PORT = 5602
DEFAULT_WORKSPACE = Path.home() / "fedops-workspace"


class AgentStudioError(RuntimeError):
    """Expected startup failure with a user-facing explanation."""


def _asset_path(filename: str) -> Path:
    return Path(__file__).resolve().parent / "assets" / filename


def print_banner() -> None:
    parts = [
        path.read_text(encoding="utf-8").rstrip()
        for path in (
            _asset_path("fedops_logo_ascii.txt"),
            _asset_path("fedops-agent-studio-ascii.txt"),
        )
        if path.is_file()
    ]
    print("\n\n".join(parts) or "FedOps Agent Studio", flush=True)


def _status(label: str, detail: str, state: str = "OK") -> None:
    print("[{:<5}] {:<22} {}".format(state, label, detail), flush=True)


def _run(
    command: Sequence[str],
    *,
    capture: bool = True,
    env: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(command),
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        env=env,
        check=False,
    )


def _command_text(command: Iterable[str]) -> str:
    return " ".join(str(part) for part in command)


def _docker_cli() -> str:
    docker = shutil.which("docker")
    if not docker:
        raise AgentStudioError(
            "Docker CLI was not found. Install Docker Desktop or Docker Engine first."
        )
    result = _run([docker, "--version"])
    if result.returncode != 0:
        raise AgentStudioError(
            (result.stderr or result.stdout or "Docker check failed").strip()
        )
    _status("Docker CLI", (result.stdout or "installed").strip())
    return docker


def _daemon_version(docker: str) -> Optional[str]:
    result = _run(
        [docker, "info", "--format", "{{.ServerVersion}}|{{.OSType}}|{{.Architecture}}"]
    )
    if result.returncode != 0:
        return None
    return (result.stdout or "").strip() or None


def _start_docker_application() -> bool:
    system = platform.system()
    if system == "Darwin":
        return _run(["open", "-a", "Docker"]).returncode == 0
    if system == "Windows":
        candidates = [
            Path(os.environ.get("ProgramFiles", "C:/Program Files"))
            / "Docker"
            / "Docker"
            / "Docker Desktop.exe",
            Path(os.environ.get("LOCALAPPDATA", "")) / "Docker" / "Docker Desktop.exe",
        ]
        executable = next((path for path in candidates if path.is_file()), None)
        if executable:
            subprocess.Popen(
                [str(executable)],
                creationflags=getattr(subprocess, "DETACHED_PROCESS", 0),
            )
            return True
        return False
    if system == "Linux":
        desktop = _run(["systemctl", "--user", "start", "docker-desktop"])
        if desktop.returncode == 0:
            return True
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            return _run(["systemctl", "start", "docker"]).returncode == 0
    return False


def _ensure_daemon(docker: str, start_docker: bool, timeout: int) -> str:
    version = _daemon_version(docker)
    if version:
        _status("Docker Engine", version)
        return version
    _status("Docker Engine", "not reachable", "WAIT")
    if not start_docker or not _start_docker_application():
        raise AgentStudioError(
            "Docker Engine is not running. Start Docker Desktop or Docker Engine and retry."
        )
    _status("Docker Engine", "starting Docker application", "WAIT")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(2)
        version = _daemon_version(docker)
        if version:
            _status("Docker Engine", version)
            return version
    raise AgentStudioError(
        "Docker Engine did not become ready within {} seconds.".format(timeout)
    )


def _image_id(docker: str, image: str) -> Optional[str]:
    result = _run([docker, "image", "inspect", "--format", "{{.Id}}", image])
    return (result.stdout or "").strip() if result.returncode == 0 else None


def _short_digest(value: str) -> str:
    prefix = "sha256:"
    return (value[len(prefix) :] if value.startswith(prefix) else value)[:12]


def _prepare_image(
    docker: str, image: str, pull: bool, dry_run: bool
) -> Tuple[Optional[str], Optional[str]]:
    previous = _image_id(docker, image)
    if dry_run:
        _status(
            "Agent Studio image",
            "would {} {}".format("pull" if pull else "use", image),
            "PLAN",
        )
        return previous, previous
    if pull:
        _status("Agent Studio image", "checking latest {}".format(image), "WAIT")
        result = _run([docker, "pull", image], capture=False)
        if result.returncode != 0:
            if previous:
                _status("Agent Studio image", "update failed; using cached image", "WARN")
                return previous, previous
            raise AgentStudioError("Could not pull Agent Studio image: {}".format(image))
    current = _image_id(docker, image)
    if not current:
        raise AgentStudioError("Agent Studio image is unavailable: {}".format(image))
    _status("Agent Studio image", "ready {} ({})".format(image, _short_digest(current)))
    return previous, current


def _nvidia_runtime_available(docker: str) -> bool:
    if platform.system() == "Darwin" or not shutil.which("nvidia-smi"):
        return False
    result = _run([docker, "info", "--format", "{{json .Runtimes}}"])
    if result.returncode != 0:
        return False
    try:
        runtimes = json.loads((result.stdout or "{}").strip())
    except ValueError:
        return False
    return isinstance(runtimes, dict) and "nvidia" in runtimes


def _use_nvidia(docker: str, requested: str) -> bool:
    available = _nvidia_runtime_available(docker)
    if requested == "nvidia" and not available:
        raise AgentStudioError(
            "NVIDIA GPU mode was requested, but Docker has no usable NVIDIA runtime. "
            "Install the NVIDIA Container Toolkit or use --gpu cpu."
        )
    enabled = requested == "nvidia" or (requested == "auto" and available)
    if enabled:
        _status("Container GPU", "NVIDIA runtime exposed with --gpus all")
    else:
        detail = "CPU mode"
        if platform.system() == "Darwin":
            detail += " (Docker Desktop cannot expose the macOS GPU)"
        _status("Container GPU", detail)
    return enabled


def _runtime_directory() -> Path:
    return Path.home() / ".fedops-agent-studio" / "runtime"


def _bridge_payload(token_file: Path, bridge_port: int) -> Optional[Dict[str, Any]]:
    try:
        token = token_file.read_text(encoding="utf-8").strip()
        request = urllib.request.Request(
            "http://127.0.0.1:{}/health".format(bridge_port),
            headers={"Authorization": "Bearer {}".format(token)},
        )
        with urllib.request.urlopen(request, timeout=1.5) as response:
            if response.status != 200:
                return None
            payload = json.loads(response.read().decode("utf-8"))
            return payload if isinstance(payload, dict) else None
    except (OSError, ValueError, urllib.error.URLError, urllib.error.HTTPError):
        return None


def _host_bridge_command(
    workspace: Path,
    token_file: Path,
    pid_file: Path,
    log_file: Path,
    bridge_port: int,
    *,
    stop: bool = False,
) -> Sequence[str]:
    command = [
        sys.executable,
        "-m",
        "fedops.agent_studio_host",
        "--workspace",
        str(workspace),
        "--token-file",
        str(token_file),
        "--port",
        str(bridge_port),
        "--pid-file",
        str(pid_file),
    ]
    if stop:
        command.append("--stop")
    else:
        command.extend(["--host", "0.0.0.0", "--daemon", "--log-file", str(log_file)])
    return command


def _port_available(port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", port))
    except OSError:
        return False
    return True


def _prepare_host_bridge(
    workspace: Path, dry_run: bool, bridge_port: int
) -> Tuple[Optional[Path], int]:
    workspace = workspace.expanduser().resolve()
    runtime_dir = _runtime_directory()
    token_file = runtime_dir / "host-token"
    pid_file = runtime_dir / "host.pid"
    log_file = runtime_dir / "host.log"
    if dry_run:
        _status("Host integration", "would use authenticated bridge on {}".format(bridge_port), "PLAN")
        return token_file, bridge_port
    runtime_dir.mkdir(parents=True, exist_ok=True)
    payload = _bridge_payload(token_file, bridge_port)
    if payload and payload.get("workspace") == str(workspace):
        _status("Host integration", "folder opener and hardware bridge running")
        return token_file, bridge_port
    if pid_file.is_file():
        _run(
            _host_bridge_command(
                workspace, token_file, pid_file, log_file, bridge_port, stop=True
            )
        )
    candidates = [bridge_port]
    candidates.extend(
        candidate
        for candidate in range(DEFAULT_BRIDGE_PORT, DEFAULT_BRIDGE_PORT + 20)
        if candidate != bridge_port
    )
    for candidate in candidates:
        if not _port_available(candidate):
            continue
        result = _run(
            _host_bridge_command(workspace, token_file, pid_file, log_file, candidate)
        )
        if result.returncode != 0:
            continue
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            payload = _bridge_payload(token_file, candidate)
            if payload and payload.get("workspace") == str(workspace):
                detail = "folder opener and hardware bridge running"
                if candidate != bridge_port:
                    detail += " on fallback port {}".format(candidate)
                _status("Host integration", detail)
                return token_file, candidate
            time.sleep(0.25)
        _run(
            _host_bridge_command(
                workspace, token_file, pid_file, log_file, candidate, stop=True
            )
        )
    _status("Host integration", "bridge unavailable; using container fallback", "WARN")
    return None, bridge_port


def build_container_command(
    docker: str,
    *,
    image: str,
    container_name: str,
    workspace: Path,
    studio_port: int,
    token_file: Optional[Path],
    bridge_port: int,
    nvidia: bool,
) -> Sequence[str]:
    command = [
        docker,
        "run",
        "-d",
        "--name",
        container_name,
        "--restart",
        "unless-stopped",
        "-p",
        "127.0.0.1:{}:24368".format(studio_port),
        "-p",
        "127.0.0.1:{}-{}:{}-{}".format(
            AGENT_PORT_START, AGENT_PORT_END, AGENT_PORT_START, AGENT_PORT_END
        ),
        "-v",
        "{}:/workspace".format(workspace),
        "-e",
        "UV_LINK_MODE=copy",
        "-e",
        "UV_CACHE_DIR=/workspace/.fedops-studio/uv/cache",
        "-e",
        "UV_PYTHON_INSTALL_DIR=/workspace/.fedops-studio/uv/python",
        "-e",
        "STUDIO_HOST_WORKSPACE_DIR={}".format(workspace),
    ]
    if platform.system() == "Linux":
        command.extend(["--add-host", "host.docker.internal:host-gateway"])
    if nvidia:
        command.extend(["--gpus", "all"])
    if token_file is not None:
        command.extend(
            [
                "-v",
                "{}:/run/secrets/agent-studio-host-token:ro".format(token_file),
                "-e",
                "STUDIO_FOLDER_OPENER_URL=http://host.docker.internal:{}".format(
                    bridge_port
                ),
                "-e",
                "STUDIO_FOLDER_OPENER_TOKEN_FILE=/run/secrets/agent-studio-host-token",
            ]
        )
    command.append(image)
    return command


def _replace_container(docker: str, container_name: str, dry_run: bool) -> None:
    if dry_run:
        _status("Existing container", "would replace {}".format(container_name), "PLAN")
        return
    inspect = _run([docker, "container", "inspect", container_name])
    if inspect.returncode != 0:
        return
    _status("Existing container", "stopping {}".format(container_name), "WAIT")
    _run([docker, "stop", "--timeout", "10", container_name])
    removed = _run([docker, "rm", container_name])
    if removed.returncode != 0:
        detail = (removed.stderr or removed.stdout or "unknown error").strip()
        raise AgentStudioError("Could not replace existing container: {}".format(detail))


def _stop_container(docker: str, container_name: str, dry_run: bool) -> None:
    inspect = _run(
        [docker, "container", "inspect", "--format", "{{.State.Running}}", container_name]
    )
    if inspect.returncode != 0:
        _status(
            "Agent Studio",
            "already stopped; container not found ({})".format(container_name),
            "INFO",
        )
        return
    if (inspect.stdout or "").strip().lower() != "true":
        _status("Agent Studio", "already stopped ({})".format(container_name), "INFO")
        return
    if dry_run:
        _status("Agent Studio", "would stop {}".format(container_name), "PLAN")
        return
    _status("Agent Studio", "stopping {}".format(container_name), "WAIT")
    stopped = _run([docker, "stop", "--timeout", "10", container_name])
    if stopped.returncode != 0:
        detail = (stopped.stderr or stopped.stdout or "unknown error").strip()
        raise AgentStudioError("Could not stop Agent Studio: {}".format(detail))
    state = _run(
        [docker, "container", "inspect", "--format", "{{.State.Running}}", container_name]
    )
    if state.returncode == 0 and (state.stdout or "").strip().lower() == "true":
        raise AgentStudioError("Agent Studio container is still running after the stop request.")
    _status("Agent Studio", "stopped successfully ({})".format(container_name))


def _stop_host_bridge(workspace: Path, bridge_port: int, dry_run: bool) -> None:
    runtime_dir = _runtime_directory()
    token_file = runtime_dir / "host-token"
    pid_file = runtime_dir / "host.pid"
    log_file = runtime_dir / "host.log"
    if not pid_file.is_file():
        _status("Host integration", "already stopped", "INFO")
        return
    if dry_run:
        _status("Host integration", "would stop host bridge", "PLAN")
        return
    result = _run(
        _host_bridge_command(
            workspace, token_file, pid_file, log_file, bridge_port, stop=True
        )
    )
    if result.returncode == 0:
        _status("Host integration", "stopped successfully")
    else:
        detail = (result.stderr or result.stdout or "unknown error").strip()
        _status("Host integration", "could not stop: {}".format(detail), "WARN")


def _remove_previous_image(
    docker: str, previous: Optional[str], current: Optional[str]
) -> None:
    if not previous or previous == current:
        return
    result = _run([docker, "image", "rm", previous])
    if result.returncode == 0:
        _status("Previous image", "removed {}".format(_short_digest(previous)))
    else:
        _status("Previous image", "still referenced; kept safely", "WARN")


def _wait_for_studio(port: int, timeout: int = 45) -> None:
    url = "http://127.0.0.1:{}/api/v1/health".format(port)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.5) as response:
                if response.status == 200:
                    _status("Agent Studio", "ready at http://localhost:{}".format(port))
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(0.5)
    raise AgentStudioError(
        "Agent Studio did not become ready within {} seconds.".format(timeout)
    )


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--container-name", default=DEFAULT_CONTAINER)
    parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    parser.add_argument("--port", type=int, default=DEFAULT_STUDIO_PORT)
    parser.add_argument("--gpu", choices=("auto", "cpu", "nvidia"), default="auto")
    parser.add_argument("--no-pull", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--no-start-docker", action="store_true")
    parser.add_argument("--docker-timeout", type=int, default=120)
    parser.add_argument("--bridge-port", type=int, default=DEFAULT_BRIDGE_PORT)
    parser.add_argument("--dry-run", action="store_true")


def add_stop_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--container-name", default=DEFAULT_CONTAINER)
    parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    parser.add_argument("--no-start-docker", action="store_true")
    parser.add_argument("--docker-timeout", type=int, default=120)
    parser.add_argument("--bridge-port", type=int, default=DEFAULT_BRIDGE_PORT)
    parser.add_argument("--dry-run", action="store_true")


def _validate_args(args: argparse.Namespace) -> None:
    for label, value in (("Studio port", args.port), ("Host bridge port", args.bridge_port)):
        if not 1 <= value <= 65535:
            raise AgentStudioError("{} must be between 1 and 65535.".format(label))
    if AGENT_PORT_START <= args.port <= AGENT_PORT_END:
        raise AgentStudioError("Studio port must not overlap Agent serving ports.")
    if args.bridge_port == args.port or AGENT_PORT_START <= args.bridge_port <= AGENT_PORT_END:
        raise AgentStudioError("Host bridge port must not overlap Studio or Agent ports.")


def run_agent_studio(args: argparse.Namespace) -> int:
    print_banner()
    try:
        _validate_args(args)
        workspace = Path(args.workspace).expanduser().resolve()
        system = "{} {} ({})".format(
            platform.system(), platform.release(), platform.machine()
        )
        _status("Host", system)
        _status("Workspace", str(workspace))
        if not args.dry_run:
            workspace.mkdir(parents=True, exist_ok=True)
        docker = _docker_cli()
        _ensure_daemon(docker, not args.no_start_docker, args.docker_timeout)
        previous, current = _prepare_image(
            docker, args.image, not args.no_pull, args.dry_run
        )
        nvidia = _use_nvidia(docker, args.gpu)
        token_file, active_bridge_port = _prepare_host_bridge(
            workspace, args.dry_run, args.bridge_port
        )
        command = build_container_command(
            docker,
            image=args.image,
            container_name=args.container_name,
            workspace=workspace,
            studio_port=args.port,
            token_file=token_file,
            bridge_port=active_bridge_port,
            nvidia=nvidia,
        )
        _replace_container(docker, args.container_name, args.dry_run)
        if args.dry_run:
            _status("Container start", _command_text(command), "PLAN")
            return 0
        started = _run(command)
        if started.returncode != 0:
            detail = (started.stderr or started.stdout or "unknown error").strip()
            raise AgentStudioError("Container start failed: {}".format(detail))
        container_id = (started.stdout or "").strip()
        _status("Container", "started {}".format(container_id[:12]))
        try:
            _wait_for_studio(args.port)
        except AgentStudioError:
            logs = _run([docker, "logs", "--tail", "60", args.container_name])
            detail = (logs.stdout or logs.stderr or "").strip()
            if detail:
                _status("Container logs", detail, "ERROR")
            raise
        _remove_previous_image(docker, previous, current)
        url = "http://localhost:{}".format(args.port)
        if not args.no_browser:
            webbrowser.open(url)
            _status("Browser", "opened {}".format(url))
        _status("Stop command", "fedops stop agent-studio", "INFO")
        return 0
    except AgentStudioError as error:
        _status("Startup failed", str(error), "ERROR")
        return 1
    except KeyboardInterrupt:
        _status("Startup", "interrupted", "STOP")
        return 130


def stop_agent_studio(args: argparse.Namespace) -> int:
    try:
        workspace = Path(args.workspace).expanduser().resolve()
        docker = _docker_cli()
        _ensure_daemon(docker, not args.no_start_docker, args.docker_timeout)
        _stop_container(docker, args.container_name, args.dry_run)
        _stop_host_bridge(workspace, args.bridge_port, args.dry_run)
        return 0
    except AgentStudioError as error:
        _status("Stop failed", str(error), "ERROR")
        return 1
    except KeyboardInterrupt:
        _status("Stop", "interrupted", "STOP")
        return 130


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fedops run agent-studio",
        description="Start FedOps Agent Studio on this device.",
    )
    add_arguments(parser)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run_agent_studio(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
