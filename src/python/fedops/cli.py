"""FedOps command line interface."""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Optional, Sequence

DEFAULT_LAUNCHER_IMAGE = "joseongjin311/fedops-launcher:v1.0.0"
DEFAULT_CONTAINER_NAME = "fedops-launcher"
DEFAULT_PORT = 5600
DEFAULT_WORKSPACE = Path.home() / "fedops-workspace"


class CliError(RuntimeError):
    """Expected command line failure with a user-facing message."""


def _log(message: str) -> None:
    print(f"[FedOps Launcher] {message}", flush=True)


def _run_command(
    command: Sequence[str],
    *,
    check: bool = False,
    capture: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def _command_to_text(command: Iterable[str]) -> str:
    return " ".join(str(part) for part in command)


def _docker_available() -> str:
    docker_path = shutil.which("docker")
    if not docker_path:
        raise CliError("Docker command not found. Please install Docker first.")

    result = _run_command([docker_path, "--version"])
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown error").strip()
        raise CliError(f"Docker version check failed: {detail}")

    version_text = (result.stdout or "").strip()
    _log(f"Docker installed: {version_text}")
    return docker_path


def _docker_daemon_ready(docker_path: str) -> None:
    result = _run_command([docker_path, "info", "--format", "{{.ServerVersion}}"])
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown error").strip()
        raise CliError(f"Docker daemon is not running or not reachable: {detail}")

    server_version = (result.stdout or "").strip() or "unknown"
    _log(f"Docker daemon: running (server={server_version})")


def _docker_image_ready(docker_path: str, image: str, dry_run: bool) -> None:
    inspect = _run_command([docker_path, "image", "inspect", image])
    if inspect.returncode == 0:
        _log(f"Docker image: already available ({image})")
        return

    if dry_run:
        _log(f"Docker image: not found; would pull {image}")
        return

    _log(f"Docker image: not found; pulling {image}")
    pull = _run_command([docker_path, "pull", image], capture=False)
    if pull.returncode != 0:
        raise CliError(f"Docker image pull failed: {image}")
    _log(f"Docker image: ready ({image})")


def run_fedops_launcher(args: argparse.Namespace) -> int:
    os_text = f"{platform.system()} {platform.release()} ({platform.machine()})"
    _log(f"OS: {os_text}")

    workspace = Path(args.workspace).expanduser().resolve()
    _log(f"Workspace: {workspace}")
    if not args.dry_run:
        workspace.mkdir(parents=True, exist_ok=True)

    docker_path = _docker_available()
    _docker_daemon_ready(docker_path)
    _docker_image_ready(docker_path, args.image, args.dry_run)

    remove_command = [docker_path, "rm", "-f", args.container_name]
    run_command = [
        docker_path,
        "run",
        "-d",
        "--name",
        args.container_name,
    ]
    if args.gpu:
        run_command.extend(["--gpus", "all"])
    run_command.extend(
        [
            "-p",
            f"{args.port}:5600",
            "-e",
            "WORKSPACE_DIR=/workspace",
            "-v",
            f"{workspace}:/workspace",
            args.image,
        ]
    )

    if args.dry_run:
        _log(f"Dry run: would remove existing container: {_command_to_text(remove_command)}")
        _log(f"Dry run: would start launcher: {_command_to_text(run_command)}")
        _log(f"Now Running On: http://127.0.0.1:{args.port} (dry-run preview)")
        return 0

    _log(f"Container cleanup: removing existing {args.container_name} if present")
    _run_command(remove_command)

    _log("Container start: launching FedOps Launcher")
    started = _run_command(run_command)
    if started.returncode != 0:
        detail = (started.stderr or started.stdout or "unknown error").strip()
        raise CliError(f"FedOps Launcher container start failed: {detail}")

    container_id = (started.stdout or "").strip()
    if container_id:
        _log(f"Container started: {container_id[:12]}")
    _log(f"Now Running On: http://127.0.0.1:{args.port}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fedops",
        description="FedOps command line tools.",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run FedOps helper services.")
    run_subparsers = run_parser.add_subparsers(dest="target")

    launcher_parser = run_subparsers.add_parser(
        "fedops-launcher",
        help="Start the FedOps Launcher Docker container.",
    )
    launcher_parser.add_argument("--gpu", action="store_true", help="Run the launcher with Docker GPU support.")
    launcher_parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Host port to bind to launcher port 5600.")
    launcher_parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE), help="Host workspace directory mounted to /workspace.")
    launcher_parser.add_argument("--image", default=DEFAULT_LAUNCHER_IMAGE, help="FedOps Launcher Docker image.")
    launcher_parser.add_argument("--container-name", default=DEFAULT_CONTAINER_NAME, help="Docker container name.")
    launcher_parser.add_argument("--dry-run", action="store_true", help="Print checks and Docker commands without starting the container.")
    launcher_parser.set_defaults(func=run_fedops_launcher)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        return 0

    try:
        return int(args.func(args))
    except CliError as exc:
        _log(f"ERROR: {exc}")
        return 1
    except KeyboardInterrupt:
        _log("Interrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
