"""Authenticated host integration used by the local Agent Studio container."""

from __future__ import annotations

import argparse
import ctypes
import hmac
import json
import os
import platform
import re
import secrets
import shutil
import signal
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _run_text(command: Sequence[str], timeout: int = 8) -> str:
    try:
        completed = subprocess.run(
            list(command),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            creationflags=(
                getattr(subprocess, "CREATE_NO_WINDOW", 0)
                if sys.platform == "win32"
                else 0
            ),
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout.strip()


def _cpu_model(system: str) -> str:
    if system == "Darwin":
        value = _run_text(["sysctl", "-n", "machdep.cpu.brand_string"])
        if value:
            return value
    if system == "Windows":
        value = os.environ.get("PROCESSOR_IDENTIFIER", "").strip()
        if value:
            return value
    if system == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.lower().startswith(("model name", "hardware")) and ":" in line:
                    value = line.split(":", 1)[1].strip()
                    if value:
                        return value
        except OSError:
            pass
    return (platform.processor() or platform.machine() or "Unknown CPU").strip()


def _physical_memory_bytes(system: str) -> int:
    if system == "Darwin":
        value = _run_text(["sysctl", "-n", "hw.memsize"])
        if value.isdigit():
            return int(value)
    if system == "Windows":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memoryLoad", ctypes.c_ulong),
                ("totalPhysical", ctypes.c_ulonglong),
                ("availablePhysical", ctypes.c_ulonglong),
                ("totalPageFile", ctypes.c_ulonglong),
                ("availablePageFile", ctypes.c_ulonglong),
                ("totalVirtual", ctypes.c_ulonglong),
                ("availableVirtual", ctypes.c_ulonglong),
                ("availableExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        try:
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
                return int(status.totalPhysical)
        except (AttributeError, OSError):
            pass
    try:
        return int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return 0


def _memory_string_bytes(value: Any) -> Optional[int]:
    match = re.search(r"([0-9.]+)\s*(TB|GB|MB|KB)", str(value or ""), re.IGNORECASE)
    if not match:
        return None
    units = {"KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
    return int(float(match.group(1)) * units[match.group(2).upper()])


def _nvidia_devices() -> List[Dict[str, Any]]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return []
    output = _run_text(
        [
            executable,
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    devices = []
    for line in output.splitlines():
        name, separator, memory = line.partition(",")
        if not separator:
            continue
        try:
            memory_bytes = int(float(memory.strip()) * 1024**2)
        except ValueError:
            memory_bytes = None
        devices.append(
            {"name": name.strip() or "NVIDIA GPU", "memoryBytes": memory_bytes, "computeUnits": None}
        )
    return devices


def _gpu_devices(system: str) -> List[Dict[str, Any]]:
    nvidia = _nvidia_devices()
    if nvidia:
        return nvidia
    if system == "Darwin":
        output = _run_text(["system_profiler", "SPDisplaysDataType", "-json"], timeout=12)
        try:
            items = json.loads(output).get("SPDisplaysDataType", [])
        except (AttributeError, ValueError):
            return []
        devices = []
        seen = set()
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("sppci_model") or item.get("_name") or "").strip()
            if not name or name in seen:
                continue
            seen.add(name)
            cores = item.get("sppci_cores")
            try:
                compute_units = int(cores) if cores is not None else None
            except (TypeError, ValueError):
                compute_units = None
            devices.append(
                {
                    "name": name,
                    "memoryBytes": _memory_string_bytes(
                        item.get("spdisplays_vram") or item.get("_spdisplays_vram")
                    ),
                    "computeUnits": compute_units,
                }
            )
        return devices
    if system == "Windows":
        executable = shutil.which("powershell") or shutil.which("pwsh")
        if not executable:
            return []
        output = _run_text(
            [
                executable,
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM | ConvertTo-Json -Compress",
            ],
            timeout=10,
        )
        try:
            payload = json.loads(output)
        except ValueError:
            return []
        items = payload if isinstance(payload, list) else [payload]
        return [
            {
                "name": str(item.get("Name")),
                "memoryBytes": item.get("AdapterRAM"),
                "computeUnits": None,
            }
            for item in items
            if isinstance(item, dict) and item.get("Name")
        ]
    if system == "Linux":
        executable = shutil.which("lspci")
        if not executable:
            return []
        devices = []
        for line in _run_text([executable, "-mm"]).splitlines():
            if re.search(r'"(?:VGA compatible|3D|Display) controller"', line, re.IGNORECASE):
                devices.append({"name": line.strip(), "memoryBytes": None, "computeUnits": None})
        return devices
    return []


def collect_hardware() -> Dict[str, Any]:
    system = platform.system() or "Unknown"
    devices = _gpu_devices(system)
    return {
        "source": "host",
        "platform": {
            "system": "macOS" if system == "Darwin" else system,
            "release": platform.release() or "Unknown",
            "architecture": platform.machine() or "Unknown",
        },
        "cpu": {"model": _cpu_model(system), "logicalCores": max(1, os.cpu_count() or 1)},
        "memory": {"totalBytes": max(0, _physical_memory_bytes(system))},
        "gpu": {"detected": bool(devices), "devices": devices},
    }


def open_directory(directory: Path) -> None:
    target = directory.expanduser().resolve()
    if not target.is_dir():
        raise FileNotFoundError("The local directory was not found.")
    if sys.platform == "darwin":
        command = ["open", str(target)]
    elif sys.platform == "win32":
        os.startfile(str(target))  # type: ignore[attr-defined]
        return
    else:
        executable = shutil.which("xdg-open")
        if not executable:
            raise FileNotFoundError("xdg-open is not installed on this Linux host.")
        command = [executable, str(target)]
    subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


class HostBridgeServer(ThreadingHTTPServer):
    def __init__(self, address: Tuple[str, int], workspace: Path, token: str):
        super().__init__(address, HostBridgeHandler)
        self.workspace = workspace.resolve()
        self.token = token
        self.hardware = collect_hardware()


class HostBridgeHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path not in {"/health", "/hardware"}:
            self._json(404, {"error": "not found"})
            return
        if not self._authorized():
            self._json(401, {"error": "unauthorized"})
            return
        server = self.server  # type: HostBridgeServer
        if self.path == "/hardware":
            self._json(200, server.hardware)
            return
        self._json(200, {"status": "ok", "workspace": str(server.workspace)})

    def do_POST(self) -> None:
        if self.path != "/open":
            self._json(404, {"error": "not found"})
            return
        if not self._authorized():
            self._json(401, {"error": "unauthorized"})
            return
        server = self.server  # type: HostBridgeServer
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 4096:
                raise ValueError("invalid request size")
            payload = json.loads(self.rfile.read(length))
            relative = Path(str(payload.get("relativePath", "")))
            if relative.is_absolute() or not relative.parts or ".." in relative.parts:
                raise ValueError("invalid relative path")
            target = (server.workspace / relative).resolve()
            target.relative_to(server.workspace)
            workspace_project = (
                len(relative.parts) == 4
                and relative.parts[0] == "accounts"
                and relative.parts[1].startswith("account-")
                and relative.parts[2] == "projects"
            )
            task_data = (
                len(relative.parts) == 6
                and relative.parts[0] == "accounts"
                and relative.parts[1].startswith("account-")
                and relative.parts[2] == ".local-data"
                and relative.parts[3] == "federated-tasks"
                and not relative.parts[4].startswith(".")
                and relative.parts[5] == "dataset"
            )
            if not workspace_project and not task_data:
                raise ValueError("only account-scoped project and Task data directories can be opened")
            open_directory(target)
        except (OSError, ValueError, TypeError) as error:
            self._json(400, {"error": str(error)})
            return
        self._json(200, {"opened": True})

    def _authorized(self) -> bool:
        server = self.server  # type: HostBridgeServer
        header = self.headers.get("Authorization", "")
        prefix = "Bearer "
        provided = header[len(prefix) :] if header.startswith(prefix) else ""
        return bool(provided) and hmac.compare_digest(provided, server.token)

    def _json(self, status: int, payload: Dict[str, Any]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: Any) -> None:
        print("[agent-studio-host] " + (format % args), flush=True)


def ensure_token(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        path.write_text(secrets.token_urlsafe(32) + "\n", encoding="utf-8")
        if os.name != "nt":
            path.chmod(0o600)
    token = path.read_text(encoding="utf-8").strip()
    if not token:
        raise ValueError("Agent Studio host token is empty.")
    return token


def _remove_pid_file(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _stop(pid_file: Path) -> None:
    if not pid_file.is_file():
        return
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
        os.kill(pid, signal.SIGTERM)
    except (OSError, ValueError, ProcessLookupError):
        _remove_pid_file(pid_file)
        return
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        except OSError:
            break
        time.sleep(0.05)
    _remove_pid_file(pid_file)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FedOps Agent Studio host integration")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5602)
    parser.add_argument("--daemon", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--pid-file", type=Path, required=True)
    parser.add_argument("--log-file", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    workspace = args.workspace.expanduser().resolve()
    token_file = args.token_file.expanduser().resolve()
    pid_file = args.pid_file.expanduser().resolve()
    if args.stop:
        _stop(pid_file)
        return 0
    if not workspace.is_dir():
        raise FileNotFoundError("Workspace does not exist: {}".format(workspace))
    ensure_token(token_file)
    if args.daemon:
        if args.log_file is None:
            raise ValueError("--daemon requires --log-file")
        _stop(pid_file)
        command = [
            sys.executable,
            "-m",
            "fedops.agent_studio_host",
            "--workspace",
            str(workspace),
            "--token-file",
            str(token_file),
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--pid-file",
            str(pid_file),
        ]
        log_file = args.log_file.expanduser().resolve()
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("ab") as log:
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=sys.platform != "win32",
                creationflags=(
                    getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                    if sys.platform == "win32"
                    else 0
                ),
            )
        pid_file.write_text("{}\n".format(process.pid), encoding="utf-8")
        time.sleep(0.2)
        if process.poll() is not None:
            _remove_pid_file(pid_file)
            raise RuntimeError("Agent Studio host integration failed; inspect {}".format(log_file))
        return 0
    token = ensure_token(token_file)
    server = HostBridgeServer((args.host, args.port), workspace, token)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
