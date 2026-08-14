"""FedOps command line interface."""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from .agent_studio_runner import add_arguments as add_agent_studio_arguments
from .agent_studio_runner import add_stop_arguments as add_agent_studio_stop_arguments
from .agent_studio_runner import run_agent_studio, stop_agent_studio


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fedops",
        description="FedOps command line tools.",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run FedOps services.")
    run_subparsers = run_parser.add_subparsers(dest="target")
    agent_studio_parser = run_subparsers.add_parser(
        "agent-studio",
        help="Start the FedOps Agent Studio Docker container.",
    )
    add_agent_studio_arguments(agent_studio_parser)
    agent_studio_parser.set_defaults(func=run_agent_studio)

    stop_parser = subparsers.add_parser("stop", help="Stop FedOps services.")
    stop_subparsers = stop_parser.add_subparsers(dest="target")
    agent_studio_stop_parser = stop_subparsers.add_parser(
        "agent-studio",
        help="Stop the FedOps Agent Studio Docker container.",
    )
    add_agent_studio_stop_arguments(agent_studio_stop_parser)
    agent_studio_stop_parser.set_defaults(func=stop_agent_studio)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("[STOP ] FedOps                 interrupted", flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
