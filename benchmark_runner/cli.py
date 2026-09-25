"""Command-line entry point for the benchmark runner.

Standard library only: this module must stay importable without lm_eval,
datasets, or torch on the host's stdlib-only virtualenv.
"""

from __future__ import annotations

import argparse
import sys


def _not_implemented(args: argparse.Namespace) -> int:
    print("not implemented", file=sys.stderr)
    return 2


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with its subcommands."""
    parser = argparse.ArgumentParser(prog="benchmark_runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a benchmark (not implemented)")
    run_parser.set_defaults(func=_not_implemented)

    worker_parser = subparsers.add_parser(
        "_worker", help="Internal worker entry point (not implemented)"
    )
    worker_parser.set_defaults(func=_not_implemented)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the selected subcommand."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
