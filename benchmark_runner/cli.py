"""Command-line entry point for the benchmark runner.

Standard library only: this module must stay importable without lm_eval,
datasets, or torch on the host's stdlib-only virtualenv.
"""

from __future__ import annotations

import argparse
import getpass
import os
import subprocess
import sys
from pathlib import Path

from benchmark_runner import pipeline
from benchmark_runner.config import DEFAULT_URL
from benchmark_runner.docker import SECRET_ENV


# Model and generation arguments with their defaults; --config fixes them all.
GENERATION_DEFAULTS = {
    "model": None,
    "base_url": None,
    "temperature": 0.0,
    "max_gen_toks": 4096,
    "reasoning_effort": None,
}


def add_resource_arguments(cli: argparse.ArgumentParser) -> None:
    """Container resource, deadline and artifact-limit flags of every stage."""
    cli.add_argument("--memory-mib", type=int, default=512)
    cli.add_argument("--tmpfs-mib", type=int, default=128)
    cli.add_argument("--pids-limit", type=int, default=128)
    cli.add_argument("--cpus", type=float, default=1.0)
    cli.add_argument("--generation-timeout", type=int, default=86400)
    cli.add_argument("--scoring-timeout", type=int, default=86400)
    cli.add_argument(
        "--max-artifact-mib",
        type=int,
        default=128,
        help="Maximum bytes per transferred artifact; overflow fails the run",
    )


def add_run_arguments(cli: argparse.ArgumentParser) -> None:
    cli.add_argument("model", nargs="?", help="Model identifier sent to the API")
    cli.add_argument("--base-url", help="Credential-free /v1/chat/completions URL")
    cli.add_argument(
        "--image",
        default=os.environ.get("LM_EVAL_IMAGE", "lm-eval-fork:local"),
    )
    # No argparse defaults here: None marks "not given" so --config can reject
    # it. GENERATION_DEFAULTS fills them for the smoke-then-full route.
    cli.add_argument("--temperature", type=float, help="Default: 0.0")
    cli.add_argument("--max-gen-toks", type=int, help="Default: 4096")
    cli.add_argument("--reasoning-effort", choices=("low", "medium", "high"))
    cli.add_argument(
        "--smoke-only",
        action="store_true",
        help="Stop after one document from each task",
    )
    cli.add_argument(
        "--config",
        type=Path,
        help=(
            "Run one complete single-stage JSON config instead of the "
            "smoke-then-full sequence; requires --output-dir"
        ),
    )
    cli.add_argument(
        "--output-dir",
        type=Path,
        help="With --config: new run directory; parent must exist",
    )
    add_resource_arguments(cli)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with its public subcommands.

    `_worker` is not a public subcommand: it is intercepted in `main` before
    this parser is ever built, so it does not appear here or in `--help`.
    """
    parser = argparse.ArgumentParser(prog="benchmark_runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Run the benchmark set, or one stage from --config",
        description=(
            "Prompt for a model when omitted, run a one-sample check, then run "
            "the full HumanEval, MBPP, IFEval and GSM8K datasets."
        ),
    )
    add_run_arguments(run_parser)
    run_parser.set_defaults(func=run, parser=run_parser)

    return parser


def checked_run_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> argparse.Namespace:
    """Reject flags that do not belong to the chosen route; fill defaults."""
    if args.config is not None:
        given = [
            name for name in GENERATION_DEFAULTS if getattr(args, name) is not None
        ]
        if given:
            parser.error(
                "--config fixes the model and generation settings; remove "
                + ", ".join(
                    "MODEL" if name == "model" else "--" + name.replace("_", "-")
                    for name in given
                )
            )
        if args.output_dir is None:
            parser.error("--config requires --output-dir")
        if args.smoke_only:
            parser.error("--smoke-only does not apply to --config")
        return args
    if args.output_dir is not None:
        parser.error("--output-dir requires --config")
    for name, default in GENERATION_DEFAULTS.items():
        if getattr(args, name) is None:
            setattr(args, name, default)
    return args


def resolve_prompts(args: argparse.Namespace) -> argparse.Namespace:
    if args.model is None:
        args.model = input("Model name: ").strip()
    if not args.model:
        raise ValueError("A model name is required")
    if args.base_url is None:
        entered = input(f"API URL [{DEFAULT_URL}]: ").strip()
        args.base_url = entered or DEFAULT_URL
    if not os.environ.get("OPENAI_API_KEY"):
        entered = getpass.getpass(
            "API key (press Enter for an unauthenticated local server): "
        )
        os.environ["OPENAI_API_KEY"] = entered or "local-no-auth"
    return args


def redacted(text: str) -> str:
    for key in SECRET_ENV:
        if os.environ.get(key):
            text = text.replace(os.environ[key], "[REDACTED]")
    return text


def run(args: argparse.Namespace) -> int:
    args = checked_run_arguments(args.parser, args)
    try:
        if args.config is None:
            pipeline.run_campaign(resolve_prompts(args))
        else:
            pipeline.run_configured(args)
        return 0
    except (
        EOFError,
        OSError,
        RuntimeError,
        ValueError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as error:
        print(
            f"Benchmark run failed ({type(error).__name__}): "
            f"{redacted(str(error))[:4096]}\n"
            "If the run directory exists, inspect its .partial diagnostics and "
            "container ID files.",
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:
        print("\nBenchmark run cancelled.", file=sys.stderr)
        return 130


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the selected subcommand.

    `_worker` is handled here, ahead of the public parser, and imports the
    worker module lazily so the public CLI path never pays for it.
    """
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "_worker":
        from benchmark_runner.runtime import worker

        return worker.main(argv[1:])
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
