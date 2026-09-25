"""Cross-platform prompt-and-run entry point for the normal benchmark set."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_runner.config import (
    DEFAULT_URL,
    TASKS,
    benchmark_config,
    model_slug,
    write_json,
)


# Names re-exported from benchmark_runner.config for existing callers.
__all__ = ["DEFAULT_URL", "TASKS", "benchmark_config", "model_slug", "write_json"]


LAUNCHER = ROOT / "scripts" / "run_benchmark.py"
PORTABLE_COMPOSE = ROOT / "docker" / "compose.yaml"


def call(argv: list[str], *, environment=None) -> None:
    subprocess.run(argv, cwd=ROOT, env=environment, check=True)  # noqa: S603


def ensure_image(docker: str, image: str) -> None:
    inspection = subprocess.run(  # noqa: S603
        [docker, "image", "inspect", image],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if inspection.returncode == 0:
        return
    print(f"Docker image {image!r} is missing; building it now...")
    environment = os.environ.copy()
    environment["LM_EVAL_IMAGE"] = image
    call(
        [docker, "compose", "-f", str(PORTABLE_COMPOSE), "build"],
        environment=environment,
    )


def completed_summary(run_directory: Path) -> dict:
    path = run_directory / "completed" / "scores.json"
    with path.open(encoding="utf-8") as stream:
        results = json.load(stream)["results"]
    return {
        task: {**result["metrics"], **result.get("stderr", {})}
        for task, result in results.items()
    }


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(
        description=(
            "Prompt for a model when omitted, run a one-sample check, then run "
            "the full HumanEval, MBPP, IFEval and GSM8K datasets."
        )
    )
    cli.add_argument("model", nargs="?", help="Model identifier sent to the API")
    cli.add_argument("--base-url", help="Credential-free /v1/chat/completions URL")
    cli.add_argument(
        "--image",
        default=os.environ.get("LM_EVAL_IMAGE", "lm-eval-fork:local"),
    )
    cli.add_argument("--temperature", type=float, default=0.0)
    cli.add_argument("--max-gen-toks", type=int, default=4096)
    cli.add_argument("--reasoning-effort", choices=("low", "medium", "high"))
    cli.add_argument(
        "--smoke-only",
        action="store_true",
        help="Stop after one document from each task",
    )
    return cli


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


def run(args: argparse.Namespace) -> Path:
    docker = shutil.which("docker")
    if docker is None:
        raise RuntimeError("Docker Desktop is not installed or docker is not on PATH")
    ensure_image(docker, args.image)

    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    run_root = ROOT / "results" / "model-runs" / model_slug(args.model)
    run_root.mkdir(parents=True, exist_ok=True)
    smoke_config = run_root / f"{stamp}-smoke-config.json"
    full_config = run_root / f"{stamp}-full-config.json"
    smoke_run = run_root / f"{stamp}-smoke"
    full_run = run_root / f"{stamp}-full"

    write_json(smoke_config, benchmark_config(args, limit=1))
    print(f"\nRunning a one-sample smoke check for {args.model}...")
    call(
        [
            sys.executable,
            str(LAUNCHER),
            "--config",
            str(smoke_config),
            "--output-dir",
            str(smoke_run),
            "--image",
            args.image,
        ]
    )
    if args.smoke_only:
        return smoke_run

    write_json(full_config, benchmark_config(args))
    print("\nSmoke check passed. Running all four complete benchmarks...")
    call(
        [
            sys.executable,
            str(LAUNCHER),
            "--config",
            str(full_config),
            "--output-dir",
            str(full_run),
            "--image",
            args.image,
        ]
    )
    return full_run


def main() -> int:
    try:
        destination = run(resolve_prompts(parser().parse_args()))
        print(f"\nCompleted results: {destination / 'completed'}")
        print("\nScore summary:")
        print(json.dumps(completed_summary(destination), indent=2))
        return 0
    except (EOFError, OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Benchmark run failed: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nBenchmark run cancelled.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
