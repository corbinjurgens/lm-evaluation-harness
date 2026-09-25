"""Internal worker operations, run inside the scoring/generation containers.

Reached only through `benchmark_runner.cli.main(["_worker", ...])`, which the
public parser never advertises. `metadata` and `idle` are standard library
only; `generate` and `score` import `benchmark_runner.runtime.bundle`, which
requires lm_eval, datasets, and torch and must stay out of this module's
top-level imports so the other operations remain usable without them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


LOCK_PATH = Path("/opt/lm-eval/requirements.lock")


def _metadata(args: argparse.Namespace) -> int:
    import hashlib
    import platform

    from benchmark_runner.runtime.bundle import source_identity, validate_config

    if args.config is not None:
        validate_config(json.loads(Path(args.config).read_text(encoding="utf-8")))
    payload = {"source": source_identity(), "python": platform.python_version()}
    if LOCK_PATH.exists():
        payload["lock_sha256"] = hashlib.sha256(LOCK_PATH.read_bytes()).hexdigest()
    print(json.dumps(payload))
    return 0


def _idle(args: argparse.Namespace) -> int:
    import time

    time.sleep(604800)
    return 0


def _stream_artifact(args: argparse.Namespace) -> int:
    import os
    import shutil
    import stat

    fd = os.open(args.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("Artifact is not a regular file")
    if info.st_size > args.byte_limit:
        raise ValueError("Artifact exceeds transfer limit")
    stream = os.fdopen(fd, "rb")
    shutil.copyfileobj(stream, sys.stdout.buffer, 65536)
    return 0


def _generate(args: argparse.Namespace) -> int:
    from benchmark_runner.runtime.bundle import generate

    generate(
        json.loads(Path(args.config).read_text(encoding="utf-8")),
        args.output,
        args.transport_log,
    )
    return 0


def _score(args: argparse.Namespace) -> int:
    from benchmark_runner.runtime.bundle import score

    score(json.loads(Path(args.input).read_text(encoding="utf-8")), args.output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the worker's own argument parser for its five operations."""
    parser = argparse.ArgumentParser(prog="benchmark_runner _worker")
    ops = parser.add_subparsers(dest="op", required=True)

    metadata = ops.add_parser("metadata")
    metadata.add_argument("--config")
    metadata.set_defaults(func=_metadata)

    idle = ops.add_parser("idle")
    idle.set_defaults(func=_idle)

    stream_artifact = ops.add_parser("stream-artifact")
    stream_artifact.add_argument("path")
    stream_artifact.add_argument("byte_limit", type=int)
    stream_artifact.set_defaults(func=_stream_artifact)

    generation = ops.add_parser("generate")
    for flag in ("config", "output", "transport-log"):
        generation.add_argument(f"--{flag}", required=True, dest=flag.replace("-", "_"))
    generation.set_defaults(func=_generate)

    scoring = ops.add_parser("score")
    for flag in ("input", "output"):
        scoring.add_argument(f"--{flag}", required=True)
    scoring.set_defaults(func=_score)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse worker arguments and dispatch to the selected operation."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
