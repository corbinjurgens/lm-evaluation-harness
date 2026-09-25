"""Host-only, Python 3.9+ launcher for immutable-image two-stage benchmarks.

Only completed/ is a published run. .partial/ retains bounded diagnostics on
failure. Docker reduces risk; it is not an absolute hostile-code sandbox.

Compatibility shim: the launcher lives in benchmark_runner.pipeline.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_runner import pipeline
from benchmark_runner.docker import (
    CONTAINER_TMP,
    LOG_LIMIT,
    WORKER,
    Docker,
    image_identity,
)
from benchmark_runner.io import private_text, safe_path


# Names re-exported for existing callers and tests.
__all__ = [
    "CONTAINER_TMP",
    "LOG_LIMIT",
    "WORKER",
    "Docker",
    "image_identity",
    "private_text",
    "safe_path",
]


def run(args, docker=None):
    """Generate and score; return the atomically published completed directory."""
    return pipeline.run_stage(
        args.config, args.output_dir, args.image, vars(args), docker
    )


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--config", required=True, type=Path)
    cli.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New directory; parent must exist",
    )
    cli.add_argument("--image", default="lm-eval-fork:local")
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
    return cli


def main():
    try:
        destination = run(parser().parse_args())
    except (
        OSError,
        ValueError,
        RuntimeError,
        subprocess.TimeoutExpired,
        KeyboardInterrupt,
    ) as error:
        detail = str(error)
        for key in ("OPENAI_API_KEY", "HF_TOKEN"):
            if os.environ.get(key):
                detail = detail.replace(os.environ[key], "[REDACTED]")
        print(
            f"Benchmark did not complete ({type(error).__name__}): {detail[:4096]}\n"
            "If the run directory exists, inspect its .partial diagnostics and container ID files.",
            file=sys.stderr,
        )
        return 1
    print(f"Completed benchmark: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
