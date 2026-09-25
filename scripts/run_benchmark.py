"""Host-only, Python 3.9+ launcher for immutable-image two-stage benchmarks.

Only completed/ is a published run. .partial/ retains bounded diagnostics on
failure. Docker reduces risk; it is not an absolute hostile-code sandbox.

Compatibility shim: forwards to `scripts/benchmark.py run --config`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_runner import cli, pipeline
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
    launcher = argparse.ArgumentParser(description=__doc__)
    launcher.add_argument("--config", required=True, type=Path)
    launcher.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New directory; parent must exist",
    )
    launcher.add_argument("--image", default="lm-eval-fork:local")
    cli.add_resource_arguments(launcher)
    return launcher


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    # Keep the old required flags and image default; `run` owns the rest.
    args = parser().parse_args(argv)
    return cli.main(["run", *argv, "--image", args.image])


if __name__ == "__main__":
    raise SystemExit(main())
