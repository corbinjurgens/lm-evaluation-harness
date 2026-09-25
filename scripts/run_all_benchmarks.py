"""Cross-platform prompt-and-run entry point for the normal benchmark set.

Compatibility shim: forwards to `scripts/benchmark.py run`.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_runner import cli
from benchmark_runner.cli import resolve_prompts
from benchmark_runner.config import (
    DEFAULT_URL,
    TASKS,
    benchmark_config,
    model_slug,
    write_json,
)
from benchmark_runner.pipeline import (
    PORTABLE_COMPOSE,
    call,
    completed_summary,
    ensure_image,
)


# Names re-exported from benchmark_runner for existing callers.
__all__ = [
    "DEFAULT_URL",
    "PORTABLE_COMPOSE",
    "TASKS",
    "benchmark_config",
    "call",
    "completed_summary",
    "ensure_image",
    "model_slug",
    "resolve_prompts",
    "write_json",
]


def main() -> int:
    return cli.main(["run", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
