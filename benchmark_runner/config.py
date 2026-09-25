"""Benchmark configuration building blocks shared across entry points."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import argparse
    from pathlib import Path


DEFAULT_URL = "http://host.docker.internal:8888/v1/chat/completions"
TASKS = ["humaneval", "mbpp", "ifeval", "gsm8k_cot_zeroshot"]


def model_slug(model: str) -> str:
    """Return a stable cross-platform folder component for a model name."""
    value = re.sub(r"[^a-z0-9._-]+", "-", model.lower()).strip("-")
    if not value:
        raise ValueError("The model name cannot produce a result-folder name")
    return value


def benchmark_config(args: argparse.Namespace, limit=None) -> dict:
    """Build one launcher config without credentials or host-specific paths."""
    gen_kwargs: dict[str, object] = {
        "temperature": args.temperature,
        "max_gen_toks": args.max_gen_toks,
    }
    if args.reasoning_effort is not None:
        gen_kwargs["reasoning_effort"] = args.reasoning_effort
    config = {
        "tasks": TASKS,
        "model": args.model,
        "base_url": args.base_url,
        "gen_kwargs": gen_kwargs,
        "seed": 1234,
        "timeout": 300,
        "num_concurrent": 1,
        "max_retries": 3,
        "profile": "practical",
    }
    if limit is not None:
        config["limit"] = limit
    return config


def write_json(path: Path, value: object) -> None:
    """Create, rather than overwrite, one human-readable run input."""
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def canonical_json(value: object) -> bytes:
    """Return a deterministic UTF-8 JSON encoding for hashing and comparison."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
