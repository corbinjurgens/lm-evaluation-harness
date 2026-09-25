"""Two-stage benchmark pipeline: generate, score offline, publish atomically.

Standard library only. Only completed/ is a published run. .partial/ retains
bounded diagnostics on failure.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from urllib.parse import urlsplit

from benchmark_runner.docker import (
    BUNDLE_PATH,
    SCORE_PATH,
    TRANSPORT_PATH,
    WORKER,
    Docker,
    OwnedContainers,
    check_worker,
    image_identity,
    limits,
    remaining,
    scoring_inspection,
)
from benchmark_runner.io import json_read, private_text, safe_path


def checked_config(path):
    config = json_read(path)
    allowed = {
        "tasks",
        "model",
        "base_url",
        "gen_kwargs",
        "seed",
        "timeout",
        "num_concurrent",
        "max_retries",
        "limit",
        "samples",
        "num_fewshot",
        "profile",
        "server",
    }
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError(
            "Unknown config fields; credentials belong only in generation environment"
        )
    url = urlsplit(config.get("base_url", ""))
    if (
        url.scheme not in {"http", "https"}
        or not url.hostname
        or url.username
        or url.password
        or url.query
        or url.fragment
        or url.path.rstrip("/") != "/v1/chat/completions"
    ):
        raise ValueError("Use a credential-free /v1/chat/completions URL")

    # The image validates the complete schema before inference. Reject credential
    # fields recursively here, before copying any user config into run artifacts.
    def no_credentials(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if any(
                    word in key.lower()
                    for word in (
                        "password",
                        "secret",
                        "token",
                        "api_key",
                        "header",
                        "authorization",
                    )
                ):
                    raise ValueError("Credentials are not permitted in configuration")
                no_credentials(child)
        elif isinstance(value, list):
            for child in value:
                no_credentials(child)

    no_credentials(config)
    return config


def run_stage(config_path, output_dir, image, resources, docker=None):
    """Generate and score; return the atomically published completed directory."""
    resources = limits(resources)
    config_path, output = safe_path(config_path), safe_path(output_dir)
    config = checked_config(config_path)
    docker = docker or Docker()
    identity = image_identity(docker, image)
    check_worker(docker, image, identity["id"], resources)
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    partial = output / ".partial"
    partial.mkdir(mode=0o700)
    docker.log_directory = partial
    staged_config = partial / "config.json"
    private_text(staged_config, json.dumps(config, allow_nan=False))
    staged_config.chmod(0o644)  # File-only bind mount; parent remains host-private.
    containers = OwnedContainers(docker, identity["id"], resources, partial)

    try:
        generation_deadline = time.monotonic() + resources["generation_timeout"]
        print("Starting generation container...", file=sys.stderr, flush=True)
        generation = containers.create(
            "generate", staged_config, remaining(generation_deadline)
        )
        metadata = json.loads(
            docker.call(
                [
                    "exec",
                    generation,
                    *WORKER,
                    "metadata",
                    "--config",
                    "/input/config.json",
                ],
                timeout=remaining(generation_deadline),
            )
        )
        if "lock_sha256" not in metadata:
            raise ValueError("Image metadata lacks lock_sha256")
        try:
            docker.call(
                [
                    "exec",
                    generation,
                    *WORKER,
                    "generate",
                    "--config",
                    "/input/config.json",
                    "--output",
                    BUNDLE_PATH,
                    "--transport-log",
                    TRANSPORT_PATH,
                ],
                timeout=remaining(generation_deadline),
                log=partial / "generation.log",
                stream=True,
            )
        finally:
            # Missing diagnostics should not hide the primary generation error.
            try:
                containers.copy(
                    generation,
                    TRANSPORT_PATH,
                    partial / "transport.jsonl",
                    generation_deadline,
                )
            except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired):
                private_text(
                    partial / "transport-unavailable.txt",
                    "Transport copy was unavailable; inspect generation.log.\n",
                )
        containers.copy(
            generation, BUNDLE_PATH, partial / "bundle.json", generation_deadline
        )
        bundle = json_read(partial / "bundle.json")
        if bundle.get("format") != "lm-eval-offline-bundle-v1" or bundle.get(
            "source"
        ) != metadata.get("source"):
            raise ValueError("Generated bundle identity mismatch")
        containers.remove(generation)
        print("Generation finished. Scoring offline...", file=sys.stderr, flush=True)
        score_input = partial / "score-input.json"
        shutil.copyfile(partial / "bundle.json", score_input)
        score_input.chmod(0o444)
        scoring_deadline = time.monotonic() + resources["scoring_timeout"]
        scoring = containers.create("score", score_input, remaining(scoring_deadline))
        actual_isolation = scoring_inspection(
            docker, scoring, identity["id"], resources
        )
        docker.call(
            [
                "exec",
                scoring,
                *WORKER,
                "score",
                "--input",
                "/input/score-input.json",
                "--output",
                SCORE_PATH,
            ],
            timeout=remaining(scoring_deadline),
            log=partial / "scoring.log",
            stream=True,
        )
        containers.copy(
            scoring,
            SCORE_PATH,
            partial / "scores.pending.json",
            scoring_deadline,
        )
        result = json_read(partial / "scores.pending.json")
        if (
            result.get("format") != "lm-eval-offline-results-v1"
            or result.get("bundle_sha256") != bundle.get("sha256")
            or result.get("source") != bundle.get("source")
        ):
            raise ValueError("Scored result identity mismatch")
        containers.cleanup()  # Failed cleanup must not look like a fully completed run.
        publication = partial / "publication"
        publication.mkdir(mode=0o700)
        for source, target in (
            ("bundle.json", "bundle.json"),
            ("scores.pending.json", "scores.json"),
            ("transport.jsonl", "transport.jsonl"),
        ):
            shutil.copyfile(partial / source, publication / target)
            (publication / target).chmod(0o600)
        manifest = {
            "format": "lm-eval-isolated-run-v1",
            "status": "complete",
            "image": identity,
            "runtime": metadata,
            "limits": resources,
            "bundle_sha256": bundle["sha256"],
            "generation": bundle["provenance"],
            "server": config.get("server", "unknown"),
            "tasks": {
                name: {
                    key: value[key]
                    for key in ("version", "repeats", "dataset", "doc_ids")
                }
                for name, value in bundle["tasks"].items()
            },
            "score_sha256": hashlib.sha256(
                (publication / "scores.json").read_bytes()
            ).hexdigest(),
            "isolation": actual_isolation,
        }
        private_text(
            publication / "manifest.json",
            json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        )
        publication.rename(output / "completed")
        return output / "completed"
    finally:
        containers.cleanup()
