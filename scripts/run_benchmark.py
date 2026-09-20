"""Host-only, Python 3.9+ launcher for immutable-image two-stage benchmarks.

Only completed/ is a published run. .partial/ retains bounded diagnostics on
failure. Docker reduces risk; it is not an absolute hostile-code sandbox.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit


LOG_LIMIT = 1024 * 1024
# This is a private, size-bounded container tmpfs, never the host's shared /tmp.
CONTAINER_TMP = "/tmp"  # noqa: S108
BUNDLE_PATH = f"{CONTAINER_TMP}/bundle.json"
TRANSPORT_PATH = f"{CONTAINER_TMP}/transport.jsonl"
SCORE_PATH = f"{CONTAINER_TMP}/scores.json"
STREAM_ARTIFACT = (
    "import os,shutil,stat,sys;"
    "fd=os.open(sys.argv[1],os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK);"
    "info=os.fstat(fd);"
    "\nif not stat.S_ISREG(info.st_mode): raise ValueError('Artifact is not a regular file')"
    "\nif info.st_size>int(sys.argv[2]): raise ValueError('Artifact exceeds transfer limit')"
    "\nstream=os.fdopen(fd,'rb');"
    "shutil.copyfileobj(stream,sys.stdout.buffer,65536)"
)
KEEPALIVE = "import time; time.sleep(604800)"
METADATA = (
    "import hashlib,json,pathlib,platform;"
    "from lm_eval.benchmark_bundle import source_identity,validate_config;"
    "validate_config(json.loads(pathlib.Path('/input/config.json').read_text()));"
    "print(json.dumps({'source':source_identity(),'python':platform.python_version(),"
    "'lock_sha256':hashlib.sha256(pathlib.Path('/opt/lm-eval/requirements.lock').read_bytes()).hexdigest()}))"
)
IMAGE_ENV = {
    "PATH",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONUNBUFFERED",
    "NLTK_DATA",
    "HF_HOME",
    "LANG",
    "LC_ALL",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "PYTHON_VERSION",
}


class Docker:
    """One fixed executable; no shell, bounded captured output and deadlines."""

    def __init__(self):
        executable = shutil.which("docker")
        if executable is None:
            raise RuntimeError("Docker executable not found")
        self.executable = str(Path(executable).resolve())
        self.log_directory = None
        self.operation = 0
        self.secrets = [
            os.environ[k] for k in ("OPENAI_API_KEY", "HF_TOKEN") if os.environ.get(k)
        ]

    def call(self, args, timeout=60, log=None, binary_output=None, max_bytes=None):
        if binary_output is None:
            return self._call(args, timeout, log)
        if not isinstance(max_bytes, int) or max_bytes <= 0:
            raise ValueError("Binary transfer requires a positive byte bound")
        created = completed = False
        try:
            with Path(binary_output).open("xb") as transfer:
                created = True
                Path(binary_output).chmod(0o600)
                output = self._call(args, timeout, log, transfer, max_bytes)
            completed = True
            return output
        finally:
            if created and not completed:
                Path(binary_output).unlink()

    def _call(self, args, timeout, log, transfer=None, max_bytes=None):
        if not args or args[0] not in {
            "image",
            "create",
            "start",
            "exec",
            "rm",
            "inspect",
        }:
            raise ValueError("Unsupported Docker operation")
        captured = bytearray()
        transfer_errors = []
        transferred = 0
        committed = False
        self.operation += 1
        if log is None and self.log_directory is not None:
            log = self.log_directory / f"docker-{self.operation:03d}-{args[0]}.log"
        # Evaluator-only, validated fixed executable/argv; never invokes a shell.
        process = subprocess.Popen(  # noqa: S603
            [self.executable, *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        def drain():
            nonlocal transferred
            try:
                while True:
                    chunk = process.stdout.read(65536)
                    if not chunk:
                        break
                    captured.extend(chunk[: max(0, LOG_LIMIT - len(captured))])
                    if transfer is not None:
                        transferred += len(chunk)
                        if transferred > max_bytes:
                            raise ValueError("Artifact exceeds transfer limit")
                        transfer.write(chunk)
            except (OSError, ValueError) as error:
                transfer_errors.append(error)
                process.kill()

        reader = threading.Thread(target=drain, daemon=True)
        reader.start()
        try:
            process.wait(timeout=timeout)
            reader.join()
            if transfer_errors:
                raise transfer_errors[0]
            if transfer is not None and not process.returncode:
                transfer.flush()
                os.fsync(transfer.fileno())
                committed = True
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            reader.join()
            process.stdout.close()
            output = captured.decode("utf-8", errors="replace")
            for secret in self.secrets:
                output = output.replace(secret, "[REDACTED]")
            if log is not None:
                logged = output
                if transfer is not None:
                    logged = f"Binary artifact transfer: {transferred} bytes; complete={committed}.\n"
                    if not committed:
                        logged += output[-2000:]
                if "inspect" in args:
                    try:
                        records = json.loads(output)
                        for record in records:
                            config = record.get("Config", {})
                            if "Env" in config:
                                config["Env"] = [
                                    value.split("=", 1)[0]
                                    for value in config["Env"] or []
                                ]
                        logged = json.dumps(records)
                    except (ValueError, TypeError, AttributeError):
                        logged = "Inspect output was not valid JSON; omitted to avoid exposing environment values.\n"
                private_text(
                    log,
                    logged.encode("utf-8")[:LOG_LIMIT].decode("utf-8", errors="ignore"),
                )
        if process.returncode:
            raise RuntimeError(f"Docker {args[0]} failed: {output[-2000:]}")
        return output


def private_text(path, text):
    with Path(path).open("x", encoding="utf-8") as stream:
        Path(path).chmod(0o600)
        stream.write(text)


def json_read(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def safe_path(path):
    path = Path(path).expanduser().absolute()
    if any(c in str(path) for c in (",", "\n", "\r", "\x00")):
        raise ValueError(
            "Docker mount paths cannot contain commas or control characters"
        )
    # Reject symlink ancestors as well as the leaf, including dangling links.
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Use real paths, not symlinks")
    return path


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


def image_identity(docker, image):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/@:+-]*", image):
        raise ValueError("Invalid image reference")
    records = json.loads(docker.call(["image", "inspect", image]))
    if len(records) != 1 or not re.fullmatch(r"sha256:[0-9a-f]{64}", records[0]["Id"]):
        raise ValueError("Expected one immutable image ID")
    record = records[0]
    if record["Config"].get("User") != "65532:65532":
        raise ValueError("Expected the fork's nonroot image")
    if record["Config"].get("Entrypoint") or record["Config"].get("Volumes"):
        raise ValueError("Image must have no entrypoint or declared volumes")
    if any(
        item.split("=", 1)[0] not in IMAGE_ENV
        for item in record["Config"].get("Env", [])
    ):
        raise ValueError("Image contains unexpected environment variables")
    return {"id": record["Id"], "repo_digests": record.get("RepoDigests") or []}


def limits(args):
    for name, lower, upper in (
        ("memory_mib", 256, 1048576),
        ("tmpfs_mib", 64, 1048576),
        ("pids_limit", 32, 4096),
        ("generation_timeout", 1, 604800),
        ("scoring_timeout", 1, 604800),
        ("max_artifact_mib", 1, 1048576),
    ):
        if not lower <= getattr(args, name) <= upper:
            raise ValueError(f"{name} must be between {lower} and {upper}")
    if not math.isfinite(args.cpus) or not 0.1 <= args.cpus <= 256:
        raise ValueError("cpus must be between 0.1 and 256")
    if args.tmpfs_mib >= args.memory_mib:
        raise ValueError("tmpfs_mib must be smaller than memory_mib")
    return {
        key: getattr(args, key)
        for key in (
            "memory_mib",
            "tmpfs_mib",
            "pids_limit",
            "cpus",
            "generation_timeout",
            "scoring_timeout",
            "max_artifact_mib",
        )
    }


def tmpfs_options(size_mib):
    """Share the exact requested private-scratch policy with its inspect guard."""
    return f"rw,nosuid,nodev,size={size_mib}m,mode=1777"


def scoring_inspection(docker, container, image, resources):
    record = json.loads(docker.call(["inspect", container]))[0]
    host, config = record["HostConfig"], record["Config"]
    mounts = [
        {key: mount.get(key) for key in ("Type", "Destination", "RW")}
        for mount in record.get("Mounts", [])
    ]
    environment = sorted(item.split("=", 1)[0] for item in config.get("Env", []))
    expected_host = {
        "PidsLimit": resources["pids_limit"],
        "Memory": resources["memory_mib"] * 1024 * 1024,
        "MemorySwap": resources["memory_mib"] * 1024 * 1024,
        "NanoCpus": round(resources["cpus"] * 1000000000),
        "Tmpfs": {CONTAINER_TMP: tmpfs_options(resources["tmpfs_mib"])},
    }
    if (
        record["Image"] != image
        or config.get("User") != "65532:65532"
        or host.get("NetworkMode") != "none"
        or not host.get("ReadonlyRootfs")
        or host.get("CapDrop") != ["ALL"]
        or not host.get("Init")
        or host.get("SecurityOpt")
        not in (["no-new-privileges"], ["no-new-privileges=true"])
        or any(host.get(key) != value for key, value in expected_host.items())
        or mounts
        != [{"Type": "bind", "Destination": "/input/score-input.json", "RW": False}]
        or set(environment)
        - IMAGE_ENV
        - {"HF_ALLOW_CODE_EVAL", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE"}
    ):
        raise ValueError(
            "Actual scoring container isolation differs from required configuration"
        )
    return {
        "image": record["Image"],
        "user": config["User"],
        "environment_keys": environment,
        "mounts": mounts,
        "host": {
            key: host.get(key)
            for key in (
                "NetworkMode",
                "ReadonlyRootfs",
                "CapDrop",
                "SecurityOpt",
                "Init",
                "PidsLimit",
                "Memory",
                "MemorySwap",
                "NanoCpus",
                "Tmpfs",
            )
        },
    }


def run(args, docker=None):
    """Generate and score; return the atomically published completed directory."""
    resources = limits(args)
    config_path, output = safe_path(args.config), safe_path(args.output_dir)
    config = checked_config(config_path)
    docker = docker or Docker()
    identity = image_identity(docker, args.image)
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    partial = output / ".partial"
    partial.mkdir(mode=0o700)
    docker.log_directory = partial
    staged_config = partial / "config.json"
    private_text(staged_config, json.dumps(config, allow_nan=False))
    staged_config.chmod(0o644)  # File-only bind mount; parent remains host-private.
    owner = uuid.uuid4().hex
    containers = []
    pending_name = None

    def create(phase, source, timeout):
        nonlocal pending_name
        pending_name = f"lm-eval-{owner}-{phase}"
        private_text(partial / f"{phase}-container-name.txt", pending_name + "\n")
        argv = [
            "create",
            "--name",
            pending_name,
            "--label",
            f"lm-eval.run={owner}",
            "--user",
            "65532:65532",
            "--network",
            "none" if phase == "score" else "bridge",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--init",
            "--pids-limit",
            str(args.pids_limit),
            "--memory",
            f"{args.memory_mib}m",
            "--memory-swap",
            f"{args.memory_mib}m",
            "--cpus",
            str(args.cpus),
            "--tmpfs",
            f"{CONTAINER_TMP}:{tmpfs_options(args.tmpfs_mib)}",
            "--log-driver",
            "none",
            "--mount",
            f"type=bind,src={source},dst=/input/{source.name},readonly",
        ]
        if phase == "score":
            argv += [
                "--env",
                "HF_ALLOW_CODE_EVAL=1",
                "--env",
                "HF_HUB_OFFLINE=1",
                "--env",
                "HF_DATASETS_OFFLINE=1",
            ]
        else:
            argv += ["--env", "HF_ALLOW_CODE_EVAL="]
            for key in ("OPENAI_API_KEY", "HF_TOKEN"):
                if os.environ.get(key):
                    argv += ["--env", key]
        argv += [identity["id"], "python", "-c", KEEPALIVE]
        container = docker.call(argv, timeout=min(timeout, 60)).strip()
        if not re.fullmatch(r"[0-9a-f]{64}", container):
            raise ValueError("Docker did not return a full container ID")
        containers.append(container)
        private_text(partial / f"{phase}-container-id.txt", container + "\n")
        pending_name = None
        docker.call(["start", container])
        return container

    def remaining(deadline):
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise subprocess.TimeoutExpired("benchmark phase", 0)
        return seconds

    def copy(container, source, destination, deadline):
        byte_limit = args.max_artifact_mib * 1024 * 1024
        docker.call(
            [
                "exec",
                container,
                "python",
                "-c",
                STREAM_ARTIFACT,
                source,
                str(byte_limit),
            ],
            timeout=min(60, remaining(deadline)),
            binary_output=destination,
            max_bytes=byte_limit,
        )
        if destination.is_symlink() or not destination.is_file():
            raise ValueError("Expected a regular artifact file")
        destination.chmod(0o600)

    def cleanup():
        nonlocal pending_name
        errors = []
        # A create timeout can happen after Docker allocated the container. Only
        # recover our random name when its independently checked owner label agrees.
        if pending_name is not None:
            try:
                records = json.loads(docker.call(["inspect", pending_name]))
                for record in records:
                    if record["Config"].get("Labels", {}).get("lm-eval.run") != owner:
                        raise RuntimeError("Container cleanup ownership mismatch")
                    container = record["Id"]
                    if not re.fullmatch(r"[0-9a-f]{64}", container):
                        raise RuntimeError("Invalid cleanup container ID")
                    containers.append(container)
            except (
                RuntimeError,
                OSError,
                ValueError,
                subprocess.TimeoutExpired,
            ) as error:
                errors.append(error)
            pending_name = None
        for container in containers[:]:
            try:
                docker.call(["rm", "--force", container])
                containers.remove(container)
            except (RuntimeError, OSError, subprocess.TimeoutExpired) as error:
                errors.append(error)
        if errors:
            raise RuntimeError(
                "Owned-container cleanup failed; inspect Docker before retrying"
            ) from errors[0]

    try:
        generation_deadline = time.monotonic() + args.generation_timeout
        generation = create("generate", staged_config, remaining(generation_deadline))
        metadata = json.loads(
            docker.call(
                ["exec", generation, "python", "-c", METADATA],
                timeout=remaining(generation_deadline),
            )
        )
        try:
            docker.call(
                [
                    "exec",
                    generation,
                    "python",
                    "-m",
                    "lm_eval.benchmark_bundle",
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
            )
        finally:
            # Missing diagnostics should not hide the primary generation error.
            try:
                copy(
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
        copy(generation, BUNDLE_PATH, partial / "bundle.json", generation_deadline)
        bundle = json_read(partial / "bundle.json")
        if bundle.get("format") != "lm-eval-offline-bundle-v1" or bundle.get(
            "source"
        ) != metadata.get("source"):
            raise ValueError("Generated bundle identity mismatch")
        docker.call(["rm", "--force", generation])
        containers.remove(generation)
        score_input = partial / "score-input.json"
        shutil.copyfile(partial / "bundle.json", score_input)
        score_input.chmod(0o444)
        scoring_deadline = time.monotonic() + args.scoring_timeout
        scoring = create("score", score_input, remaining(scoring_deadline))
        actual_isolation = scoring_inspection(
            docker, scoring, identity["id"], resources
        )
        docker.call(
            [
                "exec",
                scoring,
                "python",
                "-m",
                "lm_eval.benchmark_bundle",
                "score",
                "--input",
                "/input/score-input.json",
                "--output",
                SCORE_PATH,
            ],
            timeout=remaining(scoring_deadline),
            log=partial / "scoring.log",
        )
        copy(
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
        cleanup()  # Failed cleanup must not look like a fully completed run.
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
        cleanup()


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
