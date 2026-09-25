"""Docker client and owned-container lifecycle for the two-stage launcher.

Standard library only. Docker reduces risk; it is not an absolute
hostile-code sandbox.
"""

from __future__ import annotations

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

from benchmark_runner.io import StreamRedactor, private_text, redact


LOG_LIMIT = 1024 * 1024
# This is a private, size-bounded container tmpfs, never the host's shared /tmp.
CONTAINER_TMP = "/tmp"  # noqa: S108
BUNDLE_PATH = f"{CONTAINER_TMP}/bundle.json"
TRANSPORT_PATH = f"{CONTAINER_TMP}/transport.jsonl"
SCORE_PATH = f"{CONTAINER_TMP}/scores.json"
WORKER = ["python", "-m", "benchmark_runner", "_worker"]
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
SCORING_ENV = {
    "HF_ALLOW_CODE_EVAL": "1",
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
}
SECRET_ENV = ("OPENAI_API_KEY", "HF_TOKEN")
RESOURCE_BOUNDS = (
    ("memory_mib", 256, 1048576),
    ("tmpfs_mib", 64, 1048576),
    ("pids_limit", 32, 4096),
    ("generation_timeout", 1, 604800),
    ("scoring_timeout", 1, 604800),
    ("max_artifact_mib", 1, 1048576),
)
RESOURCE_KEYS = (
    "memory_mib",
    "tmpfs_mib",
    "pids_limit",
    "cpus",
    "generation_timeout",
    "scoring_timeout",
    "max_artifact_mib",
)


class Docker:
    """One fixed executable; no shell, bounded captured output and deadlines."""

    def __init__(self):
        executable = shutil.which("docker")
        if executable is None:
            raise RuntimeError("Docker executable not found")
        self.executable = str(Path(executable).resolve())
        self.log_directory = None
        self.operation = 0
        self.secrets = [os.environ[k] for k in SECRET_ENV if os.environ.get(k)]

    def call(
        self,
        args,
        timeout=60,
        log=None,
        binary_output=None,
        max_bytes=None,
        stream=False,
    ):
        if binary_output is None:
            return self._call(args, timeout, log, stream=stream)
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

    def redact(self, text):
        return redact(text, self.secrets)

    def _call(self, args, timeout, log, transfer=None, max_bytes=None, stream=False):
        if not args or args[0] not in {
            "image",
            "create",
            "run",
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
            redactor = StreamRedactor(self.secrets)
            echoing = stream
            try:
                while True:
                    if stream:
                        chunk = process.stdout.read1(65536)
                    else:
                        chunk = process.stdout.read(65536)
                    if not chunk:
                        break
                    captured.extend(chunk[: max(0, LOG_LIMIT - len(captured))])
                    if echoing:
                        echoing = echo(redactor.feed(chunk))
                    if transfer is not None:
                        transferred += len(chunk)
                        if transferred > max_bytes:
                            raise ValueError("Artifact exceeds transfer limit")
                        transfer.write(chunk)
                if echoing:
                    echo(redactor.finish())
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
            output = self.redact(captured.decode("utf-8", errors="replace"))
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


def echo(text):
    """Best-effort live progress; return False once stderr cannot take more."""
    if not text:
        return True
    target = sys.stderr
    try:
        buffer = getattr(target, "buffer", None)
        if buffer is None:
            target.write(text)
        else:
            # A redirected Windows stderr may be cp1252; never fail on tqdm glyphs.
            target.flush()
            buffer.write(text.encode(target.encoding or "utf-8", errors="replace"))
            buffer.flush()
        target.flush()
    except (OSError, ValueError, LookupError):
        return False
    return True


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


def limits(values):
    """Validate a resources mapping; return only the known resource keys."""
    for name, lower, upper in RESOURCE_BOUNDS:
        if not lower <= values[name] <= upper:
            raise ValueError(f"{name} must be between {lower} and {upper}")
    if not math.isfinite(values["cpus"]) or not 0.1 <= values["cpus"] <= 256:
        raise ValueError("cpus must be between 0.1 and 256")
    if values["tmpfs_mib"] >= values["memory_mib"]:
        raise ValueError("tmpfs_mib must be smaller than memory_mib")
    return {key: values[key] for key in RESOURCE_KEYS}


def tmpfs_options(size_mib):
    """Share the exact requested private-scratch policy with its inspect guard."""
    return f"rw,nosuid,nodev,size={size_mib}m,mode=1777"


def isolation_flags(resources):
    """Hardening shared by every container this launcher starts."""
    return [
        "--user",
        "65532:65532",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--init",
        "--pids-limit",
        str(resources["pids_limit"]),
        "--memory",
        f"{resources['memory_mib']}m",
        "--memory-swap",
        f"{resources['memory_mib']}m",
        "--cpus",
        str(resources["cpus"]),
        "--tmpfs",
        f"{CONTAINER_TMP}:{tmpfs_options(resources['tmpfs_mib'])}",
        "--log-driver",
        "none",
    ]


def check_worker(docker, image, image_id, resources):
    """Run `_worker metadata` offline in the image before any run container."""
    try:
        metadata = json.loads(
            docker.call(
                [
                    "run",
                    "--rm",
                    "--network",
                    "none",
                    *isolation_flags(resources),
                    image_id,
                    *WORKER,
                    "metadata",
                ]
            )
        )
        if not isinstance(metadata, dict) or "source" not in metadata:
            raise ValueError("Worker metadata lacks source")
    except (RuntimeError, ValueError) as error:
        raise RuntimeError(
            f"Docker image {image} is out of date; rebuild with: "
            "docker compose -f docker/compose.yaml build"
        ) from error
    return metadata


def scoring_inspection(docker, container, image, resources):
    record = json.loads(docker.call(["inspect", container]))[0]
    host, config = record["HostConfig"], record["Config"]
    mounts = [
        {key: mount.get(key) for key in ("Type", "Destination", "RW")}
        for mount in record.get("Mounts", [])
    ]
    # Values are checked here only; the manifest and logs keep key names.
    environment = {}
    for item in config.get("Env") or []:
        key, _, value = item.partition("=")
        environment[key] = value
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
        or any(environment.get(key) != value for key, value in SCORING_ENV.items())
        or set(environment) - IMAGE_ENV - set(SCORING_ENV)
        or set(environment) & set(SECRET_ENV)
    ):
        raise ValueError(
            "Actual scoring container isolation differs from required configuration"
        )
    return {
        "image": record["Image"],
        "user": config["User"],
        "environment_keys": sorted(environment),
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


def remaining(deadline):
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise subprocess.TimeoutExpired("benchmark phase", 0)
    return seconds


class OwnedContainers:
    """Containers one run creates; cleanup removes only these."""

    def __init__(self, docker, image_id, resources, partial):
        self.docker = docker
        self.image_id = image_id
        self.resources = resources
        self.partial = partial
        self.owner = uuid.uuid4().hex
        self.ids = []
        self.pending_name = None

    def create(self, phase, source, timeout):
        self.pending_name = f"lm-eval-{self.owner}-{phase}"
        private_text(
            self.partial / f"{phase}-container-name.txt", self.pending_name + "\n"
        )
        argv = [
            "create",
            "--name",
            self.pending_name,
            "--label",
            f"lm-eval.run={self.owner}",
            "--network",
            "none" if phase == "score" else "bridge",
            *isolation_flags(self.resources),
            "--mount",
            f"type=bind,src={source},dst=/input/{source.name},readonly",
        ]
        if phase == "score":
            for key, value in SCORING_ENV.items():
                argv += ["--env", f"{key}={value}"]
        else:
            argv += ["--env", "HF_ALLOW_CODE_EVAL="]
            for key in SECRET_ENV:
                if os.environ.get(key):
                    argv += ["--env", key]
        argv += [self.image_id, *WORKER, "idle"]
        container = self.docker.call(argv, timeout=min(timeout, 60)).strip()
        if not re.fullmatch(r"[0-9a-f]{64}", container):
            raise ValueError("Docker did not return a full container ID")
        self.ids.append(container)
        private_text(self.partial / f"{phase}-container-id.txt", container + "\n")
        self.pending_name = None
        self.docker.call(["start", container])
        return container

    def copy(self, container, source, destination, deadline):
        byte_limit = self.resources["max_artifact_mib"] * 1024 * 1024
        self.docker.call(
            ["exec", container, *WORKER, "stream-artifact", source, str(byte_limit)],
            timeout=min(60, remaining(deadline)),
            binary_output=destination,
            max_bytes=byte_limit,
        )
        if destination.is_symlink() or not destination.is_file():
            raise ValueError("Expected a regular artifact file")
        destination.chmod(0o600)

    def remove(self, container):
        self.docker.call(["rm", "--force", container])
        self.ids.remove(container)

    def cleanup(self):
        errors = []
        # A create timeout can happen after Docker allocated the container. Only
        # recover our random name when its independently checked owner label agrees.
        if self.pending_name is not None:
            try:
                records = json.loads(self.docker.call(["inspect", self.pending_name]))
                for record in records:
                    labels = record["Config"].get("Labels", {})
                    if labels.get("lm-eval.run") != self.owner:
                        raise RuntimeError("Container cleanup ownership mismatch")
                    container = record["Id"]
                    if not re.fullmatch(r"[0-9a-f]{64}", container):
                        raise RuntimeError("Invalid cleanup container ID")
                    self.ids.append(container)
            except (
                RuntimeError,
                OSError,
                ValueError,
                subprocess.TimeoutExpired,
            ) as error:
                errors.append(error)
            self.pending_name = None
        for container in self.ids[:]:
            try:
                self.remove(container)
            except (RuntimeError, OSError, subprocess.TimeoutExpired) as error:
                errors.append(error)
        if errors:
            raise RuntimeError(
                "Owned-container cleanup failed; inspect Docker before retrying"
            ) from errors[0]
