"""Offline pass@k evaluation in fresh Python processes, not a security sandbox.

Run only inside a separately secured, disposable environment. Process groups and
resource limits contain ordinary accidents, not deliberately hostile Python.
The metric implements the estimator from Chen et al., arXiv:2107.03374, and the
candidate + newline + reference contract of Hugging Face's code_eval metric.
No metric implementation is downloaded and importing this module executes no code.
"""

import math
import os
import selectors
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


_WORKER = Path(__file__).with_name("_code_eval_worker.py")
_WARNING = (
    "Code evaluation executes untrusted Python. This backend is NOT a security "
    "sandbox. Use a disposable, network-isolated environment without credentials "
    "or writable host mounts, then explicitly set HF_ALLOW_CODE_EVAL=1."
)


class CodeEvalInfrastructureError(RuntimeError):
    """The evaluator could not reliably start or supervise a candidate."""


def _check(
    program: str,
    task_id: int,
    completion_id: int,
    timeout: float,
    startup_timeout: float,
    maximum_memory_bytes: int | None,
    maximum_file_bytes: int,
) -> dict:
    # stdout/stderr are discarded by the OS: neither unbounded buffers nor a
    # candidate printing "passed" can affect the private control channel.
    with tempfile.TemporaryDirectory(prefix="lm-eval-code-") as scratch:
        source = Path(scratch) / "program.py"
        source.write_text(program, encoding="utf-8")
        read_fd, write_fd = os.pipe()
        process = None
        try:
            # Fixed interpreter/worker argv, no shell; submitted code stays in a file.
            process = subprocess.Popen(  # noqa: S603
                [
                    sys.executable,
                    "-I",
                    "-B",
                    str(_WORKER),
                    str(source),
                    str(write_fd),
                    str(timeout),
                    str(maximum_memory_bytes or 0),
                    str(maximum_file_bytes),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=scratch,
                env={
                    "PATH": os.defpath,
                    "LANG": "C.UTF-8",
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                },
                start_new_session=True,
                pass_fds=(write_fd,),
            )
            os.close(write_fd)
            write_fd = None
            deadline = time.monotonic() + startup_timeout
            ready = False
            verdict = None
            with selectors.DefaultSelector() as selector:
                selector.register(read_fd, selectors.EVENT_READ)
                while verdict is None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        if not ready:
                            raise CodeEvalInfrastructureError(
                                "Worker startup timed out"
                            )
                        verdict = "timed out"
                        break
                    if not selector.select(remaining):
                        continue
                    message = os.read(read_fd, 3)
                    if not message:
                        if not ready:
                            raise CodeEvalInfrastructureError(
                                "Worker exited before candidate execution"
                            )
                        verdict = "failed: worker exited during candidate execution"
                        break
                    for byte in message:
                        if byte == ord("R") and not ready:
                            ready = True
                            deadline = time.monotonic() + timeout
                        elif ready and verdict is None and byte in (ord("P"), ord("F")):
                            verdict = "passed" if byte == ord("P") else "failed"
                        else:
                            raise CodeEvalInfrastructureError("Invalid worker protocol")
            if verdict == "passed":
                # A completed verdict must also be followed by a clean exit.
                try:
                    returncode = process.wait(
                        timeout=max(0.01, deadline - time.monotonic())
                    )
                except subprocess.TimeoutExpired:
                    verdict = "timed out"
                else:
                    if returncode != 0:
                        verdict = "failed: worker exited abnormally"
        except OSError as exc:
            raise CodeEvalInfrastructureError(
                "Unable to supervise code worker"
            ) from exc
        finally:
            if process is not None:
                # Clean the entire original group even when its leader exited.
                # Escaping the group is hostile behavior, not a sandbox guarantee.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            os.close(read_fd)
            if write_fd is not None:
                os.close(write_fd)
    return {
        "task_id": task_id,
        "completion_id": completion_id,
        "passed": verdict == "passed",
        "result": verdict,
    }


def compute(
    *,
    references: list[str],
    predictions: list[list[str]],
    k: list[int] | tuple[int, ...] = (1, 10, 100),
    num_workers: int = 4,
    timeout: float = 3.0,
    startup_timeout: float = 10.0,
    maximum_memory_bytes: int | None = None,
    maximum_file_bytes: int = 1024 * 1024,
) -> tuple[dict[str, float], dict[int, list[tuple[int, dict]]]]:
    """Return HF-compatible (pass@k metrics, per-task candidate details).

    The timeout is per candidate after worker setup; startup has its own deadline.
    Each worker has a CPU backstop of ceil(timeout)+1 seconds, a file-size limit,
    and optionally an address-space limit. Output is discarded, never captured.
    Infrastructure errors raise; candidate exceptions, exits and timeouts fail.
    Values of k exceeding any task's candidate count are omitted, like code_eval.
    """
    if os.environ.get("HF_ALLOW_CODE_EVAL") != "1":
        raise ValueError(_WARNING)
    if os.name != "posix":
        raise NotImplementedError("The code evaluator requires POSIX process groups")
    if not references or len(references) != len(predictions):
        raise ValueError("Provide matching nonempty references and predictions")
    if any(not isinstance(reference, str) for reference in references):
        raise ValueError("Each reference must be a string")
    if any(
        not isinstance(candidates, (list, tuple))
        or not candidates
        or any(not isinstance(candidate, str) for candidate in candidates)
        for candidates in predictions
    ):
        raise ValueError("Each prediction must be a nonempty sequence of strings")
    if not k or any(type(value) is not int or value <= 0 for value in k):
        raise ValueError("k must contain positive integers")
    if type(num_workers) is not int or num_workers <= 0:
        raise ValueError("num_workers must be a positive integer")
    if any(
        not math.isfinite(value) or value <= 0 for value in (timeout, startup_timeout)
    ):
        raise ValueError("Timeouts must be finite and positive")
    for value in (maximum_memory_bytes, maximum_file_bytes):
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError("Resource limits must be positive integers")
    if maximum_file_bytes is None:
        raise ValueError("maximum_file_bytes cannot be None")
    work = [
        (candidate + "\n" + reference, task_id, completion_id)
        for task_id, (reference, candidates) in enumerate(
            zip(references, predictions, strict=True)
        )
        for completion_id, candidate in enumerate(candidates)
    ]
    results = {task_id: [] for task_id in range(len(references))}
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                _check,
                program,
                task_id,
                completion_id,
                timeout,
                startup_timeout,
                maximum_memory_bytes,
                maximum_file_bytes,
            )
            for program, task_id, completion_id in work
        ]
        for future in futures:
            result = future.result()
            results[result["task_id"]].append((result["completion_id"], result))
    totals = [len(items) for items in results.values()]
    correct = [sum(item[1]["passed"] for item in items) for items in results.values()]
    metrics = {}
    for count in k:
        if min(totals) >= count:
            # Exact integer combinations avoid cancellation for small pass rates.
            scores = [
                1.0 - math.comb(total - passed, count) / math.comb(total, count)
                if total - passed >= count
                else 1.0
                for total, passed in zip(totals, correct, strict=True)
            ]
            metrics[f"pass@{count}"] = sum(scores) / len(scores)
    return metrics, results
