"""Trusted synthetic programs exercise the actual fresh-process evaluator."""

import math
import sys
import time
from pathlib import Path

import filelock
import pytest

from lm_eval.tasks import _code_eval as backend


GOOD = "def add(a, b): return a + b"
WRONG = "def add(a, b): return a * b"
REFERENCE = "assert add(2, 3) == 5"
KS = [1, 2, 8, 16, 32, 64]


@pytest.fixture(autouse=True)
def allow_code(monkeypatch):
    monkeypatch.setenv("HF_ALLOW_CODE_EVAL", "1")


def evaluate(candidates, **kwargs):
    return backend.compute(references=[REFERENCE], predictions=[candidates], **kwargs)


def test_opt_in_is_checked_at_runtime(monkeypatch):
    monkeypatch.delenv("HF_ALLOW_CODE_EVAL")
    with pytest.raises(ValueError, match="NOT a security sandbox"):
        evaluate([GOOD], k=[1])


def test_installed_filelock_fork_guard_remains_active():
    # Emit an audit event, without actually forking. The deployment pins filelock
    # with this guard; a missing/disabled guard must not make stress tests pass.
    with (
        filelock._api._fork_transition(),
        pytest.raises(RuntimeError, match="unsafe while filelock"),
    ):
        sys.audit("os.fork")
    assert evaluate([GOOD], k=[1])[0] == {"pass@1": 1.0}
    with (
        filelock._api._fork_transition(),
        pytest.raises(RuntimeError, match="unsafe while filelock"),
    ):
        sys.audit("os.fork")


def test_one_hundred_sequential_singletons_with_filelock(tmp_path):
    # Keep the installed filelock protection active throughout real worker starts.
    with filelock.FileLock(tmp_path / "guard.lock"):
        for index in range(100):
            code = [GOOD, WRONG, ""][index % 3]
            metrics, details = evaluate([code], k=[1])
            expected = float(index % 3 == 0)
            assert metrics == {"pass@1": expected}
            assert details[0][0][0] == 0
            assert details[0][0][1]["passed"] is bool(expected)


@pytest.mark.parametrize(
    ("candidates", "correct"),
    [
        ([GOOD] * 64, 64),
        ([WRONG] * 64, 0),
        ([GOOD, WRONG] * 32, 32),
        (["def add("] * 64, 0),
        (["while True: pass"] * 64, 0),
    ],
    ids=["correct", "wrong", "mixed", "malformed", "timeout"],
)
def test_sixty_four_candidates_estimator_with_filelock(tmp_path, candidates, correct):
    with filelock.FileLock(tmp_path / "guard.lock"):
        metrics, details = evaluate(candidates, k=KS, timeout=0.2)
    for k in KS:
        failures = math.comb(64 - correct, k) if k <= 64 - correct else 0
        expected = 1 - failures / math.comb(64, k)
        assert metrics[f"pass@{k}"] == expected
    assert len(details[0]) == 64
    assert [item[0] for item in details[0]] == list(range(64))
    assert sum(item[1]["passed"] for item in details[0]) == correct
    if candidates[0] == "while True: pass":
        assert {item[1]["result"] for item in details[0]} == {"timed out"}


def test_hf_shape_multi_task_averaging_and_unavailable_k():
    metrics, details = backend.compute(
        references=[REFERENCE, REFERENCE],
        predictions=[[GOOD, WRONG], [GOOD]],
        k=[1, 2],
    )
    assert metrics == {"pass@1": 0.75}
    assert list(details) == [0, 1]
    assert details[1][0] == (
        0,
        {"task_id": 1, "completion_id": 0, "passed": True, "result": "passed"},
    )


@pytest.mark.parametrize(
    "code",
    [
        "raise SystemExit(0)",
        "import os; os._exit(0)",
        "import os; os._exit(1)",
        "import ctypes; ctypes.string_at(0)",
        "print('passed'); print('RP'); raise AssertionError",
        "import os; os.write(1, b'P'); os.write(2, b'passed'); raise AssertionError",
    ],
)
def test_candidate_exit_crash_or_printed_verdict_cannot_pass(code):
    metrics, details = evaluate([code], k=[1])
    assert metrics == {"pass@1": 0.0}
    assert details[0][0][1]["passed"] is False


def test_output_is_discarded_and_streams_cannot_block_completion():
    code = (
        "import os\nfor _ in range(100):\n os.write(1, b'x' * 100000)\n os.write(2, b'x' * 100000)\n"
        + GOOD
    )
    assert evaluate([code], k=[1])[0] == {"pass@1": 1.0}


def test_parent_credentials_are_not_inherited(monkeypatch):
    monkeypatch.setenv("CODE_EVAL_TEST_SECRET", "not-a-real-secret")
    code = "import os\nassert 'CODE_EVAL_TEST_SECRET' not in os.environ\n" + GOOD
    assert evaluate([code], k=[1])[0] == {"pass@1": 1.0}


@pytest.mark.parametrize(
    ("candidate", "reference"),
    [
        ("import numpy; a = numpy.array([1, 2, 3])", "assert a.sum() == 6"),
        ("from scipy.special import comb", "assert comb(5, 2) == 10"),
        ("import pandas; a = pandas.Series([1, 2, 3])", "assert a.sum() == 6"),
    ],
    ids=["numpy", "scipy", "pandas"],
)
def test_installed_numerical_dependencies_work_with_guard(candidate, reference):
    candidate += (
        "\nimport os, subprocess\n"
        "assert os.putenv is None\n"
        "assert os.fork is None\n"
        "assert subprocess.Popen is None\n"
    )
    metrics, details = backend.compute(
        references=[reference], predictions=[[candidate]], k=[1]
    )
    assert metrics == {"pass@1": 1.0}
    assert details[0][0][1]["passed"] is True


def test_incorrect_numpy_candidate_still_fails():
    metrics, details = backend.compute(
        references=["assert a.sum() == 6"],
        predictions=[["import numpy; a = numpy.array([1, 2, 4])"]],
        k=[1],
    )
    assert metrics == {"pass@1": 0.0}
    assert details[0][0][1]["passed"] is False


def test_dependency_startup_failure_is_infrastructure_failure(monkeypatch, tmp_path):
    # Run the real worker, injecting only a broken required dependency import.
    # It must fail before the worker's R signal and never score a candidate zero.
    wrapper = tmp_path / "missing_dependency.py"
    wrapper.write_text(
        "import importlib, runpy\n"
        "original_import = importlib.import_module\n"
        "def broken_import(name, *args, **kwargs):\n"
        " if name == 'numpy': raise ImportError('injected broken numpy')\n"
        " return original_import(name, *args, **kwargs)\n"
        "importlib.import_module = broken_import\n"
        f"runpy.run_path({str(backend._WORKER)!r}, run_name='__main__')\n"
    )
    monkeypatch.setattr(backend, "_WORKER", wrapper)
    with pytest.raises(backend.CodeEvalInfrastructureError, match="before candidate"):
        evaluate([GOOD], k=[1])


def test_resource_limits_are_effective():
    code = (
        "import resource\n"
        "assert resource.getrlimit(resource.RLIMIT_AS) == (268435456, 268435456)\n"
        "assert resource.getrlimit(resource.RLIMIT_FSIZE) == (4096, 4096)\n"
        "assert resource.getrlimit(resource.RLIMIT_CORE) == (0, 0)\n"
        "assert resource.getrlimit(resource.RLIMIT_CPU) == (2, 2)\n" + GOOD
    )
    assert evaluate(
        [code],
        k=[1],
        timeout=0.5,
        maximum_memory_bytes=268435456,
        maximum_file_bytes=4096,
    )[0] == {"pass@1": 1.0}
    oversized = "with open('large', 'wb') as f:\n f.write(b'x' * 8192)\n" + GOOD
    assert evaluate([oversized], k=[1], maximum_file_bytes=4096)[0] == {"pass@1": 0.0}


def test_popen_failure_is_infrastructure_failure(monkeypatch):
    def unavailable(*args, **kwargs):
        raise OSError("worker unavailable")

    monkeypatch.setattr(backend.subprocess, "Popen", unavailable)
    with pytest.raises(backend.CodeEvalInfrastructureError, match="supervise"):
        evaluate([GOOD], k=[1])


@pytest.mark.parametrize("worker", ["raise SystemExit(0)", "while True: pass"])
def test_failure_before_ready_is_infrastructure_failure(monkeypatch, tmp_path, worker):
    fake_worker = tmp_path / "broken_worker.py"
    fake_worker.write_text(worker)
    monkeypatch.setattr(backend, "_WORKER", fake_worker)
    with pytest.raises(backend.CodeEvalInfrastructureError, match="Worker"):
        evaluate([GOOD], k=[1], startup_timeout=0.2)


def test_invalid_control_protocol_is_infrastructure_failure(monkeypatch, tmp_path):
    fake_worker = tmp_path / "broken_worker.py"
    fake_worker.write_text("import os, sys\nos.write(int(sys.argv[2]), b'WRONG')\n")
    monkeypatch.setattr(backend, "_WORKER", fake_worker)
    with pytest.raises(backend.CodeEvalInfrastructureError, match="protocol"):
        evaluate([GOOD], k=[1])


@pytest.mark.parametrize("case", ["pass", "timeout", "crash", "startup", "popen"])
def test_scratch_and_fds_are_cleaned(monkeypatch, tmp_path, case):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    original = backend.tempfile.TemporaryDirectory

    def in_scratch(**kwargs):
        return original(dir=scratch, **kwargs)

    monkeypatch.setattr(backend.tempfile, "TemporaryDirectory", in_scratch)
    before = len(list(Path("/proc/self/fd").iterdir()))
    if case == "startup":
        monkeypatch.setattr(backend, "_WORKER", tmp_path / "missing.py")
    elif case == "popen":

        def fail(*args, **kwargs):
            raise OSError("unavailable")

        monkeypatch.setattr(backend.subprocess, "Popen", fail)
    code = {"timeout": "while True: pass", "crash": "import os; os._exit(1)"}.get(
        case, GOOD
    )
    if case in ("startup", "popen"):
        with pytest.raises(backend.CodeEvalInfrastructureError):
            evaluate([code], k=[1], timeout=0.2)
    else:
        assert evaluate([code], k=[1], timeout=0.2)[0] == {
            "pass@1": float(case == "pass")
        }
    assert list(scratch.iterdir()) == []
    assert len(list(Path("/proc/self/fd").iterdir())) == before


@pytest.mark.parametrize("tail", [GOOD, "while True: pass", "os._exit(0)"])
def test_descendants_in_worker_process_group_are_stopped(tmp_path, tail):
    # Trusted fixture deliberately bypasses the accident guard to test supervision.
    # A process may briefly remain a zombie until the container's init reaps it;
    # it must not remain running after compute returns.
    pid_file = tmp_path / "descendant.pid"
    code = (
        "import ctypes, os, time\n"
        "pid = ctypes.CDLL(None).fork()\n"
        "if pid == 0:\n"
        " time.sleep(30)\n"
        " os._exit(0)\n"
        f"with open({str(pid_file)!r}, 'w') as f: f.write(str(pid))\n" + tail
    )
    metrics, _ = evaluate([code], k=[1], timeout=0.2)
    assert metrics == {"pass@1": float(tail == GOOD)}
    pid = int(pid_file.read_text())
    stat = Path(f"/proc/{pid}/stat")
    deadline = time.monotonic() + 2
    while True:
        try:
            state = stat.read_text().split()[2]
        except FileNotFoundError:
            break
        if state == "Z":
            break
        assert time.monotonic() < deadline, f"descendant {pid} still running"
        time.sleep(0.01)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"references": [], "predictions": []},
        {"references": [REFERENCE], "predictions": []},
        {"predictions": [[]]},
        {"predictions": [[None]]},
        {"k": [0]},
        {"k": [1.5]},
        {"num_workers": 0},
        {"timeout": float("inf")},
        {"startup_timeout": 0},
        {"maximum_memory_bytes": 0},
        {"maximum_file_bytes": None},
    ],
)
def test_invalid_inputs_raise_instead_of_producing_scores(kwargs):
    options = {"references": [REFERENCE], "predictions": [[GOOD]], "k": [1]}
    options.update(kwargs)
    with pytest.raises(ValueError):
        backend.compute(**options)
