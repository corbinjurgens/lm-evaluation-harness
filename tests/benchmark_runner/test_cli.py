"""Tests for the benchmark_runner CLI skeleton."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from benchmark_runner.cli import main


ROOT = Path(__file__).resolve().parents[2]


def test_help_exits_zero():
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0


def test_help_hides_worker_subcommand(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    assert "_worker" not in out
    assert "==SUPPRESS==" not in out


def test_run_not_implemented(capsys):
    assert main(["run"]) == 2
    assert "not implemented" in capsys.readouterr().err


def test_worker_dispatches_to_runtime_worker(monkeypatch):
    calls = []

    def fake_worker_main(argv):
        calls.append(argv)
        return 0

    from benchmark_runner.runtime import worker

    monkeypatch.setattr(worker, "main", fake_worker_main)
    assert main(["_worker", "idle"]) == 0
    assert calls == [["idle"]]


def test_worker_missing_op_exits_nonzero():
    with pytest.raises(SystemExit) as excinfo:
        main(["_worker"])
    assert excinfo.value.code == 2


def test_worker_unknown_op_exits_nonzero():
    with pytest.raises(SystemExit) as excinfo:
        main(["_worker", "not-a-real-op"])
    assert excinfo.value.code == 2


def test_cli_module_does_not_import_heavy_dependencies():
    """Import benchmark_runner.cli in a fresh process and check sys.modules.

    A fresh process is required because other test modules import lm_eval
    into the pytest process, which would otherwise poison this check. ROOT is
    passed via an environment variable, and the checker code below is a
    literal, so the argument list stays fully static.
    """
    env = dict(os.environ)
    env["BENCHMARK_RUNNER_TEST_ROOT"] = str(ROOT)
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            (
                "import os, sys; "
                "sys.path.insert(0, os.environ['BENCHMARK_RUNNER_TEST_ROOT']); "
                "import benchmark_runner.cli; "
                "bad = {'datasets', 'lm_eval', 'torch'} & set(sys.modules); "
                "sys.exit(1 if bad else 0)"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0, result.stderr
