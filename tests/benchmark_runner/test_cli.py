"""Tests for the benchmark_runner CLI."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from benchmark_runner import pipeline
from benchmark_runner.cli import main


ROOT = Path(__file__).resolve().parents[2]
URL = "http://host.docker.internal:8888/v1/chat/completions"
# Independent oracle: scripts/run_benchmark.py's historical resource defaults.
RESOURCE_DEFAULTS = {
    "memory_mib": 512,
    "tmpfs_mib": 128,
    "pids_limit": 128,
    "cpus": 1.0,
    "generation_timeout": 86400,
    "scoring_timeout": 86400,
    "max_artifact_mib": 128,
}


def _unexpected(*args, **kwargs):
    raise AssertionError("must not be called")


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_config_route_passes_launcher_resource_defaults(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(pipeline, "run_configured", seen.append)
    config, output = tmp_path / "config.json", tmp_path / "run"

    assert main(["run", "--config", str(config), "--output-dir", str(output)]) == 0

    (args,) = seen
    assert (args.config, args.output_dir) == (config, output)
    resources = {key: getattr(args, key) for key in RESOURCE_DEFAULTS}
    assert resources == RESOURCE_DEFAULTS


@pytest.mark.parametrize(
    ("extra", "flag"),
    [
        (["some/model"], "MODEL"),
        (["--base-url", URL], "--base-url"),
        (["--temperature", "0"], "--temperature"),
        (["--max-gen-toks", "4096"], "--max-gen-toks"),
        (["--reasoning-effort", "low"], "--reasoning-effort"),
    ],
)
def test_config_rejects_model_and_generation_arguments(
    monkeypatch, capsys, tmp_path, extra, flag
):
    monkeypatch.setattr(pipeline, "run_configured", _unexpected)
    argv = ["run", "--config", str(tmp_path / "c.json")]
    argv += ["--output-dir", str(tmp_path / "run"), *extra]

    with pytest.raises(SystemExit) as excinfo:
        main(argv)

    assert excinfo.value.code == 2
    assert flag in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["run", "--config", "c.json"],
        ["run", "--config", "c.json", "--output-dir", "run", "--smoke-only"],
        ["run", "model", "--output-dir", "run"],
    ],
)
def test_route_specific_flags_are_rejected(monkeypatch, argv):
    monkeypatch.setattr(pipeline, "run_configured", _unexpected)
    monkeypatch.setattr(pipeline, "run_campaign", _unexpected)
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    assert excinfo.value.code == 2


def test_campaign_route_fills_generation_defaults(monkeypatch):
    seen = []
    monkeypatch.setattr(pipeline, "run_campaign", seen.append)
    monkeypatch.setenv("OPENAI_API_KEY", "set")

    assert main(["run", "some/model", "--base-url", URL, "--memory-mib", "1024"]) == 0

    (args,) = seen
    assert (args.model, args.base_url) == ("some/model", URL)
    assert (args.temperature, args.max_gen_toks) == (0.0, 4096)
    assert args.reasoning_effort is None
    assert args.memory_mib == 1024


def test_handled_error_exits_one_with_redacted_detail(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-hidden-value")

    def fail(args):
        raise RuntimeError("server said sk-hidden-value")

    monkeypatch.setattr(pipeline, "run_campaign", fail)

    assert main(["run", "some/model", "--base-url", URL]) == 1
    err = capsys.readouterr().err
    assert "Benchmark run failed" in err
    assert "sk-hidden-value" not in err
    assert "[REDACTED]" in err


def test_timeout_is_a_handled_error(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "set")

    def fail(args):
        raise subprocess.TimeoutExpired("benchmark phase", 0)

    monkeypatch.setattr(pipeline, "run_campaign", fail)
    assert main(["run", "some/model", "--base-url", URL]) == 1


def test_ctrl_c_exits_130(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "set")

    def interrupt(args):
        raise KeyboardInterrupt

    monkeypatch.setattr(pipeline, "run_campaign", interrupt)
    assert main(["run", "some/model", "--base-url", URL]) == 130


def fake_stages(monkeypatch, tmp_path):
    """Record in-process run_stage calls; publish a minimal scores.json each."""
    calls = []

    def run_stage(config_path, output_dir, image, resources, docker=None):
        calls.append((config_path, output_dir, image, resources))
        completed = output_dir / "completed"
        completed.mkdir(parents=True)
        scores = {"results": {"task": {"metrics": {"acc,none": 0.5}}}}
        (completed / "scores.json").write_text(json.dumps(scores), encoding="utf-8")
        return completed

    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "require_image", lambda image: None)
    monkeypatch.setattr(pipeline, "run_stage", run_stage)
    # In-process: no stage may be launched as a subprocess.
    monkeypatch.setattr(pipeline.subprocess, "run", _unexpected)
    return calls


@pytest.mark.parametrize("smoke_only", [True, False])
def test_campaign_runs_stages_in_process_and_reports_each(
    monkeypatch, capsys, tmp_path, smoke_only
):
    calls = fake_stages(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "set")
    argv = ["run", "Pub/Model", "--base-url", URL, "--image", "img:tag"]

    assert main(argv + (["--smoke-only"] if smoke_only else [])) == 0

    run_root = tmp_path / "results" / "model-runs" / "pub-model"
    stages = ["smoke"] if smoke_only else ["smoke", "full"]
    assert len(calls) == len(stages)
    out = capsys.readouterr().out
    for (config_path, output_dir, image, resources), stage in zip(
        calls, stages, strict=True
    ):
        assert output_dir.parent == run_root
        assert output_dir.name.endswith(f"-{stage}")
        assert config_path == run_root / f"{output_dir.name}-config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        assert config["model"] == "Pub/Model"
        assert ("limit" in config) is (stage == "smoke")
        assert image == "img:tag"
        assert resources["generation_timeout"] == 86400
        assert f"Completed benchmark: {output_dir / 'completed'}" in out
    assert out.count('"acc,none": 0.5') == len(stages)


def test_run_all_benchmarks_shim_forwards_to_run(monkeypatch):
    runner = load_script("run_all_benchmarks")
    seen = []
    monkeypatch.setattr(runner.cli, "main", lambda argv: seen.append(argv) or 7)
    monkeypatch.setattr(sys, "argv", ["run_all_benchmarks.py", "m", "--smoke-only"])

    assert runner.main() == 7
    assert seen == [["run", "m", "--smoke-only"]]


def test_run_benchmark_shim_forwards_to_config_route(monkeypatch, tmp_path):
    launcher = load_script("run_benchmark")
    seen = []
    monkeypatch.setattr(pipeline, "run_configured", seen.append)
    config, output = tmp_path / "config.json", tmp_path / "run"

    argv = ["--config", str(config), "--output-dir", str(output), "--cpus", "2"]
    assert launcher.main(argv) == 0

    (args,) = seen
    assert (args.config, args.output_dir) == (config, output)
    assert args.image == "lm-eval-fork:local"
    assert args.cpus == 2.0


def test_run_benchmark_shim_still_requires_config(monkeypatch):
    launcher = load_script("run_benchmark")
    monkeypatch.setattr(pipeline, "run_configured", _unexpected)
    with pytest.raises(SystemExit) as excinfo:
        launcher.main(["--output-dir", "run"])
    assert excinfo.value.code == 2


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
