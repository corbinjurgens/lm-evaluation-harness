"""Cross-platform contracts for the interactive benchmark runner."""

import argparse
import importlib.util
import json
import subprocess
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_all_benchmarks.py"
SPEC = importlib.util.spec_from_file_location("run_all_benchmarks", SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

# Importable once the shim above has put the checkout root on sys.path.
from benchmark_runner import pipeline


def arguments(**overrides):
    values = {
        "model": "Publisher/New Model",
        "base_url": "http://host.docker.internal:8888/v1/chat/completions",
        "temperature": 0.0,
        "max_gen_toks": 4096,
        "reasoning_effort": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_model_slug_is_safe_on_mac_and_windows():
    assert runner.model_slug("Publisher/New Model:Q4") == "publisher-new-model-q4"


def test_config_uses_normal_tasks_and_optional_reasoning():
    config = runner.benchmark_config(
        arguments(reasoning_effort="low", temperature=1.0), limit=1
    )

    assert config["tasks"] == [
        "humaneval",
        "mbpp",
        "ifeval",
        "gsm8k_cot_zeroshot",
    ]
    assert config["gen_kwargs"] == {
        "temperature": 1.0,
        "max_gen_toks": 4096,
        "reasoning_effort": "low",
    }
    assert config["limit"] == 1
    assert not any("key" in key or "token" in key for key in config)


def test_write_json_is_utf8_without_bom_and_never_overwrites(tmp_path):
    destination = tmp_path / "config.json"
    runner.write_json(destination, {"model": "test"})

    assert destination.read_bytes().startswith(b"{")
    assert json.loads(destination.read_text(encoding="utf-8")) == {"model": "test"}
    try:
        runner.write_json(destination, {"model": "replacement"})
    except FileExistsError:
        pass
    else:
        raise AssertionError("Existing config was overwritten")
    assert json.loads(destination.read_text(encoding="utf-8")) == {"model": "test"}


def test_completed_summary_merges_stderr_into_metrics(tmp_path):
    completed = tmp_path / "completed"
    completed.mkdir()
    scores = {
        "results": {
            "gsm8k": {
                "metrics": {"exact_match,strict-match": 0.5},
                "stderr": {"exact_match_stderr,strict-match": 0.25},
            },
            "legacy": {"metrics": {"acc,none": 1.0}},
        }
    }
    (completed / "scores.json").write_text(json.dumps(scores), encoding="utf-8")

    assert runner.completed_summary(tmp_path) == {
        "gsm8k": {
            "exact_match,strict-match": 0.5,
            "exact_match_stderr,strict-match": 0.25,
        },
        "legacy": {"acc,none": 1.0},
    }


def test_missing_image_builds_with_portable_compose_and_requested_tag():
    inspection = subprocess.CompletedProcess([], 1)
    with (
        mock.patch.object(pipeline.subprocess, "run", return_value=inspection),
        mock.patch.object(pipeline, "call") as call,
    ):
        runner.ensure_image("docker.exe", "private/evaluator:test")

    argv = call.call_args.args[0]
    environment = call.call_args.kwargs["environment"]
    assert argv == [
        "docker.exe",
        "compose",
        "-f",
        str(runner.PORTABLE_COMPOSE),
        "build",
    ]
    assert environment["LM_EVAL_IMAGE"] == "private/evaluator:test"
