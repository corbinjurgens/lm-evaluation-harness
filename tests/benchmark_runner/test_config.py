"""Contracts for the benchmark configuration building blocks."""

import argparse
import json

from benchmark_runner import config


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
    assert config.model_slug("Publisher/New Model:Q4") == "publisher-new-model-q4"


def test_config_uses_normal_tasks_and_optional_reasoning():
    built = config.benchmark_config(
        arguments(reasoning_effort="low", temperature=1.0), limit=1
    )

    assert built["tasks"] == [
        "humaneval",
        "mbpp",
        "ifeval",
        "gsm8k_cot_zeroshot",
    ]
    assert built["gen_kwargs"] == {
        "temperature": 1.0,
        "max_gen_toks": 4096,
        "reasoning_effort": "low",
    }
    assert built["limit"] == 1
    assert not any("key" in key or "token" in key for key in built)


def test_write_json_is_utf8_without_bom_and_never_overwrites(tmp_path):
    destination = tmp_path / "config.json"
    config.write_json(destination, {"model": "test"})

    assert destination.read_bytes().startswith(b"{")
    assert json.loads(destination.read_text(encoding="utf-8")) == {"model": "test"}
    try:
        config.write_json(destination, {"model": "replacement"})
    except FileExistsError:
        pass
    else:
        raise AssertionError("Existing config was overwritten")
    assert json.loads(destination.read_text(encoding="utf-8")) == {"model": "test"}
