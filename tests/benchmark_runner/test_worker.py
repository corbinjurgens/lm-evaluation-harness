"""Tests for the benchmark_runner internal worker operations."""

from __future__ import annotations

import json

import pytest

from benchmark_runner.runtime.worker import main


VALID_CONFIG = {
    "tasks": ["gsm8k_cot_zeroshot"],
    "model": "test-model",
    "base_url": "http://localhost:8888/v1/chat/completions",
    "gen_kwargs": {"temperature": 0, "max_gen_toks": 16},
}


def test_metadata_without_config_prints_source(capsys):
    assert main(["metadata"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "source" in payload
    assert "python" in payload


def test_metadata_with_valid_config_succeeds(tmp_path, capsys):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(VALID_CONFIG), encoding="utf-8")
    assert main(["metadata", "--config", str(config_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "source" in payload


def test_metadata_with_invalid_config_raises(tmp_path):
    # An uncaught exception here, like everywhere else in this worker and in
    # runtime/bundle.py, exits the process non-zero when run as `_worker`.
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"tasks": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        main(["metadata", "--config", str(config_path)])


def test_stream_artifact_streams_file_byte_for_byte(tmp_path, capsysbinary):
    content = b"exact contents streamed byte for byte"
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(content)
    assert main(["stream-artifact", str(artifact), str(len(content))]) == 0
    assert capsysbinary.readouterr().out == content


def test_stream_artifact_fails_above_byte_limit(tmp_path, capsysbinary):
    content = b"more than the limit allows"
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(content)
    with pytest.raises(ValueError, match="exceeds transfer limit"):
        main(["stream-artifact", str(artifact), str(len(content) - 1)])
