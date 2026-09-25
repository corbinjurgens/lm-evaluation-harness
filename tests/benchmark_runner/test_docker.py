"""Tests for the strengthened scoring-container check in benchmark_runner.docker."""

from __future__ import annotations

import json

import pytest

from benchmark_runner.docker import CONTAINER_TMP, scoring_inspection


IMAGE = "sha256:" + "a" * 64
RESOURCES = {"pids_limit": 128, "memory_mib": 512, "cpus": 1.0, "tmpfs_mib": 128}
REQUIRED = ["HF_ALLOW_CODE_EVAL=1", "HF_HUB_OFFLINE=1", "HF_DATASETS_OFFLINE=1"]


class InspectOnly:
    def __init__(self, env):
        self.env = env

    def call(self, argv, **kwargs):
        assert argv[0] == "inspect"
        return json.dumps(
            [
                {
                    "Image": IMAGE,
                    "Config": {"User": "65532:65532", "Env": self.env},
                    "HostConfig": {
                        "NetworkMode": "none",
                        "ReadonlyRootfs": True,
                        "CapDrop": ["ALL"],
                        "SecurityOpt": ["no-new-privileges"],
                        "Init": True,
                        "PidsLimit": 128,
                        "Memory": 536870912,
                        "MemorySwap": 536870912,
                        "NanoCpus": 1000000000,
                        "Tmpfs": {CONTAINER_TMP: "rw,nosuid,nodev,size=128m,mode=1777"},
                    },
                    "Mounts": [
                        {
                            "Type": "bind",
                            "Destination": "/input/score-input.json",
                            "RW": False,
                        }
                    ],
                }
            ]
        )


def inspect(env):
    return scoring_inspection(InspectOnly(env), "c" * 64, IMAGE, RESOURCES)


def test_required_offline_environment_and_image_keys_pass():
    result = inspect(["PATH=/opt/venv/bin:/usr/bin", "LANG=C.UTF-8", *REQUIRED])
    assert result["environment_keys"] == [
        "HF_ALLOW_CODE_EVAL",
        "HF_DATASETS_OFFLINE",
        "HF_HUB_OFFLINE",
        "LANG",
        "PATH",
    ]
    assert "/opt/venv/bin" not in json.dumps(result)


@pytest.mark.parametrize(
    "env",
    [
        pytest.param(["HF_ALLOW_CODE_EVAL=1", "HF_DATASETS_OFFLINE=1"], id="missing"),
        pytest.param(
            ["HF_ALLOW_CODE_EVAL=1", "HF_HUB_OFFLINE=0", "HF_DATASETS_OFFLINE=1"],
            id="wrong-value",
        ),
        pytest.param(["HF_ALLOW_CODE_EVAL=", *REQUIRED[1:]], id="code-eval-empty"),
        pytest.param([*REQUIRED, "OPENAI_API_KEY=secret"], id="openai-key"),
        pytest.param([*REQUIRED, "HF_TOKEN=secret"], id="hf-token"),
        pytest.param([*REQUIRED, "EXTRA=1"], id="unknown-key"),
        pytest.param([*REQUIRED[:2], "HF_DATASETS_OFFLINE"], id="no-value"),
    ],
)
def test_weakened_scoring_environment_raises(env):
    with pytest.raises(ValueError, match="isolation differs"):
        inspect(env)
