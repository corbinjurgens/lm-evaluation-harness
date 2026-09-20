"""Normal coding tasks load without execution and score through the owned backend."""

import importlib
import math
import subprocess
import sys
from unittest.mock import Mock

import evaluate
import filelock
import pytest
from datasets import Dataset

from lm_eval import evaluator
from lm_eval.api.instance import Instance
from lm_eval.api.task import ConfigurableTask
from lm_eval.models.dummy import DummyLM
from lm_eval.tasks import TaskManager, _code_eval


DOC = {
    "prompt": 'def add(a, b):\n    """Add two numbers."""\n',
    "entry_point": "add",
    "test": "def check(candidate):\n    assert candidate(2, 3) == 5",
    "text": "Add two numbers.",
    "test_list": ["assert add(2, 3) == 5"] * 3,
}
GOOD = "def add(a, b): return a + b"
WRONG = "def add(a, b): return a * b"


@pytest.fixture
def synthetic_dataset(monkeypatch):
    def download(task, *args, **kwargs):
        task.dataset = {"test": Dataset.from_list([DOC])}

    monkeypatch.setattr(ConfigurableTask, "download", download)


def filter_responses(task, responses):
    task._instances = [
        Instance("generate_until", DOC, ("prompt", {}), 0, resps=responses)
    ]
    task.apply_filters()
    return next(iter(task.instances[0].filtered_resps.values()))


@pytest.mark.parametrize("family", ["humaneval", "mbpp"])
def test_import_without_opt_in_cannot_download_metric_or_execute(monkeypatch, family):
    monkeypatch.delenv("HF_ALLOW_CODE_EVAL", raising=False)
    download = Mock(side_effect=AssertionError("dynamic metric download"))
    launch = Mock(side_effect=AssertionError("candidate process at import"))
    score = Mock(side_effect=AssertionError("code scoring at import"))
    monkeypatch.setattr(evaluate, "load", download)
    monkeypatch.setattr(subprocess, "Popen", launch)
    monkeypatch.setattr(_code_eval, "compute", score)
    module = importlib.import_module(f"lm_eval.tasks.{family}.utils")
    importlib.reload(module)
    assert callable(module.build_predictions)
    download.assert_not_called()
    launch.assert_not_called()
    score.assert_not_called()


@pytest.mark.parametrize(
    "name,metric", [("humaneval", "pass@1"), ("mbpp", "pass_at_1")]
)
@pytest.mark.parametrize("response,expected", [(GOOD, 1.0), (WRONG, 0.0), ("", 0.0)])
def test_public_tasks_score_real_programs(
    monkeypatch, synthetic_dataset, name, metric, response, expected
):
    monkeypatch.setenv("HF_ALLOW_CODE_EVAL", "1")
    task = TaskManager().load([name])["tasks"][name]
    filtered = filter_responses(task, [response])
    assert filtered == [response]
    assert task.process_results(DOC, [filtered]) == {metric: expected}


@pytest.mark.parametrize("name", ["humaneval", "mbpp"])
def test_task_loading_is_safe_but_scoring_requires_runtime_opt_in(
    monkeypatch, synthetic_dataset, name
):
    monkeypatch.delenv("HF_ALLOW_CODE_EVAL", raising=False)
    task = TaskManager().load([name])["tasks"][name]
    filtered = filter_responses(task, [GOOD])
    with pytest.raises(ValueError, match="NOT a security sandbox"):
        task.process_results(DOC, [filtered])


def test_humaneval_64_real_scoring_with_installed_fork_guard(
    monkeypatch, synthetic_dataset, tmp_path
):
    monkeypatch.setenv("HF_ALLOW_CODE_EVAL", "1")
    task = TaskManager().load(["humaneval_64"])["tasks"]["humaneval_64"]
    filtered = filter_responses(task, [GOOD, WRONG] * 32)
    assert len(filtered) == 64
    with filelock.FileLock(tmp_path / "scoring.lock"):
        with (
            filelock._api._fork_transition(),
            pytest.raises(RuntimeError, match="unsafe while filelock"),
        ):
            sys.audit("os.fork")
        metrics = task.process_results(DOC, [filtered])
        with (
            filelock._api._fork_transition(),
            pytest.raises(RuntimeError, match="unsafe while filelock"),
        ):
            sys.audit("os.fork")
    assert metrics == {
        f"pass@{k}": 1 - math.comb(32, k) / math.comb(64, k)
        for k in [2, 8, 16, 32, 64]
    }


@pytest.mark.parametrize(
    "name,repeats", [("humaneval", 1), ("humaneval_64", 64), ("mbpp", 1)]
)
def test_public_predict_only_preserves_samples_without_execution(
    monkeypatch, synthetic_dataset, name, repeats
):
    monkeypatch.delenv("HF_ALLOW_CODE_EVAL", raising=False)
    score = Mock(side_effect=AssertionError("generation must not execute code"))
    download = Mock(side_effect=AssertionError("dynamic metric download"))
    launch = Mock(side_effect=AssertionError("generation must not start subprocesses"))
    monkeypatch.setattr(_code_eval, "compute", score)
    monkeypatch.setattr(evaluate, "load", download)
    # Environment metadata collection is independent of generation and may invoke
    # external commands. Keep only that reporting outside the subprocess trap.
    monkeypatch.setattr(evaluator, "get_git_commit_hash", lambda: "fixture")
    monkeypatch.setattr(evaluator, "add_env_info", lambda results: None)
    monkeypatch.setattr(evaluator, "add_tokenizer_info", lambda results, model: None)
    monkeypatch.setattr(subprocess, "Popen", launch)
    model = DummyLM()
    model.generate_until = Mock(side_effect=lambda requests: [GOOD for _ in requests])
    results = evaluator.simple_evaluate(
        model=model,
        tasks=[name],
        num_fewshot=0,
        predict_only=True,
        confirm_run_unsafe_code=True,
        bootstrap_iters=0,
        torch_random_seed=None,
        limit=1,
    )
    assert len(results["samples"][name]) == 1
    sample = results["samples"][name][0]
    assert sample["doc"] == DOC
    assert sample["resps"] == [[GOOD] * repeats]
    assert sample["filtered_resps"] == [[GOOD] * repeats]
    assert not any(key.startswith("pass") for key in results["results"][name])
    model.generate_until.assert_called_once()
    score.assert_not_called()
    download.assert_not_called()
    launch.assert_not_called()
