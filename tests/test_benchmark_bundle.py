"""Offline scoring uses real task semantics and retains all recorded draws."""

import copy
import json
import os
import random
import socket
from pathlib import Path, PurePosixPath, PureWindowsPath
from unittest.mock import Mock

import datasets
import evaluate
import pytest

from benchmark_runner.runtime import bundle as runtime
from lm_eval import benchmark_bundle as bundle, evaluator
from lm_eval.api.task import ConfigurableTask
from lm_eval.models.dummy import DummyLM
from lm_eval.tasks import _code_eval


GOOD = "def add(a, b): return a + b"
REAL_SOURCE_IDENTITY = bundle.source_identity
WRONG = "def add(a, b): return a * b"
CODE_DOC = {
    "prompt": 'def add(a, b):\n    """Add two numbers."""\n',
    "entry_point": "add",
    "test": "def check(candidate):\n    assert candidate(2, 3) == 5",
    "text": "Add two numbers.",
    "test_list": ["assert add(2, 3) == 5"] * 3,
}
MATH_DOC = {"question": "What is 20 plus 22?", "answer": "Adding gives #### 42"}
IF_DOC = {
    "key": 1,
    "prompt": "Include apple and banana.",
    "instruction_id_list": ["keywords:existence", "punctuation:no_comma"],
    "kwargs": [{"keywords": ["apple", "banana"]}, {}],
}


@pytest.fixture(autouse=True)
def stable_source(monkeypatch):
    monkeypatch.setattr(
        evaluate, "load", Mock(side_effect=AssertionError("dynamic metric download"))
    )
    # Hashing the entire installed package is exercised separately; other tests
    # focus on the bundle boundary, not filesystem traversal cost.
    monkeypatch.setattr(
        runtime,
        "source_identity",
        lambda: {
            "package_sha256": "a" * 64,
            "helpers_sha256": "b" * 64,
        },
    )


def config(names):
    return bundle.validate_config(
        {
            "tasks": names,
            "model": "fixture",
            "base_url": "http://localhost:8888/v1/chat/completions",
            "gen_kwargs": {"temperature": 0.2, "max_gen_toks": 4096},
        }
    )


def fixture_bundle(name="humaneval", documents=None, responses=None, ids=None):
    cfg = bundle._config(name)
    documents = documents or [
        CODE_DOC
        if name.startswith(("humaneval", "mbpp"))
        else MATH_DOC
        if name == "gsm8k_cot_zeroshot"
        else IF_DOC
    ]
    repeats = cfg.get("repeats", 1)
    responses = (
        responses if responses is not None else [[GOOD] * repeats for _ in documents]
    )
    ids = ids or list(range(len(documents)))
    filters = bundle._filters(cfg)
    rows = [
        {
            "doc_id": doc_id,
            "doc": doc,
            "resps": [raw],
            "filter": name_filter,
            "doc_hash": "c" * 64,
            "prompt_hash": "d" * 64,
            "target_hash": "e" * 64,
        }
        for doc_id, doc, raw in zip(ids, documents, responses, strict=True)
        for name_filter in filters
    ]
    info = {
        name: {
            "version": cfg["metadata"]["version"],
            "repeats": repeats,
            "filters": filters,
            "doc_ids": ids,
            "dataset": {
                "path": cfg["dataset_path"],
                "name": cfg.get("dataset_name"),
                "split": cfg["test_split"],
                "fingerprint": "fixture-fingerprint",
                "num_rows": max(ids) + 1,
            },
        }
    }
    provenance_config = config([name])
    if ids != list(range(len(documents))):
        provenance_config["samples"] = {name: ids}
    return bundle.build_bundle(
        {name: rows}, info, bundle._provenance(provenance_config)
    )


def reseal(value):
    value["sha256"] = bundle._hash({k: v for k, v in value.items() if k != "sha256"})


@pytest.mark.parametrize("name", list(bundle.TASK_FILES))
def test_all_normal_tasks_validate_without_dataset_access(monkeypatch, name):
    monkeypatch.setattr(
        datasets, "load_dataset", Mock(side_effect=AssertionError("network dataset"))
    )
    value = fixture_bundle(name)
    bundle.validate_bundle(value)
    assert value["tasks"][name]["records"][0]["responses"] == [GOOD] * (
        64 if "64" in name else 1
    )


@pytest.mark.parametrize(
    "name,metric",
    [("humaneval", "pass@1,create_test"), ("mbpp", "pass_at_1,extract_code")],
)
def test_real_code_scoring_preserves_wrong_and_empty(monkeypatch, name, metric):
    monkeypatch.setenv("HF_ALLOW_CODE_EVAL", "1")
    monkeypatch.setattr(
        datasets, "load_dataset", Mock(side_effect=AssertionError("download"))
    )
    monkeypatch.setattr(
        socket, "create_connection", Mock(side_effect=AssertionError("network"))
    )
    value = fixture_bundle(name, [CODE_DOC] * 3, [[GOOD], [WRONG], [None]])
    assert value["tasks"][name]["records"][2]["responses"] == [""]
    result = bundle.score(value)
    assert result["format"] == "lm-eval-offline-results-v1"
    assert result["results"][name]["metrics"][metric] == pytest.approx(1 / 3)
    assert [r["doc_id"] for r in result["results"][name]["samples"]] == [0, 1, 2]


def test_humaneval_64_keeps_every_ordered_draw(monkeypatch):
    monkeypatch.setenv("HF_ALLOW_CODE_EVAL", "1")
    value = fixture_bundle("humaneval_64", responses=[[GOOD] + [WRONG] * 63])
    result = bundle.score(value)["results"]["humaneval_64"]
    assert result["metrics"] == {
        f"pass@{k},create_test": k / 64 for k in [2, 8, 16, 32, 64]
    }
    assert (
        result["samples"][0]["filters"]["create_test"]["filtered_response"]
        == [GOOD] + [WRONG] * 63
    )


def test_gsm_keeps_both_filters_and_wrong_document():
    value = fixture_bundle(
        "gsm8k_cot_zeroshot",
        [MATH_DOC] * 3,
        [["The answer is 42."], ["42"], ["The answer is 99."]],
    )
    result = bundle.score(value)["results"]["gsm8k_cot_zeroshot"]
    assert result["metrics"] == {
        "exact_match,strict-match": 1 / 3,
        "exact_match,flexible-extract": 2 / 3,
    }
    assert len(result["samples"]) == 3
    assert set(result["samples"][0]["filters"]) == {"strict-match", "flexible-extract"}


def test_scored_task_reports_mean_stderr_per_metric_and_filter():
    value = fixture_bundle(
        "gsm8k_cot_zeroshot",
        [MATH_DOC] * 3,
        [["The answer is 42."], ["42"], ["The answer is 99."]],
    )
    result = bundle.score(value)["results"]["gsm8k_cot_zeroshot"]
    # Per-document values are [1, 0, 0] and [1, 1, 0]; for both, the sample
    # variance is 1/3, so the standard error of the mean is sqrt(1/3 / 3) = 1/3.
    assert result["stderr"] == {
        "exact_match_stderr,strict-match": pytest.approx(1 / 3),
        "exact_match_stderr,flexible-extract": pytest.approx(1 / 3),
    }
    assert set(result) == {"version", "num_documents", "metrics", "stderr", "samples"}


def test_stderr_is_none_without_estimator_or_with_one_value():
    second = {
        "key": 2,
        "prompt": "Use lowercase.",
        "instruction_id_list": ["change_case:english_lowercase"],
        "kwargs": [{}],
    }
    value = fixture_bundle(
        "ifeval", [IF_DOC, second], [["apple banana"], ["UPPERCASE"]]
    )
    stderr = bundle.score(value)["results"]["ifeval"]["stderr"]
    # Prompt-level values [1, 0] have sample variance 1/2: sqrt(1/2 / 2) = 1/2.
    # Instruction-level accuracy has no registered stderr estimator.
    assert stderr == {
        "prompt_level_strict_acc_stderr,none": pytest.approx(0.5),
        "prompt_level_loose_acc_stderr,none": pytest.approx(0.5),
        "inst_level_strict_acc_stderr,none": None,
        "inst_level_loose_acc_stderr,none": None,
    }
    single = bundle.score(
        fixture_bundle("gsm8k_cot_zeroshot", [MATH_DOC], [["The answer is 42."]])
    )["results"]["gsm8k_cot_zeroshot"]
    assert single["stderr"] == {
        "exact_match_stderr,strict-match": None,
        "exact_match_stderr,flexible-extract": None,
    }


def test_ifeval_instruction_weighting_is_not_prompt_mean():
    second = {
        "key": 2,
        "prompt": "Use lowercase.",
        "instruction_id_list": ["change_case:english_lowercase"],
        "kwargs": [{}],
    }
    value = fixture_bundle(
        "ifeval", [IF_DOC, second], [["apple banana"], ["UPPERCASE"]]
    )
    metrics = bundle.score(value)["results"]["ifeval"]["metrics"]
    assert metrics == {
        "prompt_level_strict_acc,none": 0.5,
        "prompt_level_loose_acc,none": 0.5,
        "inst_level_strict_acc,none": 2 / 3,
        "inst_level_loose_acc,none": 2 / 3,
    }


def test_selected_original_ids_and_json_roundtrip():
    value = fixture_bundle(
        "gsm8k_cot_zeroshot", [MATH_DOC] * 2, [["The answer is 42."]] * 2, [8, 3]
    )
    result = bundle.score(json.loads(bundle._json(value)))
    assert [
        s["doc_id"] for s in result["results"]["gsm8k_cot_zeroshot"]["samples"]
    ] == [8, 3]


@pytest.mark.parametrize(
    "mutation",
    [
        "late_missing",
        "extra",
        "reorder_candidates",
        "wrong_version",
        "source",
        "helpers",
        "duplicate_ids",
        "reordered_records",
        "missing_doc",
        "wrong_dataset",
        "bad_doc",
        "hash",
    ],
)
def test_invalid_complete_bundle_never_executes(monkeypatch, tmp_path, mutation):
    value = fixture_bundle("humaneval_64", [CODE_DOC] * 2, [[GOOD, WRONG] * 32] * 2)
    info = value["tasks"]["humaneval_64"]
    if mutation == "late_missing":
        info["records"][1]["responses"].pop()
    elif mutation == "extra":
        info["records"][1]["responses"].append(GOOD)
    elif mutation == "reorder_candidates":
        info["records"][1]["responses"].reverse()
    elif mutation == "wrong_version":
        info["version"] = 999
    elif mutation in {"source", "helpers"}:
        value["source"][
            "package_sha256" if mutation == "source" else "helpers_sha256"
        ] = "f" * 64
    elif mutation == "duplicate_ids":
        info["doc_ids"] = [0, 0]
    elif mutation == "reordered_records":
        info["records"].reverse()
    elif mutation == "missing_doc":
        info["records"].pop()
        info["doc_ids"].pop()
    elif mutation == "wrong_dataset":
        info["dataset"]["path"] = "other"
    elif mutation == "bad_doc":
        info["records"][1]["doc"] = {}
    else:
        info["records"][0]["harness_hashes"]["doc_hash"] = "not-a-hash"
    reseal(value)
    execute = Mock(side_effect=AssertionError("must not execute"))
    monkeypatch.setattr(_code_eval, "compute", execute)
    output = tmp_path / "scores.json"
    with pytest.raises((ValueError, TypeError)):
        bundle.score(value, output)
    execute.assert_not_called()
    assert not output.exists()


def test_unsealed_response_change_rejected():
    value = fixture_bundle()
    value["tasks"]["humaneval"]["records"][0]["responses"][0] = WRONG
    with pytest.raises(ValueError, match="integrity"):
        bundle.validate_bundle(value)


@pytest.mark.parametrize(
    "kind",
    ["conflict", "same_filter", "missing_filter", "missing_task", "null_mismatch"],
)
def test_duplicate_filter_rows_only_merge_complete_equivalent_samples(kind):
    value = fixture_bundle("gsm8k_cot_zeroshot", responses=[["42"]])
    info = value["tasks"]["gsm8k_cot_zeroshot"]
    info = {k: v for k, v in info.items() if k != "records"}
    row = {
        "doc_id": 0,
        "doc": MATH_DOC,
        "resps": [["42"]],
        "filter": "strict-match",
        "doc_hash": "c" * 64,
        "prompt_hash": "d" * 64,
        "target_hash": "e" * 64,
    }
    other = copy.deepcopy(row)
    other["filter"] = "flexible-extract"
    rows = [row, other]
    if kind == "conflict":
        other["prompt_hash"] = "f" * 64
    elif kind == "null_mismatch":
        row["resps"], other["resps"] = [[None]], [[""]]
    elif kind == "same_filter":
        other["filter"] = "strict-match"
    elif kind == "missing_filter":
        rows.pop()
    samples = {} if kind == "missing_task" else {"gsm8k_cot_zeroshot": rows}
    with pytest.raises(ValueError):
        bundle.build_bundle(samples, {"gsm8k_cot_zeroshot": info}, value["provenance"])


@pytest.mark.parametrize(
    "extra",
    [
        {"use_cache": "cache"},
        {"header": {"Authorization": "secret"}},
        {"tasks": ["../plugin.yaml"]},
        {"base_url": "http://user:password@host/v1/chat/completions"},
        {"base_url": "http://host/v1/chat/completions?api_key=secret"},
        {"gen_kwargs": {"max_gen_toks": 10}},
        {"gen_kwargs": {"temperature": float("nan"), "max_gen_toks": 10}},
        {"gen_kwargs": {"temperature": 0, "max_gen_toks": 10, "header": "secret"}},
    ],
)
def test_config_rejects_cache_credentials_plugins_and_implicit_temperature(extra):
    with pytest.raises(ValueError):
        bundle.validate_config(config(["humaneval"]) | extra)


def test_atomic_publication_refuses_existing_and_cleans_failure(monkeypatch, tmp_path):
    output = tmp_path / "result.json"
    bundle.write_artifact(output, {"complete": True})
    with pytest.raises(FileExistsError):
        bundle.write_artifact(output, {"complete": False})
    assert json.loads(output.read_text()) == {"complete": True}
    new_output = tmp_path / "new.json"
    monkeypatch.setattr(os, "link", Mock(side_effect=OSError("publication failed")))
    with pytest.raises(OSError, match="publication failed"):
        bundle.write_artifact(new_output, {"complete": True})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["result.json"]


def test_backend_infrastructure_error_aborts_without_partial_result(
    monkeypatch, tmp_path
):
    value = fixture_bundle(documents=[CODE_DOC, CODE_DOC])
    monkeypatch.setattr(
        _code_eval,
        "compute",
        Mock(
            side_effect=[
                ({"pass@1": 1.0}, {}),
                RuntimeError("worker startup failure"),
            ]
        ),
    )
    output = tmp_path / "scores.json"
    with pytest.raises(RuntimeError, match="worker startup failure"):
        bundle.score(value, output)
    assert not output.exists()


@pytest.mark.parametrize(
    "name", ["humaneval", "humaneval_64", "mbpp", "gsm8k_cot_zeroshot", "ifeval"]
)
def test_actual_predict_only_generation_without_code_opt_in(
    monkeypatch, tmp_path, name
):
    monkeypatch.delenv("HF_ALLOW_CODE_EVAL", raising=False)
    doc = (
        CODE_DOC
        if name.startswith(("humaneval", "mbpp"))
        else MATH_DOC
        if name.startswith("gsm8k")
        else IF_DOC
    )

    def download(task, *args, **kwargs):
        data = datasets.Dataset.from_list([doc])
        task.dataset = {"test": data, "train": data}

    monkeypatch.setattr(ConfigurableTask, "download", download)
    monkeypatch.setattr(evaluator, "get_git_commit_hash", lambda: "fixture")
    monkeypatch.setattr(evaluator, "add_env_info", lambda results: None)
    monkeypatch.setattr(evaluator, "add_tokenizer_info", lambda results, model: None)
    execute = Mock(side_effect=AssertionError("generation executed candidate"))
    monkeypatch.setattr(_code_eval, "compute", execute)
    actual_evaluate = evaluator.simple_evaluate
    captured = {}

    def evaluate(**kwargs):
        captured.update(kwargs)
        model = DummyLM()
        model.apply_chat_template = lambda messages, **options: json.dumps(messages)
        model.generate_until = lambda requests: [GOOD for _ in requests]
        kwargs["model"] = model
        return actual_evaluate(**kwargs)

    monkeypatch.setattr(evaluator, "simple_evaluate", evaluate)
    output, transport = tmp_path / "bundle.json", tmp_path / "transport.jsonl"
    cfg = config([name]) | {"num_fewshot": 0, "limit": 1}
    result = bundle.generate(cfg, output, transport)
    assert captured["model"] == "local-chat-completions"
    assert captured["use_cache"] is None and captured["cache_requests"] is False
    assert captured["predict_only"] is True and captured["apply_chat_template"] is True
    assert captured["model_args"]["seed"] == 1234
    assert captured["model_args"]["tokenized_requests"] is False
    assert "base_url" not in result["provenance"]
    assert result["provenance"]["gen_kwargs"]["temperature"] == 0.2
    assert result["tasks"][name]["dataset"]["fingerprint"]
    assert result["tasks"][name]["records"][0]["responses"] == [GOOD] * (
        64 if "64" in name else 1
    )
    assert json.loads(output.read_text()) == result
    assert transport.stat().st_mode & 0o777 == 0o600
    execute.assert_not_called()


def test_generation_failure_keeps_log_but_no_bundle(monkeypatch, tmp_path):
    monkeypatch.setattr(
        runtime,
        "ConfigurableTask",
        Mock(side_effect=RuntimeError("dataset unavailable")),
    )
    output, transport = tmp_path / "bundle.json", tmp_path / "transport.jsonl"
    with pytest.raises(RuntimeError, match="dataset unavailable"):
        bundle.generate(config(["humaneval"]), output, transport)
    assert not output.exists() and transport.exists()


def test_cli_score_roundtrip(tmp_path):
    value = fixture_bundle("gsm8k_cot_zeroshot", responses=[["The answer is 42."]])
    source, output = tmp_path / "bundle.json", tmp_path / "scores.json"
    bundle.write_artifact(source, value)
    bundle.main(["score", "--input", str(source), "--output", str(output)])
    assert json.loads(output.read_text())["results"]["gsm8k_cot_zeroshot"][
        "metrics"
    ] == {
        "exact_match,strict-match": 1.0,
        "exact_match,flexible-extract": 1.0,
    }


def test_source_identity_detects_installed_helper_and_yaml_changes(
    monkeypatch, tmp_path
):
    package = tmp_path / "lm_eval"
    helpers = package / "tasks" / "humaneval"
    helpers.mkdir(parents=True)
    helper = helpers / "utils.py"
    helper.write_text("x = 1\n")
    (package / "task.yaml").write_text("task: fixture\n")
    monkeypatch.setattr(runtime, "PACKAGE", package)
    before = REAL_SOURCE_IDENTITY()
    (package / "ignored.pyc").write_bytes(b"cache")
    assert REAL_SOURCE_IDENTITY() == before
    helper.write_text("x = 2\n")
    after = REAL_SOURCE_IDENTITY()
    assert before["package_sha256"] != after["package_sha256"]
    assert before["helpers_sha256"] != after["helpers_sha256"]
    (package / "task.yaml").write_text("task: changed\n")
    assert REAL_SOURCE_IDENTITY()["package_sha256"] != after["package_sha256"]


def test_source_identity_detects_benchmark_runtime_changes(monkeypatch, tmp_path):
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    module = runtime_dir / "bundle.py"
    module.write_text("x = 1\n")
    monkeypatch.setattr(runtime, "RUNTIME", runtime_dir)
    before = REAL_SOURCE_IDENTITY()
    (runtime_dir / "notes.txt").write_text("not source\n")
    assert REAL_SOURCE_IDENTITY() == before
    module.write_text("x = 2\n")
    after = REAL_SOURCE_IDENTITY()
    assert before["benchmark_runtime"] != after["benchmark_runtime"]
    assert before["package_sha256"] == after["package_sha256"]


@pytest.mark.parametrize(
    ("windows", "posix", "helper"),
    [
        ("tasks\\humaneval\\utils.py", "tasks/humaneval/utils.py", True),
        ("api\\task.py", "api/task.py", True),
        ("tasks\\arc\\utils.py", "tasks/arc/utils.py", False),
    ],
)
def test_source_keys_match_across_windows_and_posix(windows, posix, helper):
    windows_key = bundle._source_key(PureWindowsPath(windows))
    posix_key = bundle._source_key(PurePosixPath(posix))
    assert windows_key == posix_key == posix
    assert bundle._is_helper(windows_key) is helper
    assert bundle._is_helper(posix_key) is helper


@pytest.mark.parametrize("nested", [False, True])
def test_reasoning_controls_are_explicit_and_preserved(nested):
    cfg = config(["humaneval"])
    controls = {"reasoning_effort": "high", "enable_thinking": False}
    cfg["gen_kwargs"].update({"chat_template_kwargs": controls} if nested else controls)
    assert bundle.validate_config(cfg)["gen_kwargs"] == cfg["gen_kwargs"]
    controls["reasoning_effort"] = "invalid"
    cfg["gen_kwargs"].update({"chat_template_kwargs": controls} if nested else controls)
    with pytest.raises(ValueError, match="reasoning_effort"):
        bundle.validate_config(cfg)


def test_missing_nltk_resource_fails_before_helper_can_download(monkeypatch):
    import nltk

    monkeypatch.setattr(
        nltk.data, "find", Mock(side_effect=LookupError("missing punkt_tab"))
    )
    download = Mock(side_effect=AssertionError("network download"))
    monkeypatch.setattr(nltk, "download", download)
    with pytest.raises(LookupError, match="missing punkt_tab"):
        bundle._config("ifeval")
    download.assert_not_called()


def test_mixed_task_roundtrip_and_late_invalid_instruction_cannot_execute(monkeypatch):
    code = fixture_bundle()
    instruction = fixture_bundle("ifeval", responses=[["apple banana"]])
    code["tasks"].update(instruction["tasks"])
    code["provenance"]["tasks"].append("ifeval")
    reseal(code)
    bundle.validate_bundle(json.loads(bundle._json(code)))
    code["tasks"]["ifeval"]["records"][0]["doc"] = copy.deepcopy(IF_DOC)
    code["tasks"]["ifeval"]["records"][0]["doc"]["kwargs"][0] = {
        "unknown_argument": "bad"
    }
    reseal(code)
    execute = Mock(side_effect=AssertionError("must not execute"))
    monkeypatch.setattr(_code_eval, "compute", execute)
    with pytest.raises(TypeError):
        bundle.score(code)
    execute.assert_not_called()


def test_publication_race_does_not_replace_other_completed_artifact(
    monkeypatch, tmp_path
):
    output = tmp_path / "result.json"
    real_link = os.link

    def competing_publish(source, target):
        output.write_text('{"other":true}')
        real_link(source, target)

    monkeypatch.setattr(os, "link", competing_publish)
    with pytest.raises(FileExistsError):
        bundle.write_artifact(output, {"mine": True})
    assert json.loads(output.read_text()) == {"other": True}
    assert list(tmp_path.iterdir()) == [output]


def randomized_instruction_bundle(seed):
    # Missing num_bullets is an accepted checker input with a random default.
    # This tests the callable contract, not prevalence in the published dataset.
    doc = {
        "key": 0,
        "prompt": "Write exactly two bullet points.",
        "instruction_id_list": ["detectable_format:number_bullet_lists"],
        "kwargs": [{}],
    }
    response = "* one\n* two"
    value = fixture_bundle("ifeval", [doc], [[response]])
    value["provenance"]["seed"] = seed
    reseal(value)
    return value, doc, response


@pytest.mark.parametrize("seed", range(6))
def test_replay_matches_direct_instruction_scoring_under_recorded_seed(seed):
    from lm_eval.tasks.ifeval.utils import process_results

    value, doc, response = randomized_instruction_bundle(seed)
    outer_state = random.getstate()
    try:
        random.seed(seed)
        direct = process_results(doc, [response])
        random.seed(9001)
        before = random.getstate()
        first = bundle.score(value)
        assert random.getstate() == before
        # Repeated validation and replay cannot shift either caller state or
        # scoring draws, including after a JSON serialization round trip.
        bundle.validate_bundle(value)
        assert random.getstate() == before
        second = bundle.score(json.loads(bundle._json(value)))
        assert random.getstate() == before
        assert first == second
        assert (
            first["results"]["ifeval"]["samples"][0]["filters"]["none"]["metrics"]
            == direct
        )
        if seed == 1:
            assert direct["prompt_level_strict_acc"] is True
            assert direct["prompt_level_loose_acc"] is False
    finally:
        random.setstate(outer_state)


def test_validation_failure_after_randomized_defaults_restores_caller_rng():
    value, _, _ = randomized_instruction_bundle(1)
    record = value["tasks"]["ifeval"]["records"][0]
    # The response hash is checked after checker construction consumed draws.
    record["response_hashes"] = ["wrong"]
    reseal(value)
    before = random.getstate()
    with pytest.raises(ValueError, match="Changed or reordered candidates"):
        bundle.validate_bundle(value)
    assert random.getstate() == before
    with pytest.raises(ValueError, match="Changed or reordered candidates"):
        bundle.score(value)
    assert random.getstate() == before


def test_scoring_failure_restores_caller_rng(monkeypatch, tmp_path):
    value, _, _ = randomized_instruction_bundle(1)

    def fail_publication(*args):
        random.random()
        raise OSError("publication failed after scoring")

    monkeypatch.setattr(runtime, "write_artifact", fail_publication)
    before = random.getstate()
    with pytest.raises(OSError, match="publication failed after scoring"):
        bundle.score(value, tmp_path / "scores.json")
    assert random.getstate() == before


def test_postpublication_cleanup_failure_is_warning_and_success(
    monkeypatch, tmp_path, caplog
):
    output = tmp_path / "scores.json"
    monkeypatch.setattr(Path, "unlink", Mock(side_effect=OSError("cleanup failed")))
    bundle.write_artifact(output, {"complete": True})
    assert json.loads(output.read_text()) == {"complete": True}
    temporaries = list(tmp_path.glob(".scores.json.*"))
    assert len(temporaries) == 1
    assert temporaries[0].stat().st_ino == output.stat().st_ino
    assert "publication committed" in caplog.text
    assert "cleanup failed" in caplog.text


@pytest.mark.parametrize("failure_stage", ["fsync", "link"])
def test_prepublication_cleanup_failure_preserves_original_error(
    monkeypatch, tmp_path, caplog, failure_stage
):
    output = tmp_path / "scores.json"
    original_error = OSError(f"{failure_stage} failed")
    monkeypatch.setattr(os, failure_stage, Mock(side_effect=original_error))
    monkeypatch.setattr(
        Path, "unlink", Mock(side_effect=OSError("cleanup also failed"))
    )
    with pytest.raises(OSError) as raised:
        bundle.write_artifact(output, {"complete": True})
    assert raised.value is original_error
    assert not output.exists()
    assert len(list(tmp_path.glob(".scores.json.*"))) == 1
    assert "not published" in caplog.text
    assert "cleanup also failed" in caplog.text
