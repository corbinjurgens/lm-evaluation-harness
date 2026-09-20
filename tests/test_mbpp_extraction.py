"""Direct MBPP tasks accept chat answers and raw complete Python programs."""

import importlib
from pathlib import Path

import pytest
from datasets import Dataset

from lm_eval.api.task import ConfigurableTask
from lm_eval.tasks import TaskManager
from lm_eval.tasks._yaml_loader import load_yaml
from lm_eval.utils import apply_template


@pytest.fixture(scope="module")
def mbpp():
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("HF_ALLOW_CODE_EVAL", "1")
        yield importlib.import_module("lm_eval.tasks.mbpp.utils")


@pytest.mark.parametrize(
    "response,expected",
    [
        ("def add(a, b):\n    return a + b", "def add(a, b):\n    return a + b"),
        (
            "Here is the solution:\n```python\ndef add(a, b):\n    return a + b\n```\nThis returns the sum.",
            "def add(a, b):\n    return a + b",
        ),
        (
            "```PYTHON\r\ndef add(a, b):\r\n    return a + b\r\n```",
            "def add(a, b):\n    return a + b",
        ),
        ("```\ndef add(a, b): return a + b\n```", "def add(a, b): return a + b"),
        ("def add(a, b): return a + b\n```", "def add(a, b): return a + b"),
        (
            "Here is the solution:\n\ndef add(a, b):\n    return a + b\n\nThis returns the sum.",
            "def add(a, b):\n    return a + b",
        ),
        (
            "```python\nimport math\n```\n```python\ndef root(n):\n    return math.sqrt(n)\n```",
            "import math\n\ndef root(n):\n    return math.sqrt(n)",
        ),
        (
            "```text\nAn illustrative example\n```\n```py\ndef add(a, b): return a + b\n```",
            "def add(a, b): return a + b",
        ),
        (
            "assert False\ndef add(a, b): return a + b",
            "assert False\ndef add(a, b): return a + b",
        ),
        (
            "def add(a, b): return a + b\nassert this is wrong.",
            "def add(a, b): return a + b\nassert this is wrong.",
        ),
        ("", ""),
        ("  \n ", ""),
        ("```python\ndef broken(\n```", "def broken("),
        ("```python\ndef unfinished():", "```python\ndef unfinished():"),
    ],
)
def test_extract_complete_code_preserves_program(mbpp, response, expected):
    assert mbpp.extract_code_blocks(response) == expected


@pytest.mark.parametrize(
    "response,expected_score",
    [
        ("def add(a, b): return a + b", 1.0),
        ("```python\ndef add(a, b): return a + b\n```", 1.0),
        ("Here is the solution:\n```python\ndef add(a, b): return a + b\n```", 1.0),
        ("def add(a, b): return a * b", 0.0),
        ("```python\ndef add(a, b): return a * b\n```", 0.0),
        ("", 0.0),
        ("pass", 0.0),
        ("assert True", 0.0),
        ("def add(a, b): return a + b\nassert False", 0.0),
        ("def add(a, b): return a + b\nassert this is wrong.", 0.0),
        ("```python\ndef add(a, b):\n```", 0.0),
    ],
)
def test_original_execution_metric_scores_extracted_answers(
    mbpp, response, expected_score
):
    predictions = mbpp.build_predictions([[response]], [{"test_list": ["ignored"]}])
    assert mbpp.pass_at_1(["assert add(2, 3) == 5"], predictions) == expected_score


def test_imports_and_helpers_reach_original_execution_metric(mbpp):
    response = """```python
import math

def helper(n):
    return math.isqrt(n)
```
```python
def root(n):
    return helper(n)
```"""
    predictions = mbpp.build_predictions([[response]], [{}])
    assert mbpp.pass_at_1(["assert root(16) == 4"], predictions) == 1.0


def test_filter_does_not_infer_function_name_from_reference_assertion(mbpp):
    response = "def answer(): return [2, 1]"
    docs = [{"test_list": ["assert sorted(answer()) == [1, 2]"]}]
    assert mbpp.build_predictions([[response, "pass"]], docs) == [[response, "pass"]]
    assert mbpp.pass_at_1(docs[0]["test_list"], [[response]]) == 1.0


@pytest.mark.parametrize("name", ["mbpp", "mbpp_instruct", "mbpp_plus"])
def test_fewshot_context_separates_complete_answers(mbpp, monkeypatch, name):
    doc = {
        "text": "Add two numbers.",
        "test_list": [
            "assert add(2, 3) == 5",
            "assert add(0, 1) == 1",
            "assert add(2, 2) == 4",
        ],
    }

    def download(task, *args, **kwargs):
        task.dataset = {"test": Dataset.from_list([doc])}

    monkeypatch.setattr(ConfigurableTask, "download", download)
    task = TaskManager().load([name])["tasks"][name]
    samples = mbpp.list_fewshot_samples()
    text_context = task.fewshot_context(doc, num_fewshot=3)
    for sample in samples:
        assert (
            sample["test_list"][2] + "\n```python\n" + sample["code"] + "\n```"
            in text_context
        )
    assert text_context.endswith("assert add(2, 2) == 4")

    messages = []

    def chat_template(turns, **kwargs):
        assert kwargs["add_generation_prompt"] is True
        messages.extend(turns)
        return "rendered chat"

    assert (
        task.fewshot_context(
            doc,
            num_fewshot=3,
            apply_chat_template=True,
            chat_template=chat_template,
        )
        == "rendered chat"
    )
    assert messages == [{"role": "user", "content": text_context}]

    messages.clear()
    assert (
        task.fewshot_context(
            doc,
            num_fewshot=3,
            apply_chat_template=True,
            fewshot_as_multiturn=True,
            chat_template=chat_template,
        )
        == "rendered chat"
    )
    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    for index, sample in enumerate(samples):
        assert messages[2 * index]["content"].endswith(sample["test_list"][2])
        assert (
            messages[2 * index + 1]["content"]
            == "```python\n" + sample["code"] + "\n```"
        )
    assert messages[-1]["content"].endswith("assert add(2, 2) == 4")


@pytest.mark.parametrize(
    "name,dataset,shots",
    [
        ("mbpp", "google-research-datasets/mbpp", 3),
        ("mbpp_instruct", "google-research-datasets/mbpp", 3),
        ("mbpp_plus", "evalplus/mbppplus", 3),
        ("mbpp_plus_instruct", "evalplus/mbppplus", 0),
    ],
)
def test_task_family_prompts_and_metrics(mbpp, name, dataset, shots):
    path = Path(__file__).parent.parent / "lm_eval" / "tasks" / "mbpp" / f"{name}.yaml"
    config = load_yaml(path)
    assert config["task"] == name
    assert config["dataset_path"] == dataset
    assert config["num_fewshot"] == shots
    assert config["metadata"]["version"] == 2.0
    assert config.get("gen_prefix", "") == ""
    assert config["generation_kwargs"] == {
        "max_gen_toks": 4096,
        "until": [],
        "do_sample": False,
    }
    assert config["metric_list"][0]["metric"].__name__ == "pass_at_1"
    assert (
        config["filter_list"][0]["filter"][0]["filter_fn"].__name__
        == "build_predictions"
    )
    doc = {
        "text": "Add two numbers.",
        "test_list": [
            "assert add(2, 3) == 5",
            "assert add(0, 1) == 1",
            "assert add(2, 2) == 4",
        ],
    }
    prompt = apply_template(config["doc_to_text"], doc)
    assert "Write complete Python code" in prompt
    assert "Task: Add two numbers." in prompt
    assert not prompt.endswith("```python\n")
    expected_tests = (
        "assert add(2, 3) == 5"
        if name == "mbpp_plus_instruct"
        else "assert add(2, 3) == 5\nassert add(0, 1) == 1\nassert add(2, 2) == 4"
    )
    assert apply_template(config["doc_to_target"], doc) == expected_tests
    if "plus" in name:
        doc["prompt"] = "Add two numeric inputs."
        assert "Task: Add two numeric inputs." in apply_template(
            config["doc_to_text"], doc
        )
    if shots:
        samples = mbpp.list_fewshot_samples()
        assert [sample["task_id"] for sample in samples] == [2, 3, 4]
        for sample in samples:
            target = apply_template(config["doc_to_target"], sample)
            assert target == "```python\n" + sample["code"] + "\n```"
            assert "Write complete Python code" in apply_template(
                config["doc_to_text"], sample
            )
