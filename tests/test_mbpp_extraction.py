"""Direct MBPP tasks accept chat answers and raw complete Python programs."""

import importlib
from pathlib import Path

import pytest
from datasets import Dataset

from lm_eval.api.instance import Instance
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
            "```python\ndef add(a, b): return a + b\n```\n   Explanation\n Explanation",
            "def add(a, b): return a + b\n",
        ),
        (
            "Here is the solution:\n```python\ndef add(a, b):\n    return a + b\n```\nThis returns the sum.",
            "def add(a, b):\n    return a + b\n",
        ),
        (
            "```PYTHON\r\ndef add(a, b):\r\n    return a + b\r\n```",
            "def add(a, b):\r\n    return a + b\r\n",
        ),
        ("```\ndef add(a, b): return a + b\n```", "def add(a, b): return a + b\n"),
        ("def add(a, b): return a + b\n```", "def add(a, b): return a + b\n"),
        (
            "Here is the solution:\n\ndef add(a, b):\n    return a + b\n\nThis returns the sum.",
            "Here is the solution:\n\ndef add(a, b):\n    return a + b\n\nThis returns the sum.",
        ),
        (
            "```python\nimport math\n```\n```python\ndef root(n):\n    return math.sqrt(n)\n```",
            "import math\n\ndef root(n):\n    return math.sqrt(n)\n",
        ),
        (
            "```text\nAn illustrative example\n```\n```py\ndef add(a, b): return a + b\n```",
            "def add(a, b): return a + b\n",
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
        ("```python\ndef broken(\n```", "def broken(\n"),
        ("```python\ndef unfinished():", ""),
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
def test_owned_execution_metric_scores_extracted_answers(
    mbpp, response, expected_score
):
    predictions = mbpp.build_predictions([[response]], [{"test_list": ["ignored"]}])
    assert mbpp.pass_at_1(["assert add(2, 3) == 5"], predictions) == expected_score


def test_imports_and_helpers_reach_owned_execution_metric(mbpp):
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


@pytest.mark.parametrize(
    "source",
    [
        'def add(a, b):\n    return """\n```python\ndef add(a, b):\n    return a + b\n```\n"""\n',
        'def add(a, b):\n    return """a\n    \nb"""\n',
        'def add(a, b):\n    return f"""a\n```\n{a}\n"""\n',
        "match True, False:\n    case _:\n        def add(a, b):\n            return a + b\n",
        "def add(a, b): return a + b\nresult is not None:\n",
        "\n\ndef add(a, b): return a + b\n\n",
    ],
)
@pytest.mark.parametrize("fenced", [False, True])
def test_program_bytes_survive_raw_or_longer_fenced_envelope(mbpp, source, fenced):
    response = f"````python\n{source}````" if fenced else source
    assert mbpp.build_predictions([[response, "", source]], [{}]) == [
        [source, "", source]
    ]


@pytest.mark.parametrize("mark", ["```", "````", "~~~"])
def test_fence_delimiters_pair_by_type_and_length(mbpp, mark):
    source = "def add(a, b): return a + b\n"
    response = f"{mark}text\nExample\n{mark}\n{mark}python\n{source}{mark}"
    assert mbpp.extract_code_blocks(response) == source


def test_legacy_closing_fence_cannot_hide_a_later_revision(mbpp):
    response = "def add(a, b): return a + b\n```\n```python\ndef add(a, b): return (\n"
    assert mbpp.extract_code_blocks(response) == response


@pytest.mark.parametrize("prefix", ["", "f"])
def test_fence_lines_inside_fenced_strings_are_not_delimiters(mbpp, prefix):
    source = f'def add(a, b):\n    return {prefix}"""a\n```\n{{a}}\n"""\n'
    assert mbpp.extract_code_blocks(f"```python\n{source}```") == source
    malformed = source + "result is not None:\n"
    assert mbpp.extract_code_blocks(malformed) == malformed


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
@pytest.mark.parametrize(
    "literal_char", ["\u2028", "\u2029", "\f", "\v", "\x85", "\x1c", "\x1d", "\x1e"]
)
def test_physical_newlines_preserve_literal_fences_and_execution(
    mbpp, monkeypatch, newline, literal_char
):
    value = literal_char * 3 + "\n```\n"
    source = newline.join(
        [
            "def add(a, b):",
            '    note = """' + literal_char * 3,
            "```",
            '"""',
            f"    assert note == {value!r}",
            "    return a + b",
            "",
        ]
    )
    compile(source, "<original>", "exec")
    response = "```python" + newline + source + "```"
    assert mbpp.build_predictions([[response, source]], [{}]) == [[source, source]]
    doc = {"text": "Add two numbers.", "test_list": ["assert add(2, 3) == 5"] * 3}

    def download(task, *args, **kwargs):
        task.dataset = {"test": Dataset.from_list([doc])}

    monkeypatch.setattr(ConfigurableTask, "download", download)
    task_path = Path(__file__).resolve().parents[1] / "lm_eval/tasks/mbpp/mbpp.yaml"
    task = ConfigurableTask(config=load_yaml(task_path))
    task._instances = [
        Instance("generate_until", doc, ("prompt", {}), 0, resps=[response])
    ]
    task.apply_filters()
    filtered = next(iter(task.instances[0].filtered_resps.values()))
    assert filtered == [source]
    assert task.process_results(doc, [filtered]) == {"pass_at_1": 1.0}


@pytest.mark.parametrize(
    "response,expected_score",
    [
        (
            'def add(a, b):\n    return """\n```python\ndef add(a, b):\n    return a + b\n```\n"""\n',
            0.0,
        ),
        ("def add(a, b): return a + b\nresult is not None:\n", 0.0),
        (
            "match True, False:\n    case _:\n        def add(a, b):\n            return a + b\n",
            1.0,
        ),
        (
            "```python\nassert False\n```\n```python\ndef add(a, b): return a + b\n```",
            0.0,
        ),
        (
            "```python\ndef add(a, b): return a + b\n```\n```python\ndef add(a, b): return a * b\n```",
            0.0,
        ),
        (
            "```python\ndef add(a, b): return a + b\nsaved_add = add\n```\n```python\ndef add(a, b): return saved_add(a, b)\n```",
            1.0,
        ),
        (
            "```python\ndef add(a, b): return a + b\n```\n```python\ndef add(a, b): return (\n",
            0.0,
        ),
        ("def add[T](a: T, b: T): return a + b\n", 1.0),
    ],
)
def test_public_filter_and_process_results_preserve_functional_outcomes(
    mbpp, monkeypatch, response, expected_score
):
    doc = {"text": "Add two numbers.", "test_list": ["assert add(2, 3) == 5"] * 3}

    def download(task, *args, **kwargs):
        task.dataset = {"test": Dataset.from_list([doc])}

    monkeypatch.setattr(ConfigurableTask, "download", download)
    task_path = Path(__file__).resolve().parents[1] / "lm_eval/tasks/mbpp/mbpp.yaml"
    task = ConfigurableTask(config=load_yaml(task_path))
    task._instances = [
        Instance("generate_until", doc, ("prompt", {}), 0, resps=[response])
    ]
    task.apply_filters()
    filtered = next(iter(task.instances[0].filtered_resps.values()))
    assert len(filtered) == 1
    assert task.process_results(doc, [filtered]) == {"pass_at_1": expected_score}


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
    assert config["metadata"]["version"] == 3.0
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
