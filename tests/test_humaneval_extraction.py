"""HumanEval's public tasks accept full answers and legacy continuations."""

import importlib.util
import runpy
import sys
from pathlib import Path
from unittest.mock import Mock

import evaluate
import pytest
from datasets import Dataset

from lm_eval.api.instance import Instance
from lm_eval.api.task import ConfigurableTask
from lm_eval.tasks import TaskManager
from lm_eval.tasks._yaml_loader import load_yaml


TASK_DIR = Path(__file__).resolve().parents[1] / "lm_eval/tasks/humaneval"
DOC = {
    "prompt": 'from typing import List\n\ndef total(values: List[int]) -> int:\n    """Sum the values."""\n',
    "entry_point": "total",
    "test": "def check(candidate):\n    assert candidate([1, 2, 3]) == 6",
}
FULL = "def total(values):\n    return sum(values)\n"


@pytest.fixture
def utils(monkeypatch):
    # Extraction/config tests do not download or execute the external code metric.
    monkeypatch.setattr(evaluate, "load", Mock(return_value=Mock()))
    path = TASK_DIR / "utils.py"
    name = "lm_eval.tasks.humaneval.utils"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.__mtime__ = path.stat().st_mtime_ns
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.mark.parametrize(
    "response,expected",
    [
        (FULL, FULL),
        (f"Here is the solution:\n```python\n{FULL}```\nDone.", FULL),
        (f"```py\n{FULL}```", FULL),
        (f"```\n{FULL}```", FULL),
        (
            "```Python\r\ndef total(values):\r\n    return sum(values)\r\n```\r\n",
            "def total(values):\r\n    return sum(values)\n",
        ),
        (
            "    ```python\n    def total(values):\n        return sum(values)\n    ```",
            FULL,
        ),
        ("    return sum(values)", DOC["prompt"] + "    return sum(values)\n"),
        (
            "\n    return sum(values)\n```\nAn explanation.",
            DOC["prompt"] + "    return sum(values)\n",
        ),
        (
            "```python\n    return sum(values)\n```",
            DOC["prompt"] + "    return sum(values)\n",
        ),
        (f"{FULL}```\nAn explanation.", FULL),
        ("", ""),
        ("   \n", ""),
        ("I cannot solve this problem.", ""),
        ("```python\ndef another(values):\n    return 6\n```", ""),
    ],
)
def test_response_formats(utils, response, expected):
    assert utils.build_predictions([[response]], [DOC]) == [[expected]]
    assert utils.build_predictions_instruct([[response]], [DOC]) == [[expected]]


def test_complete_module_preserves_imports_helpers_and_decorators(utils, tmp_path):
    code = (
        "from functools import lru_cache\n"
        "from typing import List\n\n"
        "def helper(values):\n    return sum(values)\n\n"
        "@lru_cache()\n"
        "def total(values: tuple[int, ...]) -> int:\n    return helper(values)\n"
    )
    result = utils.build_predictions([[f"```python\n{code}```"]], [DOC])[0][0]
    assert result == code
    candidate = tmp_path / "candidate.py"
    candidate.write_text(result)
    namespace = runpy.run_path(str(candidate))
    assert namespace["total"]((1, 2, 3)) == 6


def test_python3_fence_is_executable_python(utils, tmp_path):
    code = "import math\ndef root(n):\n    return math.isqrt(n)\n"
    doc = {**DOC, "entry_point": "root"}
    result = utils.build_predictions([[f"```python3\n{code}```"]], [doc])[0][0]
    assert result == code
    candidate = tmp_path / "candidate.py"
    candidate.write_text(result)
    namespace = runpy.run_path(str(candidate))
    assert namespace["root"](17) == 4


@pytest.mark.parametrize("dependencies_first", [True, False])
def test_separate_dependency_blocks_are_retained(utils, tmp_path, dependencies_first):
    dependencies = "import math\ndef helper(n):\n    return math.isqrt(n)"
    function = "def root(n):\n    return helper(n)"
    blocks = [dependencies, function] if dependencies_first else [function, dependencies]
    response = "\nExplanation between blocks.\n".join(
        f"```python\n{block}\n```" for block in blocks
    )
    doc = {**DOC, "entry_point": "root"}
    result = utils.build_predictions([[response]], [doc])[0][0]
    assert result == "\n\n".join(blocks) + "\n"
    candidate = tmp_path / "candidate.py"
    candidate.write_text(result)
    namespace = runpy.run_path(str(candidate))
    assert namespace["root"](17) == 4


def test_dependency_blocks_do_not_change_last_revision_selection(utils, tmp_path):
    dependencies = "def helper(values):\n    return -1\n"
    wrong = "def total(values):\n    return helper(values)\n"
    response = (
        f"```python\n{dependencies}```\n"
        f"```python\n{FULL}```\nRevision:\n```python3\n{wrong}```"
    )
    result = utils.build_predictions([[response]], [DOC])[0][0]
    assert result == dependencies + "\n" + wrong
    candidate = tmp_path / "candidate.py"
    candidate.write_text(result + "\n" + DOC["test"])
    namespace = runpy.run_path(str(candidate))
    with pytest.raises(AssertionError):
        namespace["check"](namespace["total"])


def test_legacy_nested_indentation_and_prompt_imports(utils, tmp_path):
    body = (
        "    result: List[int] = []\n"
        "    for value in values:\n"
        "        if value:\n"
        "            result.append(value)\n"
        "    return sum(result)\n"
    )
    result = utils.build_predictions([[body]], [DOC])[0][0]
    assert result == DOC["prompt"] + body
    candidate = tmp_path / "candidate.py"
    candidate.write_text(result)
    namespace = runpy.run_path(str(candidate))
    assert namespace["total"]([1, 2, 3]) == 6


def test_multiple_blocks_select_last_entry_point_without_testing_candidates(
    utils, tmp_path
):
    wrong = "def total(values):\n    return -1\n"
    response = f"```python\n{FULL}```\nRevision:\n```python\n{wrong}```"
    result = utils.build_predictions([[response]], [DOC])[0][0]
    assert result == wrong
    candidate = tmp_path / "candidate.py"
    candidate.write_text(result + "\n" + DOC["test"])
    namespace = runpy.run_path(str(candidate))
    with pytest.raises(AssertionError):
        namespace["check"](namespace["total"])


def test_malformed_code_is_not_repaired(utils):
    malformed = "def total(values):\n    return (\n"
    result = utils.build_predictions([[f"```python\n{malformed}```"]], [DOC])[0][0]
    assert result == malformed
    with pytest.raises(SyntaxError):
        compile(result, "<prediction>", "exec")


def test_batch_shape_and_entry_points(utils):
    other_doc = {**DOC, "entry_point": "product"}
    assert utils.build_predictions([[FULL, ""], [FULL]], [DOC, other_doc]) == [
        [FULL, ""],
        [""],
    ]


@pytest.mark.parametrize(
    "name,version,repeats,k,dataset",
    [
        ("humaneval", 2.0, 1, [1], "openai/openai_humaneval"),
        ("humaneval_instruct", 5.0, 1, [1], "openai/openai_humaneval"),
        ("humaneval_64", 2.0, 64, [2, 8, 16, 32, 64], "openai/openai_humaneval"),
        (
            "humaneval_64_instruct",
            4.0,
            64,
            [2, 8, 16, 32, 64],
            "openai/openai_humaneval",
        ),
        ("humaneval_plus", 2.0, 1, [1], "evalplus/humanevalplus"),
    ],
)
def test_public_task_config_and_prompt_filter_seam(
    utils, monkeypatch, name, version, repeats, k, dataset
):
    def download(task, *args, **kwargs):
        task.dataset = {"test": Dataset.from_list([DOC])}

    monkeypatch.setattr(ConfigurableTask, "download", download)
    task = TaskManager().load([name])["tasks"][name]
    assert task.config.dataset_path == dataset
    assert task.config.metadata["version"] == version
    assert task.config.repeats == repeats
    assert task.config.metric_list[0]["k"] == k
    assert task.config.gen_prefix is None
    assert task.config.generation_kwargs["until"] == []
    assert task.config.generation_kwargs["max_gen_toks"] == 4096
    assert task.doc_to_target(DOC) == DOC["test"] + "\ncheck(total)"

    messages = []

    def chat_template(turns, **kwargs):
        messages.extend(turns)
        return "rendered prompt"

    assert task.fewshot_context(
        DOC,
        num_fewshot=0,
        apply_chat_template=True,
        chat_template=chat_template,
    ) == "rendered prompt"
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert (
        "Return the full function as one fenced Python code block"
        in messages[0]["content"]
    )
    assert DOC["prompt"] in messages[0]["content"]

    task._instances = [
        Instance(
            "generate_until", DOC, ("prompt", {}), 0, resps=[f"```python\n{FULL}```"]
        )
    ]
    task.apply_filters()
    assert task.instances[0].filtered_resps == {"create_test": [FULL]}


def test_token_budget_can_be_overridden(utils, monkeypatch):
    def download(task, *args, **kwargs):
        task.dataset = {"test": Dataset.from_list([DOC])}

    monkeypatch.setattr(ConfigurableTask, "download", download)
    config = load_yaml(TASK_DIR / "humaneval.yaml")
    config["generation_kwargs"]["max_gen_toks"] = 8192
    task = ConfigurableTask(config=config)
    assert task.config.generation_kwargs["max_gen_toks"] == 8192
