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
        (f"```python\n{FULL}```\n   Explanation\n Explanation", FULL),
        (f"```py\n{FULL}```", FULL),
        (f"```\n{FULL}```", FULL),
        (
            "```Python\r\ndef total(values):\r\n    return sum(values)\r\n```\r\n",
            "def total(values):\r\n    return sum(values)\r\n",
        ),
        (
            "    ```python\n    def total(values):\n        return sum(values)\n    ```",
            "    ```python\n    def total(values):\n        return sum(values)\n    ```",
        ),
        ("    return sum(values)", DOC["prompt"] + "    return sum(values)"),
        (
            "\n    return sum(values)\n```\nAn explanation.",
            DOC["prompt"] + "\n    return sum(values)\n",
        ),
        (
            "```python\n    return sum(values)\n```",
            DOC["prompt"] + "    return sum(values)\n",
        ),
        (f"{FULL}```\nAn explanation.", FULL),
        ("", ""),
        ("   \n", ""),
        ("I cannot solve this problem.", "I cannot solve this problem."),
        (
            "```python\ndef another(values):\n    return 6\n```",
            "def another(values):\n    return 6\n",
        ),
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
    blocks = (
        [dependencies, function] if dependencies_first else [function, dependencies]
    )
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


def test_dependency_and_revision_blocks_keep_source_order(utils, tmp_path):
    dependencies = "def helper(values):\n    return -1\n"
    wrong = "def total(values):\n    return helper(values)\n"
    response = (
        f"```python\n{dependencies}```\n"
        f"```python\n{FULL}```\nRevision:\n```python3\n{wrong}```"
    )
    result = utils.build_predictions([[response]], [DOC])[0][0]
    assert result == dependencies + "\n" + FULL + "\n" + wrong
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


def test_multiple_blocks_preserve_redefinitions_without_testing_candidates(
    utils, tmp_path
):
    wrong = "def total(values):\n    return -1\n"
    response = f"```python\n{FULL}```\nRevision:\n```python\n{wrong}```"
    result = utils.build_predictions([[response]], [DOC])[0][0]
    assert result == FULL + "\n" + wrong
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
        [FULL],
    ]


@pytest.mark.parametrize(
    "source",
    [
        'def total(values):\n    return """\n```python\ndef total(values):\n    return sum(values)\n```\n"""\n',
        'def total(values):\n    return """a\n    \nb"""\n',
        'def total(values):\n    return f"""a\n```\n{values}\n"""\n',
        "def total[T](values: list[T]):\n    return sum(values)\n",
        "match True, False:\n    case _:\n        def total(values):\n            return sum(values)\n",
        FULL + "result is not None:\n",
        "\n" + FULL + "\n\n",
        "    # A comment may be indented in a complete raw module.\n" + FULL,
        "    def total(values):\n        return sum(values)\n",
    ],
)
@pytest.mark.parametrize("fenced", [False, True])
def test_program_bytes_survive_raw_or_longer_fenced_envelope(utils, source, fenced):
    response = f"````python\n{source}````" if fenced else source
    assert utils.build_predictions([[response]], [DOC]) == [[source]]


@pytest.mark.parametrize("mark", ["```", "````", "~~~"])
def test_foreign_fences_are_consumed_before_python(utils, mark):
    response = f"{mark}text\nExample\n{mark}\n{mark}python\n{FULL}{mark}"
    assert utils.build_predictions([[response]], [DOC]) == [[FULL]]


@pytest.mark.parametrize("ending", ["    return -1\n", "    return (\n"])
def test_unclosed_revision_rejects_earlier_complete_answer(utils, ending):
    response = (
        f"```python\n{FULL}```\nRevision:\n```python\ndef total(values):\n{ending}"
    )
    assert utils.build_predictions([[response, FULL, ""]], [DOC]) == [["", FULL, ""]]


def test_saved_binding_survives_earlier_entry_point_block(utils):
    first = FULL + "\nsaved_total = total\n"
    second = "def total(values):\n    return saved_total(values)\n"
    response = f"```python\n{first}```\n```python\n{second}```"
    assert utils.build_predictions([[response]], [DOC]) == [[first + "\n" + second]]


def test_legacy_closing_fence_cannot_hide_a_later_revision(utils):
    response = FULL + "```\n```python\ndef total(values):\n    return -1\n"
    assert utils.build_predictions([[response]], [DOC]) == [[response]]


@pytest.mark.parametrize("prefix", ["", "f"])
def test_fence_lines_inside_fenced_strings_are_not_delimiters(utils, prefix):
    source = f'def total(values):\n    return {prefix}"""a\n```\n{{values}}\n"""\n'
    assert utils.build_predictions([[f"```python\n{source}```"]], [DOC]) == [[source]]
    malformed = source + "result is not None:\n"
    assert utils.build_predictions([[malformed]], [DOC]) == [[malformed]]


@pytest.fixture(scope="module")
def execution_metric():
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("HF_ALLOW_CODE_EVAL", "1")
        yield evaluate.load("code_eval")


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
@pytest.mark.parametrize("literal_char", ["\u2028", "\u2029", "\f", "\v", "\x85", "\x1c", "\x1d", "\x1e"])
def test_physical_newlines_preserve_literal_fences_and_execution(
    execution_metric, utils, monkeypatch, newline, literal_char
):
    value = literal_char * 3 + "\n```\n"
    source = newline.join(
        [
            "def total(values):",
            '    note = """' + literal_char * 3,
            "```",
            '"""',
            f"    assert note == {value!r}",
            "    return sum(values)",
            "",
        ]
    )
    compile(source, "<original>", "exec")
    response = "```python" + newline + source + "```"
    assert utils.build_predictions([[response, source]], [DOC]) == [[source, source]]
    monkeypatch.setattr(utils, "compute_", execution_metric)

    def download(task, *args, **kwargs):
        task.dataset = {"test": Dataset.from_list([DOC])}

    monkeypatch.setattr(ConfigurableTask, "download", download)
    task = ConfigurableTask(config=load_yaml(TASK_DIR / "humaneval.yaml"))
    task._instances = [Instance("generate_until", DOC, ("prompt", {}), 0, resps=[response])]
    task.apply_filters()
    filtered = task.instances[0].filtered_resps["create_test"]
    assert filtered == [source]
    assert task.process_results(DOC, [filtered]) == {"pass@1": 1.0}


@pytest.mark.parametrize(
    "response,expected_score",
    [
        (
            'def total(values):\n    return """\n```python\ndef total(values):\n    return sum(values)\n```\n"""\n',
            0.0,
        ),
        (FULL + "result is not None:\n", 0.0),
        (f"```python\nassert False\n```\n```python\n{FULL}```", 0.0),
        (
            f"```python\n{FULL}```\n```python\ndef total(values):\n    return -1\n```",
            0.0,
        ),
        (
            f"```python\n{FULL}\nsaved_total = total\n```\n```python\ndef total(values):\n    return saved_total(values)\n```",
            1.0,
        ),
        ("def total[T](values: list[T]):\n    return sum(values)\n", 1.0),
        (
            "match True, False:\n    case _:\n        def total(values):\n            return sum(values)\n",
            1.0,
        ),
        (f"```text\nExample\n```\n```python\n{FULL}```", 1.0),
        (f"```python\n{FULL}```\n```python\ndef total(values):", 0.0),
        ("    return sum(values)\n", 1.0),
    ],
)
def test_public_filter_and_process_results_preserve_functional_outcomes(
    execution_metric, utils, monkeypatch, response, expected_score
):
    monkeypatch.setattr(utils, "compute_", execution_metric)

    def download(task, *args, **kwargs):
        task.dataset = {"test": Dataset.from_list([DOC])}

    monkeypatch.setattr(ConfigurableTask, "download", download)
    task = ConfigurableTask(config=load_yaml(TASK_DIR / "humaneval.yaml"))
    task._instances = [
        Instance("generate_until", DOC, ("prompt", {}), 0, resps=[response])
    ]
    task.apply_filters()
    filtered = task.instances[0].filtered_resps["create_test"]
    assert len(filtered) == 1
    assert task.process_results(DOC, [filtered]) == {"pass@1": expected_score}


def test_string_whitespace_preserves_runtime_value(utils, tmp_path):
    source = 'def total(values):\n    return """a\n    \nb"""\n'
    candidate = tmp_path / "candidate.py"
    candidate.write_text(
        utils.build_predictions([[f"```python\n{source}```"]], [DOC])[0][0]
    )
    assert runpy.run_path(str(candidate))["total"]([]) == "a\n    \nb"


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

    assert (
        task.fewshot_context(
            DOC,
            num_fewshot=0,
            apply_chat_template=True,
            chat_template=chat_template,
        )
        == "rendered prompt"
    )
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
