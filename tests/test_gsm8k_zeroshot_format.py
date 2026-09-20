from pathlib import Path

import pytest
import yaml

from lm_eval.api.instance import Instance
from lm_eval.api.registry import get_metric
from lm_eval.filters import build_filter_ensemble
from lm_eval.utils import apply_template


@pytest.fixture(scope="module")
def task_config():
    path = Path(__file__).parent.parent / "lm_eval/tasks/gsm8k/gsm8k-cot-zeroshot.yaml"
    return yaml.safe_load(path.read_text())


def test_normal_task_requests_strict_format_and_reasoning(task_config):
    prompt = apply_template(task_config["doc_to_text"], {"question": "What is 2 + 3?"})

    assert prompt.startswith("Q: What is 2 + 3?\n")
    assert "End your response with the sentence 'The answer is <number>.'" in prompt
    assert "without a currency symbol, units, or boxing" in prompt
    assert prompt.endswith("A: Let's think step by step.")
    assert task_config["task"] == "gsm8k_cot_zeroshot"
    assert task_config["metadata"]["version"] == 4.0
    assert [f["name"] for f in task_config["filter_list"]] == [
        "strict-match",
        "flexible-extract",
    ]


@pytest.mark.parametrize(
    "response,reference,expected_strict,expected_flexible,strict_score,flexible_score",
    [
        ("2 + 3 = 5. The answer is 5.", "5", "5", "5.", 1, 1),
        ("The answer is -5.", "-5", "-5", "-5.", 1, 1),
        ("The answer is 12.5.", "12.5", "12.5", "12.5.", 1, 1),
        ("The answer is -12.5.", "-12.5", "-12.5", "-12.5.", 1, 1),
        ("The answer is 1,234.", "1234", "1,234", "1,234.", 1, 1),
        ("The answer is -1,234.5.", "-1234.5", "-1,234.5", "-1,234.5.", 1, 1),
        ("The answer is 6.", "5", "6", "6.", 0, 0),
        ("The answer is -5.", "5", "-5", "-5.", 0, 0),
        ("The answer is 1.25.", "125", "1.25", "1.25.", 0, 0),
        ("The total is 5.", "5", "[invalid]", "5.", 0, 1),
        ("I cannot solve this.", "5", "[invalid]", "[invalid]", 0, 0),
    ],
)
def test_task_filters_and_metric(
    task_config,
    response,
    reference,
    expected_strict,
    expected_flexible,
    strict_score,
    flexible_score,
):
    doc = {"question": "A math question", "answer": f"Worked solution\n#### {reference}"}
    instance = Instance(
        request_type="generate_until",
        doc=doc,
        arguments=(),
        idx=0,
        resps=[response],
    )
    for pipeline in task_config["filter_list"]:
        components = [
            (step["function"], {k: v for k, v in step.items() if k != "function"})
            for step in pipeline["filter"]
        ]
        build_filter_ensemble(pipeline["name"], components).apply([instance])

    assert instance.filtered_resps == {
        "strict-match": expected_strict,
        "flexible-extract": expected_flexible,
    }
    metric_config = task_config["metric_list"][0]
    metric = get_metric(metric_config["metric"])
    metric_kwargs = {
        k: v
        for k, v in metric_config.items()
        if k not in {"metric", "aggregation", "higher_is_better"}
    }
    target = apply_template(task_config["doc_to_target"], doc)
    for name, expected_score in [
        ("strict-match", strict_score),
        ("flexible-extract", flexible_score),
    ]:
        assert metric(
            predictions=[instance.filtered_resps[name]],
            references=[target],
            **metric_kwargs,
        ) == {"exact_match": expected_score}
