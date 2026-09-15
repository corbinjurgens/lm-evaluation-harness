"""Filters for complete fenced Python answers in the versioned chat tasks."""

import re

from lm_eval.tasks.humaneval.utils import pass_at_k
from lm_eval.tasks.mbpp.utils import list_fewshot_samples, pass_at_1


FENCE = re.compile(r"```(?:python|py)?[ \t]*\r?\n(.*?)\r?\n?```", re.DOTALL | re.IGNORECASE)


def complete_function(response: str, name: str) -> str:
    if not name:
        return ""
    definition = re.compile(rf"(?m)^\s*(?:async\s+)?def\s+{re.escape(name)}\s*\(")
    matches = [
        block.group(1).strip()
        for block in FENCE.finditer(response)
        if definition.search(block.group(1))
    ]
    return matches[-1] + "\n" if matches else ""


def humaneval_complete(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [
        [complete_function(response, doc["entry_point"]) for response in responses]
        for responses, doc in zip(resps, docs)
    ]


def mbpp_complete(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    predictions = []
    for responses, doc in zip(resps, docs):
        match = re.search(r"\bassert\s+([A-Za-z_]\w*)\s*\(", doc["test_list"][0])
        name = match.group(1) if match else ""
        predictions.append([complete_function(response, name) for response in responses])
    return predictions
