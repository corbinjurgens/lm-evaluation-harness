import re
from textwrap import dedent

import evaluate as hf_evaluate


compute_ = hf_evaluate.load("code_eval")
test_cases = ["assert add(2, 3)==5"]
candidates = [["def add(a,b): return a*b"]]
results = compute_.compute(references=test_cases, predictions=candidates, k=[1])

_FENCE = re.compile(
    r"^[ \t]*```(?:python3?|py)?[ \t]*\r?\n(.*?)^[ \t]*```[ \t]*\r?$",
    re.MULTILINE | re.DOTALL | re.IGNORECASE,
)
_CLOSING_FENCE = re.compile(r"^[ \t]*```[ \t]*(?:\r?\n|$)", re.MULTILINE)


def pass_at_k(
    references: list[str], predictions: list[list[str]], k: list[int] | int | None = None
):
    assert k is not None
    if isinstance(k, int):
        k = [k]
    res = compute_.compute(
        references=references,
        predictions=predictions,
        k=k,
    )
    return res[0]


def _prediction(response: str, doc: dict) -> str:
    """Select the last complete answer, or retain an indented body continuation.

    Selection uses only the response and entry-point name, never the reference
    tests. Keep separate dependency blocks in response order alongside the last
    entry-point block. Legacy continuations retain the original prompt and their
    leading indentation.
    """
    definition = re.compile(
        rf"^(?:async[ \t]+)?def[ \t]+{re.escape(doc['entry_point'])}[ \t]*\(",
        re.MULTILINE,
    )
    blocks = [match.group(1) for match in _FENCE.finditer(response)]
    complete_blocks = [dedent(block).strip() for block in blocks]
    entry_blocks = [bool(definition.search(block)) for block in complete_blocks]
    if any(entry_blocks):
        last_entry = max(i for i, is_entry in enumerate(entry_blocks) if is_entry)
        return "\n\n".join(
            block
            for i, block in enumerate(complete_blocks)
            if block and (i == last_entry or not entry_blocks[i])
        ) + "\n"

    # With an assistant prefill, older clients may return only the closing fence.
    code = blocks[-1] if blocks else _CLOSING_FENCE.split(response, maxsplit=1)[0]
    if definition.search(code):
        return code.strip() + "\n"
    code = code.lstrip("\r\n").rstrip()
    if code and code[0] in " \t":
        return doc["prompt"] + code + "\n"
    return ""


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [
        [_prediction(r, doc) for r in resp]
        for resp, doc in zip(resps, docs, strict=True)
    ]


def build_predictions_instruct(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    return build_predictions(resps, docs)
