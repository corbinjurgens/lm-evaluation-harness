import ast

from lm_eval.tasks import _code_eval as compute_
from lm_eval.tasks._code_extraction import extract_python


def pass_at_k(
    references: list[str],
    predictions: list[list[str]],
    k: list[int] | int | None = None,
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
    """Preserve complete programs and the original legacy body indentation."""
    code = extract_python(response, allow_body=True)
    try:
        ast.parse(code)
    except IndentationError:
        if code.lstrip("\r\n").startswith((" ", "\t")) and not code.lstrip().startswith(
            ("```", "~~~", "def ", "async def ", "class ")
        ):
            return doc["prompt"] + code
    except (SyntaxError, ValueError):
        pass
    return code


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [
        [_prediction(r, doc) for r in resp]
        for resp, doc in zip(resps, docs, strict=True)
    ]


def build_predictions_instruct(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    return build_predictions(resps, docs)
