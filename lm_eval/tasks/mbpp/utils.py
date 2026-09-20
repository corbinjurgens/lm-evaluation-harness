import ast
import keyword
import re

import evaluate as hf_evaluate


pass_at_k = hf_evaluate.load("code_eval")

# run simple test to check code execution is enabled before model generation
test_cases = ["assert add(2, 3)==5"]
candidates = [["def add(a,b): return a*b"]]
results = pass_at_k.compute(references=test_cases, predictions=candidates, k=[1])


def pass_at_1(references: str | list[str], predictions: str | list[list[str]]) -> float:
    if isinstance(references, str):
        references = [references]
    if isinstance(predictions[0], str):
        predictions = [[p] for p in predictions]
    return pass_at_k.compute(
        references=references,
        predictions=predictions,
        k=[1],
    )[0]["pass@1"]


_CODE_FENCE = re.compile(
    r"^[ \t]{0,3}```(?P<language>[^\n`]*)\n(?P<code>.*?)^[ \t]{0,3}```[ \t]*(?=\n|$)",
    re.MULTILINE | re.DOTALL,
)


def _is_prose(line: str) -> bool:
    """Recognize plain explanatory text without discarding Python statements."""
    if line != line.lstrip() or not re.fullmatch(
        r"[A-Za-z][A-Za-z0-9 ,.!?'’:-]*", line
    ):
        return False
    if len(line.split()) < 3 or keyword.iskeyword(line.split(maxsplit=1)[0]):
        return False
    try:
        ast.parse(line)
    except SyntaxError:
        return True
    return False


def extract_code_blocks(text: str) -> str:
    """Extract complete Python answers without consulting reference tests.

    Keep all Python blocks in response order so imports and helper definitions in
    separate blocks survive. Raw code is also accepted. Only unambiguous prose
    at its edges is removed; assertions and malformed code remain executable
    candidates and are judged by the original code execution metric.
    """
    text = text.replace("\r\n", "\n").strip()
    blocks = list(_CODE_FENCE.finditer(text))
    if blocks:
        return "\n\n".join(
            block["code"].strip()
            for block in blocks
            if block["language"].strip().lower() in {"", "python", "py", "python3"}
        )

    # Older assistant-prefill responses may contain only the closing fence.
    if text.endswith("\n```") and "```" not in text[:-4]:
        text = text[:-4].rstrip()
    lines = text.splitlines()
    while lines and (not lines[0].strip() or _is_prose(lines[0])):
        lines.pop(0)
    while lines and (not lines[-1].strip() or _is_prose(lines[-1])):
        lines.pop()
    if lines and lines[0].casefold() in {"solution:", "python:", "code:"}:
        lines.pop(0)
    return "\n".join(lines)


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [[extract_code_blocks(r) for r in resp] for resp in resps]


def list_fewshot_samples():
    return [
        {
            "task_id": 2,
            "text": "Write a function to find the similar elements from the given two tuple lists.",
            "code": "def similar_elements(test_tup1, test_tup2):\r\n  res = tuple(set(test_tup1) & set(test_tup2))\r\n  return (res) ",
            "test_list": [
                "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)",
                "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)",
                "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)",
            ],
            "is_fewshot": True,
        },
        {
            "task_id": 3,
            "text": "Write a python function to identify non-prime numbers.",
            "code": "import math\r\ndef is_not_prime(n):\r\n    result = False\r\n    for i in range(2,int(math.sqrt(n)) + 1):\r\n        if n % i == 0:\r\n            result = True\r\n    return result",
            "test_list": [
                "assert is_not_prime(2) == False",
                "assert is_not_prime(10) == True",
                "assert is_not_prime(35) == True",
            ],
            "is_fewshot": True,
        },
        {
            "task_id": 4,
            "text": "Write a function to find the largest integers from a given list of numbers using heap queue algorithm.",
            "code": "import heapq as hq\r\ndef heap_queue_largest(nums,n):\r\n  largest_nums = hq.nlargest(n, nums)\r\n  return largest_nums",
            "test_list": [
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] ",
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] ",
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]",
            ],
            "is_fewshot": True,
        },
    ]
