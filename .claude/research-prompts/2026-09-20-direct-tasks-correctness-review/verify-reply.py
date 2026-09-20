"""Reproduce review claims on trusted fixtures; does not modify task code."""
import json
from pathlib import Path

import yaml

from lm_eval.api.instance import Instance
from lm_eval.filters import build_filter_ensemble
from lm_eval.tasks.humaneval import utils as humaneval
from lm_eval.tasks.mbpp import utils as mbpp


reference = ["assert add(2, 3) == 5"]
cases = {
    "raw_string_fence_changes_wrong_answer_to_correct": (
        'def add(a, b):\n    return """\n```python\ndef add(a, b):\n    return a + b\n```\n"""\n',
        (0.0, 1.0),
    ),
    "match_header_deleted_from_valid_python": (
        "match True, False:\n    case _:\n        def add(a, b):\n            return a + b\n",
        (1.0, 0.0),
    ),
    "malformed_statement_deleted": (
        "def add(a, b): return a + b\nresult is not None:",
        (0.0, 1.0),
    ),
}
for name, (response, expected) in cases.items():
    prediction = mbpp.build_predictions([[response]], [{}])[0][0]
    before = float(mbpp.pass_at_1(reference, [[response]]))
    after = float(mbpp.pass_at_1(reference, [[prediction]]))
    print(json.dumps(dict(claim=name, input=response, extracted=prediction,
                          original_score=before, extracted_score=after)), flush=True)
    assert (before, after) == expected

# Public adapter/cache paths with mocked I/O: no server request or cache write.
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

from lm_eval.api.task import ConfigurableTask
from lm_eval.models.openai_completions import LocalChatCompletion
from lm_eval._cli.utils import request_caching_arg_to_dict

old = Instance("generate_until", {}, ("OLD PROMPT", {"temperature": 0}), 0,
               metadata=("demo", 0, 1))
keys = []
for version, prompt, temperature in [(1, "OLD PROMPT", 0), (2, "NEW PROMPT", 1)]:
    task = object.__new__(ConfigurableTask)
    task._config = SimpleNamespace(task="demo", num_fewshot=0,
        metadata={"version": version}, doc_to_text=prompt,
        generation_kwargs={"temperature": temperature})
    with patch("lm_eval.api.task.load_from_cache", return_value=[[old]]) as loader:
        task.build_all_requests(cache_requests=True)
        keys.append(loader.call_args.kwargs["file_name"])
        print(json.dumps(dict(claim="request_cache_identity", version=version,
                              cache_key=keys[-1], returned_args=task.instances[0].args)),
              flush=True)
assert keys[0] == keys[1]
assert request_caching_arg_to_dict(None) == {}

model = object.__new__(LocalChatCompletion)
model.model = "probe"
model._seed = 1234
model._max_gen_toks = 4096
model.base_url = "http://invalid.test"
model.verify_certificate = True
model.timeout = 300
model.eos_string = None
model.header = {}
model.think_end_token = None
model.create_message = lambda x: x
kwargs = {"do_sample": True, "max_gen_toks": 10, "until": [], "extra": {"a": 1}}
original_kwargs = deepcopy(kwargs)
wire = {"choices": [{"index": 0,
    "message": {"content": "answer", "reasoning_content": "hidden"},
    "finish_reason": "length"}], "usage": {"total_tokens": 12}}
reply = Mock(ok=True)
reply.json.return_value = wire
with patch("lm_eval.models.api_models.requests.post", return_value=reply) as post:
    outputs = [model.model_call([{"role": "user", "content": "x"}], gen_kwargs=kwargs)
               for _ in range(2)]
    payloads = [call.kwargs["json"] for call in post.call_args_list]
    parsed = model.parse_generations(outputs[0])
    print(json.dumps(dict(claim="adapter_request_and_response",
                          kwargs_unchanged=kwargs == original_kwargs,
                          payloads=payloads, parsed=parsed)), flush=True)
assert kwargs == original_kwargs
assert all(p["temperature"] == 0 and p["seed"] == 1234 for p in payloads)
assert parsed == ["answer"]

response = "```python\ndef add(a,b): return a+b\n```\nRevision:\n```python\ndef add(a,b): return -1"
prediction = mbpp.build_predictions([[response]], [{}])[0][0]
print(json.dumps(dict(claim="dangling_revision_ignored", input=response,
                      extracted=prediction,
                      score=float(mbpp.pass_at_1(reference, [[prediction]])))), flush=True)

config = yaml.safe_load(Path("lm_eval/tasks/gsm8k/gsm8k-cot-zeroshot.yaml").read_text())
for response in ["The answer is 4. After checking, The answer is 5.",
                 "The answer is 42", "The answer is 42."]:
    instance = Instance("generate_until", {}, (), 0, resps=[response])
    for pipeline in config["filter_list"]:
        steps = [(step["function"], {k:v for k,v in step.items() if k != "function"})
                 for step in pipeline["filter"]]
        build_filter_ensemble(pipeline["name"], steps).apply([instance])
    print(json.dumps(dict(claim="gsm_existing_filter_behavior", input=response,
                          extracted=instance.filtered_resps)), flush=True)

human_cases = [
    ("raw_embedded_fence", cases["raw_string_fence_changes_wrong_answer_to_correct"][0],
     cases["raw_string_fence_changes_wrong_answer_to_correct"][0],
     "add", "assert add(2, 3) == 5", (0.0, 1.0)),
    ("dedent_changes_string", 'def f():\n    return """a\n    \nb"""\n',
     '```python\ndef f():\n    return """a\n    \nb"""\n```',
     "f", "assert f() == 'a\\n    \\nb'", (1.0, 0.0)),
    ("foreign_fence_pairing", "def add(a,b): return a+b\n",
     "```text\nIllustration\n```\n```python\ndef add(a,b): return a+b\n```",
     "add", "assert add(2,3) == 5", (1.0, 0.0)),
    ("lost_saved_binding",
     "def total(values): return sum(values)\nsaved_total = total\ndef total(values): return saved_total(values)\n",
     "```python\ndef total(values): return sum(values)\nsaved_total = total\n```\n```python\ndef total(values): return saved_total(values)\n```",
     "total", "assert total([1,2,3]) == 6", (1.0, 0.0)),
    ("type_parameter_entrypoint", "def identity[T](value: T) -> T:\n    return value\n",
     "def identity[T](value: T) -> T:\n    return value\n",
     "identity", "assert identity(5) == 5", (1.0, 0.0)),
]
for name, original, response, entry, assertion, expected in human_cases:
    doc = {"entry_point": entry, "prompt": f"def {entry}(*args):\n    pass\n"}
    candidate = humaneval.build_predictions([[response]], [doc])[0][0]
    before = float(humaneval.pass_at_k([assertion], [[original]], k=[1])["pass@1"])
    after = float(humaneval.pass_at_k([assertion], [[candidate]], k=[1])["pass@1"])
    print(json.dumps(dict(claim=name, family="humaneval", input=response,
                          extracted=candidate, original_score=before,
                          extracted_score=after)), flush=True)
    assert (before, after) == expected
