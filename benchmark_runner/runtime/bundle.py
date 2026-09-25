"""Generate once, then replay normal task scoring in an offline container.

Bundles are integrity-checked records, not authenticated or safe programs. Code
scoring still requires the separate restricted container and runtime opt-in.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

from datasets import Dataset

import lm_eval
from lm_eval import evaluator
from lm_eval.api.instance import Instance
from lm_eval.api.metrics import stderr_for_metric
from lm_eval.api.task import ConfigurableTask
from lm_eval.tasks._yaml_loader import load_yaml


PACKAGE = Path(lm_eval.__file__).resolve().parent
RUNTIME = Path(__file__).resolve().parent
HELPER_PREFIXES = (
    "tasks/humaneval/",
    "tasks/mbpp/",
    "tasks/ifeval/",
    "tasks/gsm8k/",
    "tasks/_code",
    "filters/",
    "api/task.py",
    "api/metrics.py",
)
TASK_FILES = {
    **{
        name: f"humaneval/{name}.yaml"
        for name in (
            "humaneval",
            "humaneval_instruct",
            "humaneval_64",
            "humaneval_64_instruct",
            "humaneval_plus",
        )
    },
    **{
        name: f"mbpp/{name}.yaml"
        for name in ("mbpp", "mbpp_instruct", "mbpp_plus", "mbpp_plus_instruct")
    },
    "ifeval": "ifeval/ifeval.yaml",
    "gsm8k_cot_zeroshot": "gsm8k/gsm8k-cot-zeroshot.yaml",
}
BUNDLE_FORMAT = "lm-eval-offline-bundle-v1"
RESULT_FORMAT = "lm-eval-offline-results-v1"
logger = logging.getLogger(__name__)


@contextmanager
def _python_random_state(seed=None):
    """Keep validation/scoring randomness local to this sequential operation."""
    state = random.getstate()
    try:
        if seed is not None:
            random.seed(seed)
        yield
    finally:
        random.setstate(state)


def _json(value):
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _keys(value, required, optional=()):
    if (
        not isinstance(value, dict)
        or not set(required) <= value.keys()
        or (value.keys() - set(required) - set(optional))
    ):
        raise ValueError(
            f"Expected fields {sorted(required)}; optional {sorted(optional)}"
        )


def _integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"Expected integer >= {minimum}")


def _number(value, minimum=0):
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"Expected finite number >= {minimum}")


def _text(value):
    if not isinstance(value, str) or not value:
        raise ValueError("Expected nonempty text")


def _names(names):
    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(n, str) or n not in TASK_FILES for n in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("Expected unique supported normal task names")


def validate_config(config):
    """Validate the deliberately narrow generation configuration; return a copy."""
    _keys(
        config,
        {"tasks", "model", "base_url", "gen_kwargs"},
        {
            "seed",
            "timeout",
            "num_concurrent",
            "max_retries",
            "limit",
            "samples",
            "num_fewshot",
            "profile",
            "server",
        },
    )
    config = json.loads(_json(config))
    _names(config["tasks"])
    _text(config["model"])
    _text(config["base_url"])
    url = urlsplit(config["base_url"])
    if (
        url.scheme not in {"http", "https"}
        or not url.hostname
        or url.username
        or url.password
        or url.query
        or url.fragment
        or url.path.rstrip("/") != "/v1/chat/completions"
    ):
        raise ValueError(
            "base_url must be an unauthenticated HTTP(S) /v1/chat/completions URL"
        )
    gen = config["gen_kwargs"]
    _keys(
        gen,
        {"temperature", "max_gen_toks"},
        {
            "top_p",
            "top_k",
            "min_p",
            "do_sample",
            "until",
            "seed",
            "frequency_penalty",
            "presence_penalty",
            "repeat_penalty",
            "reasoning_effort",
            "enable_thinking",
            "chat_template_kwargs",
        },
    )
    _number(gen["temperature"])
    _integer(gen["max_gen_toks"], 1)
    for key in ("top_p", "min_p"):
        if key in gen:
            _number(gen[key])
            if gen[key] > 1:
                raise ValueError(f"{key} must be <= 1")
    for key in ("top_k", "seed"):
        if key in gen:
            _integer(gen[key])
    for key in ("frequency_penalty", "presence_penalty", "repeat_penalty"):
        if key in gen:
            _number(gen[key], -2 if key != "repeat_penalty" else 0)
    if "do_sample" in gen and type(gen["do_sample"]) is not bool:
        raise ValueError("do_sample must be boolean")
    if "until" in gen and (
        not isinstance(gen["until"], list)
        or any(not isinstance(item, str) for item in gen["until"])
    ):
        raise ValueError("until must be a list of strings")
    if "chat_template_kwargs" in gen:
        _keys(
            gen["chat_template_kwargs"], set(), {"enable_thinking", "reasoning_effort"}
        )
    for controls in (gen, gen.get("chat_template_kwargs", {})):
        if (
            "enable_thinking" in controls
            and type(controls["enable_thinking"]) is not bool
        ):
            raise ValueError("enable_thinking must be boolean")
        if "reasoning_effort" in controls and controls["reasoning_effort"] not in {
            "low",
            "medium",
            "high",
        }:
            raise ValueError("reasoning_effort must be low, medium, or high")
    for key, default, minimum in (
        ("seed", 1234, 0),
        ("timeout", 300, 1),
        ("num_concurrent", 1, 1),
        ("max_retries", 3, 1),
    ):
        config.setdefault(key, default)
        _integer(config[key], minimum)
    for key in ("limit", "num_fewshot"):
        if key in config:
            _integer(config[key], 1 if key == "limit" else 0)
    if "samples" in config:
        if "limit" in config:
            raise ValueError("Use either limit or samples, not both")
        _keys(config["samples"], config["tasks"])
        for ids in config["samples"].values():
            if not isinstance(ids, list) or not ids:
                raise ValueError("samples must contain nonempty document ID lists")
            for doc_id in ids:
                _integer(doc_id)
            if len(set(ids)) != len(ids):
                raise ValueError("Duplicate selected document IDs")
    if "profile" in config and config["profile"] not in {"matched", "practical"}:
        raise ValueError("profile must be matched or practical")
    if "server" in config:
        _keys(
            config["server"],
            set(),
            {
                "revision",
                "binary_sha256",
                "model_sha256",
                "template_sha256",
                "reasoning_format",
                "reasoning_effort",
                "context_size",
            },
        )
        for key, value in config["server"].items():
            if key == "context_size":
                _integer(value, 1)
            else:
                _text(value)
    return config


def _source_key(relative):
    """One key per file on every host: Windows and the Linux image must agree."""
    return relative.as_posix()


def _is_helper(key):
    return key.startswith(HELPER_PREFIXES)


def _file_hashes(root, suffixes):
    return {
        _source_key(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.suffix in suffixes
    }


def source_identity():
    """Hash installed Python/YAML sources, independently of checkout metadata."""
    hashes = _file_hashes(PACKAGE, {".py", ".yaml"})
    helpers = {key: h for key, h in hashes.items() if _is_helper(key)}
    return {
        "package_sha256": _hash(hashes),
        "helpers_sha256": _hash(helpers),
        "benchmark_runtime": _hash(_file_hashes(RUNTIME, {".py"})),
    }


def _config(name):
    if name not in TASK_FILES:
        raise ValueError("Unsupported task")
    if name == "ifeval":
        import nltk

        # Its upstream helper downloads at import when this resource is absent.
        # Fail before importing the helper; the scoring image must bake it in.
        nltk.data.find("tokenizers/punkt_tab")
    return load_yaml(PACKAGE / "tasks" / TASK_FILES[name])


def _filters(config):
    return [f["name"] for f in config.get("filter_list", [])] or ["none"]


class _RecordedTask(ConfigurableTask):
    def __init__(self, config, documents):
        self._documents = documents
        super().__init__(config=config)

    def download(self, dataset_kwargs=None, **kwargs):
        data = Dataset.from_list(self._documents)
        self.dataset = {
            split: data
            for split in (
                self.config.test_split,
                self.config.validation_split,
                self.config.training_split,
                self.fewshot_cfg.split,
            )
            if split is not None
        }


def _doc_shape(name, doc):
    if not isinstance(doc, dict):
        raise TypeError("Document must be an object")
    if name.startswith("humaneval"):
        for key in ("prompt", "test", "entry_point"):
            _text(doc.get(key))
        if not doc["entry_point"].isidentifier():
            raise ValueError("Invalid HumanEval entry point")
    elif name.startswith("mbpp"):
        _text(doc.get("prompt", doc.get("text")))
        tests = doc.get("test_list")
        if not isinstance(tests, list) or len(tests) < 3:
            raise ValueError("MBPP requires its reference tests")
        for test in tests:
            _text(test)
    elif name == "gsm8k_cot_zeroshot":
        _text(doc.get("question"))
        _text(doc.get("answer"))
    else:
        from lm_eval.tasks.ifeval.instructions_registry import INSTRUCTION_DICT

        _text(doc.get("prompt"))
        _integer(doc.get("key"))
        instructions, kwargs = doc.get("instruction_id_list"), doc.get("kwargs")
        if (
            not isinstance(instructions, list)
            or not instructions
            or not isinstance(kwargs, list)
            or len(instructions) != len(kwargs)
        ):
            raise ValueError("Malformed IFEval instructions")
        for instruction, args in zip(instructions, kwargs, strict=True):
            if (
                not isinstance(instruction, str)
                or instruction not in INSTRUCTION_DICT
                or not isinstance(args, dict)
            ):
                raise ValueError("Unknown or malformed IFEval instruction")
        from lm_eval.tasks.ifeval.utils import process_results

        # Empty responses validate instruction arguments using actual checkers,
        # without evaluating candidate code or text.
        process_results(doc, [""])


def _record(row):
    _integer(row.get("doc_id"))
    responses = row.get("resps")
    if (
        not isinstance(responses, list)
        or len(responses) != 1
        or not isinstance(responses[0], list)
    ):
        raise ValueError("Expected exactly one generation request per document")
    raw = responses[0]
    if any(value is not None and not isinstance(value, str) for value in raw):
        raise ValueError("Responses must be strings or null placeholders")
    raw = ["" if value is None else value for value in raw]
    hashes = {key: row.get(key) for key in ("doc_hash", "prompt_hash", "target_hash")}
    if any(not isinstance(value, str) or len(value) != 64 for value in hashes.values()):
        raise ValueError("Missing harness sample hashes")
    return {
        "doc_id": row["doc_id"],
        "doc": row["doc"],
        "responses": raw,
        "response_hashes": [_hash([i, value]) for i, value in enumerate(raw)],
        "harness_hashes": hashes,
    }


def build_bundle(samples, task_info, provenance):
    """Merge equivalent per-filter harness rows without losing document draws."""
    if samples.keys() != task_info.keys():
        raise ValueError("Missing or extra sampled tasks")
    tasks = {}
    for name, info in task_info.items():
        records, original_rows, seen_filters = {}, {}, defaultdict(set)
        for row in samples[name]:
            record = _record(row)
            doc_id, filter_name = record["doc_id"], row.get("filter")
            if (
                filter_name not in info["filters"]
                or filter_name in seen_filters[doc_id]
            ):
                raise ValueError("Unexpected or duplicate sample filter")
            signature = _json(
                {
                    "doc": row["doc"],
                    "resps": row["resps"],
                    "hashes": record["harness_hashes"],
                }
            )
            if doc_id in records and original_rows[doc_id] != signature:
                raise ValueError("Conflicting duplicate sample")
            records[doc_id] = record
            original_rows[doc_id] = signature
            seen_filters[doc_id].add(filter_name)
        if set(records) != set(info["doc_ids"]) or any(
            filters != set(info["filters"]) for filters in seen_filters.values()
        ):
            raise ValueError("Missing documents or filter rows")
        tasks[name] = {**info, "records": [records[i] for i in info["doc_ids"]]}
    bundle = {
        "format": BUNDLE_FORMAT,
        "source": source_identity(),
        "provenance": provenance,
        "tasks": tasks,
    }
    bundle["sha256"] = _hash(bundle)
    validate_bundle(bundle)
    return bundle


def validate_bundle(bundle):
    """Validate every task and record before any task can execute candidates."""
    # Some accepted IFEval checker defaults draw from Python's global RNG.
    # Validation must not consume draws needed by actual instruction scoring.
    with _python_random_state():
        _validate_bundle(bundle)


def _validate_bundle(bundle):
    _keys(bundle, {"format", "source", "provenance", "tasks", "sha256"})
    if bundle["format"] != BUNDLE_FORMAT or bundle["sha256"] != _hash(
        {k: v for k, v in bundle.items() if k != "sha256"}
    ):
        raise ValueError("Bundle format or integrity mismatch")
    if bundle["source"] != source_identity():
        raise ValueError("Installed source/helper identity mismatch")
    _validate_provenance(bundle["provenance"])
    tasks = bundle["tasks"]
    if not isinstance(tasks, dict):
        raise TypeError("tasks must be an object")
    _names(list(tasks))
    if set(tasks) != set(bundle["provenance"]["tasks"]):
        raise ValueError("Task selection differs from provenance")
    for name, info in tasks.items():
        _keys(info, {"version", "repeats", "filters", "dataset", "doc_ids", "records"})
        config = _config(name)
        _integer(info["repeats"], 1)
        if (
            info["version"] != config["metadata"]["version"]
            or info["repeats"] != config.get("repeats", 1)
            or info["filters"] != _filters(config)
        ):
            raise ValueError("Task version, cardinality, or filters mismatch")
        dataset = info["dataset"]
        _keys(dataset, {"path", "name", "split", "fingerprint", "num_rows"})
        if (
            dataset["path"] != config["dataset_path"]
            or dataset["name"] != config.get("dataset_name")
            or dataset["split"] != config["test_split"]
        ):
            raise ValueError("Dataset provenance mismatch")
        _text(dataset["fingerprint"])
        _integer(dataset["num_rows"], 1)
        ids, records = info["doc_ids"], info["records"]
        if (
            not isinstance(ids, list)
            or not ids
            or not isinstance(records, list)
            or len(ids) != len(records)
        ):
            raise ValueError("Invalid document cardinality")
        for doc_id in ids:
            _integer(doc_id)
            if doc_id >= dataset["num_rows"]:
                raise ValueError("Document ID outside recorded dataset")
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate document IDs")
        expected = bundle["provenance"].get("samples", {}).get(name)
        if expected is None:
            expected = list(
                range(
                    min(
                        dataset["num_rows"],
                        bundle["provenance"].get("limit", dataset["num_rows"]),
                    )
                )
            )
        if ids != expected:
            raise ValueError("Missing, extra, or reordered selected documents")
        for doc_id, record in zip(ids, records, strict=True):
            _keys(
                record,
                {"doc_id", "doc", "responses", "response_hashes", "harness_hashes"},
            )
            _integer(record["doc_id"])
            if record["doc_id"] != doc_id:
                raise ValueError("Reordered document records")
            _doc_shape(name, record["doc"])
            raw = record["responses"]
            if (
                not isinstance(raw, list)
                or len(raw) != info["repeats"]
                or any(not isinstance(r, str) for r in raw)
            ):
                raise ValueError("Missing or extra candidates")
            if record["response_hashes"] != [_hash([i, r]) for i, r in enumerate(raw)]:
                raise ValueError("Changed or reordered candidates")
            hashes = record["harness_hashes"]
            _keys(hashes, {"doc_hash", "prompt_hash", "target_hash"})
            for value in hashes.values():
                if (
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(c not in "0123456789abcdef" for c in value)
                ):
                    raise ValueError("Invalid harness hash")


def _provenance(config):
    return {k: v for k, v in config.items() if k != "base_url"} | {
        "backend": "local-chat-completions",
        "apply_chat_template": True,
        "request_cache": False,
        "response_cache": None,
        "context_check": "untokenized; client cannot verify prompt plus output token limit",
    }


def _validate_provenance(value):
    fixed = _provenance({})
    if not isinstance(value, dict) or any(value.get(k) != v for k, v in fixed.items()):
        raise ValueError("Invalid generation provenance")
    config = {k: v for k, v in value.items() if k not in fixed}
    config["base_url"] = "http://offline.invalid/v1/chat/completions"
    if _provenance(validate_config(config)) != value:
        raise ValueError("Noncanonical generation provenance")


def write_artifact(path, value):
    """Publish complete JSON atomically and never replace an existing artifact."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    payload = _json(value) + "\n"
    temporary = None
    published = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # A same-filesystem hard link is atomic and, unlike replace(), refuses
        # a concurrently published destination. Cleanup of its private name is
        # separate from this successful publication commit boundary.
        os.link(temporary, path)
        published = True
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as error:
                # The successful link is the commit boundary. Cleanup must
                # neither turn it into failure nor mask an earlier write error.
                logger.warning(
                    "Artifact temporary cleanup failed (%s): %s: %s",
                    "publication committed" if published else "not published",
                    temporary,
                    error,
                )


def generate(config, output, transport_log):
    """Generate raw samples without metrics or either harness cache."""
    config = validate_config(config)
    output, transport_log = Path(output), Path(transport_log)
    if output.resolve() == transport_log.resolve():
        raise ValueError("Bundle and transport log need distinct paths")
    if output.exists() or transport_log.exists():
        raise FileExistsError("Generation destinations must be new")
    # Reserve a private log before the adapter appends potentially sensitive text.
    with transport_log.open("x", encoding="utf-8"):
        transport_log.chmod(0o600)
    tasks, task_info = [], {}
    for name in config["tasks"]:
        task = ConfigurableTask(config=_config(name))
        tasks.append(task)
        docs = task.eval_docs
        ids = config.get("samples", {}).get(
            name, list(range(min(len(docs), config.get("limit", len(docs)))))
        )
        if not ids or any(i >= len(docs) for i in ids):
            raise ValueError("Selected document ID outside dataset")
        task_info[name] = {
            "version": task.VERSION,
            "repeats": task.get_config("repeats"),
            "filters": _filters(_config(name)),
            "doc_ids": ids,
            "dataset": {
                "path": task.DATASET_PATH,
                "name": task.get_config("dataset_name"),
                "split": task.get_config("test_split"),
                "fingerprint": docs._fingerprint,
                "num_rows": len(docs),
            },
        }
    results = evaluator.simple_evaluate(
        model="local-chat-completions",
        model_args={
            k: config[k]
            for k in (
                "model",
                "base_url",
                "seed",
                "timeout",
                "num_concurrent",
                "max_retries",
            )
        }
        | {
            "tokenizer_backend": None,
            "tokenized_requests": False,
            "transport_log": str(transport_log),
        },
        tasks=tasks,
        gen_kwargs=config["gen_kwargs"],
        num_fewshot=config.get("num_fewshot"),
        limit=config.get("limit"),
        samples=config.get("samples"),
        use_cache=None,
        cache_requests=False,
        rewrite_requests_cache=False,
        delete_requests_cache=False,
        apply_chat_template=True,
        predict_only=True,
        log_samples=True,
        confirm_run_unsafe_code=True,
        bootstrap_iters=0,
        random_seed=config["seed"],
        numpy_random_seed=config["seed"],
        fewshot_random_seed=config["seed"],
        torch_random_seed=None,
    )
    if results is None:
        raise RuntimeError("Generation returned no results")
    bundle = build_bundle(results["samples"], task_info, _provenance(config))
    write_artifact(output, bundle)
    return bundle


def score(bundle, output=None):
    """Replay under the recorded Python seed, restoring the caller's RNG state."""
    if output is not None and Path(output).exists():
        raise FileExistsError(output)
    validate_bundle(bundle)
    with _python_random_state(seed=bundle["provenance"]["seed"]):
        return _score_validated(bundle, output)


def _score_validated(bundle, output):
    prepared = []
    # Prepare every document/filter/target before invoking any code metric.
    for name, info in bundle["tasks"].items():
        task = _RecordedTask(_config(name), [r["doc"] for r in info["records"]])
        task._instances = []
        for record in info["records"]:
            task.doc_to_target(record["doc"])
            task._instances.append(
                Instance(
                    "generate_until",
                    record["doc"],
                    ("", {}),
                    0,
                    metadata=(name, record["doc_id"], info["repeats"]),
                    resps=record["responses"].copy(),
                )
            )
        task.apply_filters()
        prepared.append((name, info, task))
    results = {}
    for name, info, task in prepared:
        raw_metrics, samples = defaultdict(list), []
        for instance in task.instances:
            per_filter = {}
            for filter_name, filtered in instance.filtered_resps.items():
                metrics = task.process_results(instance.doc, [filtered])
                per_filter[filter_name] = {
                    "filtered_response": filtered,
                    "metrics": metrics,
                }
                for metric, value in metrics.items():
                    raw_metrics[(metric, filter_name)].append(value)
            samples.append({"doc_id": instance.doc_id, "filters": per_filter})
        aggregates, stderrs = {}, {}
        for (metric, filter_name), values in raw_metrics.items():
            aggregate = task.aggregation()[metric]
            aggregates[f"{metric},{filter_name}"] = aggregate(values)
            stderr_fn = stderr_for_metric(aggregate, 100000)
            stderrs[f"{metric}_stderr,{filter_name}"] = (
                stderr_fn(values) if stderr_fn and len(values) > 1 else None
            )
        results[name] = {
            "version": info["version"],
            "num_documents": len(samples),
            "metrics": aggregates,
            "stderr": stderrs,
            "samples": samples,
        }
    result = {
        "format": RESULT_FORMAT,
        "bundle_sha256": bundle["sha256"],
        "source": bundle["source"],
        "provenance": bundle["provenance"],
        "results": results,
    }
    if output is not None:
        write_artifact(output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generation = commands.add_parser("generate")
    for flag in ("config", "output", "transport-log"):
        generation.add_argument(f"--{flag}", required=True)
    scoring = commands.add_parser("score")
    for flag in ("input", "output"):
        scoring.add_argument(f"--{flag}", required=True)
    args = parser.parse_args(argv)
    if args.command == "generate":
        generate(
            json.loads(Path(args.config).read_text(encoding="utf-8")),
            args.output,
            args.transport_log,
        )
    else:
        score(json.loads(Path(args.input).read_text(encoding="utf-8")), args.output)


if __name__ == "__main__":
    raise SystemExit(main())
