import asyncio
import copy
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import requests

from lm_eval.api.instance import Instance
from lm_eval.models.api_models import (
    APIDiagnosticsError,
    APIProcessingError,
    APIResponseError,
)
from lm_eval.models.openai_completions import LocalChatCompletion


def response(content="answer", reason="stop"):
    return {
        "choices": [
            {
                "index": 0,
                "message": {
                    "content": content,
                    "reasoning_content": "private reasoning",
                },
                "finish_reason": reason,
            }
        ],
        "usage": {"prompt_tokens": 8, "completion_tokens": 2},
    }


def model(tmp_path, concurrent=1):
    return LocalChatCompletion(
        model="test-model",
        base_url="http://unused.invalid",
        seed=100,
        num_concurrent=concurrent,
        eos_string="END",
        transport_log=str(tmp_path / "transport.jsonl"),
        header={"Authorization": "do-not-log-this"},
    )


def request(api, kwargs=None):
    return Instance(
        "generate_until",
        {},
        (
            api.apply_chat_template([{"role": "user", "content": "test prompt"}]),
            kwargs if kwargs is not None else {},
        ),
        0,
    )


def records(tmp_path):
    return [
        json.loads(line)
        for line in (tmp_path / "transport.jsonl").read_text().splitlines()
    ]


class MockSession:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.payloads = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def post(self, *args, **kwargs):
        self.payloads.append(kwargs["json"])
        value = next(self.outputs)
        transport = MagicMock()
        transport.ok = not isinstance(value, Exception)
        transport.status = 503 if isinstance(value, Exception) else 200
        transport.text = AsyncMock(return_value="unavailable")
        transport.raise_for_status.side_effect = (
            value if isinstance(value, Exception) else None
        )
        transport.json = AsyncMock(return_value=value)
        context = MagicMock()
        context.__aenter__ = AsyncMock(return_value=transport)
        context.__aexit__ = AsyncMock(return_value=False)
        return context


def generate(api, inputs, outputs):
    """Exercise the public generation entrypoint with transport-only mocks."""
    if api._concurrent > 1:
        session = MockSession(outputs)
        with patch("lm_eval.models.api_models.ClientSession", return_value=session):
            return api.generate_until(inputs), session.payloads
    responses = []
    for value in outputs:
        transport = MagicMock()
        transport.ok = not isinstance(value, Exception)
        transport.status_code = 503 if isinstance(value, Exception) else 200
        transport.raise_for_status.side_effect = (
            value if isinstance(value, Exception) else None
        )
        transport.json.return_value = value
        responses.append(transport)
    with patch("requests.post", side_effect=responses) as post:
        result = api.generate_until(inputs)
        return result, [call.kwargs["json"] for call in post.call_args_list]


@pytest.mark.parametrize("concurrent", [1, 3])
@pytest.mark.parametrize("content", ["answer", "", None])
def test_success_null_empty_and_length_preserved(tmp_path, concurrent, content, caplog):
    api = model(tmp_path, concurrent)
    kwargs = {
        "max_gen_toks": 27,
        "until": ["STOP"],
        "do_sample": True,
        "temperature": 0.8,
        "extra": {"preserve": [1]},
    }
    original = copy.deepcopy(kwargs)
    result, payloads = generate(
        api, [request(api, kwargs)], [response(content, "length")]
    )
    assert result == [content if content is not None else ""]
    assert kwargs == original
    assert payloads[0]["seed"] == 100
    assert payloads[0]["temperature"] == 0.8
    assert payloads[0]["max_tokens"] == 27
    assert payloads[0]["stop"] == ["STOP", "END"]
    (entry,) = records(tmp_path)
    assert entry["error"] is None
    assert entry["http_status"] == 200
    assert entry["request"]["messages"] == [{"role": "user", "content": "test prompt"}]
    assert entry["usage"] == {"prompt_tokens": 8, "completion_tokens": 2}
    assert entry["choices"] == [
        {
            "index": 0,
            "finish_reason": "length",
            "content_present": True,
            "content": content,
        }
    ]
    assert "finish_reason=length" in caplog.text
    logged = json.dumps(entry)
    assert "Authorization" not in logged
    assert "do-not-log-this" not in logged
    assert "private reasoning" not in logged


@pytest.mark.parametrize("concurrent", [1, 3])
def test_retries_reuse_seed_but_repeats_get_new_draws(tmp_path, concurrent):
    api = model(tmp_path, concurrent)
    inputs = [request(api, {"temperature": 0.7})] * 2
    result, payloads = generate(
        api, inputs, [requests.HTTPError("503"), response(), response()]
    )
    assert result == ["answer", "answer"]
    seeds = [payload["seed"] for payload in payloads]
    assert seeds == ([100, 100, 101] if concurrent == 1 else [100, 101, 100])
    entries = records(tmp_path)
    (failed,) = [entry for entry in entries if entry["error"]]
    (retried,) = [entry for entry in entries if entry["attempt"] == 2]
    assert failed["error"] == "HTTPError"
    assert failed["http_status"] == 503
    assert failed["request_id"] == retried["request_id"]
    assert len({entry["request_id"] for entry in entries}) == 2


@pytest.mark.parametrize("concurrent", [1, 3])
@pytest.mark.parametrize(
    "kwargs, seeds",
    [
        ({}, [100, 100]),
        ({"do_sample": True}, [100, 101]),
        ({"temperature": 0.7, "seed": 17}, [17, 17]),
    ],
)
def test_seed_override_and_greedy_defaults(tmp_path, concurrent, kwargs, seeds):
    api = model(tmp_path, concurrent)
    _, payloads = generate(api, [request(api, kwargs)] * 2, [response(), response()])
    assert [payload["seed"] for payload in payloads] == seeds
    assert [payload["temperature"] for payload in payloads] == [
        kwargs.get("temperature", 0)
    ] * 2


@pytest.mark.parametrize("concurrent", [1, 3])
@pytest.mark.parametrize(
    "invalid",
    [
        None,
        {},
        {"choices": []},
        {"choices": "wrong"},
        {"choices": [{"index": 0, "message": {}}]},
        {"choices": [{"index": -1, "message": {"content": "x"}}]},
        {"choices": [{"index": 1, "message": {"content": "x"}}]},
        {"choices": [{"index": 0, "message": {"content": []}}]},
        {"choices": [{"index": 0, "message": {"content": "x"}}] * 2},
        {
            "choices": [
                {"index": 0, "message": {"content": "x"}},
                {"index": 1, "message": {"content": "y"}},
            ]
        },
    ],
)
def test_invalid_completed_responses_abort_without_resampling(
    tmp_path, concurrent, invalid
):
    api = model(tmp_path, concurrent)
    with pytest.raises(APIResponseError):
        generate(api, [request(api)], [invalid])
    (entry,) = records(tmp_path)
    assert entry["error"] == "APIResponseError"
    assert entry["attempt"] == 1


def test_parser_orders_choices_and_keeps_only_final_content(tmp_path):
    api = model(tmp_path)
    assert api.parse_generations(
        {
            "choices": [
                {
                    "index": 1,
                    "message": {"content": "second", "reasoning_content": "ignored"},
                },
                {"index": 0, "message": {"content": "first"}},
            ]
        }
    ) == ["first", "second"]


@pytest.mark.parametrize("concurrent", [1, 3])
def test_log_failure_is_fatal_and_not_retried(tmp_path, concurrent):
    api = model(tmp_path, concurrent)
    api.transport_log = str(tmp_path / "absent" / "transport.jsonl")
    with pytest.raises(APIDiagnosticsError):
        generate(api, [request(api)], [response()])


def test_async_batch_lengths_cannot_silently_truncate(tmp_path):
    api = model(tmp_path, 3)
    with pytest.raises(ValueError, match="counts must agree"):
        asyncio.run(api.get_batched_requests(["one", "two"], ["only one"]))


def test_logging_disabled_does_not_create_sidecar(tmp_path):
    api = model(tmp_path)
    api.transport_log = None
    assert generate(api, [request(api)], [response()])[0] == ["answer"]
    assert not (tmp_path / "transport.jsonl").exists()


@pytest.mark.parametrize("concurrent", [1, 3])
def test_http_failure_exhaustion_is_recorded(tmp_path, concurrent):
    api = model(tmp_path, concurrent)
    api.max_retries = 2
    with pytest.raises(requests.HTTPError):
        generate(
            api, [request(api, {"temperature": 1})], [requests.HTTPError("503")] * 2
        )
    entries = records(tmp_path)
    assert [entry["attempt"] for entry in entries] == [1, 2]
    assert [entry["request"]["seed"] for entry in entries] == [100, 100]
    assert [entry["error"] for entry in entries] == ["HTTPError", "HTTPError"]
    assert len({entry["request_id"] for entry in entries}) == 1


@pytest.mark.parametrize("concurrent", [1, 3])
def test_invalid_json_is_not_retried(tmp_path, concurrent):
    api = model(tmp_path, concurrent)
    if concurrent == 1:
        transport = MagicMock()
        transport.ok = True
        transport.status_code = 200
        transport.json.side_effect = ValueError("bad JSON")
        with patch("requests.post", return_value=transport) as post:
            with pytest.raises(APIResponseError, match="not valid JSON"):
                api.generate_until([request(api)])
            assert post.call_count == 1
    else:
        session = MockSession([response()])
        original_post = session.post

        def post(*args, **kwargs):
            context = original_post(*args, **kwargs)
            context.__aenter__.return_value.json.side_effect = ValueError("bad JSON")
            return context

        session.post = post
        with (
            patch("lm_eval.models.api_models.ClientSession", return_value=session),
            pytest.raises(APIResponseError, match="not valid JSON"),
        ):
            api.generate_until([request(api)])
        assert len(session.payloads) == 1
    (entry,) = records(tmp_path)
    assert entry["error"] == "APIResponseError"
    assert entry["http_status"] == 200


def test_async_completion_order_does_not_reorder_answers(tmp_path):
    api = model(tmp_path, 3)
    session = MockSession([response("first"), response("second")])
    original_post = session.post
    finished = []

    def post(*args, **kwargs):
        context = original_post(*args, **kwargs)
        content = "first" if len(session.payloads) == 1 else "second"

        async def delayed_json():
            if content == "first":
                await asyncio.sleep(0.02)
            finished.append(content)
            return response(content)

        context.__aenter__.return_value.json = delayed_json
        return context

    session.post = post
    with patch("lm_eval.models.api_models.ClientSession", return_value=session):
        result = api.generate_until([request(api, {"temperature": 1})] * 2)
    assert finished == ["second", "first"]
    assert result == ["first", "second"]
    assert [entry["request"]["seed"] for entry in records(tmp_path)] == [101, 100]


@pytest.mark.parametrize("concurrent", [1, 3])
@pytest.mark.parametrize(
    "failure", [OSError("cache failure"), requests.HTTPError("local cache failure")]
)
def test_cache_failure_does_not_resample_completed_answer(
    tmp_path, concurrent, failure
):
    api = model(tmp_path, concurrent)
    api.cache_hook.add_partial = MagicMock(side_effect=[failure, None])
    with pytest.raises(APIProcessingError, match="cache completed") as error:
        generate(
            api,
            [request(api, {"temperature": 1})],
            [response("first"), response("second")],
        )
    assert error.value.__cause__ is failure
    assert api.cache_hook.add_partial.call_count == 1
    (entry,) = records(tmp_path)
    assert entry["attempt"] == 1
    assert entry["choices"][0]["content"] == "first"
    assert entry["error"] is None  # HTTP succeeded; the local cache failure propagates.


@pytest.mark.parametrize("concurrent", [1, 3])
def test_list_envelope_retains_scored_content_and_usage(tmp_path, concurrent):
    api = model(tmp_path, concurrent)
    result, payloads = generate(
        api, [request(api)], [[response("list answer", "length")]]
    )
    assert result == ["list answer"]
    assert len(payloads) == 1
    (entry,) = records(tmp_path)
    assert entry["response_type"] == "list"
    assert entry["choices"] == [
        {
            "index": 0,
            "content": "list answer",
            "content_present": True,
            "finish_reason": "length",
        }
    ]
    assert entry["usage"] == {"prompt_tokens": 8, "completion_tokens": 2}


@pytest.mark.parametrize("concurrent", [1, 3])
def test_usage_accepts_only_numeric_token_fields_and_details(tmp_path, concurrent):
    api = model(tmp_path, concurrent)
    wire = response()
    wire["usage"] = {
        "prompt_tokens": 8,
        "completion_tokens": 2,
        "total_tokens": 10,
        "input_tokens": "secret-count",
        "output_tokens": True,
        "Authorization": "synthetic-secret",
        "prompt_tokens_details": {
            "cached_tokens": 4,
            "audio_tokens": -1,
            "Authorization": "nested-secret",
        },
        "completion_tokens_details": {
            "reasoning_tokens": 1,
            "audio_tokens": 0,
            "accepted_prediction_tokens": 0,
            "rejected_prediction_tokens": 1,
            "unknown": 9,
        },
        "input_tokens_details": {"cached_tokens": "secret-count"},
        "output_tokens_details": {"reasoning_tokens": 0},
    }
    assert generate(api, [request(api)], [wire])[0] == ["answer"]
    (entry,) = records(tmp_path)
    assert entry["usage"] == {
        "prompt_tokens": 8,
        "completion_tokens": 2,
        "total_tokens": 10,
        "prompt_tokens_details": {"cached_tokens": 4},
        "completion_tokens_details": {
            "reasoning_tokens": 1,
            "audio_tokens": 0,
            "accepted_prediction_tokens": 0,
            "rejected_prediction_tokens": 1,
        },
        "input_tokens_details": {},
        "output_tokens_details": {"reasoning_tokens": 0},
    }
    assert "secret" not in json.dumps(entry)
    assert "Authorization" not in json.dumps(entry)


@pytest.mark.parametrize("concurrent", [1, 3])
def test_sidecar_retains_sampling_and_thinking_controls_without_extras(
    tmp_path, concurrent
):
    api = model(tmp_path, concurrent)
    controls = {
        "min_p": 0.05,
        "top_p": 0.9,
        "top_k": 40,
        "enable_thinking": False,
        "reasoning_effort": "low",
        "repeat_penalty": 1.1,
        "chat_template_kwargs": {
            "enable_thinking": True,
            "reasoning_effort": "medium",
            "Authorization": "template-secret",
        },
        "extra": {"Authorization": "extra-secret"},
    }
    _, payloads = generate(api, [request(api, controls)], [response()])
    assert payloads[0]["min_p"] == 0.05
    assert payloads[0]["chat_template_kwargs"]["enable_thinking"] is True
    (entry,) = records(tmp_path)
    recorded = entry["request"]
    for key in (
        "min_p",
        "top_p",
        "top_k",
        "enable_thinking",
        "reasoning_effort",
        "repeat_penalty",
    ):
        assert recorded[key] == controls[key]
    assert recorded["chat_template_kwargs"] == {
        "enable_thinking": True,
        "reasoning_effort": "medium",
    }
    assert "secret" not in json.dumps(entry)
    assert "Authorization" not in json.dumps(entry)
    assert "extra" not in recorded


@pytest.mark.parametrize("concurrent", [1, 3])
def test_local_parser_exception_does_not_resample(tmp_path, concurrent):
    api = model(tmp_path, concurrent)
    api.parse_generations = MagicMock(side_effect=OSError("local parse error"))
    with pytest.raises(APIProcessingError, match="parse completed"):
        generate(api, [request(api)], [response("first"), response("second")])
    (entry,) = records(tmp_path)
    assert entry["choices"][0]["content"] == "first"
    assert entry["attempt"] == 1
    assert entry["error"] == "APIProcessingError"
