# Historical complete-answer chat coding tasks

For new runs, use the fork's normal task names: `humaneval`, `mbpp`, `ifeval`,
and `gsm8k_cot_zeroshot`. No `--include_path` is needed. HumanEval and MBPP now
request complete Python answers and extract fenced or raw code; HumanEval also
retains support for legacy indented body continuations. Closed Python blocks are
combined in source order without selecting a winning function; an unfinished
accepted block rejects the envelope rather than rescuing an earlier answer.
Datasets and reference tests are unchanged. Normal tasks now use the fork-owned
fresh-process execution backend; the old 64-candidate threaded-fork failure is
repaired locally. Helpers no longer download the metric or execute code at
import time; scoring checks the runtime execution opt-in. Safe offline deployment
is a separate, unfinished step. See the [fork README](../../README.md#this-fork-chat-model-benchmark-corrections)
for current versions, API diagnostics and deployment limitations.
GSM8K version 5 requests the strict answer format and corrects its numeric grammar.
IFEval itself is unchanged; the GPT-OSS parser repair belongs in the llama.cpp
server fork.

These additive definitions remain as historical experiment records:

- `humaneval_chat_complete_v1` and `mbpp_chat_complete_v1` preserve the
  historical GPT-OSS configuration, including `reasoning_effort: low`.
- `humaneval_chat_complete_v2` and `mbpp_chat_complete_v2` use the same prompt,
  extractor, and execution metrics without a model-family reasoning setting.
  They are not the route for new comparisons.

Their YAML imports current Python helpers. An old task name alone does not pin
the old extractor or execution metric: exact reproduction requires the original
evaluator revision, dependencies, server, data and generation settings. In
particular, v1's recorded reasoning setting does not make a current-checkout
rescore identical to the original run.

Load the historical definitions only when reproducing one of those runs:

```bash
lm-eval ls tasks --include_path /workspace/diagnostics/chat_coding_tasks
```

For current runs, supply model-specific reasoning controls through
`--gen_kwargs`, while keeping normal task names for every model. Record the
result's `git_hash`, task versions, and generation settings. The changed task
contracts require fresh cross-model runs with both request and response caching
disabled (omit `--cache_requests` and `--use_cache`); scores are not
interchangeable with upstream or historical v1/v2 results.

Executing generated code requires `HF_ALLOW_CODE_EVAL=1`, explicit
`--confirm_run_unsafe_code`, timeouts, and an isolated evaluator. The current
networked, writable-source Docker setup is not secure untrusted-code isolation;
the pinned offline workflow is still pending.
