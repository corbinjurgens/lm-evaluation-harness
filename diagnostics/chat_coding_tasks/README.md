# Historical complete-answer chat coding tasks

For new runs, use the fork's normal task names: `humaneval`, `mbpp`, `ifeval`,
and `gsm8k_cot_zeroshot`. No `--include_path` is needed. HumanEval and MBPP now
request complete Python answers and extract fenced or raw code; HumanEval also
retains support for legacy indented body continuations. The execution metrics
and datasets are unchanged. GSM8K now explicitly requests the strict filter's
answer format. IFEval itself is unchanged; the GPT-OSS parser repair belongs in
the llama.cpp server fork.

These additive tasks remain only for reproducing earlier experiments:

- `humaneval_chat_complete_v1` and `mbpp_chat_complete_v1` preserve the
  historical GPT-OSS configuration, including `reasoning_effort: low`.
- `humaneval_chat_complete_v2` and `mbpp_chat_complete_v2` use the same prompt,
  extractor, and execution metrics without a model-family reasoning setting.
  They are retained for historical reproduction, not new comparisons.

Load the historical definitions only when reproducing one of those runs:

```bash
lm-eval ls tasks --include_path /workspace/diagnostics/chat_coding_tasks
```

For current runs, supply model-specific reasoning controls through
`--gen_kwargs`, while keeping normal task names for every model. Record the
result's `git_hash`, task versions, and generation settings. The changed task
contracts require fresh caches and fresh cross-model runs; scores are not
interchangeable with upstream or historical v1/v2 results.

Executing generated code requires `HF_ALLOW_CODE_EVAL=1`, explicit
`--confirm_run_unsafe_code`, timeouts, and an isolated evaluator.
