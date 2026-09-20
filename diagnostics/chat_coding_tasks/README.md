# Complete-answer chat coding tasks

These additive tasks evaluate chat models that return complete fenced Python
functions. They do not modify lm-eval's stock prefix-completion tasks.

- `humaneval_chat_complete_v1` and `mbpp_chat_complete_v1` preserve the
  historical GPT-OSS configuration, including `reasoning_effort: low`.
- `humaneval_chat_complete_v2` and `mbpp_chat_complete_v2` use the same prompt,
  extractor, and execution metrics without a model-family reasoning setting.
  Use v2 for new cross-model comparisons and provide reasoning effort through
  the run command or model profile.

Load the directory with:

```bash
lm-eval ls tasks --include_path /workspace/diagnostics/chat_coding_tasks
```

Executing generated code requires `HF_ALLOW_CODE_EVAL=1`, explicit
`--confirm_run_unsafe_code`, timeouts, and an isolated evaluator.
