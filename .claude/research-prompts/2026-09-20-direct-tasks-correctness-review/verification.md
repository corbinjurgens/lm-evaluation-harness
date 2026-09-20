# Returned research verification — 2026-09-20

## Verdict

Keep the fork ownership split and normal task names, but do not approve the
current HumanEval/MBPP extractors unchanged. Trusted synthetic fixtures reproduce
both false passes and false failures through the actual `code_eval` metric.
The earlier regression tests missed these semantic-preservation failures.

This review changes no benchmark code, task configuration, deployment or historical
results. The supplied `response.md` is retained unedited. Findings establish that
the bugs exist, not how often they affected actual model responses.

## Evidence boundary and reproduction

Reviewed checkout: `ca482337a4b74c4437d121382aad4d284e3210e5`.
The task/model/API/test paths have no changes between the implementation review
revision `8b280768` and final implementation revision `e0c06df1`, nor uncommitted
changes at verification time. The earlier final-review-applicability evidence
already covered that revision boundary; the reply's concern is not a new gap.

The probe runs in `lm_eval_sandbox` on Python 3.13, using its editable `/workspace`
checkout and installed `evaluate` code metric. Only trusted, small synthetic
programs are executed. The API and cache probes mock I/O; no model server request
is made and no request cache is written. Root and independent native Codex
verification agents checked the load-bearing claims.

From the fork checkout:

```bash
set -o pipefail
docker exec lm_eval_sandbox python -B .claude/research-prompts/2026-09-20-direct-tasks-correctness-review/verify-reply.py 2>&1 | tee .claude/research-prompts/2026-09-20-direct-tasks-correctness-review/verification-output.txt
```

`verify-reply.py` contains exact fixtures, assertions and mocked adapter/cache
checks. `verification-output.txt` records the run. Scores below are single
synthetic reference checks, not model benchmark scores. For fenced responses,
the baseline is the explicitly supplied source program before envelope removal.

## Confirmed scoring defects

| Fixture | Task | Baseline → extracted score | Failure |
|---|---|---|---|
| Python fence inside a returned triple-quoted string | HumanEval and MBPP | 0 → 1 | Mines string contents as a new program |
| Valid `match True, False:` statement | MBPP | 1 → 0 | Deletes the soft-keyword header as prose |
| Invalid trailing `result is not None:` | MBPP | 0 → 1 | Deletes malformed Python and repairs the candidate |
| Whitespace-only line inside a triple-quoted string | HumanEval | 1 → 0 | `dedent()` changes the returned string value |
| Foreign-language fence before Python fence | HumanEval | 1 → 0 | Pairs fence delimiters incorrectly |
| Earlier function saved as an alias, used by its replacement | HumanEval | 1 → 0 | Drops the earlier block and required binding |
| Python 3.13 type-parameter function syntax | HumanEval | 1 → 0 | Entrypoint regex rejects a valid definition |

Relevant implementation: `lm_eval/tasks/humaneval/utils.py:12–63` and
`lm_eval/tasks/mbpp/utils.py:28–75`. HumanEval tests that require dropping earlier
entrypoint blocks also need revisiting; preserving dependencies is not covered
by their current expected results.

Both extractors also silently retain an earlier complete block when a later
revision has an unclosed fence. This behavior was reproduced. Whether to reject
the whole response or adopt another deterministic treatment is an extraction
contract decision, not a fact established by the reproduction.

## Other checked claims

- **GSM8K:** actual existing filters take the first strict-format match but the
  last flexible number. With `The answer is 42` (no final punctuation), strict
  extraction returns `4` and flexible extraction returns `42`. The unescaped
  final regex dot explains this. These filters were not changed by the version-4
  prompt fix; this is not evidence that the prompt fix caused a regression.
- **Adapter kwargs:** the alleged mutation concern is refuted for the checked
  public path. Two mocked calls preserve the caller's nested kwargs. The adapter
  already deep-copies them. No additional copy fix is justified.
- **Adapter defaults and diagnostics:** omitted temperature becomes zero even
  with `do_sample=True`; the request drops `do_sample` and repeats the configured
  seed. Parsed text omits wire-level finish reason, usage and reasoning metadata.
  The runbook's GPT-OSS temperature-1 and omitted-temperature profiles are not
  matched-sampling comparisons. Forwarding a control does not prove a remote
  server honors it. Untokenized API calls skip context-length checks; default
  HTTP timeout is 300 seconds.
- **Request cache:** changing task metadata version, prompt and generation
  kwargs retains the same request-cache key in the public mocked probe, allowing
  old requests to be loaded. Default CLI settings and the checked runbook do not
  enable request caching. This is a conditional hazard, not proof of stale runs.
  New output directories alone would not invalidate that separate cache.
- **Response cache:** distinct from request caching. `do_sample=True` bypasses
  reads, but temperature alone does not; cache identity does not independently
  encode all model-level state. Do not generalize request-cache findings to it.
- **Repeated HumanEval candidates:** existing 64-candidate backend failures
  remain a separate issue. Installed metric source uses a thread pool whose
  workers create processes; `num_workers=1` still does that. Evaluator repeats
  reuse the API seed. No live duplicate-output measurement or new 64-candidate
  stress run was performed here, and this does not establish universal pass@1
  failure.
- **llama.cpp:** local parser source supports protocol/final-channel handling,
  but reasoning-format NONE includes analysis in content. Final-only behavior
  depends on runtime settings; local source is not proof of the other PC's
  deployment. The local template also has date/default-reasoning behavior that
  belongs in reproducibility metadata.
- **Docker safety/reproducibility:** live inspection confirms writable source
  mounting and host networking; the image/dependency installation and metric
  loading are not fully pinned. `code_eval` process guards are not a security
  boundary for untrusted generated code. These are exposure/reproducibility
  concerns, not evidence that an attack or disclosure occurred.

Source checks for operations: `lm_eval/models/api_models.py`,
`lm_eval/models/openai_completions.py`, `lm_eval/api/task.py`,
`lm_eval/api/model.py`, `lm_eval/_cli/utils.py`, `lm_eval/evaluator.py`, the
installed `evaluate` metric, `/Users/apple/lm-evaluation-harness/docker-compose.yml`,
the benchmark runbook, and local llama.cpp parser/template/sampler source.

## Recommended next scope, not implemented

Correct extraction first while retaining ordinary task names and model-neutral
behavior. Distinguish raw Python from outer Markdown without mining strings;
preserve code contents and dependencies; remove the broad prose-deletion
heuristic. Specify source-order multi-block and dangling-fence behavior explicitly
before implementing it, and retain narrowly defined legacy body continuations.

Add semantic-preservation regression fixtures (including these counterexamples),
bump affected extraction task versions, and run targeted container checks before
expensive fresh comparisons. Preserve historical scores and label methodology
changes; fork-defined ordinary names do not make scores stock-upstream-equivalent.
Base/instruct aliases must not be counted as independent benchmark evidence.

Do not change IFEval or llama.cpp to compensate for these task-layer extraction
bugs. Treat metric concurrency, candidate independence, safer execution, dependency
pinning and richer run metadata as separate work. Keep request caching disabled
for fresh comparisons unless its invalidation behavior is addressed. No blanket
claim that all models now work is justified by the current tests.
