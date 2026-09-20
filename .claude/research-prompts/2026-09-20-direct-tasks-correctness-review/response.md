# Research reply

Paste the reply below the rule, unedited. Nothing in it is verified: the
answering model saw only the pack, so every claim about this repository is a
hypothesis until checked against the code.

---
## 1. Direct answer

**Keep the ownership split and the normal-task-name workflow, but revise the extractors before expensive fresh comparisons.** Ordinary names are scientifically defensible for an explicitly identified fork-defined suite; the names themselves are not the problem. However, the supplied code contains counterexamples where extraction changes program meaning, including turning a wrong raw program into a passing candidate, and MBPP’s prose heuristic can delete actual Python code.

The prompt, inheritance, metric-reuse, and editable-installation changes are largely sound. Operational approval remains conditional on runtime verification, cache controls, execution isolation, and a separate resolution of the demonstrated 64-candidate scoring failure.

---

## 2. Findings

Paths below refer to the supplied snapshots. Numeric line references count the pasted source, not an independently inspected checkout. The counterexamples are deductions from that source; the test outcomes cited are the attached captured outcomes.

### Per-change disposition

“High” means capable of materially changing scores or blocking a run. “Critical” denotes an execution-safety gate.

| Change | Disposition | Severity | Supporting file/lines | Smallest remedy |
|---|---|---:|---|---|
| Optional GPT-OSS final-channel constraint parsing | **Keep** | Medium verification gate | llama.cpp `common/parsers/gpt-oss.cpp`, `constraint`, `final_msg`, structured-output branch | Retain the repair; verify content/channel preservation and the loaded binary. |
| Optional schema `items` guard | **Keep** | Low for this suite | llama.cpp `models/templates/openai-gpt-oss-120b.jinja`, `render_typescript_type()` | Test absent, empty, typed, and nested `items` within the supported schema subset. |
| HumanEval full-answer prompt, removed prefill/stops, larger default | **Keep** | Methodology change | `lm_eval/tasks/humaneval/humaneval.yaml:6,13–16`; instruct YAMLs | Preserve fork/revision labeling; do not describe this as upstream-equivalent completion evaluation. |
| HumanEval raw/fenced identification | **Revise** | **High** | `lm_eval/tasks/humaneval/utils.py:12–16,41–59` | Distinguish raw Python from outer Markdown before extracting fences; consume foreign-language fences coherently. |
| HumanEval unconditional `dedent()` | **Revise** | **High** | `lm_eval/tasks/humaneval/utils.py:46` | Preserve program text; remove only explicitly recognized presentation indentation, without normalizing string contents. |
| HumanEval last-entry-block plus other-block assembly | **Revise** | High for multi-block answers | `lm_eval/tasks/humaneval/utils.py:47–54` | Prefer one source-order module policy rather than selectively deleting whole entry-point blocks. |
| Legacy indented HumanEval continuations | **Keep, narrowly scoped** | Medium | `lm_eval/tasks/humaneval/utils.py:56–63` | Preserve indentation and original prompt; ensure Markdown/string detection cannot redirect this fallback. |
| MBPP prompts, complete demonstrations, newline delimiter, inherited variants | **Keep** | Low | `lm_eval/tasks/mbpp/*.yaml`; `tests/test_mbpp_extraction.py::test_fewshot_context_separates_complete_answers` | Record the actual single-turn/multiturn setting; keep the existing reference contracts. |
| MBPP raw/fenced identification and block joining | **Revise** | **High** | `lm_eval/tasks/mbpp/utils.py:28–31,57–68` | Preserve valid raw modules; detect dangling outer fences; retain source order without code repair. |
| MBPP broad edge-prose heuristic | **Revert the heuristic; retain bounded wrappers** | **High** | `lm_eval/tasks/mbpp/utils.py:34–46,69–75` | Stop treating standalone parse failure as proof of prose. Keep only explicitly defined presentation wrappers. |
| GSM8K answer-format instruction | **Keep** | Medium interpretive limitation | `lm_eval/tasks/gsm8k/gsm8k-cot-zeroshot.yaml`, `doc_to_text`, both filters | Document that the filters remain different heuristics, not equivalent terminal-answer validators. |
| Metric wrappers and incidental cleanup | **Keep** | Low | HumanEval `utils.py:19–30`; MBPP `utils.py:16–25` | No demonstrated scoring regression in the actual nested-list caller contract. |
| `humaneval_64` production readiness | **Insufficient evidence; gate affected runs** | **High** | `logs/baseline-code-eval-64.txt`; `humaneval_64.yaml`; API seed handling | Resolve execution failure and verify independent sampling opportunities before generating expensive 64-repeat runs. |
| IFEval task | **Keep unchanged** | Medium operational gate | `lm_eval/tasks/ifeval/ifeval.yaml` | Treat effort, sampling, budget, and final-channel handling as declared run configuration. |
| Regression suite and replay | **Keep and extend** | Medium evidence limitation | `logs/final-head-gates.log`; `replay.py`; `logs/final-replay.log` | Add the counterexamples below; distinguish parseability, compilation, functional outcomes, and preserved meaning. |
| Docker installation/runbook | **Keep installation fix; revise operations** | **Critical safety / high reproducibility** | `/Users/apple/lm-evaluation-harness/docker-compose.yml`; `lm-eval Commands.md` | Pin runtime/dependencies, protect execution, capture response metadata, and control both cache layers. |

### Finding 1 — The architecture and normal names are defensible; “unchanged metrics” is not enough by itself. **Confidence: high**

The separation is correct: llama.cpp handles model protocol and transport; the harness defines the evaluated prompt, candidate extraction, and scoring procedure. Moving benchmark-aware answer rewriting into the server would obscure that procedure. Nothing in the supplied implementation requires that.

Keeping `humaneval`, `mbpp`, and `gsm8k_cot_zeroshot` is acceptable when the result identity includes **fork, commit, effective task version, settings, and runtime**. A task name is a lookup key, not a guarantee of methodology.

The changed prompts, stopping rules, token budgets, and extraction policies nevertheless define a new experiment. The shared execution metric preserves the final correctness criterion, but not the distribution of programs presented to that criterion. Your notes now acknowledge that distinction appropriately.  
Sources: `Fixes.md`, “Direct HumanEval and MBPP fixes”; the supplied implementation diff.

The base/instruct pairs are now effectively duplicate task contracts under identical invocation settings. Keep those names for compatibility, but do not count both as independent evidence when aggregating a suite.  
Sources: `humaneval_instruct.yaml`; `mbpp.yaml`; `mbpp_instruct.yaml`.

### Finding 2 — Both extractors can replace a valid raw program with code inside its string literal. **Confidence: high**

Consider this **raw Python response**, with no outer Markdown wrapper:

````python
def add(a, b):
    return """
```python
def add(a, b):
    return a + b
```
"""
````

The actual function returns a string. It should fail `assert add(2, 3) == 5`.

Both extractors search for fences before establishing that the response is already a raw Python module. They find the fence inside the triple-quoted string and extract the inner addition function. That extracted candidate would pass the assertion.

This is a concrete **wrong-program-to-correct-candidate transformation**, without any reference-aware oracle. Python treats the embedded lines as string content, not executable definitions.  
Sources: `lm_eval/tasks/humaneval/utils.py:45–54`; `lm_eval/tasks/mbpp/utils.py:57–64`; Python’s lexical rules for triple-quoted strings. 

The required correction is representation-aware extraction: recognize raw Python as raw Python, recognize an outer Markdown container as a container, and **do not recursively search a chosen program’s strings for a better candidate**.

A whole-module parse/compile probe can help recognize valid raw code. It must not become “remove lines until something compiles” or “try earlier answers until one works.”

### Finding 3 — HumanEval’s whitespace and fence handling introduce additional meaning/selection defects. **Confidence: high**

**Whitespace normalization is not always semantics-preserving.** The `dedent()` call can alter a whitespace-only line inside a triple-quoted string, even when the module itself begins at column zero. For example, this source representation contains four meaningful spaces between two newlines:

`'def f():\n    return """a\n    \nb"""\n'`

The returned string is `"a\n    \nb"`. `dedent()` normalizes the whitespace-only line, changing it to `"a\n\nb"`. Python documents this normalization explicitly. This affects an otherwise ordinary complete fenced Python answer.  
Source: `lm_eval/tasks/humaneval/utils.py:46`. 

**Foreign-language fences are not consumed as containers.** An initial `text` fence followed by a valid Python fence can make the first fence’s closing delimiter look like an unlabelled opening delimiter. HumanEval can consequently capture the later opening `python` fence as part of the candidate and produce invalid Python. MBPP’s generic-language matching avoids this particular pairing defect.  
Sources: HumanEval `utils.py:12–15,45`; MBPP `utils.py:28–31,58–64`.

**The entry-point regex is not a structural Python test.** It can match a `def` inside a string and does not recognize every valid function-definition form supported by Python 3.13, such as type-parameter syntax. Expanding that regex indefinitely is not the best remedy; reducing dependence on it for complete-module extraction is simpler.  
Source: `lm_eval/tasks/humaneval/utils.py:41–47`.

### Finding 4 — MBPP’s prose detector can delete valid Python and repair malformed candidates. **Confidence: high**

This is valid raw Python:

```python
match True, False:
    case _:
        def add(a, b):
            return a + b
```

`_is_prose("match True, False:")` meets the text pattern and three-word threshold. `match` is a soft keyword, so `keyword.iskeyword()` does not protect it; parsing the header alone fails because its suite is absent. The extractor therefore deletes the header and leaves an invalid indented `case` statement.

Thus the heuristic demonstrably deletes part of a valid program. Python distinguishes soft keywords such as `match` from ordinary reserved keywords.  
Source: `lm_eval/tasks/mbpp/utils.py:34–46,69–73`. 

Conversely, after a correct function, a malformed line such as:

`result is not None:`

is classified as prose and removed. The extractor cannot establish whether that line was explanation or malformed program logic. Its docstring’s claim that malformed code remains for the execution metric is therefore too strong.

**Remove this general heuristic.** Supporting a short, explicit set of presentation labels is an optional convenience; inferring “not independently parseable, therefore not code” is not sound.

### Finding 5 — Multi-block handling is deterministic, but HumanEval’s hybrid assembly lacks a clean program contract. **Confidence: high**

There is **no test-guided candidate selection** in either new extractor. That is good, but insufficient to guarantee validity.

HumanEval removes every earlier entry-point block wholesale while retaining every other Python block. Suppose the first block contains:

```python
def total(values):
    return sum(values)

saved_total = total
```

and the second contains:

```python
def total(values):
    return saved_total(values)
```

Source-order execution preserves the saved function and works. HumanEval drops the first block, losing a legitimate dependency. Conversely, unrelated earlier blocks containing assertions or side effects remain.

Selecting the last **independent answer** can be a defensible protocol. Combining that selection with arbitrary surviving blocks is a different protocol and can create a program that was never supplied as an independent answer.  
Source: `lm_eval/tasks/humaneval/utils.py:47–54`.

The smallest coherent policy is the one MBPP approximates: **all accepted Python blocks form one module in response order**. Later redefinitions then have their normal Python meaning. Imports, saved bindings, assertions, and side effects are not selectively removed.

That policy can retain harmful stale blocks. This is not something the evaluator should “fix” by deleting assertions or guessing which helpers are obsolete: the prompt requests one complete block. Multiple independent revisions are optional robustness, not a reason to reconstruct an ideal answer.

### Adversarial output coverage

Here, `F(language, code)` means a complete fenced block, and `F_open(...)` means an opening fence without a closing one. “Required” means preservation/failure behavior needed for an accepted representation; “optional” means additional output-format tolerance.

| Case | Concrete example | Current behavior and required decision |
|---|---|---|
| Supported language labels | `F("python3", "def add(a,b): return a+b")`, `PYTHON`, `py`, unlabelled | Supported by the supplied implementations/tests. **Required** to retain. |
| Foreign fence before Python | `F("text", "Illustration")` followed by `F("python", correct_code)` | HumanEval can mispair delimiters; MBPP selects the Python block. Supporting surrounding foreign blocks is **optional**, but delimiter pairing must not manufacture candidates. |
| Other fence styles | `python3.13`, `python linenums`, tildes, four-backtick wrappers | Acceptance is **optional**. Explicit rejection is preferable to accidental partial recognition. |
| Prose around fenced code | `"Here is the solution:" + F("python", correct_code) + "Done."` | Existing supported case. Preserve the block; do not interpret surrounding prose as executable code. |
| Raw code with edge prose | `"Here is the solution:\ndef add(...)\nThis returns the sum."` | MBPP heuristically strips it; HumanEval generally rejects it. **Optional** robustness; narrow MBPP’s rule rather than broadening code repair. |
| Truncated fence alone | `F_open("python", "def add(a,b): return a+b")` | Currently remains invalid or fails extraction. **Required:** do not invent a closing fence or completed body. |
| Completed answer then truncated revision | `F("python", correct_code) + F_open("python", wrong_or_partial_code)` | Both can ignore the trailing incomplete block and retain an earlier answer. **Revise:** do not silently substitute an earlier completed candidate for an explicit dangling code revision. |
| CRLF | A normal Python fence and function using `\r\n` | Covered. Preserve logical source and string values. |
| Presentation indentation | The whole fence indented four spaces | HumanEval accepts through `dedent`; MBPP’s delimiter rules differ. Support is **optional**; arbitrary dedenting of program text is not an acceptable implementation. |
| Fences inside strings | The raw `add()` example above | **Required blocker:** preserve the outer program, not the embedded example. Properly longer outer Markdown fences are an optional extension. |
| Multiple complete revisions | Correct `total()` block followed by wrong `total()` block | Both currently give the later definition effect in the simple case. **Required:** never choose whichever passes. A source-order module policy preserves this without deleting dependencies. |
| Split dependencies | `F("python", "import math")`, then a function using `math.isqrt` | Supported. Also test dependencies stored in a block that defines an earlier entry point; HumanEval currently loses those. |
| Decorators, annotations, future imports | `List`/decorator imports before versus after the definition | Preserve order. A helper called later can be defined later; a decorator needed immediately cannot. **Do not hoist imports to rescue the program.** |
| Stale failing block | `F("python", "assert False")` followed by correct code | Must remain a failure under a combined-module policy. Removing the assertion would change the submitted program. |
| Raw continuations | `"    return sum(values)"` versus `"return sum(values)"` | HumanEval intentionally supports the indented form with the original prompt. Do not silently invent indentation for the second form. MBPP has no body-continuation contract. |
| Few-shot chat layout | Three complete user/assistant demonstrations plus the evaluation user, versus one collapsed user message | Both layouts are exercised in the supplied MBPP tests. Choose and record one layout for comparisons; neither includes an unfinished assistant prefill. |

The table identifies source-level behaviors and proposed invariants, not failures observed in the unseen saved responses.

### Finding 6 — The GSM8K instruction repairs the missing request, not the entire answer-selection contract. **Confidence: high**

The new instruction supplies the format the old prompt omitted, without supplying the mathematical answer or additional solution information. Keep it.

However, the unchanged filters remain materially different:

- Strict extraction takes the **first** matching phrase.
- Flexible extraction takes the **last** numerical match.
- The strict regex’s final `.` is a wildcard, not an escaped literal period.
- Neither filter verifies that the extracted value is the final sentence’s intended answer.

For example:

`The answer is 4. After checking, The answer is 5.`

selects `4` strictly and `5.` flexibly. Also, `The answer is 42` without punctuation can yield strict extraction `4`, because the wildcard consumes the final digit.

These are existing filter semantics, not regressions introduced by the prompt patch. The signed-decimal fixtures correctly establish some retained behavior, not equivalence of the filters.  
Sources: `lm_eval/tasks/gsm8k/gsm8k-cot-zeroshot.yaml`; `lm_eval/filters/extraction.py::RegexFilter.apply()`; `tests/test_gsm8k_zeroshot_format.py`.

For the smallest release, retain the documented filters, predeclare the primary metric, and inspect disagreements. A true terminal-answer validator would be another explicit scoring revision.

The mathematical problems remain unchanged, but **no evidence establishes that the added instruction has negligible effects on scores or rankings**. Fresh v4 comparisons are valid; pooling them with v3 is not.

### Finding 7 — The adapter’s mutation concern is resolved, but configuration and observability gaps remain. **Confidence: high**

The earlier concern about shared kwargs mutation is refuted by the newly supplied callers: both `model_call()` and `amodel_call()` deep-copy generation kwargs, and task request construction also copies them. No additional copying patch is justified for that concern.  
Sources: `lm_eval/models/api_models.py::model_call()`, `amodel_call()`; `lm_eval/api/task.py::construct_requests()`.

Several operational distinctions remain:

**Sampling:** `LocalChatCompletion` discards `do_sample`; actual API sampling follows the transmitted temperature and other controls. The GPT-OSS GSM8K command explicitly uses temperature 1.0, whereas the Qwen command omits temperature and reaches the adapter’s default of 0 under the shown configuration. IFEval similarly compares different sampling/budget profiles. That can be a practical profile comparison, but not a matched-sampling experiment.

**Reasoning controls:** forwarding `enable_thinking` or `reasoning_effort` proves neither support nor effective behavior. The Qwen command’s simultaneous use of both needs verification against the actual Qwen template/server.

**Timeout/context:** the API timeout defaults to 300 seconds, while tokenized requests are disabled and context length is not checked. A 32,768-token ceiling is therefore neither proof of sufficient context nor proof that a long generation can finish before timeout.

**Metadata:** successful responses are reduced to processed `message.content`. `--log_samples` does not, through these methods, retain finish reason, usage, reasoning fields, or the full wire response. Malformed response structures can also become empty strings.

Sources: `lm_eval/models/openai_completions.py`; `lm_eval/models/api_models.py`; `lm-eval Commands.md`.

### Finding 8 — Normal names expose a concrete request-cache hazard that fresh result folders alone do not solve. **Confidence: high**

`Task.build_all_requests()` constructs a cache key from task name, few-shot count, rank/world size, chat flags, system instruction, and tokenizer name. It does **not** include the task version, prompt/filter implementation, dataset revision, or generation settings.

When that cache is enabled, cached instances are returned directly. Thus a new ordinary-name task can reuse an old prompt and old request kwargs while the current task configuration describes the new version. A new SQLite response-cache filename does not invalidate this separate request cache.  
Source: `lm_eval/api/task.py::build_all_requests()`.

The current commands do not explicitly enable request caching, so this is **not evidence that the captured runs used stale requests**. The smallest safeguard is to leave request caching disabled for release validation and make that an asserted launcher setting. Add revision-aware namespacing only when request caching is actually needed.

For response caches, continue using fresh namespaces, but include sampling settings and seeds in the rule—not just model, effort, and token budget. Inspect the outer cache wrapper before claiming its complete identity; that wrapper is not supplied.

### Finding 9 — The captured checks are useful but do not establish program preservation or fresh-run readiness. **Confidence: high**

`final-head-gates.log` establishes the reported 139 scoped passes at `e0c06df…`, lint/config validation, dependency consistency, and editable imports. It does not establish full server behavior, complete dataset coverage, or the adversarial invariants above.

The independent metric probes were captured at `8b280768…`. The final revision’s gates are separate evidence. Verify the relevant diff between those two revisions rather than treating the reviewer’s “SURVIVES” label as validation of every later artifact.  
Sources: `.claude/plan-runs/direct-benchmark-tasks-20260920/logs/final-head-gates.log`; `logs/final-code-review.txt`.

The replay establishes only that its selected first candidates retained or gained a nonempty AST. In particular:

- `ast.parse()` is weaker than compilation; even a top-level `return 42` can parse.
- `pass`, wrong functions, missing imports, and changed string values can remain “parseable.”
- The script chooses folders/files by precedence and lexicographic sorting, without a content-hashed manifest.
- It checks only the first flattened candidate and asserts counts, not candidate identity or functional preservation.

Python explicitly documents the parse-versus-compilation distinction.  
Source: `replay.py`; `logs/final-replay.log`. 

Do not make “zero lost parseability” the sole future gate: correcting a false-positive extraction can legitimately make a previously fabricated candidate fail.

### Finding 10 — The 64-candidate failure is a backend limitation, with broader uncertainty but no demonstrated blanket failure of pass@1. **Confidence: high on the observed failure; medium on its full scope**

The standalone reproduction imports no modified task code. It therefore refutes attribution of that failure to these extraction patches. It directly blocks approval of default 64-candidate execution in that environment.

Ordinary tasks present one candidate per document to the metric through the shown `process_results()` path. Successful single-answer fixtures provide positive evidence for that workload, but not proof that a descriptor-ownership race cannot occur during a longer run. The captured fork warnings also deserve attention: Python documents the underlying problem with forking multithreaded processes.  
Sources: `logs/baseline-code-eval-64.txt`; `lm_eval/api/task.py::process_results()`; `logs/final-head-gates.log`. 

`num_workers=1` in a diagnostic process is **not a demonstrated safe production repair**. It may reduce concurrent activity while still using a worker thread that forks.

There is a second, independent 64-repeat gate: both API paths transmit the same `self._seed` unless overridden. With a deterministic per-request seeded server, identical repeated requests can become clones rather than independent sampling opportunities. Verify the server’s behavior and the unseen repeat/cache expansion before claiming valid pass@k sampling. Do not require unique outputs; require genuinely separate sampling opportunities.  
Source: `lm_eval/models/api_models.py::model_call()`, `amodel_call()`; `humaneval_64.yaml`.

### Finding 11 — Server ownership is correct, but final-only scoring and deployment still need verification. **Confidence: high**

The final-header repair stays in the protocol parser, and the structured-output branch still routes through schema parsing. Nothing shown indicates benchmark answer rewriting.

However, when `reasoning_format` is `NONE`, the parser’s analysis branch places analysis markup/content into `p.content(...)`. Therefore, “the evaluator receives only the final answer” depends on the active reasoning-format configuration; it is not guaranteed by the final-header fix alone.

The template also injects the current date and defaults its own reasoning setting to medium. Record effective rendering/arguments rather than assuming the earlier measured server default still applies.  
Sources: llama.cpp `common/parsers/gpt-oss.cpp`, `extract_reasoning` branch; `models/templates/openai-gpt-oss-120b.jinja::build_system_message()`.

Keep IFEval unchanged. Disabling thinking or using low effort is a declared operating profile, not a universal methodological requirement. Historical empty answers should remain labeled affected/unverified unless their cause is established; a legitimately empty or budget-limited answer can still be an ordinary failure in a correctly specified bounded experiment.

### Finding 12 — Installation is improved; reproducibility and execution isolation are not yet established. **Confidence: high**

The editable-install correction is real and supported by the import checks. But the image tag remains unpinned by digest, dependencies are resolved at startup, and the metric is loaded dynamically. Two machines checking out the same harness commit can consequently run different environments.

More importantly, the supplied Compose exposes a writable canonical checkout and host networking, while the runbook supplies API credentials to the evaluator. No separate isolation boundary for generated programs is shown. Read-only legacy results are an improvement, not a complete sandbox.

Treat safe execution as a release gate independent of task correctness.  
Sources: `/Users/apple/lm-evaluation-harness/docker-compose.yml`; `lm-eval Commands.md`; coding metric helpers’ `code_eval` calls.

---

## 3. Recommendations, in order of preference

### Recommendation 1 — Make a narrow extractor correction release, retaining normal names

**Trade-off:** some permissive historical outputs may become failures, and a few existing exact-output fixtures must change.  
**Beats:** reverting to broken prefix extraction, reintroducing special task names, or building a speculative dependency-repair engine.

Use one bounded extraction contract:

1. Preserve accepted raw Python as a whole program; do not mine its strings for fences.
2. Recognize outer fenced containers consistently, including their language and closed/unclosed state.
3. Preserve code text and source order. Remove presentation delimiters—not arbitrary whitespace inside programs.
4. For multi-block compatibility, treat accepted Python blocks as one source-order module. Do not selectively remove entry-point blocks or infer obsolete dependencies.
5. Retain the specifically supported HumanEval body-continuation route without inventing indentation.
6. Remove MBPP’s general parse-failure-as-prose heuristic. Explicit wrappers are optional.
7. Never repair a dangling final code block by silently falling back to an earlier answer.

A small shared fence-recognition helper is reasonable because both task families need the same delimiter semantics. Keep task-specific continuation handling and metrics where they are.

The minimal production surface is the two extraction helpers, optionally that shared helper, and affected metadata versions. Extend the existing three regression-test files rather than adding a broad parsing subsystem. **Do not change IFEval or the server to compensate for these extractor defects.**

### Recommendation 2 — Release a reproducible evaluator/server pair with an explicit comparison policy

**Trade-off:** a small amount of launcher/manifest work and a deliberate environment update process.  
**Beats:** branch names, mutable startup installs, and manually chosen result folders being the only provenance controls.

Before full runs, record:

- Evaluator commit and dirty state; effective task versions and helper hashes.
- Dataset revisions/fingerprints and downloaded metric implementation hash.
- Python/package environment, image digest, execution limits, and worker settings.
- Loaded server executable checksum/build, GGUF hash, effective template and arguments.
- Resolved messages, few-shot layout, sampling settings, request seed, effort, token ceiling, timeout, concurrency, and actual context allocation.
- Per-request final content, finish reason, usage, and errors through a content-preserving sidecar or existing transport logging.

The template’s date-dependent rendering means a template-file hash alone does not identify the effective prompt.

Choose whether the comparison measures **matched sampling/budgets** or **each model’s declared practical profile**. Both are defensible; the current commands mix configurations that should not be described as matched compute.

Preserve the original implementation commit and historical v1/v2 routes. Publish the correction as a new evaluator release with bumped effective extraction-task versions, then regenerate comparison results with both cache layers controlled.

### Recommendation 3 — Gate 64-repeat runs separately and investigate the process boundary, not the guard

**Trade-off:** delays affected pass@k runs while ordinary pass@1 is validated independently.  
**Beats:** disabling filelock protection, hiding warnings, or choosing an arbitrary older dependency.

Inspect the exact installed primary sources:

- The loaded `code_eval.py::_compute` and `execute.py` around `multiprocessing.Manager()` and candidate-process creation.
- `filelock/_api.py::_audit_fork_safety` at the installed 4.0.1 revision, including its tests and the ownership transitions it protects.
- The actual process start method and threads active at the guarded fork.

Determine whether a supported dependency fix exists or whether the execution backend needs an explicit safe process-launch arrangement. Spawn/forkserver approaches require checking callable importability, manager/process context consistency, cleanup, and timeouts; simply setting a global flag is not a validated repair. Python documents these context differences. 

A safe repair must preserve candidate outcomes and pass@k estimation, and must pass stress tests with the guard active. Submit the minimal reproducer and resulting fix to the responsible upstream component after diagnosis; the traceback alone does not prove filelock is the faulty component.

---

## 4. Assumptions and evidence limits

**A1. Saved responses:** their contents are absent. The replay’s captured totals are evidence about its selected inputs, not independently verified semantic preservation or model attribution.

**A2. Live server:** no loaded executable, effective template, response-format setting, or fresh wire capture is supplied. Local HEAD cannot establish remote deployment.

**A3. Dataset contracts:** actual base/plus dataset rows and revisions are absent. Preserved YAML paths and fixture probes do not establish that packaged fields reproduce every claim associated with an external EvalPlus runner.

**A4. CLI/cache/repeat internals:** the CLI’s final argument merging, outer response-cache wrapper, and repeat expansion are not supplied. Claims about those must remain conditional.

**A5. Execution backend:** only the dependency traceback and outcomes are supplied, not the installed `code_eval`/filelock implementation. Its precise race and safe repair cannot be selected from the error string alone.

**A6. Isolation:** no stronger external executor or disposable VM boundary is shown. Its presence would change the operational safety assessment.

**A7. Performance effects:** no fresh cross-model run establishes format adherence, token headroom, timeout adequacy, or ranking stability under the new prompts.

---

## 5. Checks and minimal fresh-run validation matrix

### Repository checks before generating new answers

**Check the reviewed revision boundary.** Run the following against the canonical checkout and inspect any differences:

```bash
git diff 8b28076866d0dd860d9055621f26e013776335b7 \
  e0c06df1a920307f927ed704963887b4566393ab -- \
  lm_eval/tasks/humaneval lm_eval/tasks/mbpp \
  lm_eval/tasks/gsm8k/gsm8k-cot-zeroshot.yaml \
  tests/test_humaneval_extraction.py tests/test_mbpp_extraction.py \
  tests/test_gsm8k_zeroshot_format.py
```

**Add the adversarial fixtures before editing extraction.** Assert exact preserved candidate text where appropriate, plus functional outcomes for correct/wrong fixtures. In particular, the raw-string counterexample must remain wrong, and the valid `match` program must remain valid.

**Exercise actual task processing.** Extend public-seam tests through `task.process_results()`, not only direct helper calls. Verify candidate cardinality, inherited versions, unchanged references, and that base/instruct aliases do not receive accidental different generation settings.

**Prove cache behavior.** Populate an old request cache in a disposable test location, change the task prompt/version, and demonstrate either recomputation or deliberately disabled caching. Inspect actual instance messages—not only the result configuration.

**Verify current runtime identity.** Record the container image ID, installed environment, import path, server executable path/checksum, effective template, and absence of optimization settings that would disable Python assertions. Make the Compose/run assets part of the versioned deployment rather than a separately drifting local file.

### Minimal validation matrix

This is a release gate, not a demand to benchmark every model immediately.

| Stage | Minimal workload | Pass criteria |
|---|---|---|
| Extractor regression | Both helpers against the adversarial cases above | No raw-program/string substitution; no deleted program statements; preserved string values; explicit behavior for malformed/truncated outputs. |
| Task/prompt integration | All inherited variants using fixture datasets; both MBPP few-shot layouts | Correct prompt/reference/version/repeat settings; no prefill; original metrics receive the intended candidate lists. |
| Adapter integration | Mocked successful, null, malformed, and HTTP-error responses through synchronous and concurrent paths | Payload settings preserved; response cardinality correct; null/error outcomes distinguishable; final-only scored text and sidecar metadata verified. |
| Single-answer execution | At least 100 sequential one-candidate metric calls in the intended launcher, alternating correct/wrong/empty; separate timeout fixtures | Expected 1/0/0 outcomes; no guard exception, hang, or leaked child; timeout remains failure. Residual fork warnings remain recorded risks, not “fixed.” |
| Fresh cross-model smoke | GPT-OSS plus one non-GPT chat model: 12 HumanEval, 12 MBPP, 8 GSM8K, 8 IFEval items each | No unexplained transport/parser/null failures; valid supported answers preserved; actual generation settings and metadata captured. Include the known IFEval parser-triggering item in GPT-OSS’s selection. |
| Other models joining later | A small per-model wire/prompt smoke using the same release | Supported controls and final-channel behavior verified before that model’s full comparison run. |
| 64-repeat gate | After backend repair: synthetic all-correct, all-wrong, mixed, malformed, and timeout candidate sets; then a small live repeat probe | Correct pass@k outputs, stable execution with protection active, and independently generated sampling opportunities without cache collapse. |

Choose smoke document IDs in advance, distributed across the task rather than repeatedly selecting examples known to succeed. The supplied `doc_iterator(samples=...)` re-enumerates selected documents, so retain the original dataset IDs separately when recording selected-sample provenance.  
Source: `lm_eval/api/task.py::doc_iterator()`.

**Do not impose a model-accuracy threshold as a plumbing gate.** Wrong code and wrong arithmetic are allowed outcomes. Fail the release gate when the evaluator changes an answer, loses a supported answer, cannot account for a response, or uses unintended settings.

For a declared capability-oriented smoke, any `finish_reason=length` should trigger budget/profile review before full evaluation. For a deliberately fixed-budget experiment, length-limited answers remain counted outcomes—not infrastructure errors or opportunities for selective reruns.

For mixed 64-candidate metric fixtures with \(n=64\) and \(c\) passing candidates, verify the original estimator against:

\[
\operatorname{pass@k}=1-\frac{\binom{64-c}{k}}{\binom{64}{k}},
\]

with the numerator zero when \(k>64-c\). This verifies scoring independently of whether a model can produce diverse answers.

---

## 6. Open questions that would change the recommendation

**What should multiple code blocks mean: one submitted module or independent revisions?** The recommended source-order policy is the smallest conservative choice. A final-independent-answer policy would need different, explicit boundaries—not inferred dependency repair.

**Do the live server and run captures establish final-only content, supported reasoning controls, context headroom, and adequate timeouts?** Positive evidence would clear several operational gates.

**Is GSM8K “strict” intended to retain upstream first-phrase semantics or actually validate the terminal answer?** The latter warrants a separate narrow filter revision before its results become a new baseline.

**Does extended single-candidate execution reproduce the guarded-fork failure?** If so, the dependency blocker expands beyond the currently demonstrated 64-candidate workload.

**Is there an execution-isolation layer not shown in Compose?** Verified isolation would remove the most serious operational objection.

**Bottom line:** retain the two forks, ordinary task names, corrected prompts, and existing execution metrics. Correct the extraction invariants, strengthen provenance and execution gates, and keep 64-repeat readiness separate. The evidence supports that targeted path—not approval of `e0c06df…` unchanged, and not a return to the earlier broken evaluation routes.