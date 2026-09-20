# Research question: Are our normal-name lm-eval fixes correct, model-neutral, and scientifically defensible?

You are answering from this message alone. You cannot see the repository, its
history, its data, or any file not pasted below. Where an answer depends on
something not shown, say what you would need and answer conditionally rather
than guessing.

## Question

Independently audit the attached implementation and evidence: is our split between llama.cpp protocol fixes and direct fixes to the ordinary lm-evaluation-harness tasks a sound approach, and are the actual patches correct enough to use for new cross-model comparisons? Try to refute that claim, not simply confirm it. Distinguish implementation correctness, benchmark validity/comparability, and operational readiness. Recommend keep, revise, or revert for each change, with concrete counterexamples or missing evidence when warranted. The related decision is whether to keep the implemented suite or make targeted corrections before expensive fresh benchmark runs.

## Why it is being asked

We repeatedly encountered misleading coding scores, zero GSM8K strict matches, and GPT-OSS server failures. Carrying manual patches and choosing separate fixed task names became confusing. We now maintain forks, applied the fixes under normal task names, tested them, and updated the Docker/runbook setup. We want an independent correctness and evaluation-methodology review before trusting future results. This is not a request to endorse earlier advice: the earlier research reply and internal reviews are context to challenge, not authority. Do not claim tests passed because you ran them; you can only inspect the attached source and captured outputs.

## Background

The model server runs on another PC through an Unsloth/llama.cpp OpenAI-compatible endpoint. The maintained server fork is corbinjurgens/llama.cpp, branch gpt-oss-final-constrain, local HEAD 775f85037f4e5d07c5639ea68d20208192aba529. Its earlier commits fe79b40369fff7a8de63c24f5eb8173b64b864f3 and 775f85037f4e5d07c5639ea68d20208192aba529 address an optional Harmony final-channel constraint tag and an optional JSON-schema items field. Current parser/template source is attached; the full server test suite and live remote-server state are not. Server responsibilities are templating, protocol parsing, generation and API response construction; harness responsibilities are prompts, output extraction, scoring and result provenance.

The evaluator is now corbinjurgens/lm-evaluation-harness, branch codex/chat-complete-evals, implementation HEAD e0c06df1a920307f927ed704963887b4566393ab. The canonical checkout is mounted at /workspace in lm_eval_sandbox and installed editable with .[api,ifeval], package version 0.4.14.dev0, Python 3.13. A separate legacy checkout is read-only at /legacy-workspace. Compose and verification output are attached. Absolute local paths in context labels identify files, not resources available to you. The implementation diff is pinned from 3c070e77dc4725f42eb7de5f56e126927fee25f0 to e0c06df1a920307f927ed704963887b4566393ab. Current files supplement the diff where unchanged inheritance/caller behavior matters.

The user explicitly chose to put fixes directly into humaneval, mbpp, and gsm8k_cot_zeroshot, retaining ordinary names and inherited variants; ifeval remains unchanged. This later decision supersedes the earlier research prompt's additive-only constraint. No model-name checks are used in the new tasks. Model-specific reasoning settings stay in invocation arguments. Older chat_complete_v1/v2 definitions remain historical reproduction routes, not active choices. This private comparison suite is not being represented as untouched upstream benchmark methodology.

## Already known or ruled out

Historical investigations found complete fenced chat answers being lost by prefix-oriented HumanEval/MBPP extraction. Historical GSM8K version-3 prompts asked for step-by-step reasoning but not the phrase required by strict-match; both GPT-OSS and Mistral runs had all strict extractions invalid while flexible extraction found many correct answers. The notes retain this historical evidence; saved model answers are not bundled here. Do not infer that every historical failure had the same cause.

The implemented HumanEval changes request a full function including dependencies, remove assistant prefills and code-oriented stopping strings, accept raw/fenced complete functions or legacy indented body continuations, and preserve separate dependency blocks alongside the last block defining the entry point. MBPP requests full Python code, accepts raw code or concatenates Python blocks in response order, heuristically removes limited edge prose, and uses complete fenced few-shot examples with a newline delimiter. Both use the existing execution metrics/reference tests and 4,096-token task defaults, overridable at invocation. Inherited instruct/plus/64 variants were included; few-shot counts and pass@k configurations were retained. GSM8K now explicitly requests 'The answer is <number>.' without changing its strict/flexible filters or exact-match metric. Effective metadata versions are documented; IFEval stays version 4.0 unchanged.

Final scoped regression suite: 139 passed, with 30 Python multiprocessing warnings. Lint, task discovery/config validation, editable imports from outside /workspace, pip dependency consistency and Compose validation passed. Independent reviewers found and then rechecked fixes for Python3 fences, separate dependency blocks, and MBPP few-shot fence joining. The attached final reviews include public TaskManager/prompt/filter/metric probes: correct code passes, wrong/empty code fails, signed-decimal GSM8K answers retain original scoring behavior. These are fixture and source-level checks, not proof that all models or all response formats work.

Re-extraction of 2,756 saved answers from GPT-OSS, Mistral, Qwen and Ornith runs lost no previously parseable candidate and made 759 more candidates parseable. This is syntax-only evidence, not functional accuracy, an official rescore, or a fresh benchmark. The replay program and final output are attached so you can critique the measurement itself. No fresh full model runs or new remote-model-PC binary verification accompanied this implementation.

A separate unresolved issue was reproduced by direct evaluate.load('code_eval').compute using 64 candidates, importing no modified task code: evaluate 0.4.6/filelock 4.0.1 raises 'os.fork is unsafe while filelock is changing descriptor ownership'. Single-answer metric fixtures pass with default settings. An independent reviewer verified all coding variants using num_workers=1 only in the diagnostic process; no production concurrency setting or dependency pin was changed. Treat default-concurrency humaneval_64 readiness as unproven, not repaired. Distinguish this baseline dependency limitation from regressions introduced by the patch.

The earlier reply motivated the ownership separation and concerns about benchmark contamination, but its advice is not verified merely by being attached. Its original installation paths, task-name constraints and pre-implementation state are historical. The new evidence establishes only the concrete checks above. Audit the chosen implementation rather than assuming it followed every earlier recommendation correctly.

## Constraints

Preserve the user's normal-task-name workflow if correctness permits; do not reintroduce separate GPT-OSS task names merely as a preference. If an invariant truly makes that workflow unsound, explain precisely why and distinguish a necessary change from an optional convention. Changes must remain explicit, versioned and tied to evaluator revision/settings; old upstream, v1/v2 and new results cannot be silently pooled. New comparison models must be rerun under the same task contract with fresh caches. Reasoning controls may differ when model semantics require it, but do not hide GPT-OSS settings inside universal tasks.

Preserve genuine task difficulty and execution safety. No server-side rewriting of answers to satisfy a benchmark, no oracle selection among candidates using reference tests, no removal of failing program logic to inflate scores, no relabeling empty answers or parser errors as correct. Do not recommend bypassing filelock's fork-safety guard or downgrading dependencies without analyzing why it exists and how a safe fix would be validated. Prefer narrow, maintainable, testable fork changes. Do not assume that a Docker container alone is a complete sandbox for arbitrary generated code. Minimize unrelated changes and manual cross-machine patches.

Answer from the attachments. Source statements and captured results have different evidential strength; distinguish them. Do not assume current upstream PR status, dependency release behavior, a specific running Windows binary, or an unseen dataset/sample. If external investigation is needed, state the exact primary source and question rather than inventing a current fact.

## What to return

1. A direct answer, in a few sentences.
2. Findings, numbered, each with its reasoning and a confidence of high, medium
   or low.
3. Recommendations, in order of preference, each naming its trade-off and the
   alternative it beats.
4. Assumptions: every point where the answer depends on code, data or
   behaviour not shown here.
5. Checks: concrete things to verify in the repository before acting, each
   stated so someone with the code can run or read it.
6. Open questions whose answers would change the recommendation.

Cite attached files by path. Show only the lines that change, never a whole
rewritten file.

Additionally return: (a) a per-change table with keep/revise/revert/insufficient-evidence, severity, supporting file/lines and smallest remedy; (b) adversarial output examples covering alternate fenced languages, prose, truncated fences, CRLF/indentation, Markdown fences inside strings, multiple revisions, split dependencies, decorators/typing/imports, raw continuations and few-shot chat layouts, separating required support from optional robustness; (c) whether extraction can change program meaning, cherry-pick a passing answer, retain harmful stale blocks, or silently lose valid dependencies; (d) whether unchanged metrics plus changed prompts/stops/budgets are a valid fork-defined comparison and what provenance/cache controls remain missing; (e) whether GSM8K's new instruction fixes the contract without making strict/flexible results equivalent or altering the tested capability materially; (f) whether IFEval and reasoning/token-budget recommendations are appropriately scoped; (g) a minimal fresh-run validation matrix and concrete pass/fail criteria, not a request to exhaustively rerun every model; and (h) whether the 64-candidate dependency failure blocks ordinary single-answer runs, only affected variants, or broader claims, and the safe next investigation. Distinguish demonstrated bugs from plausible risks and avoid speculative wholesale redesigns.
