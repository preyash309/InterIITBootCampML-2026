# Phase X validation — 2026-10-08

The user requested a free local replacement for billed Jev/TypeSafe access.
Local typed backends and the additive event/verification layer are implemented.
**Software/offline execution passed; useful meeting-domain model accuracy did not.**
Shadow mode remains disabled by default. There is no automatic semantic gating,
record replacement, speaker correction, fine-tuning or new frontend.

## Frozen baseline and architecture

Before Phase X changes: **668 tests, 647 passed, 21 skipped**, 21.696 seconds.
Ruff, formatting, compilation and main-environment dependency checks passed.
231 existing source/test/frontend files were hashed. Final comparison shows changes
only to root CLI routing, intelligence CLI, contextual workflow and web orchestration
for optional sidecar hooks. Phase VI prompts/evidence/models, audio/ASR/diarization
modules and all frontend source match that baseline. Earlier uncommitted VIII/IX
changes remain present and were not reverted.

`semantic_reasoning/` separates frozen schemas/config, ontology, bounded candidate
selection, provider protocol/transport, local worker, deterministic graphs/evolution,
verification, coverage, source validation, atomic serialization and evaluation.
Supporting additions are two isolated dependency locks, setup/benchmark/fixture
assembly scripts, `data/semantic_benchmark.json`, documentation and 89 tests.
Optional workflow hooks preserve the six public stages and fourteen download keys.
Canonical extraction is saved before the optional layer runs.

## Environment and model comparison

Windows, Python **3.12.14**, RTX **4070 Laptop 8,188 MiB**, NVIDIA driver **596.49**.
The existing main ML stack remains Torch **2.8.0+cu128** / CUDA **12.8** and
Transformers **5.19.0**. No driver, system CUDA, cuDNN or main package replacements.

| Item | Julia-1 | GLiNER2.5-Decide |
|---|---|---|
| Model repository | `SupersonicLabs/Julia-1` | `fastino/GLiNER2.5-Decide` |
| Model/runtime | 144.3M; `supersonic-julia 0.1.0` | Published 340M encoder; `gliner2 2.0.0` |
| Checkpoint bytes | 577,189,056 | 1,945,828,140 |
| Cache including ancillary files | 611,787,804 bytes | 1,954,172,453 bytes |
| Cache | `.models/semantic/julia-1` | `.models/semantic/gliner-decide` |
| Isolated environment | `.venv-semantic` | `.venv-decision` |
| Transformers / tokenizers | 5.0.0 / 0.22.2 | 4.57.6 / 0.22.2 |
| Precision on GPU | BF16 autocast, FP32 resident weights | FP16 weights/inference |
| GPU benchmark peak allocated | 688,888,832 bytes (657 MiB) | 2,207,947,776 bytes (2,106 MiB) |
| Benchmark model loading | 8.562 s | 10.172 s |

These are per-worker PyTorch peak allocations, not whole-device memory or a promise
for arbitrary long input/concurrent models. Both fit this laptop in the tested jobs.
Julia uses its native full probability distribution. GLiNER uses the documented
native softmax classification output for all labels; native FP16 rounding is retained.
GLiNER reports an eager-attention fallback from its requested SDPA backend in this
Transformers version; this is an internal attention implementation fallback, not CPU.

Both checkpoints downloaded at pinned revisions and passed SHA-256 verification.
Two explicit GPU tests passed in **24.867 s** with local directories, offline Hub
settings and Python socket connections disabled. The Julia worker reused its loaded
model across two calls. The tests check execution/schema, not semantic accuracy.

Laya's base English 421M checkpoint was researched but not installed/tested. Its
short English context and publisher-reported weak general typed-decision results
did not justify another model download. OpenJev was also researched, not tested.
Julia remains the smaller experimental default; GLiNER remains a comparison option.
Neither is recommended as a trusted replacement for meeting claim adjudication.

## Controlled benchmark

21 authored conversations, **34 event labels** (33 substantive events plus one
ambiguous), 12 relation references and 11 authored verification challenges. Eight
conversations reuse Phase VI examples; thirteen are new. Cases include suggestions,
acceptances, explicit rejections, numbers/negation/conditions, owner/deadline traps,
shipping reversals, independent topics and task commitments.

Event/relation runs use explicitly empty evaluation records and **no Phase VI calls**.
Verification claims are separate authored challenge records, never canonical output.
No reference labels are sent as expected answers or substituted for predictions.
This small authored benchmark is not a human-meeting accuracy estimate.

| Metric | Julia | GLiNER |
|---|---:|---:|
| Completed conversations | 21/21 | 21/21 |
| Raw argmax correct, before acceptance threshold | 2/34 (5.88%) | 4/34 (11.76%) |
| Policy event macro F1 | 0.04293 | 0.00000 |
| Policy event precision / recall | 0.22222 / 0.06061 | 0 / 0 |
| Accepted correct substantive events | 2/33 | 0/33 |
| Relation F1 | 0 | 0 |
| Provider request count incl. challenges | 47 | 45 |
| Sum of provider request latency, incl. cold startup | 11.784 s | 16.953 s |
| Multiclass Brier (sum over classes), n=34 | 1.41015 | 0.93907 |
| Ten-bin top-choice ECE, n=34 | 0.60923 | 0.35171 |

Julia's per-class policy F1 is 0.33333 for INFORMATION, 0.18182 for PROPOSAL and
zero for the remaining ten substantive classes. GLiNER's policy F1 is zero for
all twelve substantive classes because its event probabilities remained below 0.85.
GLiNER has better raw argmax accuracy here, while Julia accepts two correct events;
this does not establish either model as good at meetings. Thresholds remained 0.85
acceptance / 0.60 review; they were not lowered to make the experiment look successful.

Evolution source chronology and evidence validity passed 21/21. Current-decision
set matches were 14/21 and historical matches 19/21 for both, largely because many
gold sets are empty and both models missed confirmed decisions. Zero proposal-to-
decision false positives and zero retained rejected choices are similarly weak
statistics when recall is this poor. No successful live supersession chain was found.

| Verification dimension | Julia | GLiNER |
|---|---:|---:|
| Claim supported | 5/7 | 3/7 |
| Confirmed item type | 4/7 | 1/7 |
| Owner explicit | 1/3 | 1/3 |
| Deadline explicit | 1/3 | 1/3 |
| Negation preserved | 2/2 | 2/2 |
| Numbers preserved | 1/1 | 1/1 |
| Scope preserved | 1/1 | 0/1 |

These very small dimension-specific samples are not calibration or robust quality
evidence. Coverage accuracy/recall is **unavailable**: no actual baseline records
with manually annotated coverage links were created. Unit tests exercise paraphrase,
missing event, wrong type, duplicate and unavailable behaviors using controlled
responses; they are not measured model accuracy.

Artifacts: `.validation/phase10/julia-readable/evaluation.json` and
`.validation/phase10/gliner-readable/evaluation.json`, with full frozen predictions,
probabilities, confusion matrices and reliability bins in their sidecar bundles.
Earlier input-format experiments are retained separately; the table uses readable
state v1, preserving all source text without mechanical word-index payloads.

## Existing saved meeting

The existing **63.445-second** synthetic two-voice recording was analyzed without
rerunning ASR, diarization, refinement, extraction or Phase IX. Sources include its
matching ten-utterance speaker/refined transcript, three decisions, three actions
and saved independent speaker-reliability sidecar. All four source-file hashes
remained identical before/after both model runs.

| Measurement | Julia | GLiNER |
|---|---:|---:|
| Candidates / accepted nodes | 10 / 3 | 10 / 0 |
| Accepted relations / issue threads | 0 / 3 | 0 / 0 |
| Requests / total Phase X time | 18 / 10.344 s | 16 / 13.797 s |
| Model load time | 7.688 s | 8.703 s |
| Verification | 2 REVIEW, 4 UNSUPPORTED | 6 REVIEW |
| Coverage observations | 0 | 0 |
| Availability | partial | available (provider operational; no accepted nodes) |

Julia accepted only PROPOSAL events. One was the explicit dashboard suggestion;
another misclassified a confirmed ownerless benchmarking task as a proposal.
It also labeled the earlier confirmed shipping date a proposal with a high score.
It missed all confirmed decisions and assignments. An invalid proposed supersession
between non-decision nodes was explicitly rejected and recorded as a warning.
Verification disagreement does **not** prove the canonical claims wrong.

For example, the actual accepted quote remains:
“Someone should review the dashboard. That is a suggestion, not an assignment.”
No speaker, word, number or terminology correction is added by this layer.

Phase IX cross-reference: all ten candidates fall in one RELIABLE and nine MIXED
utterances; Julia's three accepted events all fall in MIXED regions. No event falls
in UNCERTAIN here. This is a distribution, not a causal relationship or correctness
probability. Secondary diarization was not rerun and its sidecar was unchanged.

Example generated bundle paths:

```text
.validation/phase10/julia-existing-final/semantic_reasoning/<sha256>/semantic_result.json
.validation/phase10/julia-existing-final/semantic_reasoning/<sha256>/decision_evolution.txt
.validation/phase10/gliner-existing-final/semantic_reasoning/<sha256>/semantic_verification.json
```

Detailed actual results: `.validation/phase10/local-existing-report.json`.

## Software checks and failure cases

Final full Python suite: **757 tests, 733 passed, 24 skipped**, **20.502 s**.
Phase X: **89 tests, 86 passed, 3 opt-in tests skipped**. The two explicit local
GPU tests passed separately; live Jev remained skipped because access was unavailable.
The full suite includes the required thirteen real FFmpeg integration tests and
existing web regressions. New enabled-shadow failure regression confirms the six
stages/fourteen artifacts survive optional failure.

Ruff lint passed; all 210 Python files were formatted. Compile/import checks and
`pip check` passed in the main, Julia and GLiNER environments. `git diff --check`
passed. Model weights, `.env`, local environments and generated outputs are ignored
and not tracked. No commit or push was made.

Tests cover unknown labels, probabilities/nonfinite/malformed JSON, deterministic
IDs, immutable reconstruction, stale/cross-meeting sources, invalid timestamps,
unknown references, graph cycles/chronology, temporary cleanup, safe overwrite,
budgets/deduplication, missing/wrong weights, worker reuse/timeout/cleanup, encoding
overflow, per-head native scores and API error/retry translation. Controlled
software fixtures verify proposal → acceptance/rejection, explicit supersession,
questions/answers, independent issues and task evolution; live model quality must
not be inferred from these passing software tests.

No Jev billable inference was performed. The optional transport follows the
inspected current official API and retains returned usage metadata; its accuracy,
latency and cost were not measured. Local semantic inference requires no API key.
Existing upstream Groq stages still use their configured APIs.

Automatic approval review rejected creating a new Groq benchmark baseline because
it would export transcript payloads; that operation was not executed or bypassed.
Local authored evaluation and already-saved meeting records were used instead.
A temporary review-service usage failure later delayed an isolated dependency install;
after its stated reset and the user's continuation, normal review approved it.

## Genuine remaining limits and next contract

The major limitation is measured weak model generalization, including high-score
wrong labels, missed decisions and poor owner/deadline checks. One dominant event
per utterance loses multi-clause distinctions. Lexical issue grouping/pair retrieval
can miss paraphrases; call/time/token budgets can leave partial observations.
There is no annotated human-meeting evaluation or calibrated automatic gating.
Coverage gold and a valid live Jev comparison remain unavailable.

Phase XI can consume `SemanticResult.events`, `.relations`, `.event_graph`,
`.decision_evolution`, `.verification`, `.coverage` and provenance alongside the
existing MeetingRecord. `get_verification(item_id)` and
`get_decision_evolution(issue_id)` provide sidecar lookups. Any future UI must
present these as experimental review observations and retain the canonical evidence
contract. Phase XI was not implemented automatically.
