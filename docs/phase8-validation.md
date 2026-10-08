# Phase VIII validation — 2026-10-08

## Baseline and change boundary

Before implementation, the repository was an existing Python 3.12 package with immutable dataclasses, unittest, Ruff, stdlib provider transports, local Community-1/MiniLM backends, FastAPI/SQLite orchestration and React/Vite. Phase I–VII were working. The initial published commit was `5b3b64c9ed485fd6cb9ffa6e3d2fd6eeec458575` on `main`.

Fresh baseline: **542 Python tests, 523 passed, 19 skipped**, 19.174 s, including all thirteen required real FFmpeg tests. Ruff lint/format (137 files), compileall and pip check passed. Frontend: **21/21 tests**, three files, 19.60 s.

A SHA-256 snapshot covered **200** tracked source/test/frontend/packaging files, stored locally at `.validation/phase8/baseline-hashes.json`. Final comparison found exactly eight changed existing Python modules: root CLI, ASR transport, intelligence CLI/workflow, refinement models/prompt/service and web orchestration. All **47 existing test files** and **39 frontend files** in that snapshot were unchanged. Audio, diarization, grounding, protected-edit validation, existing transcript serializers and dependency metadata were unchanged.

New modules live only under `meeting_assistant.contextual_asr`: config/models/errors, bounded context extraction, retrieval/suspicion/windowing, Groq window transcription, hypothesis linkage, service, serialization, integration, workflow, CLI and evaluation. README and `.env.example` contain additive instructions. No dependency, CUDA, Torch, model or system installation changes were made. No Phase IX/X functionality or frontend redesign was introduced. Changes are local and uncommitted.

## Final automated checks

| Check | Actual result |
|---|---|
| Complete Python suite | **607 total; 587 passed, 20 skipped**, 17.016 s |
| New Phase VIII tests | **65 total; 64 passed, 1 live test skipped** in ordinary suite |
| Real Phase I FFmpeg integration | All **13 passed**, required rather than silently skipped |
| Explicit live contextual-ASR test | **1 passed**, 1.146 s; 12-second window, native words, source hash unchanged |
| Ruff lint / format | Passed; **160 files** formatted correctly |
| compileall / pip check | Passed; no broken requirements |
| Frontend tests | **21/21**, three files, 4.36 s |
| Frontend ESLint / TypeScript | Passed |
| Frontend production build | Passed; Vite build 860 ms after TypeScript check |
| Git diff whitespace | Passed |
| Temporary window / staging audit | **0** remaining contextual window directories; **0** pending bundles |

The twenty ordinary skips comprise ten billed-provider opt-ins (including the new test), eight local-model opt-ins and two Windows symlink privilege cases. Cached local Community-1 GPU diarization and MiniLM were exercised in the actual fixture evaluations; this is not a claim that every historical local-model opt-in was rerun. The frontend build retains a dependency `"use client"` bundler warning; there was no build failure or dependency upgrade.

Tests cover pack immutability/normalization/scopes/deduplication/provenance; UTF-8 text/Markdown/JSON/CSV and bounds; hostile instructions as data; scope ranking/top-K; protected and ordinary-word negatives; priorities, clipping, frame boundaries, merging and budgets; native timestamp offsets; exact PCM slicing; candidate constraints and unchanged weak-candidate vetoes; prompt/source tampering; conditional Phase V payloads; optional failure/rate-limit/crop/temp-directory fallback; atomic serialization/cleanup/no-overwrite; no-context defaults; programmatic/web-stage compatibility; identical-request evaluation caching; and incomplete evaluation reporting on 429. Ordinary tests do not call Groq.

Local logs: `.validation/phase8/baseline-checks.log`, `final-checks.log`, `frontend-final.log`, `frontend-lint.log`, `frontend-typecheck.log`, `frontend-build.log`, `live-test.log`. These generated diagnostics are ignored by Git.

## Fixtures and reference status

1. Existing **63.4450625-second** original synthetic two-voice fixture, already used in prior phases. A fresh real Groq first pass, cached local diarization and retrieval were run; the same saved first pass is shared by comparison conditions.
2. New original Windows SAPI two-voice technical/negative benchmark, **55.29925 seconds** after Phase I canonical conversion. `data/contextual_asr_benchmark.json` contains eight intended technical terms and six protected/ordinary controls. `scripts/create_contextual_asr_fixture.ps1` and the existing assembler generate audio locally; no large binaries are committed. The spoken Kubernetes/Qdrant strings deliberately exercise recognition ambiguity. This is an authored pronunciation/intended-term benchmark, not held-out natural meeting audio.

Canonical hashes in every retained ablation result still match the existing source WAV. Raw and speaker artifacts were retained, never rewritten. Neither fixture was manually patched. The source scripts were inspected against recognized text, but no manually checked acoustic reference was established; **no WER is claimed**. Term scores below use the authored intended-term annotations and case-insensitive exact vocabulary occurrence matching. They are not global transcript accuracy.

## Controlled A/B/C comparison

Conditions share identical first-pass raw/speaker artifacts:

- **A:** existing global grounding → existing Phase V.
- **B:** global + meeting context grounding → existing Phase V.
- **C:** the same context grounding + selective contextual hypotheses → existing Phase V.

Initial comparisons hit real Groq refinement 429s after bounded existing retries. Successful A/B bundles and Pass-2 artifacts survived. The comparison harness was extended to recover validated complete conditions and cache only identical Phase V JSON requests. All Pass-2 outputs had zero candidate links, so C's requests were exactly B's requests. C was then completed through the real Phase V application/validation code using the recorded decisions: **six cached requests** for the existing fixture and **ten** for the benchmark, **zero new C LLM calls**. These are controlled request-equivalence results, **not independent fresh LLM replicates**. Raw hypotheses remain real Groq responses.

### Existing 63.445-second fixture

| Observed item | Fresh Pass 1 | A baseline | B context-only | C closed-loop |
|---|---|---|---|---|
| Kubernetes, two occurrences | `kube.net ease` | Unresolved | Unresolved | Unresolved |
| Qdrant | `Drant` | Unresolved | Unresolved | Unresolved |
| Discount wording | `discount at 15%` | **`Discount rate 15%`**, awkward false substitution | Original wording retained | Original wording retained |
| Percentages | `15%`, `50%` | Preserved | Preserved | Preserved |
| Friday | `Friday` | Preserved | Preserved | Preserved |
| Negation/modality | `not`, `Do not`, proposal wording | Preserved | Preserved | Preserved |
| Literal `not Monday` | Absent from this fixture's fresh first pass | Not applicable | Not applicable | Not applicable |

Context did not solve Kubernetes or Drant. Drant was not a selected Phase IV target in this run, so Phase VIII did not invent a new target or replacement for it. Two bounded Kubernetes calls returned `KubeNetEase` / `KubeNet Ease`, with no exact Kubernetes candidate link. The initial detector also made a third unnecessary discount-window call; that exposed the ordinary-function-word targeting problem and was removed by the final general selector rule.

A applied one inspected incorrect terminology substitution and rejected two proposed edits. B/C applied **zero edits**, rejecting two. Their raw protected-field comparison reports zero number/date/negation/modality changes. B avoiding the baseline discount substitution is a result of this context-grounding/refiner run; it is not proof that context fixes that class generally or that Pass 2 caused an improvement.

### Eight-term/six-negative benchmark

| Metric | Pass 1 | A baseline | B context-only | C closed-loop |
|---|---:|---:|---:|---:|
| Correct intended term occurrences | 7/8 | 7/8 | 7/8 | 7/8 |
| Term precision | 1.000 | 1.000 | 1.000 | 1.000 |
| Term recall / one-occurrence exact accuracy | 0.875 | 0.875 | 0.875 | 0.875 |
| Term F1 | 0.9333 | 0.9333 | 0.9333 | 0.9333 |
| Excess annotated term occurrences | 0 | 0 | 0 | 0 |
| Applied refinement edits | — | 0 | 0 | 0 |
| Protected-content changes from Pass 1 | — | 0 | 0 | 0 |

Recognized: QDRANT, PyTorch, CUDA, PostgreSQL, HNSW, Newton-Raphson, plug flow reactor. Kubernetes stayed `kube net ease`; contextual Pass 2 returned `KubeNet ease`. **Targeted Pass-2 recognition: 0/1 Kubernetes targets resolved**; the final selective pipeline therefore did not improve the full transcript's 7/8 term result. It would be incorrect to score a cropped Pass-2 window against all eight meeting terms.

The six inspected negatives preserved GPU/CPU wording, 15% versus 50%, Friday versus Monday, `Do not` and `might`, model 120 versus 20, and ordinary `four quadrants`. There were zero applied technical substitutions; false-edit precision/rate has no useful denominator when no edits were applied. This is not a general semantic-correctness guarantee.

The initial detector incorrectly selected `quadrants` and `accelerates` in addition to Kubernetes. One second pass changed CUDA to `Kuda`; another omitted a numeric fragment at a crop boundary and was marked `protected_window_disagreement`. Neither affected the raw/final transcript. The final selector excludes these simple inflections, already supported acronyms and function-word fuzzy phrases. No fixture-specific text-to-term rule was added.

Evaluation reports: `.validation/phase8/existing-final/evaluation.json` and `.validation/phase8/benchmark-combined/evaluation.json`. These reflect the **initial detector**; final-selector scope experiments below are separate retained runs.

## Final-selector ablation and cost

Real selective ASR was repeated with global-only, meeting-only and combined retrieval on the shared saved first pass. Meeting-only uses the provenance-aware adapter; this was independently revalidated after correcting its initial evaluation adapter. All final saved source/grounding/term links validate with the finished code.

| Fixture / scope | Calls | Audio retranscribed | Provider latency | Total enhancement time | Exact usable candidate links |
|---|---:|---:|---:|---:|---:|
| Existing / global | 2 | 9.3400625 s | 1.784 s | 1.795 s | 0 |
| Existing / meeting | 2 | 9.3400625 s | 1.787 s | 1.797 s | 0 |
| Existing / combined | 2 | 9.3400625 s | 1.480 s | 1.492 s | 0 |
| Benchmark / global | 1 | 4.480 s | 0.827 s | 0.835 s | 0 |
| Benchmark / meeting | 1 | 4.480 s | 0.991 s | 0.999 s | 0 |
| Benchmark / combined | 1 | 4.480 s | 0.811 s | 0.821 s | 0 |

No final contextual calls failed or retried. All scopes selected only Kubernetes, with the same unresolved recognition. Context/global ablation therefore showed **no term-quality improvement** on these samples. Final combined invocation rates are 2/8 grounding records (25%) and 1/16 (6.25%); these are rates per grounding record, not calibrated error frequencies. Audio fractions are approximately 14.72% and 8.10% of the recordings.

Initial A/B/C contextual measurements: existing **3 calls / 13.9800625 s / 2.862 s provider latency**, benchmark **3 calls / 13.740 s / 2.455 s**. Those initial latency fields include small crop/preparation overhead; final instrumentation starts immediately before backend invocation. The initial comparisons and earlier debugging/provenance runs are additional development traffic, not included in the per-meeting final table. An early harness path failure left first-pass/refinement artifacts but no completed contextual result; a total paid-development-call count is not inferred from incomplete telemetry.

Actual recorded A/B Phase V processing, including bounded retries: existing A 5.794 s / 5 calls; B 68.558 s / 12 calls (six 429 responses). Benchmark A 52.502 s / 15 calls (five 429s); B 111.578 s / 20 calls (ten 429s). C's cached Phase V application took roughly 0.012 s for each fixture. Additional aborted-condition traffic is not fabricated into these retained-success counts. Contextual ASR itself performs zero retries; Phase V keeps its existing retry policy.

Provider audio seconds are **not billable seconds** or a monetary cost estimate; Groq's documented ten-second minimum per request applies. No unavailable token/billing/ASR-confidence values were invented.

## Downstream and artifact validation

The existing fixture's completed C RefinedTranscript was passed through the unchanged **real** Phase VI CLI using its retained speaker/grounding/diarization/audio evidence. It produced:

- 3 summary items, 9 minutes, 3 decisions, 3 actions;
- 25 resolved evidence references, **0 unresolved**;
- 2 provider calls, one schema repair, zero transport retries;
- 6.986 s provider latency / 7.000 s total extraction;
- a MeetingRecord with matching refined provenance and canonical audio binding.

The existing evidence resolver independently resolved all 25 spans from this saved record without provider calls. Structural evidence validity does not establish semantic accuracy of generated claims.

Representative local artifacts (ignored):

```text
.validation/phase8/meeting_context.json
.validation/phase8/final-policy/existing/combined/contextual_asr/
    eca37d3c-5607-46be-8d12-0d60eaded725/
        meeting_context.json
        contextual_asr.json
        contextual_asr.txt
.validation/phase8/existing-final/contextual_asr/<audit-uuid>/
        contextual_refinement_links.json
.validation/phase8/end-to-end/3649ef85-c2c2-471c-be86-b6a44f486b7f/
        meeting_record.json
        meeting_record.md
        evidence_manifest.json
```

Existing raw/speaker/grounding/refined/MeetingRecord contracts, evidence APIs, anonymous speaker IDs and word times are preserved. The web integration keeps six stages and fourteen registered artifacts; multipart validation and React source hashes are unchanged. Context entry/downloads are CLI/Python/on-disk only. The full ordinary web regression and frontend build passed; no new context browser workflow is claimed.

## Genuine limitations and next contract

- No measured technical-term improvement on these two synthetic recordings; stronger hints are not evidence that a term was spoken.
- Targets must already exist in Phase IV grounding. Missing/weak retrieval can leave Drant untouched; conservative inflection/function-word/acronym filters also trade recall for precision.
- Phase V vetoes remain authoritative; even a correct Pass-2 term cannot relax a weak-candidate veto.
- Whole-window protected-signature comparison can veto useful evidence because of crop-boundary omissions or equivalent number formatting. This is deliberately conservative.
- Small synthetic intended-term annotations are not checked acoustic ground truth; no WER or general human-meeting accuracy claim is made. Excess term counts do not detect every semantic error.
- Contextual inference currently uses hosted Groq; offline operation and HTTP/frontend context management are absent.
- Existing LLM quotas can delay/fail refinement. Cached identical-condition decisions reduce duplicate work but do not provide independent stochastic replications.

Phase IX may consume the existing immutable raw/speaker transcripts, canonical WAV and retained contextual hypotheses to add a second diarizer, alignment and speaker agreement/reliability. Phase X may later add Jev, a meeting-event graph, decision evolution and semantic verification. Neither was implemented here. Participant vocabulary never resolves speaker identity.
