# Phase VI validation — 2026-10-07

This report distinguishes measured results from intended behavior. No Phase VII semantic
verifier, UI, PDF, local LLM or replacement local ASR was added.

## A. Repository assessment and regression

The existing Python 3.12.14 virtual environment uses a src-layout package, frozen dataclasses,
`unittest`, Ruff 0.16.10, stdlib HTTP/JSON transports and ignored project-local `.env`.
Phases I–V already included ingestion, API Whisper ASR, cached Community-1 diarization,
cached MiniLM glossary retrieval and conservative Groq terminology refinement.

Baseline: 418 tests, 411 passed, 7 skipped, 90.642 seconds, including real FFmpeg and cached
GPU/CPU diarization/grounding integrations. Live provider tests were excluded from that run.
Final post-implementation suite: **497 tests, 487 passed, 10 skipped, 85.926 seconds**,
including cached local GPU/CPU/embedding and real FFmpeg integration tests.

SHA-256 comparison across 139 pre-existing project files found only five intended modifications:
`.env.example`, README, pyproject, root dispatcher and refinement's low-level transport wrapper.
All audio/ASR/diarization/grounding modules, legacy tests, prior docs and benchmark data remained
byte-identical. No legacy file disappeared. Refinement transport retained its original patch
point and behavior; its 118-test suite passed after promotion of HTTPS exchange.

All seven saved upstream artifacts—raw JSON, speaker JSON, diarization JSON, grounding JSON,
refined JSON, edit log and canonical WAV—were hashed before/after saved VI extraction and
remained byte-identical. Clip extraction independently repeated hash preservation checks.
Audits live under ignored `.validation/phase6/`.

## B–C. Backend and reproducibility

Authenticated GET `/openai/v1/models` returned HTTP 200 and listed `openai/gpt-oss-120b`.
Extraction uses Groq POST `/openai/v1/chat/completions`, temperature 0, strict json_schema,
reasoning effort low, include_reasoning false, stream false, 8192 completion-token cap.
Timeout 90 seconds; at most one 429/5xx retry with Retry-After capped at 60 seconds;
at most one structure/reference-only repair. No silent model changes or semantic retries.
Verified TLS/no redirects/1-MB response cap are shared with Phase V.

Versions: intelligence_v1, meeting_record_schema_v1, meeting_examples_v1,
evidence_policy_v1, consolidation_v1. Policy was developed against observed errors; benchmark
files are explicitly original controlled text regressions, not a held-out speech corpus.

An invalid nullable owner object was rejected; its subsequent repair hit a rate limit.
No final bundle was published. The provider schema was then tightened to valid discriminated
speaker/named-entity/null branches. A TLS connection reset interrupted one benchmark; the
completed prediction was retained and remaining cases resumed explicitly, not silently retried.

## D–E. Architecture and actual schema

New `meeting_assistant.intelligence` modules: models, config, exceptions, base protocol,
prompt, validation, groq_backend, service, evidence, serialization, workflow, fixtures,
evaluate, `__main__` and exports. Shared `_groq_transport` only handles HTTPS exchange.
One original SAPI fixture script and an 18-case labeled text benchmark were added.

Public APIs: `extract_meeting_record`, `resolve_meeting_record_evidence`, `get_item_evidence`,
`extract_evidence_clip`, `meeting_record_to_json/from_json`, `save_meeting_record`.
Backend: `MeetingIntelligenceBackend.extract(IntelligenceRequest) -> IntelligenceResponse`.
The frozen canonical record binds source IDs/digests, policies, model configuration, per-call
measurements, optional portable audio reference and `MeetingContent`.

Example actual content fields:

```json
{
  "summary": [{"id":"sum_0001","text":"Decision to use kube.net ease for deployment.",
               "evidence_utterance_ids":["utt_000001","utt_000002"]}],
  "minutes": [{"id":"min_0004","topic":"database","kind":"proposal",
               "text":"Suggestion to move the database to MongoDB.",
               "evidence_utterance_ids":["utt_000003"]}],
  "decisions": [{"id":"dec_0002","text":"Keep PostgreSQL as the database; do not migrate to MongoDB.",
                 "evidence_utterance_ids":["utt_000003","utt_000004"]}],
  "action_items": [{"id":"act_0001","task":"Benchmark both models by Friday, measuring latency and memory usage.",
                    "owner":{"kind":"named_entity","speaker_id":null,"display_text":"Rahul"},
                    "deadline_text":"Friday","evidence_utterance_ids":["utt_000005"]}]
}
```

The complete JSON nests these sections under `content`. Convenience typed properties expose
record.summary/minutes/decisions/action_items. IDs are assigned by code, not the provider.
Anonymous speakers and named owners are distinct; Rahul is not mapped to SPEAKER_00.

## F–G. Evidence chain and audio

LLM-supplied utterance IDs → validated source refined digest → exact RefinedUtterance →
anonymous speaker + source start/end + refined/raw text + original word/segment references →
canonical WAV bound through matching diarization ID and audio SHA-256.
Provider schema contains no timestamps, quotes, filenames or final IDs. Every final item
resolves without another API call. Separate support utterances stay separate playback spans.

Actual clip: `act_0001` → `utt_000005`, SPEAKER_00, 22.480–28.560 seconds.
Start/end frames: 359680/456960; 97280 frames, 6.080 seconds, 194604 bytes.
Independent ffprobe confirmed WAV, pcm_s16le, 16000 Hz, mono, 16 bits for both source and clip.
Canonical duration: 63.445 seconds. Exact PCM frame tests verified floor/ceil boundaries,
padding/clamping, equality to source samples, no overwrite and failed-temp cleanup.
No claim of manual auditory listening is made; metadata, source frames and textual evidence
were inspected. The source hash was unchanged before/after extraction.

Clip path: `.validation/phase6/final-evidence/act_0001-utt_000005.wav`.
Audio extraction is optional and disabled by default. Normal publication produces no clips.

## H. Controlled extraction benchmark

The preliminary 18-case run exposed a duplicated confirmed-work decision and an incorrectly
split follow-up task owned by its requester. A later run fixed these: decision P/R/F1=1/1/1;
task P/R/F1=1/0.8889/0.9412, false owners/deadlines=0, unknown references=0, evidence
validity/coverage=1. It missed one named-team assignment. Manual review also caught one
summary describing an instruction to state approval as an approval claim, although no
approval decision/task was extracted. Those distinctions were tightened before the final run.

Final complete 18-case run (`.validation/phase6/benchmark-validated/report.json`):

| Metric | Actual result |
|---|---|
| Decision precision / recall / F1 | 0.8000 / 1.0000 / 0.8889 (TP 4, FP 1, FN 0) |
| Task precision / recall / F1 | 1.0000 / 0.8889 / 0.9412 (TP 8, FP 0, FN 1) |
| Owner / deadline accuracy on matched actions | 1.0000 / 1.0000 |
| False nonnull owners / deadlines | 0 / 0 |
| Unknown evidence references | 0 |
| Evidence validity / item coverage | 1.0000 / 1.0000 |
| Annotated supporting-ID precision / recall | 1.0000 / 0.8824 |
| Unmatched decisions / tasks | 1 / 0 |
| Manually identified unsupported summary / minute descriptions | 1 / 1 |

The extra decision duplicated confirmed benchmarking work, rather than inventing a new
product choice. The missed task was the explicit Rahul assignment, which remained in
minutes/summary as a request. The team assignment was correctly extracted after the policy
clarification. The injection case still described an instruction to state approval as a
participant stating/claiming approval; this is an inaccurate description, even though it
produced no approval decision or task. The full rerun is reported, not cherry-picked
predictions from better earlier runs. Temperature zero did not guarantee semantic stability.

Manual review found no other unsupported summary statements or missed main topics in these
18 short cases. This is a narrow manual review, not an automatic entailment score. Supporting
IDs for the changed decision omitted earlier contextual utterances; the final utterance
itself states PostgreSQL selection and Redis withdrawal. Annotated ID recall therefore
must not be interpreted as a semantic grounding score.

Metrics use explicit labeled keyword matches and supporting-ID overlaps. Owners/deadlines
are scored on matched actions including null values. Final benchmark: 18 HTTP-200 calls,
45.927524 s provider latency, 36479 reported tokens, no retries/schema repairs.

## I. Actual I–VI recording

Original 10-turn two-voice Windows SAPI fixture, 63.445 seconds, generated by
`scripts/create_intelligence_fixture.ps1`; no third-party recording or manual intermediate
artifact patch. The actual full CLI ran ingestion, Groq Whisper Large-v3 ASR, cached CUDA
Community-1, local MiniLM retrieval, Groq refinement and Groq meeting extraction.

The speaker transcript retained two anonymous speakers and ten utterances. One existing
Phase V terminology edit was applied: `discount at` → `Discount rate`. Kubernetes candidates
for `kube.net ease` were rejected by Phase V's weak-candidate guard; `Drant` remained instead
of the intended Qdrant. VI faithfully retained these upstream errors rather than inventing
unspoken corrected evidence. No claim of perfect technical-term recognition is made.

- MongoDB: utt_000003, SPEAKER_00, 10.720–15.100, “We could move the database to MongoDB,
  but this is only a proposal.” It remained a proposal minute, not a migration decision.
- Decision dec_0002: keep PostgreSQL/do not migrate to MongoDB; cites utt_000003 and
  utt_000004 (SPEAKER_01, 15.740–21.220), the explicit rejection/retention statement.
- Action act_0001: benchmark both models/measure latency and memory; named owner Rahul,
  deadline Friday, utt_000005/SPEAKER_00/22.480–28.560.
- Action act_0002: write benchmark report; owner SPEAKER_01, deadline null. “Revisit timing
  later” did not become a date.
- Action act_0003: check whether Drant is faster; owner null, deadline null. Scope was not
  expanded to optimizing/deploying it.
- “Someone should review the dashboard” remained a proposal, with no invented task/owner.
- Discount 15%, not 50%, and no database migration remained explicitly supported.

The first complete run had one compound summary, ten minutes, three decisions and three
actions, with 29 resolved evidence references. An intermediate saved-evidence run omitted
two benchmark next steps from its summary while retaining them in minutes/actions. The final
unmodified provider result contains **six summary points, seven minutes, three decisions,
three actions, 28 resolved references**. Both benchmark next steps are now in the summary.
The final MongoDB minute classifies the proposal/rejection sequence as discussion; it
does not claim that migration was approved. No intermediate/result artifact was patched.

## J–K. Playback and performance

Public CLI evidence retrieval and optional extraction succeeded for act_0001. Evidence text
matched source bytes/typed fields, speaker and interval. ffprobe measurements appear above.

First full VI: 1 chunk, 2 HTTP attempts (429 then 200), no consolidation/repair, 4.956772 s
provider latency, 16.989458 s total including bounded backoff. The successful response reported
2181 input/1224 output/3405 total tokens; complete usage is null because the 429 supplied none.

Saved VI run: 1 chunk, 1 call, HTTP 200, 2437 input/1211 output/3648 tokens, 4.576255 s
provider latency, 4.613049 s total, no retries/repairs, seven upstream hashes preserved.

Final saved CLI run: 1 chunk/1 call, HTTP 200, 2510 input/1106 output/**3616 tokens**,
**3.520583 s** provider latency, **3.557176 s** total, no retries/repairs. The final persisted
manifest exactly matches the resolver after JSON tuple/array normalization. All seven
upstream hashes remained unchanged after this last call.

Live long-input test: 3 overlapping whole-utterance chunks plus 1 selection-only consolidation,
all HTTP 200, 49.997191 s total including deliberate ten-second pacing before each request.
Provider latency 9.983634 s, 8450 total tokens. Initial Redis use was removed from final
decisions; “Do not use Redis” and “Use PostgreSQL instead” remained with source IDs, and the
self-assigned PostgreSQL benchmark retained a null deadline. No new evidence/owner/deadline
was generated by consolidation.

No dollar-cost estimate or fabricated unavailable token usage is reported.

## L. Automated verification

79 new Phase VI tests: frozen schema/config, IDs/provenance, named/speaker/null owners,
deadline lexical checks, strict parsing/injection boundaries, backend metadata/lifecycle,
HTTP/auth/quota/timeout/response caps/schema repair, chunks/consolidation restrictions,
serde/CLI/escaping/atomic cleanup, labeled metrics and streaming frame-accurate clips.
Three paid live cases separately passed (proposal, explicit decision, unassigned confirmed
work). The last three-case run passed two cases before a TLS handshake reset interrupted
the third; rerunning only that interrupted test passed in 2.421 seconds. Earlier complete
three-case runs also passed. Actual benchmark calls and live consolidation are additional
manual integration runs; no ordinary unit test silently contacts Groq.

Ruff lint/format checks, compile/import checks and `pip check` passed. No dependency/system
installation was needed beyond refreshing the existing editable package for its new console
entry. No ASR/diarization/grounding weights changed. Final validation logs/audits are ignored.

## M. Artifacts and Git hygiene

Complete run output: `.validation/phase6/integrated/86521f59-3f49-48b6-8133-96d9d35dbbde/`.
Saved-evidence bundle: `.validation/phase6/final-saved/41714b65-da8f-4936-ad78-fee3ef7ea6c7/`.
Final review bundle: `.validation/phase6/accepted/6f905831-b5f3-413a-86eb-a2eb375814b0/`.
Each contains meeting_record.json, meeting_record.md, evidence_manifest.json.
Optional final clip: `.validation/phase6/final-evidence/act_0001-utt_000005.wav`.
Long integration: `.validation/phase6/long-live/b854d05f-4640-4edd-b354-044143a06110/`.
Benchmarks/audits: `.validation/phase6/` (ignored).

Git already existed with zero commits/tracked files; the whole project's normal source/docs
remain untracked. No arbitrary history/commit was created. `git diff` is therefore empty;
baseline hash comparisons establish intended changes. Secret-pattern scans found no keys in
source/tests/docs/scripts/data. `.env`, clips and weights are confirmed ignored; no pending
staging artifact remained. `.gitignore` needed no change.

## N–O. Limits and Phase VII contract

Model omissions/semantic errors remain possible despite structurally valid evidence. This is
a small controlled synthetic benchmark, not a general accuracy guarantee. Upstream acoustic,
speaker and terminology errors propagate. The final benchmark includes one redundant
decision, one missed assignment and an inaccurate injection summary/minute description.
Metadata/evidence checks prove provenance, not
truth. Long-input consolidation chooses existing points and can conservatively omit facts;
over-budget utterances/partials fail rather than truncate. Groq needs internet/account quota.
Clip publication requires hard-link support. No speaker identity or semantic verifier exists.

```python
from meeting_assistant.intelligence import resolve_meeting_record_evidence

resolved_evidence = resolve_meeting_record_evidence(meeting_record, refined_transcript)
for item in meeting_record.content.items:
    claim = item.text
    supports = resolved_evidence[item.id]
    # A future independent verifier receives:
    # meeting_record, supports, refined_transcript, raw_transcript, Phase V edit log.
    # Future labels: SUPPORTED / PARTIALLY_SUPPORTED / UNSUPPORTED.
```

Neither those classifications nor a third LLM role is implemented in Phase VI.
