# Phase V validation — 2026-10-07

## A. Regression status and preservation

Before edits: **300 tests; 296 passed, 4 skipped**, including 13 required real
FFmpeg integrations, four GPU and one CPU Community-1 tests, and three offline
MiniLM tests. The earlier ordinary run passed 288 with 12 skips.

After implementation: **418 tests; 411 passed, 7 skipped**, 25.940 s. Separate
explicit live Groq refiner tests passed **3/3**, 4.334 s. Seven full-suite skips
were three opt-in ASR API tests, three opt-in refiner API tests, and the existing
Windows directory-symlink privilege test. The cached model tests all passed.

Baseline SHA-256 snapshot: 89 existing source/test/docs/config files. Only four
legacy files changed: root CLI dispatcher, README, pyproject entry point, and
`.env.example`. No legacy file disappeared. No Phase I–IV module or existing test
changed. The previously saved Phase IV artifacts remain untouched.

An actual saved-evidence refinement additionally hashed raw/speaker/grounding
files before and after: **all bytes identical**. The refined result passed
`validate_refined_source`; raw serialization digest also resolved to Phase III.
Generated word references, original offsets, speaker IDs and times remained exact.

Ruff lint and format passed for all source/tests/scripts (98 Python files).
Compilation/import checks and `pip check` passed. `git diff --check` was clean.
No new dependency, CUDA/cuDNN/driver component or global system change.

## B. Backend and configuration

Groq account authentication and `/openai/v1/models` returned HTTP 200 and confirmed
`openai/gpt-oss-120b` active. Actual chat completions succeeded using:

- Endpoint: `https://api.groq.com/openai/v1/chat/completions`
- Temperature: 0; strict `json_schema`; max completion tokens: 4096
- Reasoning effort: low; include reasoning: false; no streaming/tools
- Timeout: 90 s per attempt; one 429/5xx retry with capped Retry-After/backoff
- Schema repair: at most one; no semantic retries or silent model/provider swap

The initial `anyOf` nullable enum form encountered Groq HTTP 400
`json_validate_failed` even though the generated decision matched the supplied
schema. The final schema uses Groq's documented `["string", "null"]` type plus
enum including null. KEEP, REPLACE and UNCERTAIN subsequently passed real requests.
Provider-side schema rejection now enters bounded schema-only repair; mocked
tests cover repair success/failure, and rejected output is never applied directly.
Successful live final runs needed **zero schema repairs**.

## C. Prompt

System policy: `refiner_v1`; examples: `refiner_examples_v1`; schema:
`refiner_decisions_v1`; safety/application policy: `conservative_v1`.
Priorities are preserve meaning and speech, correct only supported terminology
errors, and retain uncertain text. All target, neighboring and glossary content
is untrusted JSON data. No paraphrasing, grammar/punctuation polishing, inferred
identity, new facts, acronym expansion or meeting intelligence.

## D. Architecture

New isolated `src/meeting_assistant/refinement/`: config, exceptions, models,
backend protocol, prompt/schema, Groq transport, deterministic validation/service,
serialization, CLI and evaluation. Public API:

```python
refine_transcript(speaker_transcript, grounding_result, backend=backend)
save_refined_transcript(refined, output_dir)
validate_refined_source(refined, speaker_transcript, grounding_result)
```

Frozen result/utterance/edit models retain raw text and original evidence refs;
corrected terms receive no fabricated word timestamps. The protocol permits a
future local refiner without changing downstream schemas; none is implemented.

## E. Actual structured decision

Final live meeting request returned the following applied decision (shown with
only this record from the target's four-record response):

```json
{
  "utterance_id": "utt_000002",
  "decisions": [
    {
      "grounding_record_id": "grnd_000004",
      "action": "REPLACE",
      "candidate_entry_id": "global.94abcb2d2773df65",
      "reason_code": "exact_registered_alias"
    }
  ]
}
```

Application resolved that ID to **Kubernetes**. Evidence interval: **6.92–8.26 s**;
source references: `seg_000002`, word indices **4 and 5**. Original offsets were
used; no new replacement-word timing was invented.

## F. Safety results

Actual final meeting and all 20 controlled benchmark outputs were audited:
**zero number, date/time, negation, modality, protected-content, speaker or
timestamp changes**. Both live injection cases kept the correct GPU text.
Unit tests additionally force malicious replacements of numbers, decimals,
dates, times, negation/contractions, quantifiers, units, numeric model versions,
commitments, names, speaker/time evidence and protected-token ordering. All vetoed.
Unsupported global name corrections fail; a controlled explicit meeting-scope
registered ASR alias can correct a company name without changing anonymous IDs.

## G. Controlled benchmark

`data/refinement_benchmark.json`: **20 original synthetic text/candidate cases**,
not fabricated speech measurements. Four labeled positives, 15 KEEP negatives,
one missing-candidate UNCERTAIN example. Grounding scores are controlled hypotheses.

| Metric | Observed |
|---|---:|
| Correct applied edits | 3 |
| Edit precision | 1.000 |
| Edit recall | 0.750 |
| Edit F1 | 0.857143 |
| False correction rate | 0 / 16 = 0 |
| Correct KEEP decisions | 13 / 15 = 0.866667 |
| Correct REPLACE decisions | 3 / 4 = 0.750 |
| Correct UNCERTAIN decisions | 1 / 1 = 1.000 |
| Meaning-changing applied edits | 0 |
| Protected-content/speaker/time violations | 0 |

Actual decisions: **14 KEEP, 4 REPLACE, 2 UNCERTAIN**. Of four REPLACE decisions,
three were applied and one vetoed. `CubeNet Ease`, `cue drant` and `pie torch`
were corrected. **`newton rafson` was missed (KEEP).** `might` received UNCERTAIN
instead of the labeled KEEP, but remained unchanged. The model proposed
`plug flow reactor → PFR`; **the acronym-expansion guard rejected it**.
The benchmark was not rerun or relabeled to hide these observations. No optional
transcript-only comparison or Cartesian sweep was run.

## H. Real I–V example

Reused our legal/shareable, original local Microsoft SAPI David/Zira fixture
from Phase IV: six alternating turns, chemical/math/business/ML terminology,
15%, Friday/not Monday and an explicit negation. Fixture source/provenance is
documented by `scripts/create_grounding_fixture.ps1` and Phase IV validation.
The full command ran fresh ingestion, Groq Whisper ASR, local Community-1,
offline MiniLM grounding and Groq refinement, without manual intermediate edits.

Canonical audio independently verified with ffprobe: **pcm_s16le, 16,000 Hz,
mono, 16 bits, 38.415063 s**, 1,229,360 bytes. Raw output: **9 segments, 77 words**.
Speaker evidence: two anonymous speakers, six utterances. Grounding: 14 records.
Five utterances had candidate evidence; the sixth was copied without a request.

Before:

> Deploy the containers with CubeNet Ease. The NVIDIA GPU uses CUDA kernels.

After:

> Deploy the containers with Kubernetes. The NVIDIA GPU uses CUDA kernels.

Both full and saved-evidence runs produced **13 KEEP / 1 REPLACE / 0 UNCERTAIN**,
one applied edit, zero rejected edits. Correct QDRANT, PyTorch, CUDA, GPU,
Newton-Raphson/Reynolds number remained unchanged. Spoken plug flow reactor,
annual recurring revenue and customer acquisition cost were not shortened.
Friday/not Monday, 15%, anonymous speakers and every utterance time were retained.
Raw missing punctuation was deliberately not repaired. This establishes a small
baseline, not perfect transcription accuracy or general semantic guarantees.

Actual UNCERTAIN demonstration: synthetic “Deploy flarnex for the internal
index” with only unrelated Kubernetes evidence → UNCERTAIN, unchanged text.

## I. API performance and failures

Final saved-evidence meeting run:

- **5 calls**, all HTTP 200; **6.306606 s** total refinement
- **6.295856 s** summed request latency; **1.259171 s** mean per sent utterance
- **8,495 input + 1,069 output = 9,564 tokens** reported by Groq
- Zero transport failures or schema repairs in this run

Controlled benchmark: **20 calls**, all HTTP 200, explicitly paced 10 s between
cases. Summed API latency **21.678940 s**, mean **1.083947 s**;
**22,651 input + 2,925 output = 25,576 tokens**. Pacing is not included in API
latency. No estimated dollar cost was fabricated.

The first full run hit Groq's rate limit and published **no completed refinement**.
After bounded Retry-After handling, the fresh full run succeeded: six HTTP attempts
(five 200, one 429), **10.944 s** total refinement. Aggregate tokens stayed **null**
because Groq supplied no usage for the 429. Five successful attempts separately
reported 9,570 tokens. Preliminary schema diagnostics and initial failures are
not included in final-run or benchmark usage totals; these are not account-wide
spend totals. Three separate initial smoke requests and three final live tests
also succeeded. No successful live run used semantic/schema repair retries.

No `.pending-*`, canonical temp files or ingestion locks remained in the audited
Phase V validation/job directories. Mocked write/rename failures also cleaned up.

## J. Tests

**118 added tests:** 115 offline tests plus three opt-in real Groq tests.
Coverage includes strict decisions, provenance before billing, lifecycle/concurrency,
payload injection boundaries, all actions, malformed/duplicate/missing fields,
cross-record candidates, bounded schema repair/provider errors, TLS/deadline/size,
rate limits/Retry-After, protected meaning, offset order and length changes,
names/acronyms/formatting, empty/no-grounding inputs, frozen schemas, serialization,
atomic failure cleanup, saved CLI and root dispatch. No unit test downloads
weights, calls an API or loads an ML model.

## K. Artifacts and Git

Successful fresh I–V run, relative to the repository root:

```text
jobs/382c7192-97ef-4aed-814a-71519a1ae660/audio/canonical.wav
.validation/phase5/integrated-success/
  72f399cd-8727-4b53-813a-c497170895a3/raw_transcript.json
  cac78604-1998-4185-ae40-7bc3ff72f8b3/speaker_transcript.json
  8407c251-2925-4ad2-9a8a-7379e625020b/grounding.json
  0046eaee-12c1-4d4a-8724-a7bdb1dab71e/refined_transcript.json
```

Final saved-evidence run, recording all final generation options:

```text
.validation/phase5/final-refinement/4ca295b9-0668-49f2-b3e0-daf9ff469123/
  refined_transcript.json
  refined_transcript.txt
  edit_log.json
.validation/phase5/benchmark.json
.validation/phase5/benchmark-artifacts/<UUID>/...
.validation/phase5/integrated-audit.json
.validation/phase5/category-audit.json
.validation/phase5/final-full-tests.txt
.validation/phase5/live-tests.txt
```

All local outputs/provider-derived text, caches, weights and credentials are
ignored. Git is initialized with **no commits**, so project files remain untracked
and ordinary `git diff` cannot show edits to existing untracked files. The baseline
hash audit proves preservation instead. No commit was created. Secret-prefix
scanning found no real keys/tokens in source/tests/docs/data/scripts or examples.

New files: refinement package, mirrored tests, benchmark JSON and two Phase V docs.
Modified legacy files: root CLI, pyproject, `.env.example`, README. Existing ignore
rules already cover generated artifacts; no unrelated ignore change was needed.

## L. Known limitations

The small benchmark is not general meeting quality. Hosted decisions can vary.
Conservative policy can miss real corrections (observed Newton-Raphson miss), and
the model can propose unsafe equivalence edits (observed PFR proposal); validation
must remain mandatory. Missing candidates and long single utterances need explicit
future handling. Groq quota/availability and upstream ASR/speaker timestamp limits
remain. Safety token rules do not prove full semantic equivalence or authenticity.
No local refiner, replay cache, glossary expansion, summaries or frontend was added.

## M. Phase VI contract

Retain the immutable raw transcript, speaker transcript and grounding independently
from the derived refined transcript. Before consumption of saved artifacts:

```python
from meeting_assistant.refinement import refine_transcript, validate_refined_source

refined_transcript = refine_transcript(
    speaker_transcript, grounding_result, backend=backend,
)
validate_refined_source(refined_transcript, speaker_transcript, grounding_result)

for utterance in refined_transcript.utterances:
    print(utterance.utterance_id, utterance.speaker_id,
          utterance.start, utterance.end, utterance.refined_text)
for edit in refined_transcript.edit_log:
    print(edit.grounding_record_id, edit.source_word_refs, edit.validation_status)

# Future Phase VI receives all four, without mutating any evidence:
# raw_transcript, speaker_transcript, grounding_result, refined_transcript
```

Meeting summary, structured minutes, decisions and action items belong to a
**separate future language-model role**, not the refiner, and are not implemented.
