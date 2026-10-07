# Phase VII implementation and validation report

Validated on 7 October 2026 in the existing Windows project environment. This
report separates real I–VI inference from controlled fake-runner application tests.
No source transcript or meeting artifact was patched to improve the results.

## A. Repository assessment and changes

The starting repository contained isolated `audio`, `asr`, `diarization`,
`grounding`, `refinement` and `intelligence` packages, frozen typed records,
atomic artifact serializers, deterministic evidence APIs and integrated CLIs.
There was no FastAPI server or frontend. Standard unittest and Ruff were already
used. Existing environment loading, FFmpeg executables and cached local models
were reused.

Before implementation, the full suite ran **497 tests: 487 passed, 10 skipped**
in **38.576 seconds**, with FFmpeg and cached GPU/CPU model integration flags enabled.
SHA-256 snapshots of all existing Python source/test files showed **zero changes**
after application implementation. No I–VI prompts, models, retrieval settings,
thresholds or serializers were changed.

Added `src/meeting_assistant/web/` (11 modules), `tests/web/` (5 files), `frontend/`,
two Phase VII documents and a web dependency snapshot. Updated README, `.env.example`,
`.gitignore` and `pyproject.toml` for application setup/dependencies only.
Git exists but still has no tracked files/commits: normal source appears untracked,
so a content-hash audit supplements the empty ordinary Git diff. No commits were made.

## B. Backend environment

| Component | Verified version / behavior |
|---|---|
| OS / Python | Windows x64 / project venv Python 3.12.14 |
| FastAPI | 0.142.2 |
| Starlette | 1.7.0 |
| Pydantic | 2.13.5 |
| Uvicorn | 0.54.0 |
| python-multipart | 0.0.32 |
| HTTPX test client | existing 0.28.1 |
| Persistence | standard-library SQLite, WAL, short transactions, closed connections |
| Execution | single process, one serial pipeline worker; maximum 100 unfinished jobs |

All new Python libraries were installed in `.venv`. No driver, CUDA, Torch or
other global/system library changes were required. Windows HTTP serving uses the
Selector event loop; final range/playback checks produced successful 206 responses
without the earlier Proactor pipe-reset callback warnings. Starlette emits a test
client compatibility deprecation for existing HTTPX; its pinned compatibility
path passed all application tests. No new HTTP client is used in production.

## C. API endpoints and schemas

Implemented upload/history/detail/status, record, raw/speaker/refined transcript,
evidence, canonical audio, download listing and artifact-download endpoints under
`/api/meetings`. [The application guide](phase7-application.md#rest-contract) lists
every route. Upload is multipart field `file`, returning HTTP 202 with the server
UUID. Status returns stage states, durations and completed count out of six.

Responses use typed application metadata and the unchanged Phase II/III/V/VI
dataclasses. OpenAPI contains TranscriptResult, SpeakerAwareTranscript,
RefinedTranscript, MeetingRecord and EvidenceSpan schemas. Files are addressed
through logical allowlisted IDs and relative URLs, never workspace paths.

## D. Orchestration and failure behavior

`PipelineRunner` executes I→II→III→IV→V→VI through existing public APIs and
serializers. Local GPU/embedding prerequisites prepare before the first billed
ASR request, within the transcription stage. Backend instances are retained
after successful initialization and reused by the serial worker.

Every stage reports running/complete and measured duration. A failure at any
phase marks that stage failed, leaves later stages pending, retains earlier
registered artifacts and prevents a completed result. Offline tests exercise
failures at all six phases, specifically including IV stopping V/VI.
The application never automatically resumes or reruns a paid stage; existing
phase transport-retry policies remain unchanged.

## E. Persistence and restart

SQLite retains IDs, display filename, source size, timestamps, states, stage
timings, safe errors and relative artifact references. Large payloads remain in
UUID directories. Completed fixture results were reopened in the browser after
actual server restarts without transcription/model reruns.

Unit tests verify persisted completion, artifact reopening and stale queued/running
jobs becoming FAILED with `job_interrupted`. An OS-held file lock excludes a second
server and can reopen after release, including with an existing lock file. Hard
process termination releases the OS lock; startup does not resume paid work.

## F. Frontend stack and structure

React/React DOM **19.3.0**, Vite **8.3.3**, TypeScript **5.9.3**, ESLint **10.12.0**,
typescript-eslint **8.71.1**, Prettier **3.9.9**, Vitest **4.1.11**.
Node **24.20.0** and npm **11.19.0** were already available. Exact resolved versions
are retained in `frontend/package-lock.json`; npm installation reported zero
vulnerabilities at setup.

`main.tsx` contains upload/history/processing/workspace React components and local
history-based routing; `api.ts` is the single typed HTTP layer; `types.ts` describes
the consumed schema fields; `playback.ts` owns bounded playback rules/timestamps;
`style.css` provides responsive styling. No Redux, router framework or UI kit was needed.
FastAPI serves the built SPA; Vite supports proxy-based frontend development.

## G. Views

Verified upload selection, filename/size display, processing navigation/polling,
history and all five meeting views: Overview, Transcript, Minutes, Decisions and
Action Items. The real fixture displayed topic groups and distinct proposal/decision
labels. Rahul/Friday, anonymous SPEAKER_01/no deadline and unassigned/no deadline
actions rendered correctly; null fields showed Unspecified.

An isolated fake-runner server under `.validation/phase7/ui-states` verified empty
summary/minutes/decisions/actions. This was a controlled UI test, **not actual ML
inference or an evaluation dataset**. It used no providers or local models and was
stopped afterward. Loading states and the real corrupt-input failure were also
observed. Backend-unavailable messaging was checked by stopping that isolated server.

## H. Evidence and transcript behavior

Each item opens deterministic source spans using `get_item_evidence`. The tested
`act_0001` resolved to `utt_000005`, anonymous `SPEAKER_00`, with exact raw/refined
text and word/segment references. Raw/refined toggles, edit indicators, original
ASR access and jump-to-transcript focus were checked in the browser.
No new LLM call or on-demand clip was needed for evidence display/playback.

## I. Audio playback and range serving

The fresh real job's benchmark evidence was **22.480–28.560 seconds**:

> Rahul, benchmark both models by Friday. Measure latency and memory usage.

The browser audio element reported a seek to **22.480**, duration **63.445**, and
playing state. Bounded playback later paused at **28.571** (about 11 ms past the
evidence end); another final-build check paused at **28.570**. This is ordinary
browser playback, not a frame-exact alignment claim.

A manual PageUp seek while evidence played moved to **28.820** with `paused=false`,
beyond the old end: manual seeking correctly cancelled the evidence boundary.
Tests assert complete audio bytes, exact 206 ranges/Content-Range and 416 for ranges
beyond EOF. Actual browser serving used 206 responses during playback/seeking.

Independent ffprobe inspection confirmed **WAV / pcm_s16le / 16,000 Hz / mono /
16 bits / 63.445 s / 2,030,318 bytes**. Canonical SHA-256:
`f3895def20ac89bba845889a92fe62bf9edb29d55d4a1e6f6d3f0d7787066e1f`.

## J. Downloads

All **14** registered artifact downloads were compared with their published
files and matched byte for byte: canonical WAV; raw/speaker/refined JSON and TXT;
meeting JSON and Markdown; evidence manifest; diarization JSON; grounding JSON/TXT;
edit log JSON. Correct MIME types and fixed safe attachment filenames are covered
by tests. Advanced files are grouped away from the main reading views.

The meeting JSON was also downloaded through the actual browser to
the browser's local Downloads folder as `meeting_record.json`; its bytes matched the canonical file.
SHA-256: `6fcd1065e8f00e6308fff736e69dc183eec49748d7cdfde623fb833a9fc1bc26`.

## K. Errors and robustness

Covered missing file/ID/item/artifact, empty upload, oversized file/request,
chunked multipart size rejection, extra fields/multiple files, untrusted filenames,
cross-origin upload rejection, unsafe paths/reparse points, malformed/missing stored
artifacts, stage exceptions and incomplete pipelines. An injected UUID collision
was rejected without modifying or deleting the existing upload/result.

The deliberately corrupt browser recording failed in INGESTING with
`unsupported_media` (the server filename is deliberately extension-neutral), a
safe explanatory message, no registered artifacts and all later stages pending.
No misleading completed record/audio was exposed. No owned `.uploading`,
`.canonical-*` or `.pending-*` artifacts remained after handled failure.
Frontend server errors do not expose stack traces, credentials or workspace paths.

## L. Actual browser end-to-end run

Original legal/shareable two-voice Windows SAPI fixture:
`.validation/phase6/fixture/two_speakers.wav`, generated by the existing fixture
scripts. It was selected through the browser chooser and uploaded through the
normal application, **without editing any intermediate artifacts**.

Meeting ID: `0ae6f689-0dc5-4c32-92a1-881addecdbe9`.
Request persisted at 14:55:44.592 UTC; completed at 14:56:39.147 UTC:
**54.555 seconds** upload acceptance to completion. Stage sum: **54.457 seconds**.

| Stage | Measured seconds |
|---|---:|
| Audio preparation | 2.641 |
| Transcription including first model preparation | 21.093 |
| Speaker detection/reconciliation/publication | 5.796 |
| Terminology grounding/publication | 0.806 |
| Conservative refinement/publication | 6.460 |
| Meeting intelligence/publication | 17.661 |

Raw ASR: Groq `whisper-large-v3`, English hint, temperature zero, native word
timestamps; **10 segments, 122 words, 63.445 s**. ASR-only transcription time
**7.206 s**, RTF **0.114**. Diarization's retained peak allocated GPU memory was
**1,707,943,936 bytes (~1.59 GiB)**, with two anonymous speakers. The run produced
**4 summary points, 10 minutes, 4 decisions and 3 actions**.

The existing intelligence transport policy handled one HTTP 429 then a successful
retry. Successful call usage: 2,510 prompt + 1,232 completion = **3,742 tokens**;
the failed attempt's token usage was unavailable and was not fabricated. This
accounts for additional extraction-stage latency. Models/settings were unchanged.

Observed quality issues were retained: “kube.net ease” for the intended technical
term, “Drant”, and a Phase V replacement from “discount at” to “Discount rate”
that yields awkward wording. These are baseline ML outputs; the UI exposes the
original evidence rather than fixing them. Synthetic speech is not a real-meeting
accuracy benchmark and there is no independent semantic-verification claim.

Responsive checks at **1366×900**, **768×1024**, and **390×844** found no document
horizontal overflow. Narrow evidence panels stacked and focused/scrolled into view.
Saved screenshots live under ignored `.validation/phase7/`, including
`final-workspace.jpg`, `actions-tablet.jpg`, `final-mobile-evidence.jpg`,
`corrupt-input.jpg`, `empty-actions-controlled.jpg` and
`backend-unavailable-controlled.jpg`.

Example real artifacts:

```text
jobs/web/0ae6f689-0dc5-4c32-92a1-881addecdbe9/audio/canonical.wav
jobs/web/0ae6f689-0dc5-4c32-92a1-881addecdbe9/transcripts/8a7c1f48-e29b-498e-a24a-d2c928c4caa5/raw_transcript.json
jobs/web/0ae6f689-0dc5-4c32-92a1-881addecdbe9/transcripts/5bbefa81-19fe-4c14-9eb9-4acd0d7de9bc/refined_transcript.json
jobs/web/0ae6f689-0dc5-4c32-92a1-881addecdbe9/transcripts/c011d80a-b36c-465d-88a2-897fce3a0c5a/meeting_record.json
jobs/web/0ae6f689-0dc5-4c32-92a1-881addecdbe9/transcripts/c011d80a-b36c-465d-88a2-897fce3a0c5a/meeting_record.md
jobs/web/0ae6f689-0dc5-4c32-92a1-881addecdbe9/transcripts/c011d80a-b36c-465d-88a2-897fce3a0c5a/evidence_manifest.json
```

## M. Checks

The final full suite ran in **56.391 seconds**: **542 tests: 531 passed, 11 skipped**, including
**45 new web tests (44 passed, one Windows symlink-privilege skip)** and the
unchanged cached-model/FFmpeg integration tests. Real Windows-reparse protection
is additionally covered without needing symlink creation privileges.
The other skips are the existing explicit live-provider opt-ins and original
Windows symlink case. No ordinary web tests called providers or loaded models.

Ruff lint and format checks, compileall/import/OpenAPI checks and `pip check`
passed. Frontend ESLint, TypeScript, production build and **4/4 Vitest playback
tests** passed. Real browser validation is manual, not counted as an automated
Playwright test suite. Hash comparison found no pre-existing Python file changes;
Git status/diff/ignore checks showed no tracked weights/caches/runtime files.

## N. Security and configuration

Server UUIDs, exclusive workspace claiming, fixed source names, streaming limits,
single-file multipart parsing, a bounded persisted queue, allowlisted artifact IDs,
safe attachment names and link/reparse/root checks protect the application boundary.
CORS uses explicit origins without credentials/wildcards; browser POST origins are
also rejected early when unapproved. `.env`, models/caches, jobs, SQLite companions,
node_modules and frontend build outputs are excluded from Git. No secrets were
copied into frontend source or responses. The application is bound to loopback by
default and adds no authentication/cloud deployment.

## O. Genuine limitations

- Single local worker; no distributed job queue, automatic paid resume, cancellation
  or multi-user authentication. Graceful shutdown can wait for a long current job.
- Uploaded audio and text still reach the configured API providers. Quota/network
  availability affects completion; cached diarization/grounding does not make the
  full pipeline offline.
- ML transcription, refinement, diarization and extraction errors remain possible;
  evidence links are deterministic provenance, not semantic proof.
- Native/browser timestamps are approximate. Existing long-meeting limits still
  apply; history/transcripts are not paginated/virtualized.
- Retention/deletion is manual with the server stopped. Hard termination can leave
  staging files; restart recovery marks jobs interrupted instead of deleting them.
- Validation uses original synthetic speech plus controlled UI fixtures; no claim
  about real human-meeting accuracy or production/cloud scale is made.

## P. Deferred stages

No semantic-verifier LLM, speaker-name inference, local-ASR replacement,
collaborative/HITL editing, PDF export, external-document RAG, fine-tuning, agents,
or real-meeting evaluation dataset was added. They can consume the immutable
canonical records through existing contracts without moving ML logic into React.

```python
from meeting_assistant.intelligence import meeting_record_from_json, get_item_evidence
from meeting_assistant.refinement.serialization import refined_from_json

record = meeting_record_from_json(record_path.read_text(encoding="utf-8"))
refined = refined_from_json(refined_path.read_text(encoding="utf-8"))
for item in record.content.items:
    spans = get_item_evidence(record, item.id, refined)
    print(item.id, [(span.speaker_id, span.start, span.end) for span in spans])
```

The application API exposes the same immutable records and resolves source spans
without downstream knowledge of FFmpeg, CTranslate2, model caches or provider transport.
