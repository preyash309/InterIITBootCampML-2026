# Phase VII — Interactive meeting application

Phase VII is application engineering around the existing I–VI public contracts.
Earlier plans used “Phase VII” for a semantic verifier; this implementation follows
the latest request, which assigns Phase VII to the interactive application.
Independent semantic verification remains future work.

## Setup and run

Use the existing Python environment with working Phase III/IV caches and CUDA
Torch wheels. Do not replace that environment's ML dependencies to install the UI.

```powershell
python -m pip install -e ".[web,web-test]"
cd frontend
npm ci
npm run build
cd ..
python -m meeting_assistant.web
```

Visit `http://127.0.0.1:8000`. FastAPI serves both REST and built frontend assets,
including SPA fallback for `/meetings` and `/meetings/<uuid>`.
If `frontend/dist` is absent, only the API is served; build the frontend first.
API schemas are available at `/openapi.json` and interactive API documentation at
`/docs` (Swagger's documentation assets may require connectivity).

Node 24.20.0/npm 11.19.0 and Python 3.12.14 were validated. Vite's supported Node
versions are documented in [its official setup guide](https://vite.dev/guide/).
`frontend/package-lock.json` fixes exact resolved JavaScript dependencies.
`requirements/web-windows-py312.lock.txt` records the new Python application stack;
it supplements, rather than replaces, the existing ML environment snapshots.

Development: start the same backend, then run `npm run dev` under `frontend/`.
Vite serves port 5173 and proxies `/api` to `http://127.0.0.1:8000`. If changing
the API port, also change that proxy target. No browser API keys are needed.

Existing phase settings are loaded once with `asr.config.read_environment(.env)`;
process variables win. Reuse the project's ignored `.env`, `GROQ_API_KEY`, FFmpeg
paths and model caches. Credentials never enter responses or frontend bundles.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `WEB_HOST` | `127.0.0.1` | Local server bind address |
| `WEB_PORT` | `8000` | API/demo port |
| `WEB_ALLOWED_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Comma-separated explicit CORS origins |
| `WEB_MAX_UPLOAD_BYTES` | `4294967296` | 4 GiB per recording, matching Phase I's default |
| `WEB_JOB_ROOT` | `jobs/web` | SQLite, uploads and stage artifacts |
| `WEB_MAX_ACTIVE_JOBS` | `1` | Only 1 is validated/accepted; additional uploads queue |
| `WEB_FRONTEND_DIST` | `frontend/dist` | Optional built frontend directory |

Up to 100 unfinished jobs are admitted transactionally. Browser POST origins are
checked before upload processing, including the local demo origins; scripted
clients can omit Origin. CORS permits GET/POST and Content-Type/Range with no
credential wildcard. This is not authentication: keep this local application off
public networks. There is no automatic retention, deletion, cancellation or resume.
For a custom job directory, add it to `.gitignore` too.

## Execution, persistence and artifacts

```text
React → multipart upload → UUID job + SQLite → single worker
    ingest_audio → transcribe_audio → diarize + reconcile_transcript
    → ground_transcript → refine_transcript → extract_meeting_record
    → canonical files registered → completed workspace
```

`PipelineRunner` calls public phase APIs and their original atomic serializers.
The first transcription stage includes cached diarization/grounding prerequisite
preparation before a billed ASR upload, preserving the integrated CLI's policy.
The worker reuses backend/model instances across jobs and executes one job at a
time. Pipeline exceptions stop subsequent phases. Prior published artifacts remain
registered and downloadable, but failed jobs cannot expose a completed record/audio.
Existing bounded provider transport retries are unchanged; the application never
automatically restarts a paid stage or reruns an interrupted job.

`JobStore` uses standard-library SQLite with short transactions, independent
connections that always close, and WAL mode. Only job metadata, stage timings,
sanitized errors and relative artifact references are stored there. Transcripts
and recordings remain on disk. A crash-released OS file lock prevents duplicate
servers for the same job root. FastAPI's
[lifespan](https://fastapi.tiangolo.com/advanced/events/) owns the worker lifecycle.
On restart, completed jobs persist; queued or active jobs become FAILED with
`job_interrupted`. No paid rerun occurs. Graceful shutdown waits for the current
job to finish; remaining queued jobs are interrupted on restart.
Windows uses a Selector event loop for the HTTP server to avoid Proactor
connection-reset callbacks during audio range seeking; model work stays in a thread.

```text
jobs/web/
    meetings.sqlite3                # job metadata / artifacts tables
    .server.lock                   # small OS-held lock, not a completion marker
    <server-generated-meeting-uuid>/
        upload/source.media        # original bytes, fixed server filename
        audio/canonical.wav        # Phase I: 16 kHz / mono / PCM s16le / WAV
        transcripts/
            <raw-uuid>/raw_transcript.json, raw_transcript.txt
            <speaker-uuid>/diarization.json, speaker_transcript.json, speaker_transcript.txt
            <grounding-uuid>/grounding.json, grounding.txt
            <refined-uuid>/refined_transcript.json, refined_transcript.txt, edit_log.json
            <record-uuid>/meeting_record.json, meeting_record.md, evidence_manifest.json
```

Original upload names are bounded display metadata only. Streaming upload copying
uses 64 KiB blocks, with a total request-body guard before multipart disk spooling
and a separate exact file-byte limit. Multipart parsing permits one file and no
extra text fields, limiting unused-part memory and disk consumption. Completed source copies are atomically
renamed. Failed uploads clean their owned partial files and directories.
Runtime jobs, weights, caches, databases, generated transcripts, node_modules and
build outputs are ignored. Do not remove jobs while the server is using them.

## REST contract

| Method / route | Result |
|---|---|
| `POST /api/meetings` (`file` multipart field) | HTTP 202 typed queued job with server ID and URLs |
| `GET /api/meetings` | Saved history, newest first |
| `GET /api/meetings/{id}` | Typed job metadata |
| `GET /api/meetings/{id}/status` | Stage statuses, durations, completed count / 6 |
| `GET /api/meetings/{id}/record` | Canonical typed Phase VI MeetingRecord |
| `GET /api/meetings/{id}/transcript/raw` | Original Phase II TranscriptResult |
| `GET /api/meetings/{id}/transcript/speaker` | Original Phase III SpeakerAwareTranscript |
| `GET /api/meetings/{id}/transcript/refined` | Original Phase V RefinedTranscript |
| `GET /api/meetings/{id}/evidence/{item_id}` | Typed EvidenceSpan array, deterministically resolved |
| `GET /api/meetings/{id}/audio` | Completed job's canonical WAV, including byte ranges |
| `GET /api/meetings/{id}/downloads` | Logical IDs, safe filenames/content types and relative URLs |
| `GET /api/meetings/{id}/downloads/{artifact}` | Exact original registered artifact bytes |

States: QUEUED, INGESTING, TRANSCRIBING, DIARIZING, GROUNDING, REFINING,
EXTRACTING_INTELLIGENCE, COMPLETED, FAILED. Stage states are pending, running,
complete or failed. The UI polls every 1.5 seconds and stops at a terminal state.
The count reflects completed stages, not an invented inference percentage.

Errors use `{ "error": { "code": "...", "message": "..." } }`.
Typical HTTP codes: 202 queued, 400 empty/invalid upload, 413 size limit, 403 rejected
upload origin, 404 unknown UUID/item/artifact, 409 result not ready, 500 unavailable
or malformed stored resource. Pipeline failures live in the job's error field,
with current stage and previously completed stages retained.
The frontend distinguishes server connection failures, pipeline failures,
unavailable results and playback failures. “Upload a new recording” starts over
explicitly; it is not an automatic resume.

The registry accepts only the fixed artifact IDs in `web.registry.ARTIFACTS`.
No endpoint accepts a filesystem path. UUID validation, traversal/drive checks,
root containment, and symlink/Windows-reparse checks protect reads and publication.
API payloads contain no absolute workspace paths. Downloads use fixed safe names
and original content types, independent of the untrusted source filename.
Starlette FileResponse handles audio ranges; tests verify 206, exact Content-Range,
exact partial bytes and 416 for out-of-bounds ranges.

## Workspace and evidence

React has home upload, history and meeting routes. Views are Overview, Transcript,
Minutes, Decisions and Action Items. Minutes are grouped by topic with explicit
kind labels; named entities/anonymous owner IDs are displayed as retained, and
null owners/deadlines show Unspecified. Empty sections have explicit empty states.

The refined transcript is the default. Raw ASR remains available beside each edit,
in the speaker-row toggle, in exact original segments/word timestamps, and as a
download. Text search is local substring search only. All provider text is escaped
by React; no HTML interpretation or inferred identity is introduced.

Every intelligence item can open an evidence panel. The backend calls
`get_item_evidence(record, item_id, refined)` and validates source provenance.
The UI receives exact speaker, times, raw/refined text and source references.
One persistent audio element seeks to the requested start, checks playback time
every 40 ms and pauses at/just beyond end. Manual seeking cancels the bounded
interval. “Jump to transcript” selects and focuses the cited utterance.
No clip extraction is needed for ordinary playback, and evidence requests do not
call any model. Responsive narrow layouts stack the evidence panel and focus/scroll
it into view. Buttons, upload input, search, player and navigation have labels and
keyboard focus styles. Supporting text uses readable contrast and reduced motion
is respected for the loading indicator.

## Checks and limits

```powershell
python -m unittest discover -s tests -t .
python -m unittest discover -s tests/web -t .
ruff check src tests scripts
ruff format --check src tests scripts
python -m compileall -q src tests
cd frontend
npm run lint
npm run typecheck
npm test
npm run build
```

Web tests use controlled original text fixtures and fake orchestration, with real
artifact serializers; they never call Groq or load local models. Existing cached
model tests remain opt-in as documented in the earlier phase guides. Real browser
validation results are in [phase7-validation.md](phase7-validation.md).

This is a local, single-worker competition app. There is no authentication,
distributed execution, cancellation, artifact retention/delete UI, automatic crash
resume, paginated history or transcript virtualization. Long jobs still obey the
existing pipeline's duration/context limits and need disk, memory and API quota.
A hard termination can leave staging files; restart recovery changes job status
without deleting or automatically resuming those files.
Whisper word/speaker timestamps are approximate; browser playback is not a forced
alignment or frame-exact clip export. Synthesis validation does not establish real
human-meeting accuracy. Raw/refined/claim records can contain ML errors: deterministic
evidence links do not prove semantic correctness. Authentication/cloud, semantic
verification, identity inference, local-ASR migration, PDF and editing are deferred.
