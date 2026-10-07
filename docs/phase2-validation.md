# Phase II API baseline — actual validation

Validated on 2026-10-07. This supersedes the local-inference implementation plan
for current testing, following the user's request to use Groq/OpenAI Whisper
until the workflow can be replaced with local models.

## Repository assessment and changes

Phase I was inspected and its existing suite rerun before ASR edits: 62 tests,
61 passed and one Windows directory-symlink privilege skip. All 13 real FFmpeg
integration tests passed. The repository uses a Python `src` layout, frozen
dataclasses, stdlib logging/configuration, `unittest`, and Ruff. SHA-256 comparison
against the pre-edit snapshot confirms every Phase I audio module and audio test
is unchanged.

The new `src/meeting_assistant/asr/` package contains:

| Module | Responsibility |
| --- | --- |
| `base.py` | Minimal backend protocol for API/local interchangeability |
| `config.py` | Frozen provider/options config, validated environment and optional `.env` |
| `models.py` | Immutable transcript/segment/word, model, processing, artifact types |
| `exceptions.py` | Sanitized ASR-specific errors with optional HTTP status |
| `audio.py` | Canonical WAV/frame validation and bounded lossless PCM chunking |
| `transport.py` | Streaming HTTPS multipart requests, deadlines, status/JSON handling |
| `api_backend.py` | Groq/OpenAI mapping, offsets, native metadata, result construction |
| `service.py` | Public `transcribe_audio()` with backend injection |
| `serialization.py` | Versioned UTF-8 JSON, raw timestamped TXT, atomic pair publication |
| `evaluate.py` | Reference-driven WER and runtime-only evaluation |
| `__init__.py`, `__main__.py` | Public exports and canonical/integrated developer CLI |

The root `__main__.py` integrates ingestion and ASR. Packaging adds
`meeting-assistant` and `meeting-transcribe` console scripts. README and
`.env.example` describe setup, configuration, outputs, evaluation, and limits.
`.gitignore` excludes credentials, model weights, caches, jobs, transcripts, and
local validation artifacts. **No Python runtime dependency was added**: HTTPS,
streaming, schema models, serialization, and WER use the standard library.

## Default inference settings

- Provider: **Groq**, official `api.groq.com/openai/v1/audio/transcriptions`.
- Model: **`whisper-large-v3`**.
- Device: remote; compute type, VRAM, and server VAD are not reported.
- Language hint: **`en`**. No translation endpoint is used.
- Temperature: **0**; no initial prompt, hotwords, or glossary.
- Response: **`verbose_json`**, both **segment and word** timestamps.
- Maximum chunk duration: **600 seconds**; maximum upload file: **20 MB**.
- Request deadline: **300 seconds**; complete ASR job deadline: **7,200 seconds**.
- Automatic retries/provider fallback: disabled.
- OpenAI alternative: `whisper-1`; mocked/unit-tested, **not live-tested**.

The contract requests supported parameters from the official
[Groq docs](https://console.groq.com/docs/speech-to-text) and
[OpenAI docs](https://developers.openai.com/api/docs/guides/speech-to-text).
Beam size/VAD controls are unavailable through these endpoints and are not invented.

## Actual English fixture and end-to-end run

The original script was synthesized locally by Windows SAPI using Microsoft
David Desktop (en-US), via `scripts/create_speech_fixture.ps1`. No copyrighted
recording was downloaded. Original fixture: `.validation/phase2/fixture/speech.wav`,
22,050 Hz, mono, 16-bit. Synthesis script/provenance are retained alongside it.
The script has not been certified by manual listening as ground truth, so no
real-recording WER is claimed.

Executed the integrated CLI with the project Python 3.12.14 environment:

```text
python -m meeting_assistant transcribe .validation/phase2/fixture/speech.wav
```

The configured portable FFmpeg/ffprobe paths were supplied through environment
variables. The model API call was real and authenticated, not mocked.

| Measurement | Final retained run |
| --- | --- |
| Canonical duration | 14.4269375 seconds |
| Client model load | 0 seconds; inference is remote |
| API request time, including network | 0.9643532 seconds |
| Total ASR time | 0.969 seconds (rounded CLI output) |
| RTF, including network | 0.06684393 |
| Chunks | 1 |
| Segments | 4 |
| Timestamped words | 31 |
| Reported language | `English` |
| Language probability | `null`, not supplied by Groq |
| Per-word probability | `null`, not supplied by Groq |
| Native segment avg_logprob | -0.22302176 |
| Native segment no_speech_probability | 0.0070533752 |
| Native segment compression_ratio | 1.2733333 |

An earlier smoke run took 1.169 seconds, RTF 0.081, with the same segment/word
counts. These are short-fixture observations, not a meeting-length benchmark.

The final job and transcript are retained at:

```text
jobs/ccab12c3-f428-48f8-867e-e0930a30645d/
    audio/canonical.wav
    transcripts/18bc8670-c7e6-4d0b-a1a0-a84e7b10d04c/
        raw_transcript.json
        raw_transcript.txt
```

Independent external ffprobe inspection confirmed **WAV, pcm_s16le, 16,000 Hz,
one channel, 16 bits**, duration 14.426938 seconds. The JSON was loaded back into
the typed model successfully. The TXT is readable from the ordinary workspace
after fixing Windows staging-directory permission inheritance.

Actual raw text:

```text
[00:00:00.000 --> 00:00:02.000]  Today we are testing the Meeting Assistant.
[00:00:03.140 --> 00:00:07.400]  Please benchmark the speech recognition pipeline with 16,000 Hz audio.
[00:00:08.360 --> 00:00:10.820]  We need 3 results by Friday at 10.30.
[00:00:11.700 --> 00:00:13.500]  Do not delete the original recording.
```

## Quality and robustness observations

- Text inspection against the synthesis script found all four sentences,
  including the negation **“Do not delete”**, present. It capitalized “Meeting
  Assistant” and rendered “sixteen thousand hertz”, “three”, and “ten thirty” as
  `16,000 Hz`, `3`, and `10.30`. These native choices were **not corrected**.
- Native word timing for “Assistant.” ended at 2.48 seconds while its segment
  ended at 2.00 seconds. The 0.48-second discrepancy is retained under the
  documented 0.5-second tolerance, not presented as forced alignment accuracy.
- A real two-chunk run with 8-second chunks produced 4 segments/31 words,
  2.018-second API time and RTF 0.140. It rendered “three” in words rather than
  `3`, illustrating context-dependent API output. All timestamps used global
  recording offsets. JSON is retained in
  `.validation/phase2/manual-api-checks/chunked_speech/`.
- A real one-second digital-silence request hallucinated **“you”** (one segment,
  one word). This is explicitly recorded, not hidden or rewritten. The raw JSON
  is retained in `.validation/phase2/manual-api-checks/silence/`. Phase I still
  rejects exact digital silence before ASR in the integrated workflow. No speech
  detector/forced alignment/LLM correction was introduced to mask this baseline.
- A deliberately corrupt `.wav` passed to the integrated CLI exited **1** with
  `corrupt_media`; no transcript output directory was created.
- Mocked failure tests verify authentication, quota/rate limits, timeout, HTTP
  rejection, TLS/connection errors, invalid JSON, absent/malformed timestamps,
  truncated/noncanonical input, and publication failures. Original exceptions
  are chained where applicable; keys/provider bodies are absent from normal errors.
- Concurrent calls use separate temporary workspaces; failure cleanup preserves
  source recordings and leaves no completed-looking partial transcript.
- No `.pending-*` transcript staging files remained in the retained job.

Runtime-only evaluation was actually run against the saved JSON. WER was tested
with controlled strings, including substitutions/insertions/deletions and empty
references. No real-recording WER or human-meeting accuracy result is claimed.

## Test and hygiene results

The final complete suite was run with Python 3.12.14, `REQUIRE_FFMPEG_TESTS=1`,
and `RUN_ASR_API_TESTS=1`: **130 tests, 129 passed, one skipped**, in 9.992 seconds.
The skip is Windows directory-symlink creation privilege. All **13 real FFmpeg
integration tests** and all **three real Groq API tests** passed. The suite contains
62 unchanged Phase I tests and 68 ASR tests. Normal CI does not make API calls or
download any model; its three live API tests are skipped by default.

Ruff lint, Ruff formatting, compilation, baseline hash comparison, and independent
ffprobe checks passed. Git was initialized without a commit. Existing and new
project source files are untracked because no initial commit was requested;
`git diff` has no tracked baseline, so original-file hashes and `git diff --no-index`
against the pre-edit snapshot were used to inspect changes. `git check-ignore`
verified `.env`, `.models/`, `.tools/`, and `jobs/` exclusion. A scan of Git-visible
source/docs found **zero occurrences of the actual API key**. No weights or
credentials are tracked.

Packaging was also verified: built a `py3-none-any` wheel without dependency
downloads, installed it into an isolated local check directory, and successfully
ran its ASR CLI help/imports. The project wheel was then installed in `.venv` for
the developer workflow; no global Python installation was modified.

No CUDA, driver, or other global system changes were made. Local GPU/offline
inference remains deferred; the already downloaded Large-v3 cache is retained.
See [environment assessment](phase2-environment.md).

## Phase III contract

```python
from meeting_assistant.audio import ingest_audio
from meeting_assistant.asr import transcribe_audio

audio = ingest_audio("meeting.mp4")
raw_transcript = transcribe_audio(audio.canonical_audio_path)
for segment in raw_transcript.segments:
    print(segment.id, segment.start, segment.end, segment.text)
```

Phase III can attach speaker information separately using segment/word evidence.
It should preserve this immutable raw record. Switching to a future local backend
requires backend injection/configuration, not changes to these transcript consumers.
