# Phase II — ASR (temporary API backend)

The developer workflow is:

```text
audio/video → Phase I → verified canonical WAV → Whisper transcription API
            → validate segment/word timestamps → immutable raw transcript → JSON/TXT
```

**Phase II output is an uncorrected raw ASR transcript. Domain correction,
diarization, and meeting intelligence are intentionally deferred.** No local ML
packages, torch, WhisperX, pyannote, or LLMs are required for this API phase.

### Setup and run

Copy `.env.example` to `.env`, then add your own provider key locally. `.env` is
ignored. Process environment variables override file values; the file parser
supports simple `NAME=value` lines and quoted values, without executing or
interpolating anything. Never paste keys into chat or commit them.

```dotenv
ASR_PROVIDER=groq
GROQ_API_KEY=your-key
```

The default model is `whisper-large-v3`. Alternatively select `ASR_PROVIDER=openai`
and set `OPENAI_API_KEY`; its default is `whisper-1`, which uses Whisper v2 rather
than Large-v3. These are explicit alternatives with no silent provider fallback.
See the official [Groq transcription documentation](https://console.groq.com/docs/speech-to-text)
and [OpenAI transcription documentation](https://developers.openai.com/api/docs/guides/speech-to-text).

```sh
python -m pip install -e .
# Original meeting audio or video: runs BOTH phases.
python -m meeting_assistant transcribe meeting.mp4
# Already canonical audio: Phase II only.
python -m meeting_assistant.asr path/to/canonical.wav
# Provider/configuration comparison; each invocation is a new API transcription.
python -m meeting_assistant transcribe meeting.mp3 --provider groq --model whisper-large-v3-turbo
python -m meeting_assistant.asr path/to/canonical.wav --provider openai
```

From the checkout, set `PYTHONPATH=src` as described above. FFmpeg/ffprobe must be
on PATH or configured in process environment variables. The integrated CLI also
accepts `--ffmpeg`, `--ffprobe`, and those executable overrides from `.env`.
Use `--help` for `--work-dir`, `--output-dir`, `--env-file`, `--chunk-seconds`,
`--language`, and `--verbose`. The key is never a CLI argument. Requests send
audio to the selected external provider and may incur that account's API charges.

Integrated output is published into the ingestion job by default:

```text
jobs/<UUID>/
    audio/canonical.wav
    transcripts/<transcript UUID>/
        raw_transcript.json
        raw_transcript.txt
```

Canonical-only CLI output defaults to `transcripts/<transcript UUID>/`.
Override either with `ASR_OUTPUT_DIR` or `--output-dir`. Both files are written
to staging, flushed, and published together by directory rename. Completed raw
transcripts cannot be overwritten; a retry creates a new transcript UUID. A
failed request publishes neither file. Successfully ingested audio remains usable.

### Public API and future local replacement

```python
from meeting_assistant.audio import ingest_audio
from meeting_assistant.asr import transcribe_audio, save_transcript

audio = ingest_audio("meeting.mp4")
raw = transcribe_audio(audio.canonical_audio_path)
artifacts = save_transcript(raw, audio.canonical_audio_path.parent.parent / "transcripts")

for segment in raw.segments:
    print(segment.id, segment.start, segment.end, segment.text)
    for word in segment.words:
        print(word.text, word.start, word.end, word.probability)
```

`ASRBackend` is a protocol with `transcribe(Path, options=None) -> TranscriptResult`.
Inject an instance with `transcribe_audio(path, backend=backend)`. The included
`WhisperAPIBackend` is reusable and holds immutable configuration. Each call has
its own HTTPS connection and temporary chunk workspace, with no shared inference
state. A future local backend can implement the same protocol and reuse a loaded
model internally without changing downstream consumers. It is not implemented yet.

Transcript dataclasses are frozen, with tuple segments/words, deterministic
`seg_000001` IDs, schema version `1.0`, native raw text, and validated timestamps.
`model_info` and `processing_info` preserve the provider, requested language,
temperature, duration, request/chunk counts, and timings. The API exposes segment
log probabilities, no-speech probabilities, and compression ratios. Word and
language confidence are nullable when not exposed. Remote compute type, server
VAD policy, and VRAM are unknown; client model load time is zero. RTF measures
API request time divided by full canonical duration and includes the network.
`transcript_from_json()` loads persisted records with the same validation.

### Defaults and configuration

| Environment variable | Default / behavior |
| --- | --- |
| `ASR_PROVIDER` | `groq`; alternative `openai` |
| `GROQ_API_KEY`, `OPENAI_API_KEY` | Required for the explicitly selected provider |
| `ASR_MODEL` | Groq: `whisper-large-v3`; OpenAI: `whisper-1` |
| `ASR_LANGUAGE` | `en`; `auto` omits the language hint |
| `ASR_TEMPERATURE` | `0`, reproducible baseline request |
| `ASR_CHUNK_SECONDS` | `600` (10 minutes) |
| `ASR_MAX_UPLOAD_SIZE` | `20,000,000` bytes; capped at 24 MB |
| `ASR_MAX_INPUT_SIZE` | 4 GiB |
| `ASR_REQUEST_TIMEOUT` | 300 seconds; overall deadline for each upload/response |
| `ASR_TOTAL_TIMEOUT` | 7,200 seconds across the ASR job |
| `ASR_OUTPUT_DIR` | Integrated: job's `transcripts/`; canonical CLI: `transcripts/` |

Every request uses transcription, `verbose_json`, and both `segment` and `word`
timestamp granularities. There are no domain hints/hotwords, translation calls,
or grammar correction. VAD and beam size are not exposed by these Whisper API
endpoints; they are not simulated as client options. The reported language may
be a provider label such as `English`; it is not an invented detection confidence.

Long recordings are split into contiguous, lossless PCM WAV chunks, bounded by
both duration and upload size. Only one temporary chunk exists at a time and the
canonical file is untouched. Chunk offsets are applied to all timestamps and
segment IDs stay sequential across the complete transcript. Raw chunk text is
joined by newlines, without deduplication or rewriting. Requests stream file
bytes in 64-KiB blocks; neither full recordings nor HTTP upload bodies are loaded
into RAM. TLS verification is enabled and endpoints are fixed official hosts.

Errors have stable `ASRError.code` values for configuration, missing/rejected
credentials, invalid/noncanonical/truncated audio, connection failure, timeout,
rate limits, provider rejection, malformed/inconsistent timestamps, and artifact
write failure. HTTP status is retained separately. Exceptions preserve causes
where useful. Logs expose stage/provider/status/error events, not audio,
transcript text, keys, or provider response bodies. Automatic retries are disabled
to avoid duplicate requests/charges; rerun explicitly after resolving an error.

### Tests and evaluation

```sh
# Entire standard suite: offline ASR tests plus real FFmpeg tests when available.
python -m unittest discover -s tests -t . -v
python -m unittest discover -s tests/asr -t . -v
ruff check src tests
ruff format --check src tests
python -m compileall -q src tests
```

Normal ASR tests never contact providers or download models. The live API tests
are explicit and billed; set the chosen key and a canonical speech fixture:

```powershell
$env:RUN_ASR_API_TESTS = '1'
$env:ASR_TEST_AUDIO = 'path/to/canonical.wav'
python -m unittest tests.asr.test_api_integration -v
Remove-Item Env:RUN_ASR_API_TESTS
```

`scripts/create_speech_fixture.ps1` can synthesize original English test speech
with an installed Windows SAPI voice. It records its origin and synthesis script.
The source text must be checked against the audio before it is treated as a
manually verified reference. See [actual API validation](phase2-validation.md).

Evaluate a saved transcript without another API call:

```sh
python -m meeting_assistant.asr.evaluate --transcript path/to/raw_transcript.json --reference reference.txt
# Without a reference, reports runtime metrics only.
python -m meeting_assistant.asr.evaluate --transcript path/to/raw_transcript.json
# Makes a new API request using canonical WAV, then evaluates.
python -m meeting_assistant.asr.evaluate --audio path/to/canonical.wav --reference reference.txt
```

WER uses standard word edit distance after casefolding, punctuation removal, and
whitespace tokenization. It does not rewrite numbers or alter the raw record.
The output includes reference/hypothesis word counts and total word errors.
No dataset or ground truth is fabricated. Technical-term, date/time, number,
and negation metrics need future annotated data.

### Limitations and deferred local setup

- API inference needs connectivity, a valid key, account quota, and provider
  availability. The OpenAI adapter is mocked/unit-tested; live testing used Groq.
- Server decoding/VAD/compute type are controlled by the provider, so this is
  an API baseline, not a benchmark of the future local faster-whisper stack.
- Fixed PCM chunk boundaries can split sentences/words and lose cross-chunk
  context. There is no overlap/deduplication or glossary prompting in this baseline.
- Native Whisper timestamps are approximate. Word intervals may differ from
  segment boundaries by up to 0.5 seconds; out-of-order or larger inconsistencies
  fail explicitly. No forced alignment is included.
- No acoustic speech detector suppresses hallucinations on silent/noisy input.
  Raw output is retained. Phase I continues to reject exact digital silence;
  canonical-only ASR permits it for explicit no-speech tests.
- API results may vary even with temperature zero; no accuracy claim is made
  from the short synthetic fixture. No human meeting evaluation set exists yet.
- A process crash may leave staging artifacts. Handled failures clean staging
  and chunks; retention and crash recovery are not implemented.

The earlier background setup completed downloading the CTranslate2 Large-v3
weights under ignored `.models/whisper/large-v3/` (~3.1 GB, decimal). NVIDIA
archives are also retained under ignored `.tools/`. They are unused by the API
backend. faster-whisper/CTranslate2 remain uninstalled and GPU/offline inference
remains unverified; see [environment record](phase2-environment.md).
