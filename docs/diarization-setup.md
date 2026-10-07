# Phase III — Speaker Diarization

Community-1 runs locally through `pyannote.audio`. Its regular diarization retains
simultaneous speaker turns; its exclusive diarization supplies one speaker at a
time for ASR word reconciliation. The raw Phase II record is never rewritten.
Native word spelling, punctuation, timestamps and probabilities remain intact.

`SPEAKER_00` does not mean a known person. It only represents one consistent
speaker cluster inside the recording. Labels follow first observation and are
not comparable across meetings. There is no speaker identification, forced
alignment, glossary correction, LLM processing, or meeting intelligence.

### Install and cache once

The verified machine uses Windows x64, Python 3.12, an RTX 4070 Laptop GPU and
project-local torch/torchaudio 2.8.0 CUDA 12.8 wheels. The existing NVIDIA driver
was retained. A separate global CUDA toolkit is unnecessary for this wheel setup.
Use a CUDA wheel supported by your driver; do not replace drivers blindly.

```powershell
python -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[diarization]"
```

The optional group pins pyannote.audio 4.0.7, TorchCodec 0.7.0 and
huggingface-hub 1.33.0 as well. Exact resolved versions for this Windows/Python
environment are in [the dependency snapshot](../requirements/diarization-windows-py312.lock.txt).
After installing CUDA torch wheels, the snapshot can reproduce the remaining
environment with `python -m pip install -r requirements/diarization-windows-py312.lock.txt`.
It is specific to this tested stack, not a cross-platform lockfile.

Accept the access conditions on the [Community-1 model page](https://huggingface.co/pyannote/speaker-diarization-community-1),
then set `HF_TOKEN` in the ignored `.env` file. Keep tokens out of chat, source
control and logs. Download and verify the checkpoint explicitly:

```powershell
python -m meeting_assistant.diarization --prepare-model
python -m meeting_assistant.diarization --verify-model
```

The pinned model revision is `3533c8cf8e369892e6b79ff1bf80f7b0286a54ee`.
Default cache: `.models/diarization/community-1/`, approximately **32.8 MB** for
configuration, segmentation, embedding and PLDA files. Weights are ignored by
Git. A provenance manifest and checksums detect missing/corrupt cached files.
Inference only opens the local directory; missing weights fail clearly and
never trigger a download. `--prepare-model` is an explicit online setup action;
use a separate process from inference and leave `HF_HUB_OFFLINE` unset for setup.
The backend disables pyannote/Hugging Face telemetry and enforces the hub offline
flag during model loading. No audio is sent to Hugging Face or a cloud diarization
service. Integrated ASR still uploads audio to the configured Groq/OpenAI provider.

### Configure and run

Configuration reads `.env` using the existing conventions; process variables win.

| Variable | Default / meaning |
|---|---|
| `DIARIZATION_MODEL` | `pyannote/speaker-diarization-community-1` |
| `DIARIZATION_MODEL_REVISION` | pinned full revision above |
| `DIARIZATION_MODEL_CACHE` | `.models/diarization/community-1` |
| `DIARIZATION_DEVICE` | `cuda`; explicitly set `cpu` for fallback |
| `DIARIZATION_OFFLINE_ONLY` | `true`; acquisition helper respects it |
| `HF_TOKEN` | initial acquisition only; not passed to inference |
| `DIARIZATION_NUM_SPEAKERS` | unset: automatic speaker estimation |
| `DIARIZATION_MIN_SPEAKERS`, `DIARIZATION_MAX_SPEAKERS` | optional bounds; cannot combine with exact count |
| `DIARIZATION_ALIGNMENT_TOLERANCE_SECONDS` | `0.1` |
| `DIARIZATION_NEAREST_TURN_MAX_GAP_SECONDS` | `0.2`; zero disables nearest fallback |
| `DIARIZATION_UTTERANCE_MAX_GAP_SECONDS` | `1.0` |
| `DIARIZATION_MAX_DURATION_SECONDS` | `28800` (8 hours); configurable host-memory protection |

```powershell
# Media → canonical WAV → local diarization + Groq raw ASR → speaker evidence
python -m meeting_assistant diarize meeting.mp4
# Explicit speaker-count experiment; default does not assume two speakers.
python -m meeting_assistant diarize meeting.mp4 --num-speakers 2
# Reuse the exact canonical WAV and retained raw JSON, with no ASR request.
python -m meeting_assistant diarize path/to/canonical.wav --canonical --raw-transcript path/to/raw_transcript.json --output-dir path/to/transcripts
# Explicit CPU fallback, never automatic.
python -m meeting_assistant.diarization --verify-model --device cpu
```

Local diarization is checked before a billed ASR request in the integrated CLI.
Raw JSON/TXT is published separately before reconciliation. Failures preserve
the source, canonical WAV and any previously published raw transcript.

```text
jobs/<job_uuid>/
    audio/canonical.wav
    transcripts/<raw_uuid>/raw_transcript.json
                           raw_transcript.txt
    transcripts/<speaker_uuid>/diarization.json
                               speaker_transcript.json
                               speaker_transcript.txt
```

Speaker bundles use atomic directory publication and refuse overwrite. Both
turn sets, canonical audio SHA-256, model revision, device, runtime and observed
peak allocated GPU bytes are retained. Speaker evidence references the raw UUID,
raw JSON digest, segment IDs and zero-based word indices. JSON supports typed
round-trip loading; TXT uses anonymous labels and explicit `UNKNOWN_SPEAKER`.

### API and attribution policy

```python
from meeting_assistant.audio import ingest_audio
from meeting_assistant.asr import transcribe_audio
from meeting_assistant.diarization import (
    PyannoteBackend, diarize_audio, reconcile_transcript, diarize_transcript,
    save_speaker_transcript,
)

audio = ingest_audio("meeting.mp4")
raw_transcript = transcribe_audio(audio.canonical_audio_path)
backend = PyannoteBackend()  # lazy initialization; reuse this instance for requests
diarization = diarize_audio(audio.canonical_audio_path, backend=backend)
speaker_transcript = reconcile_transcript(raw_transcript, diarization)
artifacts = save_speaker_transcript(speaker_transcript, diarization, "transcripts")
# Convenience alternative (runs diarization and reconciliation):
# speaker_transcript = diarize_transcript(audio.canonical_audio_path, raw_transcript, backend=backend)

for utterance in speaker_transcript.utterances:
    print(utterance.id, utterance.speaker_id, utterance.start, utterance.end, utterance.text)
```

Attribution chooses the maximum temporal overlap of each word with exclusive
turns, summing intervals per speaker. Ties choose the earliest contributing turn.
Small no-overlap gaps use explicitly recorded tolerance or nearest-turn fallback;
larger gaps remain unknown. Speaker changes split ASR segments into utterances.
Segments without words use a marked segment fallback and retain their exact text.
Regular overlap remains separate evidence and sets per-word/utterance overlap flags.
See [the evidence contract](phase3-reconciliation.md) for precise rules.

The model loads once per backend instance and inference calls on that instance
are serialized. Convenience functions retain one backend for the most recent
configuration; use an explicit shared backend for applications. GPU failures
surface clearly; there is no silent CPU fallback. Domain errors distinguish
missing/corrupt model cache, access, device, model load, OOM, invalid canonical
audio, malformed turns, reconciliation failure and publication failure.

### Tests and evaluation

Normal discovery does not import the ML runtime, load a model or download weights:

```powershell
python -m unittest discover -s tests -t .
ruff check src tests scripts
ruff format --check src tests scripts
```

Real tests require cached weights and canonical two-speaker/overlap recordings:

```powershell
$env:RUN_DIARIZATION_MODEL_TESTS="1"
$env:RUN_DIARIZATION_CPU_TESTS="1" # optional actual CPU fallback test
$env:DIARIZATION_TEST_AUDIO="path/to/two-speaker-canonical.wav"
$env:DIARIZATION_TEST_OVERLAP_AUDIO="path/to/overlap-canonical.wav"
$env:DIARIZATION_TEST_RAW_JSON="path/to/matching/raw_transcript.json"
python -m unittest tests.diarization.test_model_integration -v
```

Tests block socket connections during model loading/inference. Model acquisition
must be completed beforehand. Fixture generators are
`scripts/create_multispeaker_fixture.ps1` (Windows SAPI, two installed English
voices) and `scripts/assemble_diarization_fixture.py` (sequential/controlled overlap).
Their output belongs under ignored `.validation/`; no binary audio is committed.
Synthesis provenance and scheduled turns are saved, but are not human-meeting
ground truth or accurate speech-boundary annotations.

```powershell
python -m meeting_assistant.diarization.evaluate --diarization path/to/diarization.json --speaker-transcript path/to/speaker_transcript.json --expected-speakers 2
# Only score DER with supplied reference RTTM annotations:
python -m meeting_assistant.diarization.evaluate --diarization path/to/diarization.json --reference-rttm reference.rttm --recording-id meeting-001
```

DER uses regular turns and standard `pyannote.metrics`, with configurable collar
and overlap scoring. Without annotations it is not scored. Assignment coverage
is a diagnostic, **not speaker accuracy**. Actual runs, errors, timings and
artifact paths are recorded in [Phase III validation](phase3-validation.md).

### Observed limitations

- Validation uses two synthetic Windows SAPI voices; it does not establish
  human-meeting accuracy or benchmark-quality DER. No annotated dataset exists.
- Whisper word timestamps are approximate and may overlap or extend beyond a
  speaker turn. Phase III preserves them, even where display utterances overlap.
- Overlap detection is imperfect; exclusive attribution cannot recover speech
  that ASR omitted. No multi-speaker speech reconstruction is attempted.
- Pyannote needs the complete waveform in CPU memory (about 220 MiB/hour at
  float32), in addition to model/inference memory. The configurable duration
  limit bounds this allocation; long-meeting streaming is not implemented.
- TorchCodec file decoding is unavailable with this machine's static FFmpeg
  build. The tested canonical PCM waveform path works without shared FFmpeg DLLs.
- A hard process termination may leave staging files or an acquisition lock.
  Handled exceptions clean owned temporary files; crash recovery is deferred.
