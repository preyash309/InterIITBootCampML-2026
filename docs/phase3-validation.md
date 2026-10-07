# Phase III — actual implementation and validation

Validated on 2026-10-07. The work ends at anonymous speaker evidence; Phase IV
domain correction and later meeting intelligence are not implemented.

## Repository assessment and regression boundary

The existing project had isolated Phase I/II Python packages, frozen dataclasses,
standard-library API ASR, dotenv-style configuration, `unittest`, Ruff and atomic
publication. Before Phase III, all 130 discovered tests completed: 126 passed,
four skipped; all 13 real FFmpeg tests passed.

SHA-256 comparison against the pre-edit snapshot confirms all Phase I modules
and existing audio tests are unchanged. The Phase II raw transcript schema,
serialization and timestamps are unchanged. One genuine ASR integration defect
was corrected in `asr/api_backend.py`: a word starting in a timestamp gap could
overlap the next segment while its midpoint still preceded that segment. The
old rule attached it to the previous segment and failed bounds validation.

Observed Groq example: `I`, 22.66–23.50 s; previous segment ended 22.38 s;
next segment began 23.16 s. Membership now chooses maximum overlap among
compatible nondecreasing segment indices, retaining every returned timestamp
and word. A regression test captures this exact interval. Existing Phase II
tests and all three real Groq tests (speech, chunking, silence) were rerun and passed.

Existing root CLI, README and packaging were extended; no other baseline source
file changed. `.env.example` received token-free Phase III settings.

## Verified environment and dependencies

| Component | Actual value |
|---|---|
| OS | Windows 11, x64, build 26200 |
| Project Python | 3.12.14, `.venv` |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU, 8,188 MiB |
| Driver | 596.49, retained |
| torch / torchaudio | 2.8.0+cu128 / 2.8.0+cu128 |
| torch CUDA runtime | 12.8; `torch.cuda.is_available()` true |
| pyannote.audio | 4.0.7 |
| TorchCodec | 0.7.0 |
| huggingface-hub | 1.33.0 |
| Phase I FFmpeg | existing portable 9.0.2, unchanged |
| Initial free project-volume space | 94,359,506,944 bytes |

Only project-venv Python packages were installed. No global driver, CUDA toolkit
or system library was changed. GPU wheel files include user-space CUDA libraries.
Exact installed package pins are recorded in
`requirements/diarization-windows-py312.lock.txt`; `pip check` passed.
Pyannote requires its numerical, lightning, metrics and telemetry packages
transitively; telemetry is disabled, not used by the application. No WhisperX
or local ASR package was installed. Core Phase I/API-ASR dependencies remain empty.

TorchCodec's direct file decoder cannot load with this machine's static FFmpeg
installation. The tested backend bypasses that decoder using canonical PCM read
with `wave`, filling one CPU float32 tensor from bounded blocks. The irrelevant
TorchCodec import warning is suppressed at that import boundary; other upstream
warnings remain visible. Pyannote emitted TF32 reproducibility and short pooling
window warnings during successful runs; no enhancement or timestamp adjustment
was applied in response.

## Model and offline operation

- Repository: `pyannote/speaker-diarization-community-1`.
- Revision: `3533c8cf8e369892e6b79ff1bf80f7b0286a54ee`.
- Local cache: `.models/diarization/community-1/`.
- Total acquired files including provenance: **32,831,705 bytes**.
- Files: config, README/model attribution, segmentation weights, embedding
  weights, PLDA model and transform, provenance manifest.
- Weight checksums were checked against Hugging Face LFS metadata during
  acquisition. Backend validation additionally checks configuration/README and
  all required artifact hashes against the pinned revision.
- Access succeeded after authorized Hugging Face setup. The token is only in
  ignored local configuration, never passed to the inference pipeline.
- Real model loading and inference succeeded with `socket.socket.connect`
  blocked. The backend also sets hub offline mode and disables pyannote/HF telemetry.
- Missing-cache offline CLI verification returned the domain-specific failure
  and exit code 1. Invalid device CLI selection returned exit code 2.

The first pre-subsystem CUDA pilot took 3.767 s to load and 2.934 s to diarize
27.687 s of audio (RTF 0.106), producing two speakers and nine regular/exclusive
turns. Subsequent integrated timings below are single runs, not statistical
performance benchmarks.

## Architecture and configuration

New `meeting_assistant.diarization` modules:

| Modules | Responsibility |
|---|---|
| `base`, `service`, `__init__` | backend protocol, public orchestration, reusable convenience backend |
| `config`, `exceptions` | typed settings, speaker bounds, sanitized domain failures |
| `audio`, `cache` | canonical validation, provenance, bounded reads, explicit acquisition |
| `pyannote_backend` | lazy initialization, locked/reused pipeline, local CPU/CUDA inference |
| `models` | frozen turns, diarization, word evidence, utterances and speaker transcript |
| `reconciliation` | indexed temporal attribution and regular-overlap sweep |
| `serialization` | stable JSON, typed reloading, TXT and atomic three-file publication |
| `__main__`, `evaluate` | developer workflow, diagnostics and supplied-RTTM DER interface |

Defaults: Community-1 pinned revision, CUDA, automatic speaker-count estimation,
exclusive turns for ownership, regular turns for overlap, 100 ms attribution
tolerance, 200 ms nearest-turn threshold, one-second utterance grouping gap,
500 ms raw/diarization duration consistency tolerance, eight-hour configurable
maximum input duration. CPU is explicit and was tested on the speech fixture.
No WhisperX, identities, glossary hints, LLMs or transcript rewriting are present.

The [evidence contract](phase3-reconciliation.md) specifies tie-breaking,
unknown attribution, absent-word fallback and exact evidence references.

## Real synthetic fixtures and results

Original project text was synthesized locally with Microsoft David Desktop and
Microsoft Zira Desktop (English US), using installed Windows SAPI voices. Six
turns include numbers, technical terms, a negation and a short `Yes` interjection.
The fixture scripts save provenance and assembly schedules under ignored
`.validation/phase3/fixture/`. These are synthetic tests, not human-meeting
benchmarks. A 1.2 s file-level overlap was assembled for the second recording;
that includes TTS padding/silence and is not a precise speech-overlap annotation.

| Measure | Sequential fixture | Controlled overlap |
|---|---:|---:|
| Audio duration | 27.687125 s | 26.487125 s |
| Expected / detected speakers | 2 / 2 | 2 / 2 |
| Regular / exclusive turns | 9 / 9 | 10 / 10 |
| Model load | 5.626911 s | 5.547450 s |
| Diarization inference | 1.055677 s | 1.001458 s |
| Total local processing, including load | 6.715797 s | 6.581083 s |
| Inference RTF | 0.038129 | 0.037809 |
| Peak allocated CUDA bytes | 1,707,583,488 | 1,707,583,488 |
| Timestamped words / direct assignments | 55 / 55 | 55 / 55 |
| Tolerance / nearest / unassigned | 0 / 0 / 0 | 0 / 0 / 0 |
| Assignment coverage | 100% | 100% |
| Derived utterances / known speaker switches | 6 / 5 | 6 / 5 |
| Detected regular overlap regions | 0 | 1 |

The overlap region is 5.751594–5.869719 s (~118 ms). Exclusive turns remain
nonoverlapping while raw ASR word intervals/display utterances can overlap.
GPU memory is allocator-observed process memory, not a system-wide VRAM peak.

Actual sequential speaker output:

```text
[00:00:13.480 --> 00:00:15.720] SPEAKER_00:
We should not change the original recording.

[00:00:16.520 --> 00:00:17.480] SPEAKER_01:
Yes.

[00:00:18.080 --> 00:00:22.280] SPEAKER_00:
Agreed. Let us check the speaker boundaries before we finish.
```

## Quality observations and provenance audit

Text/schedule inspection found both intended voice clusters, all five expected
speaker changes, and the short `Yes` preserved as Speaker 01. Technical words
and `not` were retained. The API rendered spoken `sixteen` as `16`; Phase III
retains that native ASR output rather than correcting it.

Boundaries are imperfect. In the sequential result, raw `I` starts at 22.66 s,
about 0.66 s before its corresponding diarization turn starts at 23.318 s. It
still overlaps that turn and is assigned directly; no timestamp was moved.
The overlap run contains an extra 0.031–0.166 s regular turn near the start,
before the scheduled first voice begins. A later turn also starts slightly before
its scheduled source clip. The detected overlap is much shorter than the
file-level overlap schedule; the schedule includes silence and is not speech
ground truth, so an overlap recall/DER cannot be inferred.

No missed interjection or incorrect cluster switch was observed in these two
synthetic outputs. This is not a claim of perfect diarization. No human listening
assessment or annotated meeting benchmark was performed. Assignment coverage
does not measure speaker correctness. DER was not scored on these fixtures.

Audit reloaded all output JSON into typed models, reran reconciliation and checked
that utterances matched. Every derived word equals its source ASR word, each word
occurs once, references resolve, and the raw JSON SHA-256 digest matches. Retained
raw files remained byte-for-byte unchanged after audit/reconciliation. No handled
failure left a misleading completed bundle or `.pending-*` state.

## Tests and checks actually run

- Normal complete discovery: **226 tests, 217 passed, nine skipped**, 5.286 s.
- Complete discovery with offline CUDA/CPU model tests enabled: **226 tests,
  222 passed, four skipped**, 21.890 s. Skips: three opt-in API tests and one
  Windows directory-symlink privilege test.
- Separately rerun live Groq regression tests: **3/3 passed**, 3.323 s.
- Separately run real diarization tests: **5/5 passed**, 16.361 s: offline CUDA
  multispeaker/provenance, regular/exclusive overlap, model reuse, silent WAV with
  empty speaker output, explicit CPU multispeaker fallback.
- All 13 real FFmpeg regression tests passed in complete discovery.
- New deterministic tests cover boundaries/ties, summed overlap, zero-duration
  words, tolerance/nearest/unknown, speaker splitting/backchannels, absent word
  timestamps, regular overlap, empty output, reference retention, immutability,
  schema validation, round trips, atomic failure cleanup, cache integrity,
  device/OOM/error translation, concurrent model reuse, configuration, CLI and
  diagnostics. Normal tests never load/download ML weights.
- Ruff lint and format checks passed for `src`, `tests`, `scripts` (59 Python files).
- Compilation and public-import checks passed. Public imports do not import
  torch/pyannote; editable package installation and CLI loading succeeded.
- `pip check` reported no broken requirements.

Across these runs, every test except the Windows symlink-privilege case was
executed successfully. RTTM DER scoring is implemented but no real annotated
recording was supplied/scored.

## Generated artifacts and Git

Sequential artifacts:

```text
jobs/5e36fee1-9b2f-4fbf-9106-b867dbf6abc0/audio/canonical.wav
jobs/5e36fee1-9b2f-4fbf-9106-b867dbf6abc0/transcripts/
    2198f12e-bd98-4323-b71b-1ad691364e32/raw_transcript.json
                                         raw_transcript.txt
    5ffa23a6-743a-49ae-bfd6-8266aab9a278/diarization.json
                                         speaker_transcript.json
                                         speaker_transcript.txt
```

Overlap artifacts: job `37f48a0c-d79b-4550-8de0-c6ea90532ab2`, raw UUID
`a466e714-c472-45b8-ac1e-70a2142512b8`, speaker UUID
`222c3438-478e-4f8e-bc55-e21e8ff979ef`.

The Git repository still has no initial commit; deliverables are untracked rather
than committed. Status/diff and baseline hashes were inspected. Models, `.env`,
fixtures, temporary downloads and generated job/transcript files are ignored;
`git ls-files` confirms no weights or tokens are tracked. No commit was created.

## Remaining limitations and Phase IV interface

Synthetic-only quality assessment, approximate ASR/diarization timestamps,
imperfect overlap detection, full CPU-waveform memory allocation, and absent
crash recovery remain genuine limitations. TorchCodec direct decoding is not
available on this machine; canonical tensor input was verified on CPU and CUDA.
ASR still depends on Groq connectivity/quota; local diarization is offline.

```python
from meeting_assistant.audio import ingest_audio
from meeting_assistant.asr import transcribe_audio
from meeting_assistant.diarization import diarize_transcript

audio = ingest_audio("meeting.mp4")
raw_transcript = transcribe_audio(audio.canonical_audio_path)
speaker_transcript = diarize_transcript(audio.canonical_audio_path, raw_transcript)
for utterance in speaker_transcript.utterances:
    print(utterance.id, utterance.speaker_id, utterance.start, utterance.end, utterance.text)
    for evidence in utterance.words:
        reference = evidence.source_word_reference
        original_word = raw_transcript.segments[int(reference.segment_id[4:]) - 1].words[reference.word_index]
        assert original_word == evidence.word
```

Phase IV can derive domain-grounded records from this evidence while retaining
both the immutable raw ASR transcript and separate diarization result.
