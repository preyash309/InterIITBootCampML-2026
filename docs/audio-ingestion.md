# Phase 1 — Audio Ingestion

The stage accepts a local recording path, validates its contents, inspects its
audio stream using ffprobe JSON, converts it with FFmpeg, independently checks
the resulting PCM WAV, and returns an immutable `AudioIngestionResult`.

```text
recording → file checks → ffprobe → audio stream checks → temporary WAV
          → ffprobe + streaming WAV validation → atomic publication → typed result
```

Every published recording is **WAV, `pcm_s16le`, 16,000 Hz, mono, 16 bits**.
Conversion decodes the source and writes lossless PCM; there is no lossy
recompression, denoising, silence removal, enhancement, gain change, or loudness
normalization. Channels are downmixed and samples resampled only as necessary.

### Requirements and setup

- Python 3.11 or newer.
- External `ffmpeg` **and** `ffprobe` executables, with the codecs for your inputs.
- No Python runtime dependencies. Both ingestion and the API ASR adapter use the
  standard library; dataclasses and `unittest` are also standard library.
  Local diarization is a separately installed optional dependency group.
  Packaging uses setuptools as a build dependency.

Install both executables using your operating system's package manager or a
build linked from the [official FFmpeg downloads page](https://ffmpeg.org/download.html).
They are not automatically installed by this package. No machine-specific paths
are embedded in application code.

```sh
python -m venv .venv
# Activate the virtual environment using your shell's activation command.
python -m pip install -e .
ffmpeg -version
ffprobe -version
```

Tools are resolved through PATH by default. Override either one independently
with `FFMPEG_PATH` or `FFPROBE_PATH` (an executable path or executable name).
An invalid explicit override fails rather than falling back silently. Paths with
spaces or Unicode work. Do not include extra command-line flags in these values.

### Supported input categories

Content identification comes from FFmpeg, **not the filename extension**. Common
audio inputs include WAV, MP3, M4A, AAC, FLAC, OGG/Opus, AIFF, and WMA. Video
containers such as MP4/MOV, WebM/Matroska, AVI, MPEG, and MPEG-TS are accepted
when they contain readable audio. A WAV renamed `.txt` still works.

Only local regular files are accepted. Network URLs, playlists, reference/sequence
demuxers, and ambiguous raw PCM inputs are excluded. An explicit demuxer allowlist
and a local-file protocol whitelist apply before both inspection and conversion.
Of multiple audio tracks, the default track is selected, otherwise the first.
Codec availability depends on the external FFmpeg build.

### Public API / Phase 2 contract

```python
from meeting_assistant.audio import (
    AudioIngestionConfig,
    AudioIngestionError,
    ingest_audio,
)

try:
    result = ingest_audio("meeting.mp4")
    # Future ASR needs only this pathlib.Path:
    audio_path = result.canonical_audio_path
except AudioIngestionError as error:
    print(error.code, str(error))  # sanitized application-facing error
```

`AudioIngestionResult` contains `job_id`, `source_path`, `canonical_audio_path`,
`source_metadata`, `canonical_metadata`, and `status="success"`. Both metadata
objects provide container format, codec, selected audio stream index, duration
(possibly unknown for the source), duration provenance, sample rate, channels,
sample width when known, and filesystem size. Paths are absolute. No metadata is
inferred from an extension, and no transcript data is introduced.

Supply `config=AudioIngestionConfig(...)` for explicit configuration. Without a
config object, ingestion reads the environment on each call. Explicit config
objects use their own values; they do not merge with environment overrides.

### CLI smoke test

```sh
python -m meeting_assistant.audio "path/to/meeting.mp3"
python -m meeting_assistant.audio "path/to/meeting.mp4" --work-dir jobs
# Optional installed entry point:
meeting-audio "path/to/meeting.wav"
```

The CLI prints source/canonical metadata and the published path on success. It
prints a concise error code and actionable message to stderr and exits with 1
on ingestion failure. `--job-id <uuid>` regenerates that job safely; `--verbose`
enables internal diagnostics, which can contain paths from FFmpeg.

To run straight from the source tree without installation, set `PYTHONPATH=src`:

```powershell
$env:PYTHONPATH = 'src'
python -m meeting_assistant.audio 'path/to/meeting.mp3'
```

On POSIX shells use `PYTHONPATH=src python -m meeting_assistant.audio ...`.

### Configuration

| Environment variable | Config field | Default |
| --- | --- | --- |
| `AUDIO_WORK_DIR` | `work_dir` | `jobs` relative to the process working directory |
| `FFMPEG_PATH` | `ffmpeg_path` | Resolve `ffmpeg` on PATH |
| `FFPROBE_PATH` | `ffprobe_path` | Resolve `ffprobe` on PATH |
| `AUDIO_FFMPEG_TIMEOUT` | `ffmpeg_timeout_seconds` | 3,600 seconds |
| `AUDIO_FFPROBE_TIMEOUT` | `ffprobe_timeout_seconds` | 60 seconds per probe |
| `AUDIO_MAX_INPUT_SIZE` | `max_input_size_bytes` | 4 GiB (4,294,967,296 bytes) |
| `AUDIO_MIN_DURATION` | `min_audio_duration_seconds` | 0.1 seconds |
| `AUDIO_DURATION_TOLERANCE_SECONDS` | `duration_tolerance_seconds` | 1.0 second |
| `AUDIO_DURATION_TOLERANCE_RATIO` | `duration_tolerance_ratio` | 0.02 (2%) |

Resource limits are configurable to support long meetings. Canonical encoding,
sample rate, and channels are fixed contracts rather than environment settings.
Duration comparison allows the larger of the absolute/relative tolerances. If
stream duration is unavailable in a video-capable container, container duration
is retained as metadata but not compared against audio duration: its video track
may be longer. Missing source fields remain `None`. Canonical fields must exist
and meet the full contract.

### Workspaces, retries, and cleanup

```text
src/meeting_assistant/audio/  # isolated Phase 1 package
tests/audio/                 # unit tests and generated real-media tests
docs/                        # assessment, architecture, validation record
jobs/                        # ignored local artifacts
    <generated UUID>/
        .ingestion.lock      # exists only while converting/validating
        audio/
            .canonical-<random>.wav  # temporary; never returned as success
            canonical.wav           # only verified output is published
```

Only current-stage directories exist. The caller's source stays in its original
location; it is never moved, copied, renamed, or deleted. `source/`, `transcripts/`,
`outputs/`, and `logs/` can be added when later stages actually need them.
Original filenames are not used as job-directory names.

Job IDs are generated UUIDs or caller-supplied UUIDs normalized to a safe form.
Per-job exclusive lock files reject simultaneous writers. A repeated call with
the same job ID intentionally regenerates the output (including when a new
source is explicitly supplied). FFmpeg writes a unique temporary WAV in the
same directory. Only successful validation permits an atomic `os.replace` of
`canonical.wav`. Failed retries retain the previous successful WAV; the failed
call raises and does not return that old file as its new result. Temporary files
and locks are cleaned on handled failures. Empty job directories may remain.
Symlinks, junctions, and hard-linked source/output aliases are checked before
publication. Use a service-owned workspace inaccessible to uploaders.

### Failures and logging

All expected failures inherit `AudioIngestionError` and provide a stable `code`:
missing/directory/device/empty input, excessive size, access/path/disk errors,
unrecognized/corrupt media, absent audio, unknown codecs, unavailable tools,
timeouts, malformed probe metadata, conversion failure, missing/empty/wrong-format
output, too-short/all-zero/truncated audio, inconsistent durations, invalid job
IDs, and busy jobs. Original exceptions are chained where appropriate. Errors
retain `stderr` and `returncode` separately without putting raw tool diagnostics
or absolute paths into ordinary user-facing messages.

Invalid bytes can be indistinguishable from an unsupported format. A recognized
media suffix labels a **failed** probe as `CorruptMedia`; other failed probes use
`UnsupportedMedia`. This distinction never affects admission of valid content.
Decode errors discovered after a successful probe become `AudioConversionFailed`.

Module loggers emit `ingestion_started`, `source_validated`, `metadata_extracted`,
`conversion_started`, `conversion_completed`, `canonical_validation_passed`, and
`ingestion_failed`. Event/job/error-code fields are attached through logging
`extra` for application JSON handlers. Tool logs retain return codes and stderr.
The library installs no global logging configuration and never logs audio bytes.

### Tests

```sh
python -m unittest discover -s tests -t . -v
# Pure unit tests without FFmpeg:
python -m unittest tests.audio.test_config_metadata tests.audio.test_ffmpeg tests.audio.test_ingestion -v
# Real integration tests only:
python -m unittest tests.audio.test_integration -v
```

Set `PYTHONPATH=src` if using an uninstalled checkout. Integration tests resolve
tools using the same PATH/environment mechanism as ingestion. They are skipped
when tools are absent; set `REQUIRE_FFMPEG_TESTS=1` to make absence a failure in CI.
Tests synthesize tiny WAVs, encode temporary media, and clean their temporary
directories. No audio binary fixtures are committed. Integration assertions use
an independent ffprobe command plus the Python WAV reader. The suite covers mono,
stereo, multiple rates, MP3 and other codecs, video with/without audio, spaces,
Unicode, hostile names, misleading extensions, corrupt/truncated/silent files,
missing tools, failures/timeouts, malformed JSON, resource limits, retries,
source preservation, safe paths, and partial-output cleanup.

Optional development checks, with Ruff installed separately:

```sh
ruff check src tests
ruff format --check src tests
python -m compileall -q src tests
```

Ruff is not a runtime dependency. No external static type checker is configured.

### Known limitations

- Signal validation rejects exact digital silence; it is not VAD and cannot
  establish whether nonzero audio contains speech, noise, or meaningful content.
- ffprobe is an inspection tool, not proof of integrity. Full conversion uses
  strict decoder errors, but a damaged file whose valid prefix decodes cleanly
  and lacks trustworthy duration information cannot always be recognized.
- Inputs must be fully written and remain unchanged during ingestion. Ownership
  and lifetime of uploaded input files belong to the caller.
- Standard RIFF WAV limits a single canonical file to roughly 4 GiB (about
  37 hours at this format). Larger canonical files need a future RF64-aware
  contract/validator. Limits and disk capacity should match deployment needs.
- Process termination or power loss can leave a lock/temp file. Remove stale
  artifacts only after confirming no writer is active; no unsafe automatic lock
  expiry or job-retention service is implemented.
- This stage processes one audio track; merging multiple language/audio tracks
  and preserving stereo separation are outside the canonical mono contract.

See [architecture](architecture.md) and [actual validation results](validation.md).
