# Phase 1 architecture

## Repository assessment before implementation

The project directory was empty, including hidden files. There were no README,
dependency/configuration files, application modules, tests, architecture docs,
`.gitignore`, `AGENTS.md`, or Git repository. No language, web framework, test,
logging, or FFmpeg convention existed to preserve. System Python 3.14.7 was
available; FFmpeg and ffprobe were not on PATH. These findings were reported
before creating the package.

## Implementation boundaries

Python 3.11+ with a `src` layout is suitable for eventual Python ML applications
without selecting a UI framework. Frozen standard-library dataclasses provide
typed results/configuration without adding Pydantic. The subsystem has no
dependencies on future stages or framework imports.

| Module | Responsibility |
| --- | --- |
| `audio/config.py` | Fixed canonical constants; validated resource/path configuration; environment loading |
| `audio/exceptions.py` | Application error taxonomy with separate tool diagnostics |
| `audio/models.py` | Immutable source/canonical metadata and ingestion result |
| `audio/ffmpeg.py` | Tool discovery, command arguments, execution/timeouts, JSON inspection, conversion |
| `audio/metadata.py` | Defensive ffprobe JSON parsing and audio-track selection |
| `audio/validation.py` | File constraints; independent header, frame, signal, format, and duration validation |
| `audio/ingestion.py` | Pipeline orchestration, UUID workspace, locking, temporary output, atomic publication, cleanup |
| `audio/__init__.py` | Small public interface; library logging handler |
| `audio/__main__.py` | Minimal smoke-test CLI with actionable nonzero failures |

## Processing flow

1. Normalize/generate a UUID and read validated configuration.
2. Check the source is readable, regular, nonempty, and within its size limit.
3. Resolve both executables and inspect the source through bounded ffprobe JSON.
4. Select default/first audio stream, validate mandatory identifiers, and extract
   available metadata. Unknown optional fields stay unknown.
5. Create a contained job/audio directory and acquire its exclusive lock. Check
   that the destination cannot alias or overwrite the original source.
6. Strictly decode the selected track into a same-directory temporary PCM WAV.
7. Check existence/size, re-probe, and validate the canonical encoding/rate/channels.
8. Stream the PCM WAV with the independent standard-library reader, checking
   header properties, declared versus actual frame bytes, nonzero signal, and
   duration consistency. No waveform is loaded wholesale.
9. Publish via atomic replacement, remove temporary state/lock, and return
   `AudioIngestionResult`. A failed retry preserves the previously published file.

FFmpeg processes paths and uses argument lists with `shell=False`, closed stdin,
bounded execution, restricted protocols/demuxers, and captured diagnostics.
Library events use standard loggers plus structured `extra` fields; callers own
logging destinations and job retention. The service must control its workspace
permissions; checks are not a sandbox against privileged local processes.

## Downstream boundary

```python
from meeting_assistant.audio import ingest_audio

result = ingest_audio("meeting.mp4")
canonical_audio_path = result.canonical_audio_path  # pathlib.Path
```

The next stage needs only that verified path. Phase 1 does not create transcript
models, call ASR/LLMs, use VAD/diarization, or create later-stage artifacts.
