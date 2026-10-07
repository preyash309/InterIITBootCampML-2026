# Phase 1 validation record

Executed on 2026-10-07 in the project workspace on Windows.

## Environment

- Primary test runtime: Python 3.14.7.
- Additional full-suite runtime: Python 3.12.14 (bundled development runtime).
- Actual FFmpeg/ffprobe build: 9.0.2 essentials from gyan.dev, a Windows build
  provider linked by the official FFmpeg downloads page.
- Tool executables were explicitly configured with `FFMPEG_PATH`/`FFPROBE_PATH`.
  They were not on the system PATH. The portable build lives in ignored `.tools/`;
  no machine-specific executable path appears in package code.
- Ruff 0.16.10 used as a temporary development checker, not a package dependency.

## Automated validation

```text
python -m unittest discover -s tests -t . -v
Ran 62 tests in 5.440s
OK (skipped=1)
```

**61 passed, 1 skipped, no failures.** The suite includes 49 unit tests and 13
real-FFmpeg integration tests. `REQUIRE_FFMPEG_TESTS=1` was set, so absent tools
could not silently skip integration. All 13 integration tests ran and passed.
The full suite also passed on Python 3.12.14 with the same single skip.

The skip was `test_symlink_job_is_rejected`: Windows denied creation of a real
directory symlink. Tests of unsafe audio/destination link rejection and Windows
reparse-point detection ran using mocks, and actual hard-link source/output
alias protection ran successfully. Real directory-symlink behavior remains
unverified in this environment.

Covered cases:

- Mono PCM WAV, stereo downmixing, 44.1/48-kHz resampling, MP3, M4A, AAC, FLAC,
  OGG/Opus, WebM/Opus, and MP4 with AAC audio.
- Spaces, Unicode, shell-like filenames, leading hyphens, and valid WAV content
  with a misleading `.txt` extension.
- Missing, directory, zero-byte, oversized, inaccessible, unsupported, corrupt,
  truncated, all-zero, too-short, video-only, and playlist inputs.
- Missing executables, authoritative executable overrides, launch permissions,
  subprocess failures/timeouts, malformed JSON/metadata, and default-track selection.
- Expected codec/rate/channels/sample width, independent WAV header/frame
  checks, duration tolerance and mismatches, absent/empty/invalid conversion output.
- UUID validation, busy locks, unsafe links, original/hard-linked source
  preservation, safe atomic replacement, failed retries, temporary cleanup,
  lock-cleanup failure reporting, and typed public-path/CLI contracts.

Other actual checks:

```text
ruff check src tests --output-format concise
All checks passed!

ruff format --check src tests
17 files already formatted

python -m compileall -q src tests
[exit 0]
```

No external static type checker was executed or configured. Python 3.11,
Linux, and macOS execution have not been tested here.

## Independent CLI and ffprobe smoke test

Generated a 2-second, 48,000-Hz stereo synthetic WAV with a Unicode/space filename,
encoded a small MP3 using the real FFmpeg binary, then ran the actual module CLI
using UUID `8af157b1-12bb-4471-aac9-31146c47bfbd` and an ignored validation workspace.

The CLI succeeded. Source metadata: MP3 codec/container, 48,000 Hz, 2 channels,
2.000 seconds, 32,876 bytes. An independent `ffprobe -show_format -show_streams
-of json` command, outside the ingestion wrapper, confirmed:

```json
{
  "format": "wav",
  "codec": "pcm_s16le",
  "sample_rate": "16000",
  "channels": 1,
  "bits_per_sample": 16,
  "duration": "2.000000",
  "size": "64078"
}
```

Real negative CLI checks:

- Random text named `corrupt.wav`: exit 1 with `corrupt_media`, no traceback.
- Truncated WAV: exit 1 with `audio_conversion_failed`, no traceback.
- Invalid `FFMPEG_PATH`: exit 1 with `ffmpeg_unavailable`, actionable configuration message.
- Invalid `FFPROBE_PATH`: exit 1 with `ffprobe_unavailable`, actionable configuration message.

After corrupt/truncated retries using the successful job ID, the previous valid
canonical WAV remained byte-identical. Both original bad inputs remained present.
No `.canonical-*` temporary files or `.ingestion.lock` files remained after these
failures. The ignored `.validation/` directory retains the small smoke-test
inputs, published WAV, and independent metadata summary for inspection.

## Packaging and repository review

Built a wheel and installed it into an ignored local target using setuptools and
`pip --no-build-isolation --no-deps`. Imported from that installed package (not
`src/`) and successfully ingested the real MP3. The public result provided a
readable `pathlib.Path` at `result.canonical_audio_path` without importing FFmpeg
internals. No runtime Python dependencies were installed for ingestion.

The initial directory had no Git repository, so `git status` reports that it is
not a repository and there is no tracked baseline/diff. All application, test,
configuration, and documentation files are new. Review covered the source tree,
documentation, and new-file whitespace checks using Git's `--no-index` mode.
Tooling, build/package metadata, bytecode, jobs, and manual test artifacts are
excluded by `.gitignore`. No ASR, diarization, LLM, frontend, or later-stage
implementation was added.
