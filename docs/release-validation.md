# First public release preparation — 2026-10-07

This release task froze the validated application. No ASR settings, model behavior,
prompts, glossary, schemas, evidence rules, audio playback, REST contracts or
persistence behavior were changed. No paid provider tests were run.

## Repository audit and changes

The pre-release repository had Git initialized on `master`, no tracked files,
no commits and no remote. It contained the complete I–VII application and UI,
225 intended public source/configuration/documentation/test files, plus ignored
development and runtime directories. Source, tests, scripts, frontend configuration,
packaging, dependency snapshots, authored benchmark definitions, glossary data,
historical reports and screenshot provenance were reviewed.

Changes are confined to:

- A product-focused README with architecture, evidence mechanics, complete Windows
  setup, privacy, limitations and GitHub-relative links.
- Three preserved detailed guides extracted from the former README:
  `audio-ingestion.md`, `asr.md`, `diarization-setup.md`.
- Four reviewed synthetic-demo screenshots and their provenance under `assets/`.
- Additional ignore rules for model weights, runtime output, databases, logs,
  coverage, temporary files and editor/OS clutter.
- Empty OpenAI credential and supported ASR model/temperature examples in
  `.env.example`.
- An updated package description, without dependency or API changes.
- Removal of one machine-specific download path from the historical Phase VII
  report, retaining the download verification result.

SHA-256 comparison of all pre-existing application source, tests, scripts,
frontend files and package/lock configuration found no changes except the
intentional `pyproject.toml` description update. No source file or runtime asset
was deleted. Useful research/validation reports and fixture generators remain.

## Secret and runtime protection

All intended public text was checked for credential-related terms and common
Groq, Hugging Face, OpenAI, GitHub, AWS and private-key patterns, and for exact
credential values loaded from the ignored `.env` without printing them. Scanned
small validation text was checked separately. No actual credential matches were
found. Credential configuration references and explicit dummy test values are
intentional. The final staged snapshot is checked again before commit.

Excluded directories include `.env`, `.venv`, `.tools`, `.models`, `.cache`,
`.validation`, `jobs`, `build`, `frontend/node_modules` and `frontend/dist`.
Private/runtime job contents are not public fixtures. No recordings, generated
transcripts, SQLite databases, weights, downloaded tooling or logs are included.
Only the four reviewed screenshots are copied out of validation.

Notable excluded assets: an unused ~3.09 GB Whisper checkpoint, ~90.9 MB MiniLM
weights and ~26.6 MB Community-1 embedding weights. The whole local model tree is
~3.22 GB. Portable tool/wheel archives occupy ~5.70 GB; the Python environment
occupies ~8.20 GB. These remain local. Current hosted ASR does not require the
Whisper checkpoint. Explicit model preparation commands in the README obtain
Community-1 and MiniLM; FFmpeg and Python dependencies are installed separately.

The packaged glossary JSONL, small authored benchmark JSON, generation scripts,
three Python snapshots and frontend lockfile are intentional source assets.
No intended public file exceeds 10 MB.

## Reproducibility review

All installed pins match their snapshots exactly:

| Snapshot | Checked pins |
|---|---:|
| `diarization-windows-py312.lock.txt` | 95 |
| `grounding-windows-py312.lock.txt` | 104 |
| `web-windows-py312.lock.txt` | 18 |

The grounding snapshot includes the combined Phase III/IV environment. CUDA
Torch/TorchAudio must be installed from the documented `cu128` index first.
The web extra alone cannot run the full pipeline. No ML dependencies were
reinstalled or changed for release. The frontend's existing lockfile was retained
unchanged and used for a clean `npm ci`.

This task did not test a new machine installation. Setup instructions reflect the
validated existing Windows/Python/CUDA stack; Linux/macOS full setup remains
unverified. No project LICENSE file exists; no license was selected on behalf of
the team. Third-party model/software access conditions still apply.

## Fresh final checks

| Check | Actual result |
|---|---|
| Full Python discovery | 542 tests in 20.260 s; 523 passed, 19 skipped |
| Real FFmpeg integration | All 13 passed; missing FFmpeg was configured to fail |
| Paid tests | Disabled explicitly for ASR, refinement and intelligence |
| Optional cached-model tests | Disabled for this ordinary release regression |
| Ruff 0.16.10 lint | Passed |
| Ruff format | Passed; 137 files already formatted |
| `compileall` on src/tests/scripts | Passed |
| Seven phase/application imports | Passed |
| `pip check` | No broken requirements |
| `npm ci` | 297 packages installed; 0 audit vulnerabilities |
| ESLint | Passed |
| TypeScript | Passed |
| Vitest | 3 files, 21/21 tests passed; 39.37 s |
| Production build | Passed; Vite bundling 1.36 s |

The 19 skips consist of nine live-provider tests, eight opt-in local-model tests
and two Windows symlink privilege cases. Existing mocks cover model/backend
contracts; this release did not rerun real GPU or hosted inference. The preceding
Phase VII socket-blocked cached-model regression passed 531/542, with 11 skipped;
its historical results remain in [phase7-validation.md](phase7-validation.md).

Build output: JavaScript **437.59 kB / 134.72 kB gzip**; CSS **73.70 kB / 15.46 kB
gzip**. Rolldown emitted existing module-level `use client` directive warnings
from UI dependencies; the client-only build succeeded. Starlette's test client
emitted a deprecation warning for HTTPX; checks passed with the existing validated
dependency stack. Neither warning prompted dependency changes in a release freeze.

The first PowerShell test harness stopped on a stderr deprecation warning before
running the suite; its error policy was corrected and the complete run above
passed. An initial sandboxed `npm ci` could not access npm's cache; the authorized
retry succeeded. Neither issue required application changes.

## Scope and publishing

The detailed actual browser/synthetic model run is retained in
[Phase VII validation](phase7-validation.md), with UI interactions in
[UI validation](ui-polish-validation.md). No fresh hosted call or accuracy
benchmark is claimed for release preparation. Evidence provenance remains
distinct from independent semantic verification.

The first commit is intended for `main`, with message
`Initial release: evidence-grounded AI meeting assistant`. There is no configured
remote; publishing requires the project's chosen GitHub URL. No repository,
remote URL, license or artificial historical commits are invented.
