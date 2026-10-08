# Phases VIII–XI release integration

Release checks performed on 2026-10-08. The existing public repository is
`preyash309/InterIITBootCampML-2026`; no replacement repository or remote was created.
The initial local and fetched `origin/main` base was `5b3b64c`.
The uncommitted work was preserved on `phase8-11-trust-aware-release` for one
integration commit, titled **Add trust-aware meeting reliability and review layers**.
Integration uses a fast-forward only, without rewriting published history.

## Scope and preserved conclusions

The canonical I–VII application is retained. This release adds the existing
optional context/ASR, speaker-reliability and semantic sidecars, their reproducible
setup and tests, and the read-only trust-aware workspace. No new ML behavior,
dependency installation, cloud traffic or model benchmark was introduced during
integration. Historical phase reports retain their measured results.

- [VIII](phase8-validation.md): no measured technical-term improvement on the
  synthetic contextual-ASR recordings; disabled by default.
- [IX](phase9-validation.md): independent diarizer agreement is a risk signal,
  not correctness probability. Primary speaker merges produced counterexamples;
  no automatic speaker fusion is enabled.
- [X](phase10-validation.md): event macro F1 approximately 0.043 for Julia-1 and
  0 for GLiNER2.5-Decide; relation F1 zero for both. Semantic observations remain
  experimental, disabled by default and unsuitable for automatic output gating.
- [XI](phase11-validation.md): diagnostic views preserve canonical artifacts,
  source evidence and shared audio playback. The review queue is read-only.

## Fresh ordinary checks

| Check | Result |
|---|---|
| Complete Python suite | 772 total; 747 passed, 25 skipped; 109.319 seconds |
| Real FFmpeg integration | All 13 passed; missing FFmpeg configured to fail |
| Ruff lint / format | Passed; 213 Python files already formatted |
| Python compilation / main dependency check | Passed |
| Existing secondary / Julia / GLiNER environment dependency checks | All passed |
| Frontend tests | 37 passed across four test files |
| ESLint / TypeScript | Passed |
| Vite production build | Passed; dependency module-directive warnings remain nonfatal |

Live provider and optional model test flags were disabled. No local semantic or
secondary-diarizer inference was rerun. Tests cover software behavior, not improved
recognition or semantic accuracy. Follow the commands in the root
[README](../README.md#tests-and-measured-results) to reproduce ordinary checks.

## Publication audit

Candidate files were checked for credential patterns and exact credential values
read privately from ignored environment files; no release candidate contained a
secret. Credential values were never printed. Generic hostile-path test cases
were retained; public documentation and source contained no developer-specific
absolute paths. Repository-relative Markdown links resolved successfully.

Model checkpoints, all virtual environments, Hugging Face caches, `.env`,
`.validation`, jobs, runtime databases, generated recordings/transcripts and
benchmark outputs are excluded. Authored fixture specifications and generators
are included instead of generated meeting recordings. No candidate exceeded
1 MiB. Five selected Phase XI JPEGs contain only the original synthetic fixture;
their [provenance](assets/README.md) records the source and limitations.

Third-party weights are acquired separately. Sortformer uses CC BY-NC 4.0;
Julia-1 and GLiNER2.5-Decide releases use Apache-2.0. No project license was
invented; the repository still has no source LICENSE file.

Local validation logs and audit manifests remain ignored under
`.validation/release-integration/`. The final commit and remote verification are
reported after publication rather than fabricated in advance.
