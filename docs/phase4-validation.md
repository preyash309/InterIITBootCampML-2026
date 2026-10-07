# Phase IV validation — 2026-10-07

## Assessment and preservation

The existing Python project already had Phases I–III, frozen evidence models,
`unittest`, Ruff, a `.venv`, portable FFmpeg, and cached local Community-1.
The pre-change run was **226 tests: 217 passed, 9 skipped**. All 13 real FFmpeg
integrations were required and passed. A pre-change SHA-256 snapshot covered
58 existing Python source/test files. The final comparison found only the
intentional root CLI dispatcher change; no Phase I–III module or existing test
changed. No legacy file disappeared.

Environment: Windows 11 x64, Python 3.12.14, existing Torch/Torchaudio 2.8.0+cu128,
CUDA 12.8, RTX 4070 Laptop GPU, driver 596.49. Phase IV runs on CPU; no CUDA,
NVIDIA driver, cuDNN or other global system component was installed or changed.

## Glossary

17 packaged JSONL files, **3,590 canonical entries**, **51 aliases**, **8 explicit
ASR aliases**, 3,590 contextual descriptions, 20 engineering-unit entries with
symbols retained only as metadata, and 26 entries with ambiguity/context policies.
All 3,590 canonical entries have nonempty phonetic encodings; the complete
registered variant index contains 3,649 encodings. The final static cache contains
3,590 embedding rows; project and meeting names are excluded from persistence.
Global scope has 3,590 entries; project and meeting scopes are supplied at runtime.
The validated global version is:

```text
fb45239ff3072ee4c9918d82bdc767d4184bd7cc5abc4c829c7c4682e19ecaad
```

| Domain | Canonical entries |
|---|---:|
| AI/ML | 723 |
| ML systems | 159 |
| Software engineering | 428 |
| Data engineering/analytics | 165 |
| Vector retrieval | 54 |
| Cloud | 114 |
| Organizations/products | 70 |
| Business | 286 |
| Chemical engineering | 215 |
| Mechanical engineering/units | 255 |
| Electrical/electronics/control | 260 |
| Civil engineering | 110 |
| Mathematics | 283 |
| Statistics | 195 |
| Computer science | 140 |
| Academic/research | 83 |
| Meeting/collaboration | 50 |

Representative concepts include PyTorch/RealNVP, CSTR/fugacity, Reynolds number,
PID controller/Transformer, Newton–Raphson/eigenvalues, BFS/A* search, and ARR/EBITDA.
The corpus uses original curated names and compact contextual descriptions,
with specific reviewed overrides. Counts exclude aliases and repeated concepts.
`scripts/build_glossary.py --check` passes. A built wheel contains all 17 glossary
files and 3,590 entries, without credentials or downloaded weights.

## Dependencies and model

New project-local packages: RapidFuzz **3.14.6**, Jellyfish **1.2.1**, Transformers
**5.19.0**, Tokenizers **0.23.2**, regex **2026.9.29**, Typer **0.27.3**,
shellingham **1.5.4**, annotated-doc **0.0.5**. The latter four support Transformers'
runtime/CLI dependencies, not a new application CLI framework. Existing NumPy
2.5.3, Hugging Face Hub 1.33.0, safetensors 0.8.0 and Torch were reused.
`requirements/grounding-windows-py312.lock.txt` records exact resolved versions
without local wheel paths, credentials or editable absolute paths.

One embedding model was downloaded:
`sentence-transformers/all-MiniLM-L6-v2`, pinned revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, at
`.models/grounding/minilm`. Required files plus manifest total **91,567,897 bytes**
(~92 MB; downloader metadata adds a small overhead). Output has **384 dimensions**.
Inference uses CPU, local files only, safetensors, eval mode and normalized mean
pooling. Both actual load/inference tests and the preservation/performance script
blocked socket connections successfully. The model was not downloaded by tests.

## Retrieval and controlled benchmark

Default policy: Top-K 5; combined minimum 0.64; lexical floor 0.78; phonetic floor
0.88; contextual floor 0.30. Score weights: lexical 0.45, phonetic 0.25, semantic
0.25, scope 0.05. Scope components are global 0.2 / project 0.6 / meeting 1.0,
blended with entry priority. Exact registered evidence dominates fuzzy evidence.
Spans contain 1–4 tokens; context is capped at 600 characters, with nearby
neighbors within three seconds. The effective policy is included in JSON.

The explicitly requested controlled benchmark contains **24 positive cases and
8 negative controls**, authored as synthetic retrieval examples. The final
socket-blocked run reported:

| Metric | Result |
|---|---:|
| Recall@1 | 1.000 |
| Recall@3 | 1.000 |
| Recall@5 | 1.000 |
| MRR | 1.000 |
| Controls returning a specifically forbidden term | 0/8 |
| Controls returning any candidate | 2/8 |

The two controls returning candidates were legitimate `Quadrant` and electrical
`Transformer`. Neither Qdrant for the coordinate-plane control nor Hugging Face
Transformers for the power-supply control was returned. React/attention ordinary
speech, numeric percentages, negations, deadlines and thanks returned no candidates.
Benchmark inference after setup took **0.317 s** in this run.

This is a small regression-oriented fixture with several registered aliases,
not a held-out ASR corruption corpus. It does **not** establish general correction
accuracy, calibrated confidence or 100% performance on real meetings. Weights
were not optimized against these cases. Inspection led to general fixes for
registered spacing variants, symbol-preserving normalization and unspoken
substring extensions; no benchmark-specific transcript correction was applied.

## Actual integrated recording

Original project text was synthesized with the existing Windows SAPI David and
Zira voices. Pronunciation spellings for product names are explicit in
`scripts/create_grounding_fixture.ps1`; provenance and turn scheduling are saved
under ignored `.validation/phase4/fixture`. This synthetic sample is legal to
share, but is not a human-meeting benchmark or a diarization ground-truth dataset.

The developer CLI ran ingestion → Groq `whisper-large-v3` ASR → local GPU
Community-1 speaker evidence → local grounding. Local prerequisites were checked
before the single billed ASR request. Independent ffprobe verified the resulting
38.415063-second WAV as PCM s16le, 16 kHz, mono, 16-bit; file size 1,229,360 bytes.
Groq returned **9 segments and 77 timestamped words**, English, without a language
probability. ASR inference took **1.609 s**. Community-1 inferred **2 speakers**,
yielding **6 utterances**; diarization inference took **3.005 s**, with observed
peak allocated GPU memory **1,707,452,416 bytes**. These are observations, not
speaker-accuracy scores.

Actual unedited ASR sentence:

```text
Deploy the containers with CubeNet Ease. The NVIDIA GPU uses CUDA kernels.
```

Final grounding evidence:

```text
Observed: CubeNet Ease
Candidate: Kubernetes
Matched registered ASR alias: cube net ease
Match type: normalized_exact
Score: approximately 0.927
Lexical: 0.900; phonetic: 1.000; contextual cosine: approximately 0.288
```

The exact registered-spacing match remains usable even below the fuzzy context
floor; it is still candidate evidence, not an applied replacement. The observed
span, original character offsets, raw word references, anonymous speaker and
native timestamps are retained. The final run considered **272 spans** and selected
**14 nonoverlapping records**. Other observed terms included QDRANT/PyTorch/CUDA,
plug flow reactor → PFR, Newton-Raphson method, Reynolds number, annual recurring
revenue → ARR and customer acquisition cost → CAC.

## Quality observations

- Whisper rendered the intended Kubernetes pronunciation as `CubeNet Ease`.
  Phase IV supplies the registered candidate, leaving the ASR text unchanged.
- `QDRANT` capitalization and `Newton-Raphson` punctuation are original ASR output.
  They are compared in normalized lookup space without being edited.
- Spoken fifteen percent appeared as `15%`. That raw representation, Friday,
  Monday and the phrases `not Monday` / `Do not change` were preserved.
- GPU also produced CPU as a lower-ranked phonetic/context candidate. This is
  a real acronym-related false alternative; the exact GPU candidate ranks first.
  No replacement is applied. Metaphone limitations remain visible in the scores.
- The small synthetic recording does not establish natural-meeting fidelity,
  WER or DER; no annotated human reference dataset was claimed or scored.

## Offline preservation and performance

`scripts/validate_grounding_offline.py` ran on the actual retained speaker JSON
with all socket connections blocked. It built a fresh local index, performed
grounding, published JSON/TXT, reused the index and evaluated the controlled cases.
Raw and speaker JSON were confirmed **byte-for-byte unchanged**. Their digests:

```text
raw:     bbd8850d893b0c691a3d8662338cdc8837281b6f6900be0913ac4a451778a343
speaker: d4e97d0d9583e66015c7f23c2547f4d10898863f29272fb9c5a1a1c56a3dd41c
```

| Measurement | Actual final run |
|---|---:|
| Glossary load | 0.287 s |
| CPU model initialization/import | 18.817 s |
| Fresh glossary embedding/index build | 28.540 s |
| Warm index load | 0.429 s |
| First grounding, after setup | 1.531 s |
| Reused grounding | 1.401 s |
| Utterances / spans / records | 6 / 272 / 14 |
| Float32 glossary matrix | 5,514,240 bytes (~5.26 MiB) |

These are single observations affected by machine load, not universal latency
guarantees. Earlier same-size runs were faster. Total process/native model memory
was not reliably measured; matrix storage is the directly measured quantity.
Model initialization and static index encoding are reused rather than paid per
meeting. Project/meeting entries and query context are not persisted in this cache.

## Artifacts and repeatability

Actual retained paths, relative to the project root:

```text
jobs/e16649c4-2812-4b16-aa17-efd26f2fe4f1/audio/canonical.wav
.validation/phase4/integrated/e8e7d449-b8f2-40b7-ab01-1bb557afd30f/raw_transcript.json
.validation/phase4/integrated/ee8fa8a0-60cc-4259-bc7d-6df5a1e78b0a/speaker_transcript.json
.validation/phase4/final-offline/artifacts/9f5b8500-7f48-4168-9814-f0de86e1795d/grounding.json
.validation/phase4/final-offline/artifacts/9f5b8500-7f48-4168-9814-f0de86e1795d/grounding.txt
.validation/phase4/final-offline/report.json
```

Validation commands are in the guide and scripts. Generated audio, results,
benchmarks, model weights, caches and `.env` are ignored. The fixture files intended
for version control contain only original text scripts and the controlled cases;
generated audio and transcripts remain ignored.

## Tests and code checks

The suite contains **300 tests**: audio 62, ASR 69, diarization 95, grounding 74.
Grounding tests cover glossary validation/collisions/layers/versions/units,
configuration, normalization/protected language, lexical/phonetic/context scoring,
spacing/symbols, scope priorities, timestamps and references, nonoverlap selection,
empty and segment-only fallback, incremental/cache corruption, frozen schemas,
serialization, failed-publication cleanup, missing models, CLI and three actual
offline MiniLM integrations. Normal unit tests do not load or download weights.

The final model-enabled run completed **300 tests in 63.116 s: 296 passed,
4 skipped**. All 13 real FFmpeg integrations, four GPU Community-1 tests, one
explicit CPU Community-1 test and all three socket-blocked MiniLM integrations
passed. The skips were three explicitly opt-in live ASR API tests and the Windows
directory-symlink privilege test. The separate integrated command made one real
Groq transcription request successfully; it is not counted as those skipped tests.
The 74 grounding tests include 71 unit/controlled tests and three model tests.
Ruff lint/format,
`compileall`, `pip check`, glossary regeneration check and wheel content inspection
passed. A clean-process public-package import did not import Torch, Transformers,
NumPy, Jellyfish or RapidFuzz. No `.pending-*` artifacts remained after handled
failure tests. No model weights or credentials are tracked.

## Git and Phase V contract

Git is initialized but the existing project has no committed baseline; project
files remain untracked. Consequently `git diff` alone is empty and cannot prove
preservation. The pre-change source hash audit provides that check. No arbitrary
commit was created. New files are confined to grounding code/data/tests, glossary
and fixture/validation scripts, a lock and documentation; modified shared files
are README, `.env.example`, `pyproject.toml` and the root command dispatcher.

Phase V should consume the two immutable objects:

```python
from meeting_assistant.grounding import ground_transcript, validate_grounding_source

grounding_result = ground_transcript(speaker_transcript)
validate_grounding_source(grounding_result, speaker_transcript)
for record in grounding_result.records:
    print(record.utterance_id, record.observed_text,
          [(candidate.entry_id, candidate.canonical) for candidate in record.candidates])
```

The later editor may consider KEEP / REPLACE WITH GROUNDED CANDIDATE / UNCERTAIN.
No editor, LLM request, transcript rewrite, summary, decision/action extraction,
RAG infrastructure or frontend was introduced in Phase IV.
