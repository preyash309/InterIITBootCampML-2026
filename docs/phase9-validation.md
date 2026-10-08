# Phase IX — measured validation (2026-10-08)

## A. Baseline and frozen contracts

Before editing, **607 tests ran: 587 passed, 20 skipped**, including 13 real FFmpeg
integration tests. `.validation/phase9/baseline-hashes.json` snapshots existing source,
tests, frontend, packaging and configuration, including the uncommitted Phase VIII
work. The final hash comparison confirms no Phase I/II/III/IV/V/VI schema, model,
reconciliation, evidence, or frontend source changed. Only four existing orchestration
files gained small opt-in hooks/arguments: contextual workflow, intelligence workflow,
intelligence CLI, and web orchestration. README, example environment and ignore rules
also changed. Existing tests were not rewritten. Prior Phase VIII changes remain.

Primary regular turns retain overlap; exclusive turns remain the sole canonical
word-attribution source. Original words, IDs, timestamps, utterances and MeetingRecord
are unchanged. Secondary observations occupy new sidecars.

## B. Secondary decision and compatibility

Independent **NVIDIA Sortformer 4spk-v1**, NeMo **3.0.0**, was successfully selected.
The official pinned checkpoint was downloaded from
[`nvidia/diar_sortformer_4spk-v1`](https://huggingface.co/nvidia/diar_sortformer_4spk-v1)
at revision `2617bffbd820aa29d8f1fb6ab6f9ed7f0adbc996`, 493,434,880 bytes.
SHA256: `bc74dfd8ca314240abcdc7e2949901eeaa72947a04ce1fab893e373d81f1e689`.
It is cached at `.models/sortformer/diar_sortformer_4spk-v1.nemo` with a setup manifest.
The model is CC BY-NC 4.0; no weights are committed.

Actual host: Windows 11 build 26200, Python **3.12.14**, RTX 4070 Laptop GPU with
8,188 MiB, NVIDIA driver **596.49**, driver-reported CUDA capability **13.2**.
Validated main runtime remains **Torch/torchaudio 2.8.0+cu128**, CUDA build **12.8**,
cuDNN **9.10.2**, pyannote.audio **4.0.7**, SciPy **1.18.1**, transformers **5.19.0**.
Initial free disk was 75.8 GiB. No system-level installation, driver, toolkit or cuDNN
change was made. Main packaging 26.3 and fsspec 2026.9.0 remain unchanged.

`.venv-secondary` borrows validated Torch and isolates NeMo dependencies. Lightning
2.4.0 requires older packaging/fsspec; only the secondary environment uses **packaging
24.2** / **fsspec 2025.12.0**. Inference-only installation avoided the full training
extras. NeMo's unconditional module imports also required sentencepiece, datasets /
Arrow and its small OneLogger modules. OneLogger reported disabled/no exporters.
Exact resolved local versions are in `requirements-secondary.lock`. Both main and
secondary `pip check` passed. The documented setup command was actually rerun and
passed, verifying the model digest and resolved environment.

## C–H. Implementation and policies

New isolated modules provide frozen results, secondary protocol, subprocess adapter,
standalone NeMo worker, alignment, exact interval sweep, word/utterance observations,
optional service, versioned JSON loading, atomic saving, and evaluation CLI. Setup,
fixture generation/assembly and benchmark scripts are separate developer tooling.

Primary -> secondary -> maximum-overlap assignment -> exact comparison -> sidecar;
no fusion or canonical label repair. Hungarian assignment uses deterministic residual
lexicographic tie selection, zero-weight unmatched columns, and a small exact fallback
without SciPy. Count mismatch, unmapped labels and >=20% overlap split/merge hints
are retained. Policies are `speaker_alignment_v1` / `speaker_reliability_v1`.

States: AGREE, DISAGREE, PRIMARY_ONLY, SECONDARY_ONLY, OVERLAP_AGREE,
OVERLAP_DISAGREE, UNMAPPED_SECONDARY, SILENCE. Shared silence is excluded from overall
speech-union agreement. Word fractions use immutable existing timestamps and references.
Utterance fractions use the existing complete span, including a separately reported
shared-silence fraction. This conservative choice can label pause-heavy utterances
MIXED despite correct word ownership. It was not tuned to improve these results.

RELIABLE >=.90 agreement and <=.10 overlap, MIXED >=.65, otherwise UNCERTAIN; dominant
unmapped time/zero-duration words are uncertain. These are uncalibrated temporal
observations, **not correctness probabilities**. Boundary deltas compare best
intersecting matching single turns. Different fragmentation can produce large deltas;
no turns are merged or retimed to conceal this.

Native NeMo 3.0.0 postprocessing was used unchanged: onset/offset .5, no padding and
zero minimum speech/gap duration, batch size 1, num_workers 0, native float32 model.
No dataset-specific tuning, extra VAD model, training, ASR, or paid inference was used.

## I. Existing 63.445-second recording

Reused the provenance-matching saved Phase VIII raw/primary/speaker artifacts; primary
Community-1 remains canonical. The secondary actually ran locally/offline on CUDA.

- Primary/secondary counts **2 / 2**. `speaker_0 -> SPEAKER_00` (23.233 s overlap),
  `speaker_1 -> SPEAKER_01` (19.897 s). No count mismatch, split or merge hints.
- Aligned agreement **93.690%** over speech union. AGREE 43.130 s, PRIMARY_ONLY
  1.555 s, SECONDARY_ONLY 1.350 s, shared SILENCE 17.410 s. No regular overlap or
  mapped-label disagreement was observed on this clean recording.
- 122 word observations: **97 RELIABLE (79.51%), 12 MIXED (9.84%), 13 UNCERTAIN
  (10.66%)**. Ten utterances: **1 RELIABLE / 9 MIXED / 0 UNCERTAIN**. Pause time
  explains much of the lower full-span utterance agreement.
- Boundary absolute-sum errors range .0169–5.8238 s; the largest is caused by
  different turn fragmentation. It is retained as a measurement, not suppressed.
- Saved primary inference 2.194 s. Retained secondary run: load **5.139 s**, inference
  **2.011 s**, subprocess total **37.884 s** (import/startup included), inference RTF
  **.0317**, subprocess RTF **.5971**. Comparison **1.276 s** includes the first lazy
  SciPy import; subsequent comparisons are sub-3-ms.
  Total measured Phase IX service overhead, including subprocess launch, validation,
  fingerprinting and comparison, was **42.540 s**.
- Peak secondary Torch allocations **1,009,286,144 bytes (962.531 MiB)**; sampled
  whole-device peak for this run **1,389 MiB**. Allocation is not the same as total
  device memory.

An earlier standalone offline CUDA smoke also succeeded: 24 segments / two speakers,
load 5.399 s, inference 3.139 s, total 39.672 s. The opt-in integration test later
passed in 16.724 s. These are separate measured runs, not interchangeable timings.

## J. Six controlled authored fixtures

Original scripts use Microsoft David Desktop, Microsoft Zira Desktop, and Microsoft
Mark. Exact sample scheduling, speaker/text/source voice, and SAPI word-onset events
are retained. No hosted ASR calls were added. Controlled speaker transcripts contain
no pretend ASR words: evaluation uses the authored event probes separately.

| Fixture | Duration s | Expected / primary / secondary speakers | Temporal agreement | Scored probes | Primary accuracy | Secondary accuracy |
|---|---:|---:|---:|---:|---:|---:|
| A clean alternating | 13.966 | 2 / 2 / 2 | 96.649% | 26 | 24/26, 92.31% | 23/26, 88.46% |
| B rapid turns | 10.336 | 2 / 1 / 2 | 44.699% | 12 | 5/12, 41.67% | 5/12, 41.67% |
| C overlap | 12.546 | 2 / 2 / 2 | 89.638% | 22 | 20/22, 90.91% | 19/22, 86.36% |
| D three speakers | 20.374 | 3 / 2 / 3 | 63.965% | 41 | 24/41, 58.54% | 34/41, 82.93% |
| E silence/gaps | 11.147 | 2 / 1 / 2 | 61.480% | 14 | 6/14, 42.86% | 12/14, 85.71% |
| F short interjections | 16.344 | 2 / 1 / 2 | 81.985% | 30 | 21/30, 70.00% | 23/30, 76.67% |

C excludes three source probes inside authored overlap. Its model-observed overlap
union is .3759 s: OVERLAP_AGREE .2985 s and OVERLAP_DISAGREE .0774 s. This is not an
overlap recall score against verified acoustic annotation. B/D/E/F all retain count
mismatch/unmapped-secondary observations; no labels were patched. The expected
speaker count was met on 2/6 primary and 6/6 secondary cases, but correct count is
not equivalent to correct diarization.

## K–L. Ground truth and the research question

Reference files explicitly bind canonical SHA256 and duration. The source was
checked by re-ingesting the original Phase VI recording: the canonical SHA256 exactly
matched the existing 63.445-second recording. The actual saved sidecar was also
reloaded into frozen models successfully (122 words / 10 utterances).
Source clip scheduling includes synthesis silence and has not been manually verified as acoustic speech
boundaries. Therefore **DER/JER are null for these runs**. The evaluator supports
established pyannote.metrics DER/JER only for explicitly verified acoustic reference
segments, declared collar and overlap inclusion. No number is mislabeled as DER.

Actual source-word ownership is scored at existing ASR-word midpoints (63-s case) or
SAPI word-onset events (controlled cases), only where one authored source is active.
These are speaker-attribution probes, not lexical recognition scores. Controlled
events are point observations (agreement 0 or 1), not timed ASR word spans. They
can disproportionately expose word-onset misses near the model's 80-ms resolution.

Across the six controlled cases: primary **100/145 (68.97%)**, secondary **116/145
(80.00%)**. High-agreement probes: primary **90/95 (94.74%)**; low agreement:
**10/50 (20.00%)**. That aggregate association is promising, but not a calibration or
significance result. Most low-agreement probes have one-sided/ambiguous/unmapped
coverage, rather than two different mapped speakers. **No ordinary mapped-label
disagreement probes were observed**, so that specific correctness comparison is
unmeasured.

The rapid-turn case is a strong counterexample: the primary collapsed two true
speakers; among six mapped-agree probes its accuracy was **1/6**, while low-agreement
accuracy was **4/6**. The secondary-to-primary time assignment can align a secondary
speaker to a canonical label whose best truth identity differs. Agreement cannot
repair a primary speaker merge or guarantee correctness.

On the existing fixture, primary accuracy was **122/122**, including **97/97 high**,
**12/12 medium**, and **13/13 low**. Agreement adds no demonstrated primary-accuracy
benefit there. F's yes/no/okay probes were primary **0/3**, secondary **2/3**; the
misses and unmapped secondary regions remain visible. Across controlled fixtures the
six interjection probes yielded **3/6** correct for each model.

Conclusion: independent disagreement is useful observable evidence, but these small,
synthetic, voice-limited recordings do not establish reliable calibration. Do not
enable automatic fusion or advertise confidence percentages.

## M–N. Performance and failure handling

| Fixture | Primary inference s | Secondary load s | Secondary inference s | Secondary subprocess total s | Comparison ms |
|---|---:|---:|---:|---:|---:|
| A | 2.536 | 6.073 | 2.091 | 38.250 | 1.611 |
| B | .570 | 6.719 | 2.559 | 46.765 | 2.812 |
| C | .904 | 5.277 | 1.611 | 42.124 | .607 |
| D | .941 | 2.203 | .765 | 15.094 | .831 |
| E | .607 | 2.105 | .681 | 14.615 | .839 |
| F | 1.251 | 2.376 | .665 | 14.632 | .743 |

Whole-device peak **3,954 MiB** during the controlled sequential benchmark, with the
primary cached in the parent and secondary in a child. Each child released its
memory on exit. Import/cache warm-up and concurrent CPU regression checks influenced
startup totals; these are observational measurements, not controlled latency claims.
The conservative 120-s duration guard and 600-s subprocess timeout remain explicit.
GPU fallback was not silent; GPU inference passed. Explicit CPU configuration is
supported but not benchmarked in this phase.

Fake-backend tests cover unavailable runtime/cache, load/inference failure, malformed
results, provenance mismatch, timeout, noncanonical/missing input and optional bad
configuration. They verify unavailable sidecars/no fabricated scores, sanitized
messages, temporary cleanup, source compatibility and canonical pipeline completion.
The subprocess uses argument lists, strips API credentials, and is killed/reaped on
timeout. Each reused backend serializes calls; independent callers must coordinate
GPU use. Output staging cleanup is scoped to the job, including when other jobs run.

## O–Q. Tests, artifacts and regressions

Normal suite: **668 tests, 647 passed, 21 skipped**, including all 13 real FFmpeg
integration tests. New Phase IX tests: **61 total, 60 passed normally, 1 opt-in**.
The explicit real cached offline CUDA integration test **passed separately**.
Ruff lint, Ruff format (181 Python files), compileall and both main/secondary pip
checks passed. The normal full suite includes web regressions. Frontend sources and
dependencies are byte-for-byte unchanged; frontend checks were not unnecessarily
rerun. A transient test failure while a real benchmark ran exposed a global temp-dir
assertion; it was fixed to inspect only the invocation's own directory, then rerun.

Retained ignored artifacts:

- `.validation/phase9/environment.json`, `baseline-hashes.json`, `final-checks.log`
- `.validation/phase9/first-secondary.json`, `first-inference.log`, `real-integration-test.log`
- `.validation/phase9/fixtures/<A_clean...F_interjections>/reference.json` and source WAVs
- `.validation/phase9/benchmark/<fixture>/evaluation.json`
- `.validation/phase9/benchmark/existing_63s/speaker_reliability/6baebd9fb6ee35a4293e6164ae2f62cc3f3e193c6da591b873854651797e29fc/`
- `.validation/phase9/gpu-observation-existing.json`, `gpu-observation.json`

Every reliability bundle has secondary/comparison/reliability JSON and readable TXT.
The original raw/speaker/primary artifacts were not rewritten. Git ignore checks
passed for cache, isolated environment and validation outputs; no model weights are
tracked. New source/tests/scripts/docs/fixture specs and the optional dependency lock
remain uncommitted, alongside preserved Phase VIII work. No commit/push was requested.

## R–S. Limitations and next phases

This is a four-speaker non-streaming optional baseline with a 120-s GPU guard, not
validated long-meeting diarization. CPU performance is unmeasured. Source scheduling
is insufficient for acoustic DER/JER; broader natural recordings and acoustic
annotation are needed. Count collapse can make aligned agreement misleading.
Full-span utterance measurements are conservative around pauses; single-turn boundary
pairing is sensitive to fragmentation. Thresholds remain uncalibrated and unchanged.

Phase X can independently add semantic evidence verification. Phase XI can read frozen
sidecars via `reliability_from_json` / `get_speaker_reliability(utterance_id)` for review
and display. Neither phase was implemented; canonical MeetingRecord/evidence remain
the current contract. No speaker identities, voice recognition, cross-meeting matching,
source separation, new ASR model or hosted Phase IX inference was introduced.
