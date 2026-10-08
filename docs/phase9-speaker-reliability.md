# Phase IX — independent diarization and speaker reliability

Community-1 remains the canonical diarizer. Phase IX optionally runs NVIDIA
Sortformer in shadow mode, aligns anonymous labels, and writes additive observations.
It never changes `SPEAKER_XX`, ASR words/timestamps, Phase III utterances, evidence,
refinement, or `MeetingRecord`. Phase X semantic verification and Phase XI reliability
UI are not implemented.

## Backend and installation

The selected checkpoint is [`nvidia/diar_sortformer_4spk-v1`](https://huggingface.co/nvidia/diar_sortformer_4spk-v1),
pinned at `2617bffbd820aa29d8f1fb6ab6f9ed7f0adbc996`. Its end-to-end
FastConformer/Transformer architecture is independent of Community-1. This is not a
second invocation of pyannote. The NVIDIA model license is **CC BY-NC 4.0**; weights
are downloaded separately and are not covered by the project's code license.

One checkpoint uses 493,434,880 bytes (about 471 MiB). Its SHA256 is
`bc74dfd8ca314240abcdc7e2949901eeaa72947a04ce1fab893e373d81f1e689`.
The worker rejects a different checkpoint before loading. NeMo 3.0.0 supports the
checkpoint's `restore_from` and native `diarize` interfaces; see the
[official implementation](https://github.com/NVIDIA-NeMo/Speech/blob/v3.0.0/nemo/collections/asr/models/sortformer_diar_models.py).

The optional `.venv-secondary` runs in a subprocess. It borrows the validated main
environment's Torch/torchaudio and installs other inference dependencies separately.
No Torch, torchaudio, pyannote, driver, CUDA toolkit, or cuDNN is replaced in the main
environment. The setup requires an existing validated main ML environment. It does
not install Torch automatically. Resolved optional packages are in
`requirements-secondary.lock`; these are not core runtime dependencies.

From the project root, using the main virtual environment:

```powershell
.venv/Scripts/python.exe scripts/setup_secondary_diarizer.py --download-model
```

The cache defaults to `.models/sortformer/`. Inference restores a local checkpoint,
sets `HF_HUB_OFFLINE=1`, disables hub telemetry, and passes no application provider
credentials to the child. A missing cache produces an unavailable sidecar; inference
never initiates a model download. The download is an explicit setup action.

CUDA is the default; set `SECONDARY_DIARIZER_DEVICE=cpu` explicitly for a CPU run.
There is no silent device fallback. The native model supports at most four speakers
and whole-recording inference. A conservative **120-second** shadow-only duration
guard protects the 8-GB laptop GPU; raise it explicitly after measuring longer clips.
This guard never rejects a recording from the canonical meeting pipeline. Inference
is sequential within each meeting. The child exits and releases its CUDA memory;
the main primary model may remain cached. The web worker already permits one active
job. Distinct callers must coordinate GPU jobs rather than create concurrent copies.

## Public contract

```python
from meeting_assistant.speaker_reliability import (
    assess_speaker_reliability, save_speaker_reliability,
)

# Existing Phase III records are explicit: primary regular/exclusive turns are needed.
reliability = assess_speaker_reliability(
    audio.canonical_audio_path, primary_diarization, speaker_transcript,
)
bundle = save_speaker_reliability(reliability, output_dir)
observation = reliability.get_speaker_reliability("utt_000001")
```

`SecondaryDiarizationBackend` is a small protocol returning frozen normalized
`SecondaryDiarizationResult`/`SecondarySpeakerSegment` records. Model-specific
objects stay in the subprocess. `compare_diarization(primary, speaker, secondary)`
is the pure comparison interface. `reliability_from_json` reconstructs frozen
version-1 records. These APIs do not initiate ASR, hosted inference, or intelligence.

`process_meeting(media, speaker_reliability=True)` in the existing
`contextual_asr.workflow` opts in. The integrated intelligence CLI also accepts
`--speaker-reliability`. Web orchestration accepts the same optional keyword or
environment flag internally during `DIARIZING`; the same six visible stages and
14 canonical download keys remain intact. The frontend is unchanged.

## Alignment and comparison

1. Validate canonical WAV and its SHA256 against the primary recording; verify the
   speaker transcript's diarization ID and duration.
2. Normalize secondary turns, retain model/version/revision/checkpoint digest,
   duration, load/inference/total runtime and GPU allocation observations.
3. Sweep exact turn endpoints. Sum union speaking-time intersections for each
   primary/secondary label pair, avoiding duplicate same-label interval mass.
4. Maximize total overlap using SciPy's Hungarian assignment when available. Re-solve
   residual optimal assignments to pick lexicographically earliest labels on ties;
   dummy columns allow unmatched speakers. No epsilon perturbation changes short
   fragments. A small exact assignment fallback supports core tests without SciPy.
5. Keep count mismatch, unmapped labels and possible split/merge indicators. The
   latter require at least 20% overlapping source-speaker time in each candidate
   relationship; they are observations, not automatic label repairs.
6. Compare regular speaker sets for overlap and retain primary exclusive attribution
   separately for word observations. Sweep results cover the entire recording,
   including shared silence. Adjacent intervals merge only when every observation
   is identical.

Exact states: `AGREE`, `DISAGREE`, `PRIMARY_ONLY`, `SECONDARY_ONLY`, `OVERLAP_AGREE`,
`OVERLAP_DISAGREE`, `UNMAPPED_SECONDARY`, `SILENCE`. Shared silence is excluded from
overall agreement's speech-union denominator. One-sided speech stays distinct from
speaker disagreement. There is no hidden boundary collar or timestamp retiming.

## Reliability policy

Word observations retain the original `(segment_id, word_index)` reference. Time
intersections produce agreement, disagreement, primary-only, secondary-only, shared
silence, unmapped, primary overlap, secondary overlap and overlap-set agreement
fractions. Strict word agreement requires one mapped secondary speaker matching the
canonical word speaker and primary exclusive attribution. Multi-speaker secondary
regions do not invent a sole speaker. Zero-duration words are uncertain with zero
temporal support. Fractions refer to time, **not probability of correctness**.

Utterance observations use its existing time span. Boundary deltas compare the best
intersecting matching primary exclusive turn and secondary turn, rather than falsely
comparing diarization edges to ASR utterance edges. Signed start/end differences and
their absolute sum are retained; absent matching turns produce nullable deltas.

Policy `speaker_reliability_v1`: `RELIABLE` requires agreement >= .90 and maximum
primary/secondary overlap <= .10; `MIXED` requires agreement >= .65; otherwise
`UNCERTAIN`. Dominant unmapped time or zero temporal support is uncertain. Reasons
include overlapping speech, disagreement, unmatched speech presence, shared silence,
unmapped speakers and speaker-count mismatch. Count mismatch does not automatically
invalidate unaffected correctly mapped regions. Thresholds have not been calibrated.

## Configuration and artifacts

See `.env.example`: `SPEAKER_RELIABILITY_ENABLED` defaults false;
`SECONDARY_DIARIZER_BACKEND` is sortformer; `SECONDARY_DIARIZER_DEVICE` is cuda;
`SECONDARY_DIARIZER_PYTHON` and `SECONDARY_DIARIZER_MODEL_PATH` configure local paths;
`SECONDARY_DIARIZER_MAX_DURATION_SECONDS=120`,
`SPEAKER_RELIABILITY_MAX_SPEAKERS=4`, timeout 600 seconds, reliable/mixed .90/.65.
The Python executable default adapts to Windows vs Unix venv paths.

Each bundle is atomically renamed from a staging directory into
`<output>/speaker_reliability/<result-sha256>/` with:

- `secondary_diarization.json`
- `diarization_comparison.json`
- `speaker_reliability.json`
- `speaker_reliability.txt`

Identical results save idempotently; different runs keep separate bundles. Failed
inference publishes an explicit unavailable result with no fabricated observations.
Full diagnostics stay in module logging. Canonical audio, serialized primary and
speaker records, and secondary result fingerprints plus policy/configuration
versions provide provenance. No existing artifacts are rewritten. Optional failure
cannot invalidate successful primary diarization; primary failure remains fatal.

## Tests and evaluation

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -t .
$env:RUN_SECONDARY_DIARIZER_TESTS='1'
$env:SECONDARY_TEST_AUDIO='path/to/canonical.wav'
.venv/Scripts/python.exe -m unittest tests.speaker_reliability.test_reliability.RealSecondaryTests -v
```

Ordinary tests use fake independent outputs and download/load no model. Run a saved
primary/speaker comparison without paying for ASR:

```powershell
.venv/Scripts/python.exe -m meeting_assistant.speaker_reliability.evaluation `
  --audio canonical.wav --primary diarization.json --speaker speaker_transcript.json `
  --reference reference.json --output-dir benchmarks/phase9
```

The reference must explicitly bind canonical `audio_sha256`, duration, and labeled
`segments`. `reference_kind` identifies its origin. DER/JER use pyannote.metrics only
when `verified_acoustic_annotations=true`, with declared collar and overlap included.
No DER/JER is claimed for unverified source-clip schedules.

Original authored fixture scripts and three installed SAPI voices generate A clean,
B rapid, C overlapping, D three-speaker, E gaps, F interjections. Source schedules
retain speaker/start/end/text and SAPI word-event timestamps. Existing ASR word
midpoints or authored word-onset probes are scored only in unambiguous source clips.
The latter are benchmark probes, not fabricated ASR output. Accuracy is conditional
on source ownership; it is not lexical accuracy or a natural-meeting estimate.

```powershell
scripts/create_speaker_reliability_fixture.ps1
.venv/Scripts/python.exe scripts/assemble_speaker_reliability_fixtures.py .validation/phase9/fixtures
.venv/Scripts/python.exe scripts/benchmark_speaker_reliability.py
```

See [actual validation](phase9-validation.md) for measured results and limitations.
