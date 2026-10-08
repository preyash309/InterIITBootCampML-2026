# Phase X — typed semantic observations

This layer is additive and disabled by default. It consumes matching saved
`RefinedTranscript`, `SpeakerTranscript` and canonical `MeetingRecord` objects.
It never edits raw ASR, speaker labels, refined text or MeetingRecord items.
Phase XI visualization and automatic semantic gating are deferred.

The user replaced the planned billed Jev baseline with free local models.
[Julia-1](https://huggingface.co/SupersonicLabs/Julia-1) is a 144.3M typed
choice/score/Boolean encoder with a native 2–20 choice interface.
[GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide) is an English
classifier accepting custom labels. Both are Apache-2.0 model releases.
[Laya](https://huggingface.co/convaiinnovations/laya) also fits this hardware;
its publisher reports weak zero-shot typed-decision performance for the base
English checkpoint. Published benchmarks are not meeting-domain guarantees.

Julia is the smaller experimental default, with GLiNER available for comparison.
**Both local models failed to establish useful meeting-event accuracy in our
small authored benchmark. They are experimental observers, not trusted judges.**

## Setup and offline operation

Run from the project root using the existing validated Torch environment:

```powershell
.venv/Scripts/python.exe scripts/setup_semantic_model.py --provider julia --download-model
# Optional second candidate; do not download it unless comparing models:
.venv/Scripts/python.exe scripts/setup_semantic_model.py --provider gliner --download-model
```

Linux/macOS use `.venv/bin/python`. Generated environments choose platform-specific
Python executable paths. `.venv-semantic` and `.venv-decision` borrow the main
environment's existing Torch through a local `.pth` file and install only pinned
overlays. They do not replace the main Transformers, Torch, CUDA or cuDNN stack.
The setup requires an existing usable Torch installation. Dependency locks are
`requirements-semantic.lock` and `requirements-decision.lock`.

Julia's checkpoint is approximately 551 MiB, with roughly 585 MiB of downloaded
runtime/tokenizer files in total. GLiNER's checkpoint is approximately 1.81 GiB.
Allow extra installation/cache space and activation memory. GLiNER's card calls
its encoder 340M; the distributed safetensors include additional weights, so use
actual download/VRAM measurements for capacity planning.

Models are pinned to revisions and validated by SHA-256 before worker startup:

| Provider | Repository revision | Checkpoint SHA-256 |
|---|---|---|
| Julia | `a85b127321d580d65176c89ced8273f305745d85` | `df853bf7fe424420011f3d0c47a05d7341aa9eefa7fb9f203ea4aada4ad95b72` |
| GLiNER | `5a7adf72a23b4d311abae6ce050d7f0012bb3416` | `40a5a23ff860dc3dff426cecd1048cacdd29c648c96db209dad818e9686dc997` |

An isolated resident subprocess loads one model lazily and serializes calls.
Local loading uses existing directories, `HF_HUB_OFFLINE=1`,
`TRANSFORMERS_OFFLINE=1`; Python socket connections are blocked in the worker.
There is no network fallback or automatic model download during inference.
Missing/corrupt weights, invalid encoding and CUDA errors remain explicit.
CPU fallback requires `SEMANTIC_LOCAL_DEVICE=cpu`; it is never automatic.
Device/global Torch settings remain inside the isolated worker.

## API and processing

```python
from meeting_assistant.semantic_reasoning import (
    JuliaBackend, SemanticConfig, analyze_meeting_semantics, save_semantics,
)

config = SemanticConfig(enabled=True)
with JuliaBackend(config) as backend:
    observations = analyze_meeting_semantics(
        refined, speakers, meeting_record,
        backend=backend, config=config,
        speaker_reliability=reliability,  # optional matching saved Phase IX sidecar
    )
    save_semantics(observations, output_dir)
```

Omit `backend` to let the service create and clean up the configured local backend.
Injected backends implement `SemanticDecisionBackend.decide` and expose their
provider/model identities and call ledger. Reuse an injected backend across meetings
when maintaining a resident model is appropriate.

Processing is deterministic except model scores/timing:

1. Bind source identities/hashes and resolve existing evidence.
2. Select up to 80 whole nonempty utterances, retaining previous/next context.
3. Classify one dominant event; preserve all accepted, ambiguous, abstained and
   unavailable observations. Text, timestamps and speakers come from source data.
4. Prune relation pairs by a 12-event window, 120-second proximity and shared
   evidence/terms or adjacency; adjudicate at most 100 pairs.
5. Build graph and issue components locally. Positive edges and at least two
   shared content terms join nearby events; representatives quote source text.
6. Construct chronological evolution. Only explicit accepted `SUPERSEDES` edges
   between confirmed decisions mark an earlier decision historical.
7. Verify existing decisions/actions against resolved evidence, then check inverse
   coverage for accepted decisions, assignments, commitments and blockers.

Relations describe later B relative to earlier A. `RESULTS_IN` means A leads to B.
Repeated decisions without supported supersession remain separate current nodes.
General cycles are reported explicitly; supersession must respect chronology.
This version models one dominant event per utterance, not clause-level extraction.

## Ontologies and policies

| Event | Meaning |
|---|---|
| INFORMATION | Factual update without a choice/promised task |
| QUESTION / ANSWER | Request for an answer / response to a question |
| PROPOSAL | Unaccepted option, suggestion or conditional plan |
| SUPPORT | Endorsement without confirming a choice |
| OBJECTION / REJECTION | Concern against an option / explicit refusal |
| DECISION | Explicitly confirmed choice or acceptance |
| COMMITMENT | Speaker promises their own work |
| TASK_ASSIGNMENT | Explicit work assigned to an identified person/team |
| BLOCKER | Explicit obstacle preventing progress |
| CLARIFICATION | Disambiguates meaning/scope |
| NONE / AMBIGUOUS | No applicable event / insufficient or conflicting evidence |

Relations: `SUPPORTS`, `ACCEPTS`, `ANSWERS`, `CONTRADICTS`, `REJECTS`,
`SUPERSEDES`, `CLARIFIES`, `ASSIGNS`, `RESULTS_IN`, `NONE`, `AMBIGUOUS`.

Policies: `event_ontology_v1`, `event_classification_v1`, `event_relation_v1`,
`semantic_verification_v1`, `coverage_v1`. Acceptance uses top native probability
at least 0.85; below 0.60 abstains. No threshold was tuned to improve test scores.
Scores are uncalibrated on meetings and must not be presented as user confidence.

Julia retains the complete native softmax and `max_probability` as provider_score.
GLiNER uses its documented classification `class_act=softmax`, `multi_label=True`,
zero selection cutoff to expose every native label score; the adapter selects the
argmax. It never fabricates unavailable scores or renormalizes native FP16 values.
FP16 distribution sums allow numerical error up to 0.001. GLiNER questions run
separately against the resident model, preserving each head's complete evidence;
exact combined encoding is checked against a 1,024-token cap without truncation.
Julia uses strict encoding, 4,096 combined tokens and a 768-token head budget.

Readable local state formatting preserves evidence text, speakers, target roles,
existing owners and deadlines; mechanical word-index/timestamp metadata remains
in result provenance. Runtime/format versions are recorded in provider observations.

Verification checks claim support, confirmed item type, negation, numbers and scope.
Owner/deadline explicitness is asked only when those fields already exist. Outcomes
are `SUPPORTED`, `REVIEW`, `UNSUPPORTED`, `UNAVAILABLE`; none edits the item.
Coverage uses at most three likely record items per important event, with same-event
adjudication even for shared IDs. Wrong item types, duplicate matches and truncated
candidate lists are retained as review warnings. No record item is added or removed.

## Configuration and CLI

`SEMANTIC_ENABLED=false`, provider `julia`, model/cache/runtime as above, device
`cuda`, request deadline 60s, per-meeting deadline 180s. Stage call limits are
80/100/50/70 for events/relations/verification/coverage, with 250 total physical
attempts. State cap is 14,000 characters; limits never silently truncate evidence.
See `.env.example`. Every `SemanticConfig` field accepts `SEMANTIC_<FIELD>`.
Legacy `JEV_*` variables are aliases; `SEMANTIC_*` takes precedence.

To use GLiNER set provider, model, cache and Python path consistently:

```dotenv
SEMANTIC_PROVIDER=gliner
SEMANTIC_MODEL=fastino/GLiNER2.5-Decide
SEMANTIC_MODEL_CACHE=.models/semantic/gliner-decide
SEMANTIC_LOCAL_PYTHON=.venv-decision/Scripts/python.exe
```

Process saved evidence without rerunning earlier stages:

```powershell
python -m meeting_assistant semantics meeting_record.json --refined refined_transcript.json --speaker speaker_transcript.json --enable --output-dir outputs/shadow
```

Add `--speaker-reliability speaker_reliability.json` for a matching Phase IX sidecar.
The intelligence CLI also accepts `--semantic-reasoning`; web orchestration honors
the enable flag without adding stages or download keys. Optional failures do not
invalidate the saved canonical meeting. Invalid configuration makes no model/API calls.

Output is an atomic content-addressed bundle under `semantic_reasoning/<sha256>/`:
`semantic_result.json`, `meeting_events.json`, `event_relations.json`,
`meeting_event_graph.json`, `decision_evolution.json`, `decision_evolution.txt`,
`semantic_verification.json`, `coverage_report.json`, `semantic_reasoning.txt`.
Partial files are cleaned; identical results are idempotent; existing bundles are
never silently overwritten. JSON is UTF-8 and reconstructs frozen typed models.

## Tests and evaluation

```powershell
python -m unittest discover -s tests -t .
$env:RUN_SEMANTIC_MODEL_TESTS='1'
python -m unittest tests.semantic_reasoning.test_local_backend.RealJuliaTests -v
python scripts/benchmark_semantic_reasoning.py --local --events-only --output-dir benchmarks/semantic/julia
python scripts/benchmark_semantic_reasoning.py --local --local-provider gliner --events-only --output-dir benchmarks/semantic/gliner
```

Standard tests use controlled responses, never model downloads or hosted calls.
Authored event/relation gold tests model output; it is not a real human-meeting corpus.
`--events-only` uses explicitly empty evaluation records and separate authored claim
challenges. It makes no Phase VI calls and does not measure Phase VI or coverage quality.
For coverage, provide saved actual baseline records and manually annotated links.
The evaluator reports policy F1, raw argmax accuracy, evolution, evidence integrity,
verification dimensions, Brier and ten-bin ECE. Small samples do not establish calibration.

Optional Jev support remains pinned to `jev-1.13.0` and the inspected official
`POST https://api.typesafe.ai/v1/systemone` contract. It requires backend-only
`TYPESAFE_API_KEY`. Live tests require `RUN_JEV_API_TESTS=1`; no live Jev call was
made. Jev's confidence field is distinct from local native maximum probability.
Transient API retries honor bounded Retry-After and total physical-attempt budgets.

All weights, model caches, local environments, generated meeting data and benchmark
outputs are ignored. No frontend changes, fine-tuning or automatic speaker fusion
were introduced by Phase X.
