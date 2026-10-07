# Phase V — Conservative transcript refinement

The existing `src` Python project uses frozen dataclasses, immutable tuples,
stdlib logging, `unittest`, Ruff, optional ML dependency groups, and an ignored
project-local `.env`. Before changes, the complete suite passed: **296 passed,
4 skipped**, including cached Community-1 GPU/CPU and MiniLM tests. Phase V
preserves the Phase I–IV modules, schemas, and saved evidence.

This stage is a conservative terminology editor. It is not a summary generator.
It preserves all speaker utterances in source order. It never rewrites raw ASR,
changes speaker assignments or timestamps, invents corrected word timestamps,
or extracts meeting decisions/actions. Phase VI is a separate, future LLM role.

## Architecture and public contract

`meeting_assistant.refinement` contains:

| Module | Responsibility |
|---|---|
| `config.py`, `exceptions.py` | Immutable options and sanitized stage failures |
| `models.py`, `base.py` | Frozen requests, decisions, outputs; replaceable backend protocol |
| `prompt.py` | Versioned policy, illustrative examples and constrained decision schema |
| `groq_backend.py` | Fixed-host TLS JSON transport, bounded retries, usage/latency |
| `validation.py` | Strict JSON parsing, evidence references, deterministic safety vetoes |
| `service.py` | Validate provenance, build bounded target requests, apply original offsets |
| `serialization.py` | Stable UTF-8 JSON, speaker/timestamp TXT and atomic three-file publication |
| `__main__.py` | Full I–V workflow or reuse saved speaker/grounding evidence |
| `evaluate.py` | Explicit, small live benchmark over labeled synthetic decision cases |

```python
from meeting_assistant.refinement import (
    GroqRefinerBackend, RefinementConfig, refine_transcript,
    save_refined_transcript, validate_refined_source,
)

backend = GroqRefinerBackend(RefinementConfig.from_env())
refined = refine_transcript(speaker_transcript, grounding_result, backend=backend)
validate_refined_source(refined, speaker_transcript, grounding_result)
files = save_refined_transcript(refined, "transcripts")
for utterance in refined.utterances:
    print(utterance.utterance_id, utterance.speaker_id, utterance.start,
          utterance.end, utterance.raw_text, utterance.refined_text)
```

`TranscriptRefinerBackend` exposes `model_info` and
`refine(RefinementRequest) -> RefinementResponse`. Provider responses never escape
as arbitrary rewritten text. A future local backend can implement this protocol;
one is not installed or implemented here. Backend instances reuse immutable
configuration, with independent connections/state for concurrent requests.

The existing ASR transport is specifically a streaming multipart audio uploader,
not a reusable JSON chat transport. Phase V shares its environment convention
and security policy, without making ASR own an LLM client or changing tested
upload behavior. No new Python dependency, model weight or system library is needed.

## Model and policy

Baseline: **Groq `openai/gpt-oss-120b`**, endpoint
`https://api.groq.com/openai/v1/chat/completions`, temperature **0**,
`response_format=json_schema`, `strict=true`, reasoning effort `low`,
`include_reasoning=false`, no streaming or tools. This model name refers to an
open-weight model served by **Groq**, not the OpenAI API. The exact configured
model is checked against returned completion metadata; no automatic model swap.

Groq documents strict schema support for this model in its
[structured-output guide](https://console.groq.com/docs/structured-outputs).
The [API reference](https://console.groq.com/docs/api-reference) documents the
generation parameters. Model availability can change; overrides are explicit.
Other models require explicitly choosing a supported schema mode; unsupported
requests fail instead of silently weakening constraints.

Versions: `refiner_v1`, `refiner_examples_v1`, `refiner_decisions_v1`,
`conservative_v1`. The system prompt treats all transcript and glossary text as
untrusted JSON evidence. Neighboring context is read-only. No hidden reasoning
is requested or stored; decisions use short enumerable reason codes.

For each supplied grounding record, the model returns exactly one decision:

```json
{
  "utterance_id": "utt_000002",
  "decisions": [
    {
      "grounding_record_id": "grnd_000004",
      "action": "REPLACE",
      "candidate_entry_id": "global.94abcb2d2773df65",
      "reason_code": "exact_registered_alias"
    }
  ]
}
```

This illustrates the shape; the actual record IDs depend on the grounding run.
`REPLACE` references a candidate in that exact record. Deterministic code obtains
its canonical text. `KEEP`/`UNCERTAIN` require null candidate IDs and no replacement.
No `replacement_text`, speaker, time or full-transcript field is allowed.

Strict parsing rejects duplicate JSON keys, unknown fields, missing/duplicate
decisions, changed target IDs, unknown actions/reasons or cross-record candidates.
One bounded schema-only repair can follow invalid decisions or Groq's explicit
`json_validate_failed` response. Provider-rejected text is never applied directly.
There are no semantic "think again" retries.

Safety vetoes preserve ordered protected tokens: numeric values (including named
model digits), common percentages/units, dates/times, negation, commitments,
modality and quantifiers. Source offsets and original word references are
validated before requests. Edits are applied right to left to the original text,
then reproduced from the audit log and checked again against upstream evidence.
An edit interval describes its original audio evidence, not new token alignment.

Already-supported original terms cannot become competing terms, e.g. GPU → CPU.
Case-only and dash-only formatting is left unchanged. Ordinary exact aliases and
acronym expansion/shortening are vetoed. Strong registered phonetic ASR aliases
can correct a spoken misrecognition (e.g. `see you da` → `CUDA`). Project/meeting
name corrections need strong explicit candidate evidence; global names are not
guessed. No new safe-normalization metadata is fabricated: Phase IV candidates
do not carry the glossary entry's `safe_normalization` flag.

Rejected REPLACE decisions remain in the edit log with a reason and unchanged
text. KEEP/UNCERTAIN also remain auditable. No grounding records means no request.
This policy intentionally trades recall for conservative preservation.

## Configuration

Process variables override optional `.env` values. Secrets never appear in config
repr, request JSON, logs or ordinary error messages. `.env` is already ignored.

| Environment variable | Default |
|---|---|
| `GROQ_API_KEY` | Required only when calling Groq |
| `REFINER_PROVIDER` | `groq` |
| `REFINER_MODEL` | `openai/gpt-oss-120b` |
| `REFINER_TEMPERATURE` | `0` (baseline enforces this) |
| `REFINER_STRUCTURED_MODE` | `json_schema`; explicit alternative `json_object` |
| `REFINER_REQUEST_TIMEOUT_SECONDS` | `90` per HTTP attempt |
| `REFINER_TRANSPORT_RETRIES` | `1`, maximum `2` |
| `REFINER_MAX_BACKOFF_SECONDS` | `60`; Retry-After beyond this fails safely |
| `REFINER_SCHEMA_REPAIR_RETRIES` | `1`, maximum `1` |
| `REFINER_MAX_COMPLETION_TOKENS` | `4096` |
| `REFINER_MAX_TARGET_CHARS` | `8000` |
| `REFINER_MAX_REQUEST_CHARS` | `24000` for evidence JSON |
| `REFINER_CONTEXT_CHARS` | `1000` total neighboring text |
| `REFINER_DESCRIPTION_CHARS` | `300` per candidate description/variant |
| `REFINER_TOP_K` | `3`, only existing ranked Phase IV candidates |
| `REFINER_MAX_RECORDS` | `64` per target utterance |
| `REFINER_NEIGHBOR_GAP_SECONDS` | `5` |

All budgets are prevalidated before the first billed refinement call. Oversized
targets fail explicitly; no source text is silently trimmed or split. Retries are
bounded and distinguish transport failures from schema repair. Authentication,
other 4xx errors, ambiguous connection failures and timeouts are not retried
automatically. 429/temporary 5xx can retry with capped backoff. TLS verification
is enabled; hosts are fixed, redirects are not followed, responses are capped at
1 MB and each network attempt has an overall deadline.

Per-attempt metadata stores request/response bytes, HTTP status, latency, repair
flag and exact usage when supplied. Aggregate usage is null if an attempted call
lacks provider usage, rather than inventing a token count. No dollar estimate.
Hosted temperature 0 does not guarantee identical decisions across runs.

## Run and artifacts

```sh
# Uploads audio for ASR and bounded text evidence for refinement; may incur charges.
python -m meeting_assistant refine meeting.mp4
# Reuse matching saved evidence; uploads text only, no audio/model loading.
python -m meeting_assistant refine speaker_transcript.json \
    --grounding grounding.json --output-dir transcripts
# Alternative package/installed command:
python -m meeting_assistant.refinement --help
meeting-refine --help
```

Full mode preflights cached local Community-1 and MiniLM, ingests audio, saves
raw ASR and speaker evidence, grounds terminology, then refines. See earlier
phase documentation for model setup and FFmpeg. Optional `--project-glossary` and
`--meeting-context` apply only during full grounding. `--model` is an explicit
override. Saved evidence must have matching IDs/digests and original spans/refs.

```text
jobs/<audio UUID>/audio/canonical.wav
jobs/<audio UUID>/transcripts/
    <raw UUID>/raw_transcript.json, raw_transcript.txt
    <speaker UUID>/diarization.json, speaker_transcript.json, speaker_transcript.txt
    <grounding UUID>/grounding.json, grounding.txt
    <refined UUID>/refined_transcript.json, refined_transcript.txt, edit_log.json
```

UUID output directories avoid trusting upload filenames. Files are flushed and
published together by directory rename. Existing output cannot be overwritten.
Handled write/provider failures leave no completed refinement bundle or staging
files; upstream artifacts remain usable. Interruption/power loss can leave owned
staging files; no automatic unsafe cleanup/retention daemon is introduced.

## Testing and controlled evaluation

```sh
python -m unittest discover -s tests -t .
python -m unittest discover -s tests/refinement -t .
ruff check src tests scripts
ruff format --check src tests scripts
python -m compileall -q src tests scripts
python -m pip check
# Explicit billed live tests:
# Set RUN_REFINER_API_TESTS=1 in your shell, then:
python -m unittest tests.refinement.test_api_integration -v
# Small controlled live comparison; pacing can respect free-account limits:
python -m meeting_assistant.refinement.evaluate --pause-seconds 10 \
    --output benchmarks/refinement.json --artifacts benchmarks/refinement-artifacts
```

`data/refinement_benchmark.json` contains 20 original synthetic, labeled **text
decision cases**, not a fabricated recording dataset. Controlled candidate scores
are hypotheses, not measured retrieval confidence. It isolates refiner behavior:
four terminology errors, correct originals/ordinary words, protected meaning,
acronym expansions, a missing correct candidate, and two injection attempts.
Precision/recall/F1 measure actually applied correct edits; action accuracy is
separate. False correction rate is edits on KEEP/UNCERTAIN cases divided by those
cases. Meaning-changing edits are applied changes inconsistent with labels.
Undefined precision/recall remain null. Reports and private artifact bundles are
ignored; no automatic experiment sweep, replay cache or hidden network in tests.

## Genuine limitations

- Groq service availability, model access and account quotas remain external.
- Prompt/schema/safety gates constrain behavior but do not prove semantic fidelity
  of every technical candidate. Manual review and larger labeled evaluations remain
  necessary; the small synthetic benchmark is not general meeting accuracy.
- Missing or excluded grounding candidates cannot be invented. Conservative vetoes
  can reject legitimate acronym/name/numeric corrections and formatting changes.
- Long single utterances beyond configured budgets need an explicit upstream
  segmentation policy; this phase fails safely instead of truncating evidence.
- Native Whisper timestamps and Community-1 speaker assignments retain their
  upstream accuracy limits. No new forced alignment or identity inference.
- Source SHA-256 checks bind supplied evidence; they are not authenticity signatures.
- Raw transcript, speaker transcript, grounding and refined transcript must all be
  retained. A future Phase VI consumer should call `validate_refined_source` on
  deserialized refinements before trusting their references.
