# Phase VIII — Context-aware closed-loop ASR

This optional stage supplies additional audio evidence for terminology refinement. It does not edit raw ASR, identify participants, replace diarization, or relax Phase V's existing safety rules. No dependencies or local model versions were changed.

## Runtime

```text
Phase I canonical PCM WAV → unchanged Phase II first-pass ASR → Phase III
                                                               ↓
MeetingContextPack → context-aware Phase IV grounding → suspicious spans
                                                               ↓
                                             merged, bounded PCM windows
                                                               ↓
                                    relevant vocabulary → Groq Whisper Pass 2
                                                               ↓
                          timestamped hypotheses linked to existing candidates
                                                               ↓
                unchanged Phase V KEEP / REPLACE / UNCERTAIN + deterministic vetoes
                                                               ↓
                           existing RefinedTranscript → Phase VI MeetingRecord
```

Supplying context enables context-aware retrieval. Additional ASR is **disabled by default** and requires `--context-asr`, `contextual_asr=True`, or the explicit environment feature flag with a supplied context. With neither context nor an explicit CLI flag, the existing workflow does not enter this stage. For standalone global-glossary experiments, use `contextual_retranscribe` with `enabled=True, use_global_glossary=True`; the default is false.

## Context pack

`MeetingContextPack` and its sources/terms are frozen dataclasses with immutable tuples. JSON uses `meeting_context_v1`; hypotheses use schema `1.0` and policy `context_asr_v1`.

Supported inputs: title, repeated agenda entries, free-text description, explicit terms and aliases, participant names, and UTF-8 TXT/MD/JSON/CSV documents. Documents are limited to eight regular local files, 64 KiB each and 128 KiB total. A pack supports at most 256 terms. Model weights and recordings are unnecessary for context creation.

JSON documents support `{"terms":["Qdrant",{"canonical":"PostgreSQL","aliases":["postgres"]}],"text":"Discuss CUDA."}`. CSV supports `canonical` and optional `aliases` (pipe separated), `category`, and `scope`. Unknown fields/formats and malformed values fail with domain errors. PDF/Office formats are deferred.

Each term retains canonical spelling, aliases, category, meeting/project scope, priority and source IDs. Lookup normalization reuses Phase IV and does not modify canonical spelling (`C++`, `Recall@10`, `GPT-OSS-120B`, `P99 latency`). Sources retain a safe display label, kind and SHA-256 of their supplied content; document paths/content are not sent to Whisper. Equal scoped terms merge provenance. Explicit document vocabulary takes precedence over extraction.

Extraction uses known glossary names, acronyms, CamelCase and hyphenated identifiers. It deliberately ignores ordinary title-case prose and does not exhaustively recognize entities. Document instructions remain untrusted data. Phase V receives JSON evidence under its existing untrusted-data policy; its system prompt is unchanged. Whisper receives only a bounded possible-vocabulary list, never a document as system instructions.

Participant names become vocabulary terms. They never map `SPEAKER_00` to a person.

## Public APIs and CLI

```python
from meeting_assistant.contextual_asr import build_meeting_context, process_meeting

context = build_meeting_context(
    title="Vector Database Migration",
    agenda=["Compare deployment and retrieval options"],
    terms=["Qdrant", "Kubernetes", "HNSW", "PostgreSQL"],
    participants=["Rahul", "Ananya"],
)
record = process_meeting("meeting.mp4", context=context, contextual_asr=True)
# process_meeting("meeting.mp4") preserves the existing no-context workflow.
```

```powershell
python -m meeting_assistant context --title "Vector Database Migration" --term Qdrant --term Kubernetes --participant Rahul --output transcripts/context.json
python -m meeting_assistant intelligence meeting.wav --context transcripts/context.json
# Explicitly opt in to extra hosted speech calls:
python -m meeting_assistant intelligence meeting.wav --context transcripts/context.json --context-asr
```

Context creation makes no provider calls and refuses to overwrite an existing pack. The intelligence CLI also retains the legacy `--meeting-context` input; it cannot be combined with the new format or saved-evidence mode.

For already validated artifacts:

```python
from meeting_assistant.contextual_asr import ContextualASRConfig, contextual_retranscribe
from meeting_assistant.contextual_asr.integration import context_retriever
from meeting_assistant.grounding import ground_transcript
from meeting_assistant.refinement import refine_transcript

engine = context_retriever(context, baseline_retriever)
grounding = ground_transcript(speaker_transcript, retriever=engine)
evidence = contextual_retranscribe(
    canonical_audio_path, speaker_transcript, grounding,
    context=context, config=ContextualASRConfig(enabled=True),
)
refined = refine_transcript(speaker_transcript, grounding, contextual_asr=evidence)
```

`ContextualASRBackend` is a small protocol; `GroqContextualASRBackend` reuses Phase II's existing streamed HTTPS transport and native segment/word parser. The normal Phase II request has no prompt field.

## Retrieval and target selection

The adapter preserves global/project glossary entries and MiniLM reuse, adding context entries through Phase IV's existing scope/priority scoring. Equal evidence favors meeting over project over global; a fuzzy match can still outrank a weaker meeting term. Grounding remains candidate generation. Candidate reasons include `context_source:<id>`; the pack provides the corresponding auditable sources.

The detector requires noncanonical text plus matching contextual/global acoustic evidence. A fuzzy target needs phonetic score ≥0.88 and lexical score ≥0.70 for a context term, or ≥0.78 for a global term. Registered ASR aliases are another signal. Priority is the existing candidate score plus 0.10 for supplied context and 0.05 for an ASR alias, capped at 1, with minimum 0.70. This is a ranking heuristic, **not a calibrated error probability**.

Numbers, money, dates, negation, modality, already supported terms/acronyms, ordinary aliases and simple `s`/`es` inflections are excluded as targets. Fuzzy phrases containing ordinary function words are excluded. These filters favor precision and can miss real errors. They were tightened after the synthetic benchmark exposed ordinary inflections and grammatical phrases as unnecessary targets; they contain no fixture-specific replacements.

Windows use source word times, two seconds padding, meeting-bound clipping and 16-kHz frame alignment. Nearby windows merge if the union fits the duration cap. Remaining overlapping windows are skipped, rather than generating duplicate calls. Priority determines budget admission; skipped record IDs and reasons are retained.

Only relevant candidates belonging to a selected window are retrieved. Default top-K is six; context provenance is preferred, then scope/score. Protected candidate terms are excluded from prompts. No whole glossary or contextual document is uploaded.

The deterministic prompt is `Possible vocabulary: Qdrant, Kubernetes.` It is capped at **223 UTF-8 bytes**, a conservative bound below the provider's 224-token contract for byte-based Whisper tokenization. A long term may be omitted. See the [official Groq speech contract](https://console.groq.com/docs/speech-to-text). Prompt wording makes no assertion that a term was spoken.

## Configuration and costs

| Environment variable | Default | Allowed bounds |
|---|---:|---|
| `CONTEXT_ASR_ENABLED` | false | boolean |
| `CONTEXT_ASR_USE_GLOBAL_GLOSSARY` | false | boolean |
| `CONTEXT_ASR_MAX_WINDOWS` | 6 | 1–12 |
| `CONTEXT_ASR_MAX_WINDOW_SECONDS` | 15 | 1–30 s |
| `CONTEXT_ASR_PADDING_SECONDS` | 2 | 0–3 s |
| `CONTEXT_ASR_MERGE_GAP_SECONDS` | 0.5 | 0–2 s |
| `CONTEXT_ASR_TOP_K_TERMS` | 6 | 1–8 |
| `CONTEXT_ASR_MIN_PRIORITY` | 0.70 | 0–1 |
| `CONTEXT_ASR_MAX_TOTAL_AUDIO_SECONDS` | 60 | 1–180 s |
| `CONTEXT_ASR_TOTAL_TIMEOUT_SECONDS` | 180 | 1–600 s |

Provider/model/language/temperature/authentication inherit the existing ASR configuration: default Groq `whisper-large-v3`, English, temperature zero, native segment and word timestamps. The second pass currently supports Groq only. No local ASR, WhisperX, or other models were installed.

PCM frames are copied in bounded blocks to temporary canonical WAVs, without recompression or changes to the source. Crop-relative provider timestamps are shifted by the frame-aligned window start. Raw timestamps are never retimed. An in-run cache key includes frame/prompt fingerprints, provider/model, language/temperature and window start; it is not shared across users or meetings.

Calls have existing request deadlines plus an overall enhancement deadline. Contextual calls have **zero transport retries**; 429/401/403 stops remaining calls. Other optional provider/parser failures are sanitized and recorded, then Phase V continues from Pass 1. Crop I/O failures are skipped. Invalid canonical input or mismatched source provenance remains an explicit integration error. Source/hash/write failures are not silently claimed as completed evidence.

Recorded call counts mean attempted backend transcriptions, not confirmed billable completions. Initialization failure records zero calls/audio. Recorded audio seconds are not a cost estimate: Groq documents a minimum billed duration of ten seconds per request. Latency excludes crop time in the final implementation; total stage time includes validation/cropping/fingerprinting. Billing/token metrics are not invented.

## Phase V and audit artifacts

An exact known candidate must appear in native Pass-2 words near the original span (±0.75 s). Its ID must already belong to the record and bounded prompt vocabulary. Arbitrary second-pass text introduces no replacements. Whole-window numeric/date/negation/modality disagreement vetoes candidate links, even if the technical term improved.

Phase V receives linked evidence only for candidates in its own top-K. Its weak-candidate, original-supported, acronym, name, formatting and protected-meaning vetoes remain unchanged. A good contextual hypothesis therefore may still be rejected. Saved evidence revalidates source hashes, native timestamps/text, vocabulary provenance, candidate links, prompt digest and budgets.

New UUID bundles publish atomically under:

```text
transcripts/contextual_asr/<result-uuid>/
    meeting_context.json                 # when a pack exists
    contextual_asr.json
    contextual_asr.txt
    contextual_refinement_links.json     # when Phase V completed
```

JSON retains windows, detected/skipped spans, Pass 1/2, native words, retrieved terms/source IDs, protected conflicts, model/language/temperature, prompt/PCM/canonical/source hashes, configuration, error codes and timings. Link sidecars retain hypothesis/candidate IDs and Phase V decisions/rejections. Raw/speaker/refined/MeetingRecord schemas and files stay separate. Failed publication cleans staging files and refuses overwrite.

`PipelineRunner.run(..., context=pack)` supports programmatic jobs within existing stages. **No context HTTP endpoint, React form or additional downloadable-artifact registry was added.** The hardened one-file multipart contract, six visible stages and fourteen existing artifacts remain intact; contextual bundles are available on disk/CLI. HTTP/UI context entry is deferred.

## Evaluation

```powershell
python -m meeting_assistant.contextual_asr.evaluation --audio meeting.wav --context transcripts/context.json --output-dir benchmarks/context-run
# Optional developer-checked acoustic reference:
python -m meeting_assistant.contextual_asr.evaluation --audio meeting.wav --context transcripts/context.json --reference reference.txt --output-dir benchmarks/checked-run
```

The harness shares first-pass ASR/diarization across A baseline, B context-only and C closed-loop. `--scope global|meeting|combined` supports controlled retrieval ablations; `--saved-raw`, `--saved-speaker` reuse exact matching first-pass artifacts. `--canonical` bypasses ingestion only for a verified canonical WAV.

Identical Phase V JSON requests reuse validated decisions within a comparison and record cache hits/zero additional calls. Recovery can seed the same cache from complete immutable A/B bundles with exact source/model validation. This is **not an independent fresh LLM replicate**. Nonidentical requests call the real refiner. Rate-limit failures leave contextual artifacts and completed conditions; `evaluation.json` explicitly marks incompleteness and the CLI returns 2. Use `--recover` with the same audio/context/scope/output plus saved raw/speaker inputs for an interrupted run. A completed evaluation refuses overwrite.

`--annotations data/contextual_asr_benchmark.json` reports intended-script term counts, not WER/acoustic truth. WER is computed only for a separately supplied checked reference. Output includes per-condition text, term precision/recall/F1 when applicable, protected-field changes, applied/rejected edits, call/audio/time metadata and skipped spans. False term metrics count excess annotated vocabulary occurrences, not all possible semantic errors; manual inspection remains necessary.

Synthetic fixture creation: `scripts/create_contextual_asr_fixture.ps1`, then `scripts/assemble_diarization_fixture.py <fixture-directory>`. The authored eight-term/six-negative definitions are original and shareable; generated audio stays ignored. See [measured validation and limitations](phase8-validation.md).

```powershell
python -m unittest discover -s tests -t .
# Explicit billed opt-in, with a canonical original/shareable fixture:
$env:RUN_CONTEXT_ASR_API_TESTS="1"
$env:CONTEXT_ASR_TEST_AUDIO="path/to/canonical.wav"
python -m unittest tests.contextual_asr.test_evaluation_integration.LiveContextualTests
```

Normal tests use fake contextual backends. Keep all live opt-in flags unset for offline CI.
