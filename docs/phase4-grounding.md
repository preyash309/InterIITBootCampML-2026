# Phase IV — candidate evidence contract

## Repository assessment

Before changes, this was an existing Python `src/meeting_assistant` project with
isolated `audio`, `asr` and `diarization` packages. Phase I uses external FFmpeg;
Phase II uses the replaceable Groq/OpenAI Whisper adapter; Phase III uses locally
cached Community-1 and frozen speaker evidence. Existing conventions are frozen
stdlib dataclasses/tuples, `unittest`, Ruff, module logging, `.env` with process
overrides, and UUID artifact bundles. The pre-change suite ran 226 tests:
217 passed, 9 skipped, including all 13 required FFmpeg integrations.

No existing phase implementation or schema was rewritten. Only the root CLI
dispatcher gains `ground`; all new behavior is in `grounding/`.

## Files and responsibilities

| File | Purpose |
|---|---|
| `models.py` | Frozen glossary, candidate, record, provenance, policy and runtime schemas |
| `glossary.py` | JSONL validation, deterministic version digest, layers and meeting JSON |
| `normalization.py` | Retrieval-only normalization, protected language and Metaphone |
| `embeddings.py` | Explicit checkpoint preparation, checksum-verified local CPU inference and static NumPy cache |
| `retrieval.py` | Alias, lexical, phonetic shortlist; cosine evidence and deterministic ranking |
| `service.py` | Bounded context, span selection, raw references and source-integrity verifier |
| `serialization.py` | Typed JSON loading and immutable UUID JSON/TXT publication |
| `__main__.py` | Setup, saved-speaker mode, integrated I–IV workflow |
| `evaluate.py` | Controlled Recall@K/MRR and negative-control reporting |
| `data/glossary/*.jsonl` | Human-readable, packaged global entries, one file per domain |
| `scripts/glossary_seed.txt`, `glossary_overrides.json` | Original curated concepts/context descriptions and specific aliases/definitions |
| `scripts/build_glossary.py` | Deterministic regeneration and stale-output check |
| `scripts/validate_grounding_offline.py` | Socket-blocked provenance/performance validation |

## Glossary maintenance and provenance

The corpus contains 3,590 distinct canonical concepts, 51 registered aliases and
8 explicit ASR aliases. Aliases and normalization variants are not counted as
additional concepts. IDs are stable SHA-256 prefixes of canonical retrieval
forms. The complete validated glossary has a SHA-256 version, retained in results.
The original curated seed lists are grouped by meaningful subject, without
generated permutations or plural expansion. Cross-subject repeated concepts are
retained once. Compact original subject-context descriptions support embeddings;
important ambiguous names and aliases have more specific definition overrides.
These are contextual descriptions, not an encyclopedic reference dictionary.

Examples: Qdrant/PyTorch (AI/software), CSTR/fugacity (chemical), Reynolds number/
heat exchanger (mechanical), PID controller/Transformer (electrical), eigenvalue/
Newton–Raphson method (math), BFS/A* search (CS), ARR/EBITDA (business). No arbitrary
personal names are included in the global corpus. No scraped copyrighted prose,
meeting recordings or synthetic alias permutations were added.

After editing the seed/override sources, run `python scripts/build_glossary.py`,
then `--check` and the tests. Packaged JSONL can also be read/edited directly, but
reconcile those edits with the seed sources before rebuilding. Duplicate IDs,
bad categories/scopes, empty names/descriptions and unmarked alias collisions
fail validation. Same-scope canonical duplicates require distinct `sense` fields.
Whitespace-equivalent names also trigger collision checking. `Unit test` versus
Python `unittest` is deliberately marked ambiguous.

Layers are an explicit union; records preserve the winning entry's scope and ID.
Project entries use `source="project"` in a supplied JSONL file. Meeting entries
use `source="meeting"`; only caller-supplied names are included. For example:

```json
{
  "participants": ["Asha Rao"],
  "companies": ["Example Co"],
  "projects": ["Project Aurora"],
  "entries": [{
    "id": "meeting.custom", "canonical": "ExampleProtocol",
    "aliases": ["example protocol"], "asr_aliases": [],
    "category": "custom_protocol", "domain": "software",
    "description": "ExampleProtocol: custom protocol supplied for this meeting.",
    "source": "meeting", "priority": 1.0
  }]
}
```

Names are vocabulary candidates, never inferred speaker identities. Context
files are bounded to 256 KiB and 1,000 entries; JSONL permits 20,000 entries and
32,000 characters per line. Persisted evidence intentionally contains its bounded
source text; the reusable static index never contains meeting names or text.

## Retrieval policy

1. Preserve original evidence; normalize only lookup strings using NFKC/casefold,
   spaces, hyphens and acronym forms. Registered spacing variants match compactly.
   Meaningful `+`, `#`, `*`, `&` are expanded rather than lost: C/C++/C# remain distinct.
2. Examine 1–4-token spans with cheap lexical/phonetic signals. Skip ordinary
   stop words, protected negations/commitments, dates and number words.
3. Exact canonical/alias/explicit ASR alias/normalized matches have recorded types.
4. Lexical score = 0.75 normalized Indel ratio + 0.25 token-sort ratio, via RapidFuzz.
   Phonetic score = normalized ratio of Jellyfish English Metaphone encodings.
5. Encode each bounded utterance context once, not each n-gram or whole meeting.
   Current text comes first; previous/next utterances within three seconds may
   fill spare space, to at most 600 characters. Speaker IDs are metadata only.
6. Compare normalized context and glossary vectors using NumPy cosine, clipped
   to [0,1]. This is one semantic/context signal, not two duplicated scores.
7. Fuzzy candidates need lexical >= 0.78 **or** phonetic >= 0.88, context >= 0.30,
   and combined score >= 0.64. Exact registered evidence bypasses fuzzy thresholds;
   ambiguous common words such as React still need context support.
8. Combined score = 0.45 lexical + 0.25 phonetic + 0.25 semantic + 0.05 scope.
   Scope = 0.8 × {global:0.2, project:0.6, meeting:1.0} + 0.2 × entry priority.
   Exact scores have a floor of 0.90 + 0.05 semantic + 0.05 scope. These initial
   engineering constants were not optimized against the small benchmark.
9. Reject fuzzy extensions based solely on literal containment: `vector` does
   not justify inventing `pgvector`; `transformer` cannot add `Swin`.
10. Return up to five unique entries, ordered by score, scope and stable ID.
    Select nonoverlapping spans deterministically: exact evidence first, then
    score, span length, offset and entry ID. Different interpretations remain
    candidates for the selected span; raw word references cannot be reused.

Numbers in named models can match only complete exact registered variants.
No numeric fuzzy substitutions are admitted. No text edits or safe-normalization
recommendations are applied; the `safe_normalization` glossary field is metadata
reserved for later conservative editing. Candidate lists are hypotheses, not
instructions to replace text.

## Embeddings and cache

Model: `sentence-transformers/all-MiniLM-L6-v2`, revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, Apache-2.0, 384 dimensions.
The implementation follows the model's documented Hugging Face Transformers
mean pooling over nonpadding tokens, followed by L2 normalization. It uses a
maximum of 256 wordpieces, eval mode, inference mode, CPU and float32 vectors.
Torch is reused from Phase III; no CUDA dependency is added for grounding.
The `sentence-transformers` framework itself is unnecessary for this pooling.

Preparation downloads six required files (~92 MB), pinned to a full revision,
and writes a local SHA-256 manifest. Inference verifies that manifest and calls
only local paths with `local_files_only=True`, `trust_remote_code=False` and
safetensors. Missing/damaged caches fail explicitly, without fallback or download.
Use `--prepare-model` in a fresh online process; `--verify-model` stays offline.

Glossary embedding text consists of canonical name, acronym expansion and
description. Cache keys contain model/revision/dimension/pooling recipe and
global glossary version; manifests map entry IDs to content-hashed rows. Alias
changes update the glossary/version and lexical index, while unchanged embedding
text reuses its row. Changed descriptions encode only affected rows. The latest
four versions may supply reusable rows. Static vectors are small NumPy arrays,
validated for checksum, shape, dtype and finite values, with pickle disabled.
Publication is atomic; project/meeting entries are encoded in memory only.
Normal operation does not save query embeddings or cache raw meeting context.
Old static index versions are retained on disk; manual cache cleanup is supported.

An explicit shared `GroundingRetriever` reuses the model/index. The convenience
API keeps at most two default configurations. Initialization and backend inference
are locked. CPU thread settings are restored after inference; synchronous use
avoids interference with unrelated concurrent Torch CPU workloads.

## Configuration

Existing simple `.env` convention applies; process environment takes precedence.
All fields have a `GROUNDING_` override; named defaults include:

| Variable | Default |
|---|---|
| MODEL / REVISION | pinned identity/revision above; other models rejected in this baseline |
| MODEL_CACHE | `.models/grounding/minilm` |
| INDEX_CACHE | `.cache/grounding` |
| OFFLINE_ONLY | `true`; normal inference is local regardless; explicit setup enables download |
| TOP_K / MIN_SCORE | 5 / 0.64 |
| LEXICAL_FLOOR / PHONETIC_FLOOR / CONTEXT_FLOOR | 0.78 / 0.88 / 0.30 |
| MAX_SPAN_WORDS / CONTEXT_CHARS / NEIGHBOR_GAP_SECONDS | 4 / 600 / 3 |
| BATCH_SIZE / CPU_THREADS | 32 / 4 |
| LEXICAL_WEIGHT / PHONETIC_WEIGHT / SEMANTIC_WEIGHT / SCOPE_WEIGHT | 0.45 / 0.25 / 0.25 / 0.05 |

Weights must sum to one. The effective retrieval policy is serialized with each
result. These knobs enable controlled experiments; no automatic sweep is run.
Exact resolved dependencies are in `requirements/grounding-windows-py312.lock.txt`.
The validated optional ML environment is Python 3.12; core Phase I remains separate.

## Persistence and Phase V interface

New UUID artifact bundles contain `grounding.json`/`grounding.txt`; write failures
clean owned staging files and cannot overwrite earlier bundles. A hard process
termination can leave staging files. No upstream artifact is rewritten.
Records retain raw UUID/digest, speaker UUID/digest, utterance, anonymous speaker,
original character offsets, word references and original timestamps. Segment-only
fallback uses utterance timing and empty word references explicitly.

```python
from meeting_assistant.grounding import (
    GroundingRetriever, load_glossary, ground_transcript, ground_span,
    validate_grounding_source,
)
from meeting_assistant.grounding.serialization import save_grounding

retriever = GroundingRetriever(load_glossary())
grounding_result = ground_transcript(speaker_transcript, retriever=retriever)
validate_grounding_source(grounding_result, speaker_transcript)
files = save_grounding(grounding_result, "transcripts")
candidates = ground_span("cue drant", "Use a vector database for embeddings.",
                         retriever=retriever)
```

Phase V will consume `speaker_transcript` and `grounding_result`, validating
provenance before proposing KEEP / REPLACE WITH GROUNDED CANDIDATE / UNCERTAIN.
It is not implemented here. Frozen raw and speaker records remain independent.

## Genuine limitations

- Heuristic thresholds/scores need evaluation on real annotated meetings; the
  controlled examples do not establish correction accuracy or calibrated confidence.
- Metaphone is English-oriented and imperfect for accents, spoken acronyms and
  invented names. Registered aliases cover only explicitly reviewed examples.
- Many descriptions are compact shared subject context; richer curated definitions
  could improve semantic discrimination. Scope scores do not prove relevance.
- Context truncation can omit late evidence; a 1–4-token span window can miss longer
  terms. A literal substring extension is rejected conservatively.
- Lexical/phonetic shortlist plus contextual reranking does not propose arbitrary
  semantically related concepts without surface evidence.
- API ASR still needs network access in the integrated workflow. Saved Phase III
  grounding, phonetics and embeddings work fully offline after preparation.

Official implementation references: [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
[RapidFuzz scoring](https://rapidfuzz.github.io/RapidFuzz/Usage/fuzz.html),
[Jellyfish phonetic functions](https://jamesturk.github.io/jellyfish/functions/).
