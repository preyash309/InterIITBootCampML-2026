# Phase XI — Trust-aware meeting workspace

The existing React/Vite, Tailwind, shadcn/Radix and Lucide workspace now exposes
saved context, speaker comparisons and experimental semantic observations.
Canonical transcripts, MeetingRecord, resolved evidence and audio remain authoritative.
No inference code, ML schema, model configuration or prompt changed in this phase.

## Product flow

Overview → Meeting reliability → Review → source evidence → bounded audio playback
or jump to the retained transcript. The workspace keeps seven views: Overview,
Transcript, Minutes, Decisions, Action Items, Decision Evolution and Review.

The reliability panel separates terminology provenance, speaker attribution,
evidence references and experimental semantics. It never computes an overall score.
Transcript badges open the existing evidence sheet; no extra drawer or audio element
was introduced. Decisions and actions retain their original wording with optional,
collapsed “Semantic observer” disclosures.

Context audits show pass 1, pass 2, supplied candidates, source labels and actual
Phase V edit outcomes. “Context checked” does not imply a correction. Participant
names remain context, not speaker identities. Contextual ASR showed no measured
terminology improvement in the existing baseline; the UI makes no improvement claim.

Speaker badges report Reliable / Mixed / Uncertain observations. Expanded details
contain actual primary/secondary model identities, temporal agreement, overlap and
boundary disagreement. Speaker-count mismatch gets a prominent warning and a review
entry, even when local agreement looks high. Agreement is not correctness confidence.

Decision Evolution is a vertical, source-quoted timeline with optional Current and
Historical labels confined to this experimental view. Isolated events and empty
results remain inspectable. Experimental semantic models performed poorly in Phase X;
their observations cannot create, approve, remove or alter canonical items.

## Read-only API

| Route suffix under `/api/meetings/{uuid}` | Typed presentation payload |
|---|---|
| `/contextual-asr` | `ContextualASRResult` and `ContextAudit` |
| `/speaker-reliability` | `SpeakerReliabilityResult` and `SpeakerObservation` |
| `/semantic-reasoning` | `SemanticResult`, events, relations, evolution, verification, coverage |

Each response uses `{state, message, data}`. States are `available`, `not_generated`,
`unavailable`, and `failed`; `data` is null except when available. Unknown UUIDs return
404 and unfinished meetings return 409. The client additionally distinguishes loading
and network failure. Optional failures do not prevent canonical workspace loading.

The routes only deserialize existing completed bundles under
`WEB_JOB_ROOT/<uuid>/transcripts/{contextual_asr,speaker_reliability,semantic_reasoning}/`.
They reuse UUID validation, root containment and symlink/Windows-reparse protection.
No arbitrary path parameter or inference endpoint was added. Filenames are allowlisted;
pending directories are ignored. Reading is bounded to 100 directory entries, 16 MiB
per JSON file and 32 MiB total candidate JSON per sidecar lookup.

UUID/content-hash bundle identities and typed serializers are verified. Source
transcript/audio hashes bind observations to the registered canonical sources;
primary diarization hashes are checked when registered. Context validates grounding
and pack hashes. Semantics uses the existing evidence/source validator and requires
its referenced speaker-comparison sidecar when present. Only **one** matching bundle
is accepted. Missing, mismatched or ambiguous results are unavailable; malformed or
unsafe data fails safely. Absolute local paths, configuration/runtime paths, request
payloads, credentials and native probability tables are not projected into these APIs.

Existing canonical routes, the six-stage worker, upload handling, SQLite schema and
the fourteen-download allowlist are unchanged.

## Review derivation

The client deterministically derives read-only entries from canonical data and
available sidecars, in this order:

1. Canonical empty/unresolved evidence references.
2. Global speaker-count mismatch, then every UNCERTAIN utterance.
3. At most five MIXED utterances with at least 20% disagreement, greatest disagreement
   first with utterance ID as a stable tie-break. This is a display rule, not confidence.
4. Contextual hypotheses whose linked refinement did not apply a correction.
5. Experimental semantic REVIEW / UNSUPPORTED checks on canonical items.
6. Experimental missing-coverage observations.

Filters and counts use the same list. Play uses the first referenced retained
utterance's exact bounds; View transcript reuses the existing tab switch, search
clearing, focus and temporary highlight. Canonical item evidence still uses the
existing deterministic resolver endpoint. Nothing is marked reviewed or approved
persistently, and no numerical priority is invented.

## Implementation and operation

Backend additions: `web/diagnostic_schemas.py`, `web/diagnostics.py`, three routes in
`web/app.py`, and `tests/web/test_diagnostics.py`. Frontend additions:
`diagnostics.ts`, `components/trust.tsx`, `test/trust.test.tsx`; existing workspace,
API client, cards, transcript, evidence sheet, CSS and workspace tests were extended.
Reduced-motion transcript navigation now uses automatic rather than smooth scrolling.
No dependency or model installation was required.

Use the existing server/build commands documented in README. Sidecars must already
exist inside the registered meeting workspace; the UI does not generate them or
import arbitrary external directories. Context upload, rerunning observers and
choosing among multiple matching runs are deliberately outside this display phase.

See [validation and screenshots](phase11-validation.md). The release screenshots use
the existing legal, locally synthesized two-voice recording and actual saved VIII–X
outputs. No fake successful model chain is presented as real inference.
