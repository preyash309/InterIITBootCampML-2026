"""Read saved, source-bound sidecars only. No backend/model is instantiated here."""

import logging
import re
from dataclasses import asdict

from meeting_assistant.contextual_asr.serialization import (
    context_from_json,
    context_to_json,
    contextual_asr_from_json,
)
from meeting_assistant.contextual_asr.service import validate_contextual_source
from meeting_assistant.diarization.serialization import speaker_transcript_to_json
from meeting_assistant.grounding.serialization import grounding_from_json
from meeting_assistant.intelligence.serialization import meeting_record_from_json
from meeting_assistant.refinement.serialization import refined_from_json
from meeting_assistant.refinement.service import digest
from meeting_assistant.semantic_reasoning.evidence import validate_semantic_source
from meeting_assistant.semantic_reasoning.serialization import fingerprint as semantic_fingerprint
from meeting_assistant.semantic_reasoning.serialization import from_json
from meeting_assistant.speaker_reliability.serialization import (
    fingerprint as reliability_fingerprint,
)
from meeting_assistant.speaker_reliability.serialization import reliability_from_json

from .diagnostic_schemas import (
    ContextualASRResult,
    MeetingEvent,
    SemanticResult,
    Sidecar,
    SpeakerObservation,
    SpeakerReliabilityResult,
)
from .registry import safe_path

logger = logging.getLogger(__name__)
FILES = {
    "contextual_asr": ("contextual_asr.json", contextual_asr_from_json),
    "speaker_reliability": ("speaker_reliability.json", reliability_from_json),
    "semantic_reasoning": ("semantic_result.json", from_json),
}


def read(path):
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("Diagnostic exceeds read budget.")
    with path.open(encoding="utf-8") as stream:
        return stream.read(16 * 1024 * 1024 + 1)


def label(value):
    """Source labels may have Windows or POSIX separators; never expose local paths."""
    return value.replace("\\", "/").rsplit("/", 1)[-1][:160]


class DiagnosticReader:
    def __init__(self, store, identifier):
        store.get(identifier)  # Validated UUID and registered job before touching disk.
        self.store, self.identifier = store, identifier
        self.refined = refined_from_json(read(store.artifact(identifier, "refined_json")))
        from meeting_assistant.diarization.serialization import speaker_transcript_from_json

        self.speaker = speaker_transcript_from_json(
            read(store.artifact(identifier, "speaker_json"))
        )
        self.record = meeting_record_from_json(read(store.artifact(identifier, "meeting_json")))

    def candidates(self, kind):
        relative = f"{self.identifier}/transcripts/{kind}"
        root = safe_path(self.store.root, relative, must_exist=False)
        if not root.exists():
            return []
        entries = list(root.iterdir())
        if len(entries) > 100:
            raise ValueError("Too many diagnostic bundles.")
        name, parser = FILES[kind]
        results = []
        total_size = 0
        for entry in sorted(entries):
            # Only completed UUID/hash bundles; staging files are not results.
            if not re.fullmatch(r"[a-f0-9]{64}|[a-f0-9-]{36}", entry.name):
                continue
            path = safe_path(self.store.root, f"{relative}/{entry.name}/{name}")
            total_size += path.stat().st_size
            if total_size > 32 * 1024 * 1024:
                raise ValueError("Diagnostic discovery exceeds total read budget.")
            result = parser(read(path))
            expected = (
                result.result_id
                if kind == "contextual_asr"
                else reliability_fingerprint(result)
                if kind == "speaker_reliability"
                else semantic_fingerprint(result)
            )
            if entry.name != expected:
                raise ValueError("Diagnostic bundle identity changed.")
            results.append((path.parent, result))
        return results

    def matches(self, kind, result):
        speaker_hash = digest(speaker_transcript_to_json(self.speaker))
        audio_hash = self.record.audio.sha256 if self.record.audio else None
        if kind == "speaker_reliability":
            if (
                result.provenance.speaker_transcript_sha256,
                result.provenance.canonical_audio_sha256,
            ) != (speaker_hash, audio_hash):
                return False
            ids = {u.id: u.speaker_id for u in self.speaker.utterances}
            if "diarization_json" in self.store.artifacts(self.identifier):
                from meeting_assistant.diarization.serialization import (
                    diarization_from_json,
                    diarization_to_json,
                )

                primary = diarization_from_json(
                    read(self.store.artifact(self.identifier, "diarization_json"))
                )
                if (
                    digest(diarization_to_json(primary))
                    != result.provenance.primary_diarization_sha256
                ):
                    return False
            return all(
                u.utterance_id in ids and u.primary_speaker == ids[u.utterance_id]
                for u in result.utterances
            )
        if kind == "contextual_asr":
            if result.canonical_audio_sha256 != audio_hash:
                return False
            grounding = grounding_from_json(
                read(self.store.artifact(self.identifier, "grounding_json"))
            )
            validate_contextual_source(result, self.speaker, grounding)
            return (result.source_grounding_id, result.source_grounding_sha256) == (
                self.refined.grounding_result_id,
                self.refined.grounding_sha256,
            )
        reliability = None
        if result.provenance.reliability_sha256:
            from meeting_assistant.speaker_reliability.serialization import fingerprint

            reliability = next(
                (
                    r
                    for _, r in self.candidates("speaker_reliability")
                    if fingerprint(r) == result.provenance.reliability_sha256
                    and self.matches("speaker_reliability", r)
                ),
                None,
            )
            if reliability is None:
                return False
        validate_semantic_source(result, self.refined, self.speaker, self.record, reliability)
        return True

    def get(self, kind):
        try:
            candidates = self.candidates(kind)
            if not candidates:
                return Sidecar(state="not_generated", message="This observation was not generated.")
            matches = []
            for directory, result in candidates:
                try:
                    if self.matches(kind, result):
                        matches.append((directory, result))
                except Exception:
                    continue  # A valid sidecar from another source is never presented.
            if len(matches) != 1:
                return Sidecar(
                    state="unavailable",
                    message="No unique observation matches this meeting's retained sources.",
                )
            directory, result = matches[0]
            if getattr(result, "availability", "available") in ("unavailable", "disabled"):
                return Sidecar(
                    state="unavailable",
                    message="The saved observer did not produce an available result.",
                )
            data = self.project(kind, directory, result)
            return Sidecar(
                state="available",
                message="Read from saved observations; no inference was run.",
                data=data,
            )
        except Exception as exc:
            logger.warning(
                "sidecar_read_failed", extra={"sidecar": kind, "exception_type": type(exc).__name__}
            )
            return Sidecar(
                state="failed", message="Saved observation data could not be read safely."
            )

    def project(self, kind, directory, result):
        if kind == "speaker_reliability":
            a = result.comparison.alignment
            fields = set(SpeakerObservation.model_fields)
            return SpeakerReliabilityResult(
                primary_model=result.provenance.primary_model,
                secondary_model=result.secondary.model,
                primary_count=a.primary_count,
                secondary_count=a.secondary_count,
                count_mismatch=a.count_mismatch,
                possible_merges=[p for p, _ in a.possible_merges],
                utterances=[
                    {k: v for k, v in asdict(u).items() if k in fields} for u in result.utterances
                ],
            )
        if kind == "contextual_asr":
            relative = (directory / "meeting_context.json").relative_to(self.store.root).as_posix()
            path = safe_path(self.store.root, relative, must_exist=False)
            context = context_from_json(read(path)) if path.exists() else None
            if context and digest(context_to_json(context)) != result.context_sha256:
                raise ValueError("Context binding differs.")
            sources = {s.id: label(s.label) for s in context.sources} if context else {}
            audits = []
            for h in result.hypotheses:
                links = {link.grounding_id for link in h.candidate_links}
                edits = [e for e in self.refined.edit_log if e.grounding_record_id in links]
                audits.append(
                    dict(
                        id=h.id,
                        utterance_ids=[
                            u.utterance_id
                            for u in self.refined.utterances
                            if u.start < h.window.end and u.end > h.window.start
                        ],
                        start=h.window.start,
                        end=h.window.end,
                        pass1_text=h.pass1_text,
                        pass2_text=h.pass2_text,
                        candidates=[t.canonical for t in h.terms],
                        sources=sorted(
                            {
                                sources.get(s, "Meeting context")
                                for t in h.terms
                                for s in t.context_source_ids
                            }
                        ),
                        provider=h.provider,
                        model=h.model,
                        status=h.status,
                        outcomes=[f"{e.action}: {e.validation_status}" for e in edits]
                        or ["No correction applied"],
                        unresolved=not edits
                        or any(e.validation_status != "applied" for e in edits),
                    )
                )
            return ContextualASRResult(
                title=context.title if context else None,
                terms=[t.canonical for t in (*context.explicit_terms, *context.extracted_terms)]
                if context
                else [],
                participants=list(context.participant_names) if context else [],
                sources=list(sources.values()),
                hypotheses=audits,
            )
        titles = {i.id: i.representative_text for i in result.event_graph.issue_threads}
        return SemanticResult(
            availability=result.availability,
            models=sorted(
                {d.model for e in result.events if (d := e.decision)}
                | {d.model for v in result.verification for d in v.dimensions}
            ),
            events=[
                {k: v for k, v in asdict(e).items() if k in MeetingEvent.model_fields}
                for e in result.events
                if e.outcome == "accepted"
            ],
            relations=[
                {k: v for k, v in asdict(r).items() if k != "decision"} for r in result.relations
            ],
            decision_evolution=[
                dict(
                    issue_id=e.issue_id,
                    title=titles.get(e.issue_id, "Observed issue"),
                    ordered_event_ids=e.ordered_event_ids,
                    current_decision_ids=e.current_decision_ids,
                    historical_decision_ids=e.historical_decision_ids,
                )
                for e in result.decision_evolution
            ],
            verification=[
                dict(
                    item_id=v.item_id,
                    status=v.status,
                    evidence_utterance_ids=v.evidence_utterance_ids,
                    dimensions=[
                        dict(
                            question=d.question_id,
                            choice=d.choice,
                            provider=d.provider,
                            model=d.model,
                        )
                        for d in v.dimensions
                    ],
                )
                for v in result.verification
            ],
            coverage=[
                dict(
                    event_id=c.event_id,
                    event_type=c.event_type,
                    status=c.status,
                    matched_record_ids=c.matched_record_ids,
                )
                for c in result.coverage
            ],
        )
