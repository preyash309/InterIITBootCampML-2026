"""Reuse the existing resolver; reject mixed records and mismatched source spans."""

from meeting_assistant.diarization.serialization import speaker_transcript_to_json
from meeting_assistant.intelligence.evidence import resolve_meeting_record_evidence
from meeting_assistant.intelligence.serialization import meeting_record_to_json
from meeting_assistant.refinement.service import digest

from .exceptions import SemanticSourceMismatch
from .models import Provenance
from .serialization import fingerprint


def bind_sources(refined, speaker, record, reliability=None):
    try:
        resolve_meeting_record_evidence(record, refined)
        speaker_sha = digest(speaker_transcript_to_json(speaker))
        if (
            refined.source_speaker_transcript_id != speaker.transcript_id
            or refined.source_speaker_sha256 != speaker_sha
        ):
            raise SemanticSourceMismatch("Refined and speaker transcript identities differ.")
        if len(refined.utterances) != len(speaker.utterances):
            raise SemanticSourceMismatch("Source utterance count differs.")
        for u, original in zip(refined.utterances, speaker.utterances):
            if (u.utterance_id, u.speaker_id, u.start, u.end, u.raw_text, u.source_word_refs) != (
                original.id,
                original.speaker_id,
                original.start,
                original.end,
                original.text,
                tuple(w.source_word_reference for w in original.words),
            ):
                raise SemanticSourceMismatch("Source utterance evidence changed.")
        if (
            record.audio is not None
            and abs(record.audio.duration_seconds - speaker.duration_seconds) > 1 / 16000
        ):
            raise SemanticSourceMismatch("Canonical audio binding duration differs.")
        if reliability is not None and (
            reliability.provenance.speaker_transcript_sha256 != speaker_sha
            or record.audio is None
            or reliability.provenance.canonical_audio_sha256 != record.audio.sha256
        ):
            raise SemanticSourceMismatch(
                "Phase IX sidecar belongs to different source audio/transcript."
            )
        return Provenance(
            refined.id,
            record.source_refined_sha256,
            speaker.transcript_id,
            speaker_sha,
            record.id,
            digest(meeting_record_to_json(record)),
            record.audio,
            fingerprint(reliability) if reliability is not None else None,
        )
    except SemanticSourceMismatch:
        raise
    except Exception as exc:
        raise SemanticSourceMismatch("Phase X requires matching typed evidence sources.") from exc


def validate_semantic_source(result, refined, speaker, record, reliability=None):
    if result.provenance != bind_sources(refined, speaker, record, reliability):
        raise SemanticSourceMismatch("Phase X provenance does not match supplied sources.")
    by_id = {u.utterance_id: u for u in refined.utterances}
    for event in result.events:
        if len(event.evidence_utterance_ids) != 1:
            raise SemanticSourceMismatch("Version 1 events require a whole target utterance.")
        source = by_id.get(event.evidence_utterance_ids[0])
        if source is None or (event.text, event.start, event.end, event.speaker_id) != (
            source.refined_text,
            source.start,
            source.end,
            source.speaker_id,
        ):
            raise SemanticSourceMismatch("Event text/timestamps/speaker must resolve exactly.")
        if any(uid not in by_id for uid in event.context_utterance_ids):
            raise SemanticSourceMismatch("Unknown event context evidence.")
    for relation in result.relations:
        if any(uid not in by_id for uid in relation.evidence_utterance_ids):
            raise SemanticSourceMismatch("Unknown relation evidence.")
    for observation in result.relation_observations:
        if any(uid not in by_id for uid in observation.evidence_utterance_ids):
            raise SemanticSourceMismatch("Unknown relation observation evidence.")
    items = {i.id: i for i in record.content.items}
    for verification in result.verification:
        if (
            verification.item_id not in items
            or verification.evidence_utterance_ids
            != items[verification.item_id].evidence_utterance_ids
        ):
            raise SemanticSourceMismatch("Verification evidence changed.")
    events = {e.id: e for e in result.events if e.outcome == "accepted"}
    for observation in result.coverage:
        if observation.event_id not in events or any(
            i not in items for i in observation.matched_record_ids
        ):
            raise SemanticSourceMismatch("Coverage references unknown evidence.")
