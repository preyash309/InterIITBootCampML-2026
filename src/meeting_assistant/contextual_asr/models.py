"""Immutable, bounded context and re-ASR evidence; no correction or identity inference."""

import math
import re
from dataclasses import dataclass

from meeting_assistant.asr.models import TranscriptSegment
from meeting_assistant.grounding.normalization import normalize

from .config import ContextualASRConfig
from .exceptions import InvalidContext, InvalidContextualEvidence


def text(value, limit=160, *, empty=False, multiline=False):
    if (
        not isinstance(value, str)
        or len(value) > limit
        or (not empty and not value.strip())
        or any(ord(c) < 32 and not (multiline and c in "\r\n\t") for c in value)
    ):
        raise InvalidContext("Context text must be bounded printable text.")


def typed(values, kind):
    if not isinstance(values, tuple) or any(not isinstance(v, kind) for v in values):
        raise InvalidContextualEvidence("Evidence must use immutable typed tuples.")


def number(value, minimum=0, maximum=math.inf):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not (math.isfinite(value) and minimum <= value <= maximum)
    ):
        raise InvalidContextualEvidence("Invalid finite contextual measurement.")


def digest(value):
    if not isinstance(value, str) or re.fullmatch(r"[a-f0-9]{64}", value) is None:
        raise InvalidContextualEvidence("Provenance requires a SHA-256 digest.")


@dataclass(frozen=True)
class ContextSource:
    id: str
    kind: str
    label: str
    sha256: str

    def __post_init__(self):
        text(self.id, 100)
        text(self.label, 160)
        digest(self.sha256)
        if self.kind not in (
            "title",
            "agenda",
            "description",
            "glossary",
            "participants",
            "document",
        ):
            raise InvalidContext("Unknown context source kind.")


@dataclass(frozen=True)
class ContextTerm:
    id: str
    canonical: str
    aliases: tuple[str, ...] = ()
    category: str = "technology"
    scope: str = "meeting"
    source_ids: tuple[str, ...] = ()
    priority: float = 0.8

    def __post_init__(self):
        text(self.id, 100)
        text(self.canonical)
        typed(self.aliases, str)
        typed(self.source_ids, str)
        if len(self.aliases) > 12 or not self.source_ids or len(self.canonical.split()) > 6:
            raise InvalidContext("Terms need bounded aliases, source IDs and term-shaped text.")
        for alias in self.aliases:
            text(alias)
        if (
            not normalize(self.canonical)
            or not isinstance(self.category, str)
            or re.fullmatch(r"[a-z][a-z0-9_]{0,40}", self.category) is None
        ):
            raise InvalidContext("Invalid canonical term/category.")
        if self.scope not in ("project", "meeting"):
            raise InvalidContext("Context term scope must be project or meeting.")
        number(self.priority, 0, 1)

    @property
    def lookup_form(self):
        return normalize(self.canonical)


@dataclass(frozen=True)
class MeetingContextPack:
    title: str | None
    agenda: tuple[str, ...]
    description: str | None
    participant_names: tuple[str, ...]
    explicit_terms: tuple[ContextTerm, ...]
    extracted_terms: tuple[ContextTerm, ...]
    sources: tuple[ContextSource, ...]
    version: str = "meeting_context_v1"

    def __post_init__(self):
        if self.version != "meeting_context_v1":
            raise InvalidContext("Unknown context pack version.")
        if self.title is not None:
            text(self.title, 500)
        if self.description is not None:
            text(self.description, 8000, multiline=True)
        typed(self.agenda, str)
        typed(self.participant_names, str)
        typed(self.explicit_terms, ContextTerm)
        typed(self.extracted_terms, ContextTerm)
        typed(self.sources, ContextSource)
        if len(self.agenda) > 32 or len(self.participant_names) > 100 or len(self.terms) > 256:
            raise InvalidContext("Context pack exceeds count limits.")
        for item in self.agenda:
            text(item, 1000)
        for name in self.participant_names:
            text(name, 120)
        if len(self.sources) > 32 or len({s.id for s in self.sources}) != len(self.sources):
            raise InvalidContext("Context sources must be bounded and unique.")
        if len({t.id for t in self.terms}) != len(self.terms) or len(
            {(t.scope, t.lookup_form) for t in self.terms}
        ) != len(self.terms):
            raise InvalidContext("Context term IDs and scoped names must be unique.")
        if any(set(t.source_ids) - {s.id for s in self.sources} for t in self.terms):
            raise InvalidContext("Term source does not exist in this pack.")

    @property
    def terms(self):
        return self.explicit_terms + self.extracted_terms


@dataclass(frozen=True)
class SuspiciousSpan:
    grounding_id: str
    priority: float
    signals: tuple[str, ...]

    def __post_init__(self):
        if re.fullmatch(r"grnd_\d{6,}", self.grounding_id) is None:
            raise InvalidContextualEvidence("Invalid grounding reference.")
        number(self.priority, 0, 1)
        typed(self.signals, str)


@dataclass(frozen=True)
class AudioWindow:
    id: str
    start: float
    end: float
    grounding_ids: tuple[str, ...]

    def __post_init__(self):
        text(self.id, 50)
        number(self.start)
        number(self.end, self.start)
        if self.end == self.start:
            raise InvalidContextualEvidence("Audio window must contain frames.")
        typed(self.grounding_ids, str)
        if not self.grounding_ids or len(set(self.grounding_ids)) != len(self.grounding_ids):
            raise InvalidContextualEvidence("Window references must be nonempty and unique.")


@dataclass(frozen=True)
class RetrievedTerm:
    entry_id: str
    canonical: str
    scope: str
    context_source_ids: tuple[str, ...]

    def __post_init__(self):
        text(self.entry_id, 120)
        text(self.canonical)
        typed(self.context_source_ids, str)
        if self.scope not in ("global", "project", "meeting"):
            raise InvalidContextualEvidence("Unknown vocabulary scope.")


@dataclass(frozen=True)
class CandidateLink:
    grounding_id: str
    candidate_entry_id: str
    start: float
    end: float

    def __post_init__(self):
        text(self.grounding_id, 50)
        text(self.candidate_entry_id, 120)
        number(self.start)
        number(self.end, self.start)


@dataclass(frozen=True)
class ContextualASRHypothesis:
    id: str
    window: AudioWindow
    pass1_text: str
    pass2_text: str
    terms: tuple[RetrievedTerm, ...]
    prompt_sha256: str
    audio_frames_sha256: str
    segments: tuple[TranscriptSegment, ...]
    candidate_links: tuple[CandidateLink, ...]
    conflicts: tuple[str, ...]
    provider: str
    model: str
    latency_seconds: float
    status: str
    error_code: str | None = None
    provider_called: bool = True
    requested_language: str | None = "en"
    temperature: float = 0.0

    def __post_init__(self):
        if not isinstance(self.window, AudioWindow):
            raise InvalidContextualEvidence("Hypothesis needs a typed window.")
        text(self.id, 50)
        if any(
            not isinstance(v, str) or len(v) > 10000 for v in (self.pass1_text, self.pass2_text)
        ):
            raise InvalidContextualEvidence("Hypothesis text exceeds bounds.")
        typed(self.terms, RetrievedTerm)
        typed(self.segments, TranscriptSegment)
        typed(self.candidate_links, CandidateLink)
        typed(self.conflicts, str)
        text(self.provider, 100)
        text(self.model, 120)
        if self.requested_language is not None:
            text(self.requested_language, 30)
        number(self.temperature, 0, 1)
        if type(self.provider_called) is not bool:
            raise InvalidContextualEvidence("Provider call flag must be boolean.")
        digest(self.prompt_sha256)
        digest(self.audio_frames_sha256)
        number(self.latency_seconds)
        if self.status not in ("candidate", "unmatched", "conflict", "failed"):
            raise InvalidContextualEvidence("Unknown hypothesis status.")
        if self.status == "failed" and (
            not self.error_code
            or self.segments
            or self.candidate_links
            or self.pass2_text
            or self.conflicts
        ):
            raise InvalidContextualEvidence("Failed hypotheses cannot contain usable evidence.")
        if self.conflicts and self.candidate_links:
            raise InvalidContextualEvidence("Protected conflicts cannot promote candidates.")
        expected = (
            "conflict" if self.conflicts else "candidate" if self.candidate_links else "unmatched"
        )
        if self.status != "failed" and (self.status != expected or self.error_code is not None):
            raise InvalidContextualEvidence("Hypothesis status disagrees with evidence.")
        if len(self.terms) > 8 or len({t.entry_id for t in self.terms}) != len(self.terms):
            raise InvalidContextualEvidence("Vocabulary must be bounded and unique.")


@dataclass(frozen=True)
class SkippedSpan:
    grounding_id: str
    reason: str

    def __post_init__(self):
        text(self.grounding_id, 50)
        text(self.reason, 100)


@dataclass(frozen=True)
class ContextualASRResult:
    result_id: str
    context_sha256: str | None
    source_grounding_id: str
    source_grounding_sha256: str
    source_speaker_sha256: str
    canonical_audio_sha256: str
    config: ContextualASRConfig
    suspicious_spans: tuple[SuspiciousSpan, ...]
    hypotheses: tuple[ContextualASRHypothesis, ...]
    skipped: tuple[SkippedSpan, ...]
    total_seconds: float
    schema_version: str = "1.0"

    def __post_init__(self):
        from uuid import UUID

        try:
            UUID(self.result_id)
            UUID(self.source_grounding_id)
        except (ValueError, TypeError, AttributeError) as exc:
            raise InvalidContextualEvidence("Evidence identifiers must be UUIDs.") from exc
        for value in (
            self.source_grounding_sha256,
            self.source_speaker_sha256,
            self.canonical_audio_sha256,
        ):
            digest(value)
        if self.context_sha256 is not None:
            digest(self.context_sha256)
        typed(self.suspicious_spans, SuspiciousSpan)
        typed(self.hypotheses, ContextualASRHypothesis)
        typed(self.skipped, SkippedSpan)
        number(self.total_seconds)
        if not isinstance(self.config, ContextualASRConfig) or self.schema_version != "1.0":
            raise InvalidContextualEvidence("Invalid contextual policy/schema.")
        if len(self.hypotheses) > self.config.max_windows:
            raise InvalidContextualEvidence("Call budget exceeded.")
        if not self.config.enabled and self.hypotheses:
            raise InvalidContextualEvidence("Disabled enhancement cannot contain hypotheses.")
        if len({h.window.id for h in self.hypotheses}) != len(self.hypotheses):
            raise InvalidContextualEvidence("Duplicate windows are forbidden.")
        if len({h.id for h in self.hypotheses}) != len(self.hypotheses):
            raise InvalidContextualEvidence("Duplicate hypotheses are forbidden.")
        if any(
            h.window.end - h.window.start > self.config.max_window_seconds + 1e-6
            for h in self.hypotheses
        ):
            raise InvalidContextualEvidence("Window duration budget exceeded.")
        if self.audio_seconds > self.config.max_total_audio_seconds + 1e-6:
            raise InvalidContextualEvidence("Contextual audio budget exceeded.")

    @property
    def provider_calls(self):
        return sum(h.provider_called for h in self.hypotheses)

    @property
    def audio_seconds(self):
        return sum(h.window.end - h.window.start for h in self.hypotheses if h.provider_called)
