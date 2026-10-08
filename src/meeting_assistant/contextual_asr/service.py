"""Optional bounded evidence generation with explicit safe fallback and PCM cleanup."""

import hashlib
import logging
import tempfile
import time
import wave
from contextlib import ExitStack
from pathlib import Path
from uuid import uuid4

from meeting_assistant.asr.audio import ensure_unchanged, inspect_audio
from meeting_assistant.asr.config import ASRConfig
from meeting_assistant.asr.exceptions import ASRError, ASRRateLimitError
from meeting_assistant.diarization.serialization import speaker_transcript_to_json
from meeting_assistant.grounding.serialization import grounding_to_json
from meeting_assistant.grounding.service import validate_grounding_source

from .config import ContextualASRConfig
from .context import sha256
from .exceptions import ContextualASRError, InvalidContextualEvidence
from .hypotheses import link_candidates, window_pass1_text
from .models import ContextualASRHypothesis, ContextualASRResult, SkippedSpan
from .retrieval import build_prompt, retrieve_terms
from .suspicion import detect_suspicious_spans
from .transcribe import GroqContextualASRBackend
from .windows import select_windows

logger = logging.getLogger(__name__)


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def crop_window(audio, window, destination):
    """Copy exact PCM frames, streaming, without touching the canonical input."""
    ensure_unchanged(audio)
    digest = hashlib.sha256()
    count = round(window.end * 16000) - round(window.start * 16000)
    with wave.open(str(audio.path), "rb") as source, wave.open(str(destination), "wb") as output:
        source.setpos(round(window.start * 16000))
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        while count:
            block = source.readframes(min(count, 32000))
            if not block or len(block) % 2:
                raise InvalidContextualEvidence("Canonical window became truncated.")
            digest.update(block)
            output.writeframesraw(block)
            count -= len(block) // 2
    ensure_unchanged(audio)
    return digest.hexdigest()


def validate_native_segments(segments, window):
    previous = window.start
    for segment in segments:
        if segment.start < previous or segment.end > window.end + 1:
            raise InvalidContextualEvidence("Contextual timestamps exceed ordered window bounds.")
        if any(w.start < window.start or w.end > window.end + 1 for w in segment.words):
            raise InvalidContextualEvidence("Contextual words exceed window bounds.")
        previous = segment.start


def contextual_retranscribe(
    audio_path, source, grounding, *, context=None, config=None, backend=None, asr_config=None
):
    started = time.perf_counter()
    config = config or ContextualASRConfig.from_env()
    validate_grounding_source(grounding, source)
    asr_config = asr_config or ASRConfig.from_env()
    audio = inspect_audio(Path(audio_path), asr_config)
    if abs(source.duration_seconds - audio.duration_seconds) > 0.1:
        raise InvalidContextualEvidence("Source duration differs from canonical audio.")
    from .serialization import context_to_json

    spans = (
        detect_suspicious_spans(grounding, context, config)
        if config.enabled and (context is not None or config.use_global_glossary)
        else ()
    )
    windows, skipped = select_windows(spans, grounding, audio.duration_seconds, config)
    skipped = list(skipped)
    hypotheses, cache = [], {}
    halted = None
    for window in windows:
        remaining = config.total_timeout_seconds - (time.perf_counter() - started)
        if halted or remaining <= 0:
            skipped.extend(
                SkippedSpan(identity, halted or "total_timeout")
                for identity in window.grounding_ids
            )
            continue
        prompt, terms = build_prompt(retrieve_terms(window, grounding, context, config))
        if not prompt:
            skipped.extend(
                SkippedSpan(identity, "no_safe_vocabulary") for identity in window.grounding_ids
            )
            continue
        pass1 = window_pass1_text(source, window)
        inference_started = None
        provider_called = False
        segments, pass2, links, conflicts = (), "", (), ()
        status, error_code = "failed", None
        with ExitStack() as workspace:
            try:
                temporary = workspace.enter_context(
                    tempfile.TemporaryDirectory(prefix="meeting-context-asr-")
                )
                frames_digest = crop_window(audio, window, Path(temporary) / "window.wav")
            except (OSError, wave.Error) as exc:
                logger.warning(
                    "contextual_window_failed",
                    extra={"event": "contextual_window_failed", "code": type(exc).__name__},
                )
                skipped.extend(
                    SkippedSpan(identity, "window_preparation_failed")
                    for identity in window.grounding_ids
                )
                continue
            prompt_digest = sha256(prompt)
            try:
                backend = backend or GroqContextualASRBackend(asr_config)
                key = (
                    frames_digest,
                    backend.provider,
                    backend.model,
                    prompt_digest,
                    asr_config.options.language,
                    asr_config.options.temperature,
                    window.start,
                )
                if key not in cache:
                    provider_called = True
                    inference_started = time.perf_counter()
                    cache[key] = backend.transcribe_window(
                        Path(temporary) / "window.wav",
                        start=window.start,
                        duration=window.end - window.start,
                        prompt=prompt,
                        timeout=remaining,
                    )
                pass2, segments = cache[key]
                # Even fakes/custom providers must satisfy immutable native timestamp types.
                from meeting_assistant.asr.models import TranscriptSegment

                if (
                    not isinstance(pass2, str)
                    or not isinstance(segments, tuple)
                    or any(not isinstance(segment, TranscriptSegment) for segment in segments)
                    or (pass2.strip() and (not segments or not any(s.words for s in segments)))
                ):
                    raise InvalidContextualEvidence("Malformed contextual backend output.")
                validate_native_segments(segments, window)
                from meeting_assistant.grounding.normalization import normalize

                if normalize(pass2) != normalize(" ".join(s.text for s in segments)):
                    raise InvalidContextualEvidence("Contextual text and native segments disagree.")
                links, conflicts = link_candidates(window, grounding, segments, terms, pass1, pass2)
                status = "conflict" if conflicts else "candidate" if links else "unmatched"
            except (
                ASRError,
                ContextualASRError,
                OSError,
                ValueError,
                TypeError,
                RuntimeError,
            ) as exc:
                error_code = getattr(exc, "code", "contextual_backend_failed")
                pass2, segments, links, conflicts = "", (), (), ()
                logger.warning(
                    "contextual_asr_failed",
                    extra={"event": "contextual_asr_failed", "code": error_code},
                )
                if isinstance(exc, ASRRateLimitError) or getattr(exc, "status_code", None) in (
                    401,
                    403,
                ):
                    halted = "provider_unavailable"
        hypotheses.append(
            ContextualASRHypothesis(
                f"casr_{len(hypotheses) + 1:06d}",
                window,
                pass1,
                pass2,
                terms,
                prompt_digest,
                frames_digest,
                segments,
                links,
                conflicts,
                getattr(backend, "provider", asr_config.provider),
                getattr(backend, "model", asr_config.model),
                time.perf_counter() - inference_started if inference_started is not None else 0.0,
                status,
                error_code,
                provider_called,
                asr_config.options.language,
                asr_config.options.temperature,
            )
        )
    ensure_unchanged(audio)
    result = ContextualASRResult(
        str(uuid4()),
        sha256(context_to_json(context)) if context else None,
        grounding.grounding_id,
        sha256(grounding_to_json(grounding)),
        sha256(speaker_transcript_to_json(source)),
        file_digest(audio.path),
        config,
        spans,
        tuple(hypotheses),
        tuple(skipped),
        time.perf_counter() - started,
    )
    validate_contextual_source(result, source, grounding)
    return result


def validate_contextual_source(result, source, grounding):
    if not isinstance(result, ContextualASRResult) or (
        result.source_grounding_id != grounding.grounding_id
        or result.source_grounding_sha256 != sha256(grounding_to_json(grounding))
        or result.source_speaker_sha256 != sha256(speaker_transcript_to_json(source))
    ):
        raise InvalidContextualEvidence(
            "Contextual evidence does not match III/IV source artifacts."
        )
    validate_grounding_source(grounding, source)
    record_ids = {r.id for r in grounding.records}
    if any(s.grounding_id not in record_ids for s in (*result.suspicious_spans, *result.skipped)):
        raise InvalidContextualEvidence("Unknown contextual grounding reference.")
    for hypothesis in result.hypotheses:
        from meeting_assistant.grounding.normalization import normalize

        if normalize(hypothesis.pass2_text) != normalize(
            " ".join(s.text for s in hypothesis.segments)
        ):
            raise InvalidContextualEvidence("Saved contextual text and native segments disagree.")
        if (
            hypothesis.window.end > source.duration_seconds
            or set(hypothesis.window.grounding_ids) - record_ids
            or hypothesis.pass1_text != window_pass1_text(source, hypothesis.window)
        ):
            raise InvalidContextualEvidence("Contextual window/source text changed.")
        validate_native_segments(hypothesis.segments, hypothesis.window)
        for term in hypothesis.terms:
            candidates = tuple(
                c
                for r in grounding.records
                if r.id in hypothesis.window.grounding_ids
                for c in r.candidates
                if c.entry_id == term.entry_id
            )
            if not candidates or any(
                c.canonical != term.canonical or c.scope != term.scope for c in candidates
            ):
                raise InvalidContextualEvidence(
                    "Contextual vocabulary does not match supplied candidates."
                )
            expected_sources = tuple(
                dict.fromkeys(
                    reason.removeprefix("context_source:")
                    for c in candidates
                    for reason in c.reasons
                    if reason.startswith("context_source:")
                )
            )
            if set(term.context_source_ids) != set(expected_sources):
                raise InvalidContextualEvidence("Contextual term source IDs changed.")
        _, terms = build_prompt(hypothesis.terms)
        if terms != hypothesis.terms or sha256(build_prompt(terms)[0]) != hypothesis.prompt_sha256:
            raise InvalidContextualEvidence("Contextual prompt provenance changed.")
        links, conflicts = link_candidates(
            hypothesis.window,
            grounding,
            hypothesis.segments,
            hypothesis.terms,
            hypothesis.pass1_text,
            hypothesis.pass2_text,
        )
        if hypothesis.status != "failed" and (
            links != hypothesis.candidate_links or conflicts != hypothesis.conflicts
        ):
            raise InvalidContextualEvidence(
                "Contextual candidate links do not resolve to native words."
            )
