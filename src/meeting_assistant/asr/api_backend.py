"""Groq/OpenAI Whisper adapter. Provider data stops at this boundary."""

import logging
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from .audio import ensure_unchanged, inspect_audio, iter_audio_chunks
from .config import ASRConfig, TranscriptionOptions
from .exceptions import (
    ASRAuthenticationError,
    ASRError,
    ASRTimeout,
    ASRTranscriptionError,
    InvalidTranscript,
)
from .models import (
    ModelInfo,
    ProcessingInfo,
    TranscriptResult,
    TranscriptSegment,
    TranscriptWord,
    finite_number,
    probability,
)
from .transport import request_transcription

logger = logging.getLogger(__name__)


def _word(raw: dict, offset: float) -> TranscriptWord:
    if not isinstance(raw, dict):
        raise InvalidTranscript("Provider word must be an object.")
    start, end = raw["start"], raw["end"]
    finite_number(start, minimum=0)
    finite_number(end, minimum=start)
    return TranscriptWord(raw["word"], start + offset, end + offset, raw.get("probability"))


def parse_response(
    payload: dict, *, offset: float, chunk_duration: float, segment_offset: int
) -> tuple[str, str | None, float | None, tuple[TranscriptSegment, ...]]:
    """Preserve raw text; only assign IDs, global offsets and word-to-segment membership."""
    try:
        text = payload["text"]
        if not isinstance(text, str):
            raise InvalidTranscript("Provider transcript text must be a string.")
        language = payload.get("language")
        if language is not None and not isinstance(language, str):
            raise InvalidTranscript("Provider language must be a string when available.")
        language_probability = payload.get("language_probability")
        probability(language_probability)
        duration = payload.get("duration")
        if duration is not None:
            finite_number(duration, minimum=0)
            if abs(duration - chunk_duration) > max(1.0, chunk_duration * 0.02):
                raise InvalidTranscript(
                    "Provider audio duration is inconsistent with canonical audio."
                )
        raw_segments = payload.get("segments", [])
        raw_words = payload.get("words", [])
        if not isinstance(raw_segments, list) or not isinstance(raw_words, list):
            raise InvalidTranscript("Provider segments and words must be arrays.")
        if text.strip() and (not raw_segments or not raw_words):
            raise InvalidTranscript("Provider omitted requested segment or word timestamps.")
        segments: list[TranscriptSegment] = []
        for index, raw in enumerate(raw_segments, segment_offset + 1):
            if not isinstance(raw, dict):
                raise InvalidTranscript("Provider segment must be an object.")
            start, end = raw["start"], raw["end"]
            finite_number(start, minimum=0)
            finite_number(end, minimum=start)
            if end > chunk_duration + 1.0 or (segments and start + offset < segments[-1].start):
                raise InvalidTranscript("Provider segments are outside audio or out of order.")
            segments.append(
                TranscriptSegment(
                    id=f"seg_{index:06d}",
                    start=start + offset,
                    end=end + offset,
                    text=raw["text"],
                    avg_logprob=raw.get("avg_logprob"),
                    no_speech_probability=raw.get("no_speech_prob"),
                    compression_ratio=raw.get("compression_ratio"),
                )
            )
        words_by_segment: list[list[TranscriptWord]] = [[] for _ in segments]
        previous_start = offset
        cursor = 0
        for raw in raw_words:
            word = _word(raw, offset)
            if word.start < previous_start or word.end > offset + chunk_duration + 1.0:
                raise InvalidTranscript(
                    "Provider word timestamps are outside audio or out of order."
                )
            previous_start = word.start
            if not segments:
                raise InvalidTranscript("Word timestamps have no corresponding segments.")
            midpoint = (word.start + word.end) / 2
            # Provider segment boundaries and word intervals can differ. A word
            # starting in a gap may overlap the following segment even though
            # its midpoint precedes that segment. Preserve all timestamps and
            # choose the best compatible interval, without moving backwards.
            candidates = []
            for position in range(cursor, len(segments)):
                segment = segments[position]
                if segment.start > word.end + 0.5:
                    break
                if word.start >= segment.start - 0.5 and word.end <= segment.end + 0.5:
                    overlap = max(0.0, min(word.end, segment.end) - max(word.start, segment.start))
                    distance = abs(midpoint - (segment.start + segment.end) / 2)
                    candidates.append((overlap, -distance, -position))
            if not candidates:
                raise InvalidTranscript("Provider words cannot be reconciled with segment bounds.")
            cursor = -max(candidates)[2]
            words_by_segment[cursor].append(word)
        segments = [
            replace(segment, words=tuple(words))
            for segment, words in zip(segments, words_by_segment, strict=True)
        ]
        return text, language, language_probability, tuple(segments)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise InvalidTranscript(
            "Provider returned missing or malformed transcript fields."
        ) from exc


class WhisperAPIBackend:
    """Reusable, stateless adapter. Concurrent calls have separate connections/workspaces.

    There is no automatic provider fallback or retry that might incur duplicate charges.
    """

    def __init__(self, config: ASRConfig | None = None) -> None:
        self.config = config if config is not None else ASRConfig.from_env()
        if not self.config.api_key:
            name = "GROQ_API_KEY" if self.config.provider == "groq" else "OPENAI_API_KEY"
            raise ASRAuthenticationError(f"Set {name} in your environment or ignored .env file.")

    def transcribe(
        self, audio_path: Path, options: TranscriptionOptions | None = None
    ) -> TranscriptResult:
        started = time.perf_counter()
        transcript_id = str(uuid4())
        config = self.config
        options = options if options is not None else config.options
        if not isinstance(options, TranscriptionOptions):
            from .exceptions import ASRConfigurationError

            raise ASRConfigurationError("Options must be TranscriptionOptions.")
        context = {"transcript_id": transcript_id, "provider": config.provider}
        logger.info("asr_started", extra={**context, "event": "asr_started"})
        try:
            audio = inspect_audio(audio_path, config)
            segments: list[TranscriptSegment] = []
            texts: list[str] = []
            languages: list[str | None] = []
            probabilities: list[float | None] = []
            inference_seconds = 0.0
            with tempfile.TemporaryDirectory(prefix="meeting-asr-") as temporary:
                for chunk, offset, duration in iter_audio_chunks(audio, config, Path(temporary)):
                    remaining = config.total_timeout_seconds - (time.perf_counter() - started)
                    if remaining <= 0:
                        raise ASRTimeout("ASR job exceeded ASR_TOTAL_TIMEOUT.")
                    ensure_unchanged(audio)
                    inference_started = time.perf_counter()
                    payload = request_transcription(
                        chunk, config, options, min(config.request_timeout_seconds, remaining)
                    )
                    inference_seconds += time.perf_counter() - inference_started
                    text, language, confidence, parsed = parse_response(
                        payload,
                        offset=offset,
                        chunk_duration=duration,
                        segment_offset=len(segments),
                    )
                    texts.append(text)
                    languages.append(language)
                    probabilities.append(confidence)
                    segments.extend(parsed)
                    logger.info(
                        "asr_chunk_completed",
                        extra={**context, "event": "asr_chunk_completed", "chunk": len(texts)},
                    )
            ensure_unchanged(audio)
            result = TranscriptResult(
                transcript_id=transcript_id,
                language=languages[0] if len(set(languages)) == 1 else None,
                # No invented or aggregated detection confidence across requests.
                language_probability=probabilities[0] if len(probabilities) == 1 else None,
                duration_seconds=audio.duration_seconds,
                text="\n".join(texts),
                segments=tuple(segments),
                model_info=ModelInfo(
                    provider=config.provider,
                    model=config.model,
                    requested_language=options.language,
                    temperature=options.temperature,
                ),
                processing_info=ProcessingInfo(
                    model_load_seconds=0.0,
                    transcription_seconds=inference_seconds,
                    total_seconds=time.perf_counter() - started,
                    chunk_count=len(texts),
                ),
            )
            logger.info("asr_completed", extra={**context, "event": "asr_completed"})
            return result
        except ASRError as exc:
            logger.warning(
                "asr_failed", extra={**context, "event": "asr_failed", "error_code": exc.code}
            )
            raise
        except OSError as exc:
            raise ASRTranscriptionError(
                "Cannot create or clean the ASR temporary workspace."
            ) from exc
