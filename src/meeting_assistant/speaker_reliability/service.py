"""Shadow assessment with explicit source linkage and safe optional failure."""

import logging
from collections import Counter
from dataclasses import replace
from pathlib import Path
from time import perf_counter

from meeting_assistant.diarization.audio import ensure_unchanged, inspect_audio
from meeting_assistant.diarization.serialization import (
    diarization_to_json,
    speaker_transcript_to_json,
)

from .alignment import align_speakers
from .comparison import compare_intervals
from .config import SpeakerReliabilityConfig
from .exceptions import InvalidReliability, SpeakerReliabilityError
from .models import (
    DiarizationComparisonResult,
    Provenance,
    SecondaryDiarizationResult,
    SpeakerReliabilityResult,
)
from .reliability import reliability_records
from .serialization import fingerprint

logger = logging.getLogger(__name__)


def source_provenance(primary, speaker, config):
    import hashlib

    if (
        primary.diarization_id != speaker.diarization_id
        or primary.duration_seconds != speaker.duration_seconds
    ):
        raise InvalidReliability("Primary diarization and speaker transcript do not match.")
    return Provenance(
        primary.audio_sha256,
        hashlib.sha256(diarization_to_json(primary).encode("utf-8")).hexdigest(),
        hashlib.sha256(speaker_transcript_to_json(speaker).encode("utf-8")).hexdigest(),
        None,
        primary.model_info.model,
        primary.model_info.revision,
        primary.model_info.package_version,
        primary.model_info.device,
        primary.processing_info.diarization_seconds,
        config,
    )


def compare_diarization(primary, speaker_transcript, secondary, *, config=None):
    """Pure comparison API; all three records are immutable and remain unchanged."""
    started = perf_counter()
    config = config or SpeakerReliabilityConfig()
    provenance = source_provenance(primary, speaker_transcript, config)
    if not isinstance(secondary, SecondaryDiarizationResult):
        raise InvalidReliability("Secondary backend returned an untyped result.")
    if (
        secondary.audio_sha256 != primary.audio_sha256
        or abs(secondary.duration_seconds - primary.duration_seconds) > 1e-6
    ):
        raise InvalidReliability("Secondary result belongs to a different canonical recording.")
    if secondary.speaker_count > config.max_speakers:
        raise InvalidReliability("Secondary output exceeds configured speaker capacity.")
    alignment = align_speakers(primary, secondary, config)
    intervals = compare_intervals(primary, secondary, alignment)
    states = Counter()
    for item in intervals:
        states[item.state] += item.end - item.start
    silence = states["SILENCE"]
    speech = primary.duration_seconds - silence
    agreement = (states["AGREE"] + states["OVERLAP_AGREE"]) / speech if speech else 0.0
    comparison = DiarizationComparisonResult(
        alignment,
        intervals,
        agreement,
        speech,
        silence,
        tuple(sorted(states.items())),
        perf_counter() - started,
    )
    words, utterances = reliability_records(
        speaker_transcript, primary, secondary, comparison, config
    )
    warnings = ("speaker_count_mismatch",) if alignment.count_mismatch else ()
    return SpeakerReliabilityResult(
        "available",
        replace(provenance, secondary_result_sha256=fingerprint(secondary)),
        secondary,
        comparison,
        words,
        utterances,
        warnings,
        perf_counter() - started,
    )


def assess_speaker_reliability(
    canonical_audio_path, primary, speaker_transcript, *, backend=None, config=None
):
    """Explicit invocation runs shadow inference; pipeline opt-in is handled by callers.

    Source linkage errors are caller defects and fail explicitly. Optional runtime/media
    errors produce an unavailable sidecar without rewriting any primary evidence.
    """
    started = perf_counter()
    config = config or SpeakerReliabilityConfig()
    provenance = source_provenance(primary, speaker_transcript, config)
    try:
        audio = inspect_audio(
            Path(canonical_audio_path), max_duration_seconds=config.max_duration_seconds
        )
        if audio.sha256 != primary.audio_sha256:
            raise InvalidReliability(
                "Canonical recording fingerprint does not match the primary result."
            )
        if backend is None:
            from .secondary import SortformerBackend

            backend = SortformerBackend(config)
        secondary = backend.diarize(audio.path)
        ensure_unchanged(audio)
        result = compare_diarization(primary, speaker_transcript, secondary, config=config)
        logger.info(
            "speaker_reliability_completed",
            extra={"agreement_fraction": result.comparison.agreement_fraction},
        )
        return replace(result, total_seconds=perf_counter() - started)
    except Exception as exc:
        logger.warning("speaker_reliability_unavailable", exc_info=True)
        # No raw path/token/model-library traceback in user-facing sidecars.
        return SpeakerReliabilityResult(
            "unavailable",
            provenance,
            None,
            None,
            (),
            (),
            (
                f"Secondary diarization unavailable ({type(exc).__name__}); primary evidence retained.",
            ),
            perf_counter() - started,
        )


def run_optional(
    canonical_audio_path, primary, speaker_transcript, output_dir, *, environ, enabled=False
):
    """Small integration seam. Default paths do not import/load any secondary libraries."""
    if not enabled and environ.get("SPEAKER_RELIABILITY_ENABLED", "false").lower() not in (
        "true",
        "1",
    ):
        return None
    from .serialization import save_speaker_reliability

    try:
        try:
            config = SpeakerReliabilityConfig.from_env(environ=environ)
        except SpeakerReliabilityError as exc:
            logger.warning("speaker_reliability_configuration_invalid", exc_info=True)
            result = SpeakerReliabilityResult(
                "unavailable",
                source_provenance(primary, speaker_transcript, SpeakerReliabilityConfig()),
                None,
                None,
                (),
                (),
                (
                    f"Secondary configuration unavailable ({type(exc).__name__}); primary evidence retained.",
                ),
                0,
            )
        else:
            result = assess_speaker_reliability(
                canonical_audio_path, primary, speaker_transcript, config=config
            )
        return save_speaker_reliability(result, output_dir)
    except Exception:
        logger.warning("speaker_reliability_optional_step_failed", exc_info=True)
        return None
