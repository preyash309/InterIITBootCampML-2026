"""Programmatic I–VI workflow with optional Phase VIII context. Frontend stays unchanged."""

from pathlib import Path
from types import SimpleNamespace

from meeting_assistant.asr.config import read_environment
from meeting_assistant.intelligence import extract_meeting_record, save_meeting_record
from meeting_assistant.intelligence.config import IntelligenceConfig
from meeting_assistant.intelligence.models import MeetingRecord
from meeting_assistant.intelligence.workflow import run_upstream


def process_meeting(
    media,
    *,
    context=None,
    contextual_asr=False,
    speaker_reliability=False,
    semantic_reasoning=False,
    output_dir=None,
    environ=None,
) -> MeetingRecord:
    values = read_environment() if environ is None else environ
    args = SimpleNamespace(
        input=Path(media),
        project_glossary=None,
        meeting_context=None,
        context=None,
        context_pack=context,
        context_asr=contextual_asr,
        speaker_reliability=speaker_reliability,
        output_dir=Path(output_dir) if output_dir else None,
    )
    refined, speaker, grounding, canonical, diarization, output = run_upstream(args, values)
    record = extract_meeting_record(
        refined,
        speaker,
        grounding,
        config=IntelligenceConfig.from_env(environ=values),
        canonical_audio_path=canonical,
        diarization=diarization,
    )
    save_meeting_record(record, refined, output)
    from meeting_assistant.semantic_reasoning.service import run_optional

    run_optional(refined, speaker, record, output, environ=values, enabled=semantic_reasoning)
    return record
