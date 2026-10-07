"""Full I–V developer workflow, or reuse saved III/IV evidence without audio upload."""

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

from meeting_assistant.asr.config import read_environment

from .config import RefinementConfig
from .exceptions import RefinementError
from .groq_backend import GroqRefinerBackend
from .serialization import save_refined_transcript
from .service import refine_transcript
from .validation import protected_signature


def read_json(path: Path) -> str:
    with path.open(encoding="utf-8") as stream:
        text = stream.read(128 * 1024**2 + 1)
    if len(text) > 128 * 1024**2:
        raise RefinementError("Evidence JSON exceeds the 128 MiB character limit.")
    return text


def _upstream(args, values):
    from meeting_assistant.asr import ASRConfig, transcribe_audio
    from meeting_assistant.asr.api_backend import WhisperAPIBackend
    from meeting_assistant.asr.serialization import save_transcript
    from meeting_assistant.audio import AudioIngestionConfig, ingest_audio
    from meeting_assistant.diarization import DiarizationConfig, PyannoteBackend
    from meeting_assistant.diarization.reconciliation import reconcile_transcript
    from meeting_assistant.diarization.serialization import save_speaker_transcript
    from meeting_assistant.grounding import (
        GroundingConfig,
        GroundingRetriever,
        ground_transcript,
        load_glossary,
    )
    from meeting_assistant.grounding.glossary import read_meeting_context
    from meeting_assistant.grounding.serialization import save_grounding

    diar_config = DiarizationConfig.from_env(environ=values)
    diar_backend = PyannoteBackend(diar_config)
    asr_backend = WhisperAPIBackend(ASRConfig.from_env(environ=values))
    glossary = load_glossary(
        project_glossary=args.project_glossary,
        meeting_entries=read_meeting_context(args.meeting_context) if args.meeting_context else (),
    )
    engine = GroundingRetriever(glossary, GroundingConfig.from_env(environ=values))
    # Check existing local prerequisites before uploading paid audio.
    diar_backend.load_model()
    engine.prepare()
    overrides = {
        name: values[key]
        for name, key in (
            ("ffmpeg_path", "FFMPEG_PATH"),
            ("ffprobe_path", "FFPROBE_PATH"),
            ("work_dir", "AUDIO_WORK_DIR"),
        )
        if values.get(key)
    }
    audio = ingest_audio(args.input, config=replace(AudioIngestionConfig.from_env(), **overrides))
    output = args.output_dir or audio.canonical_audio_path.parent.parent / "transcripts"
    raw = transcribe_audio(audio.canonical_audio_path, backend=asr_backend)
    raw_files = save_transcript(raw, output)
    diarization = diar_backend.diarize(audio.canonical_audio_path)
    speaker = reconcile_transcript(raw, diarization, config=diar_config.reconciliation)
    speaker_files = save_speaker_transcript(speaker, diarization, output)
    grounding = ground_transcript(speaker, retriever=engine)
    grounding_files = save_grounding(grounding, output)
    print(f"Canonical WAV: {audio.canonical_audio_path}")
    print(
        f"Raw JSON: {raw_files.json_path}\nSpeaker JSON: {speaker_files.json_path}\nGrounding JSON: {grounding_files.json_path}"
    )
    return speaker, grounding, output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Conservative terminology refinement; no meeting intelligence."
    )
    parser.add_argument(
        "input", type=Path, help="Meeting media, or saved speaker JSON with --grounding."
    )
    parser.add_argument(
        "--grounding",
        type=Path,
        help="Reuse matching Phase IV JSON; no audio upload, ASR or diarization.",
    )
    parser.add_argument("--project-glossary", type=Path)
    parser.add_argument("--meeting-context", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--model")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.grounding and (args.project_glossary or args.meeting_context):
        parser.error(
            "Saved grounding already contains its glossary evidence; do not add new layers."
        )
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    try:
        values = read_environment(args.env_file)
        config = RefinementConfig.from_env(environ=values)
        if args.model:
            config = replace(config, model=args.model)
        if args.grounding:
            from meeting_assistant.diarization.serialization import speaker_transcript_from_json
            from meeting_assistant.grounding.serialization import grounding_from_json

            speaker = speaker_transcript_from_json(read_json(args.input))
            grounding = grounding_from_json(read_json(args.grounding))
            output = args.output_dir or Path("transcripts")
            backend = None  # Empty grounding can be copied without credentials.
        else:
            backend = GroqRefinerBackend(config)
            speaker, grounding, output = _upstream(args, values)
        result = refine_transcript(speaker, grounding, backend=backend, config=config)
        files = save_refined_transcript(result, output)
    except Exception as exc:
        from meeting_assistant.asr.exceptions import ASRError
        from meeting_assistant.audio.exceptions import AudioIngestionError
        from meeting_assistant.diarization.exceptions import DiarizationError
        from meeting_assistant.grounding.exceptions import GroundingError

        if isinstance(
            exc, (RefinementError, ASRError, AudioIngestionError, DiarizationError, GroundingError)
        ):
            print(f"Refinement failed [{exc.code}]: {exc}", file=sys.stderr)
            return 1
        if isinstance(exc, OSError):
            print("Refinement failed: check input/output access and disk space.", file=sys.stderr)
            return 1
        raise
    info = result.processing_info
    print(f"Refinement completed - {result.model_info.provider} / {result.model_info.model}")
    print(
        f"Target utterances: {info.utterance_count}; sent: {info.sent_utterance_count}; grounding records: {len(grounding.records)}"
    )
    print(
        "Decisions: "
        + "; ".join(
            f"{action}={sum(e.action == action for e in result.edit_log)}"
            for action in ("KEEP", "REPLACE", "UNCERTAIN")
        )
    )
    print(
        f"Applied: {sum(e.validation_status == 'applied' for e in result.edit_log)}; rejected: {sum(e.validation_status == 'rejected' for e in result.edit_log)}; applied protected-content violations: {sum(protected_signature(u.raw_text) != protected_signature(u.refined_text) for u in result.utterances)}"
    )
    print(
        f"API calls: {len(info.calls)}; total refinement: {info.total_seconds:.3f} s; tokens: {info.usage('total_tokens')}"
    )
    print(f"JSON: {files.json_path}\nText: {files.text_path}\nEdit log: {files.edit_log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
