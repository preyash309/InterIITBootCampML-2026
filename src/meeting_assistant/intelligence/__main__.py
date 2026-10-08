"""Developer extraction/evidence CLI; no frontend or semantic verifier."""

import argparse
import json
import logging
import sys
from dataclasses import asdict, replace
from pathlib import Path

from meeting_assistant.asr.config import read_environment
from meeting_assistant.refinement.__main__ import read_json
from meeting_assistant.refinement.exceptions import RefinementError
from meeting_assistant.refinement.serialization import refined_from_json

from .config import IntelligenceConfig
from .evidence import extract_evidence_clip, get_item_evidence
from .exceptions import IntelligenceError
from .groq_backend import GroqMeetingIntelligenceBackend
from .serialization import meeting_record_from_json, save_meeting_record
from .service import extract_meeting_record


def evidence_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Resolve immutable meeting evidence; no API calls."
    )
    parser.add_argument("record", type=Path)
    parser.add_argument("item_id")
    parser.add_argument("--refined", type=Path, required=True)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--extract-clips", type=Path)
    parser.add_argument("--padding-seconds", type=float, default=0)
    args = parser.parse_args(argv)
    if args.extract_clips and not args.audio:
        parser.error("--extract-clips requires --audio.")
    try:
        record = meeting_record_from_json(read_json(args.record))
        refined = refined_from_json(read_json(args.refined))
        spans = get_item_evidence(record, args.item_id, refined)
        for span in spans:
            print(json.dumps(asdict(span), ensure_ascii=False, indent=2))
            if args.extract_clips:
                if record.audio is None:
                    raise IntelligenceError(
                        "Record has no audio binding; re-extract with validated audio provenance."
                    )
                clip = extract_evidence_clip(
                    args.audio,
                    span,
                    args.extract_clips / f"{args.item_id}-{span.utterance_id}.wav",
                    padding_seconds=args.padding_seconds,
                    audio_reference=record.audio,
                )
                print(
                    f"Clip: {clip.path}; playback: {clip.playback_start:.6f} --> {clip.playback_end:.6f}; duration: {clip.duration_seconds:.6f} s"
                )
        return 0
    except (IntelligenceError, RefinementError, OSError) as exc:
        print(f"Evidence retrieval failed: {exc}", file=sys.stderr)
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Meeting intelligence from I-VI or saved validated evidence."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--speaker", type=Path)
    parser.add_argument("--grounding", type=Path)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--diarization", type=Path)
    parser.add_argument("--project-glossary", type=Path)
    parser.add_argument("--meeting-context", type=Path)
    parser.add_argument("--context", type=Path, help="Phase VIII structured context pack.")
    parser.add_argument(
        "--context-asr", action="store_true", help="Opt in to bounded extra Whisper calls."
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--speaker-reliability",
        action="store_true",
        help="Optional independent local diarization sidecars.",
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--semantic-reasoning", action="store_true", help="Optional Jev shadow sidecars."
    )
    parser.add_argument("--model")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if bool(args.speaker) != bool(args.grounding):
        parser.error("Saved mode requires both --speaker and --grounding.")
    if args.speaker and (
        args.project_glossary
        or args.meeting_context
        or args.context
        or args.context_asr
        or args.speaker_reliability
    ):
        parser.error("Saved evidence cannot be combined with new glossary layers.")
    if args.context and args.meeting_context:
        parser.error("Choose --context or the legacy --meeting-context format.")
    if args.audio and not args.diarization or args.diarization and not args.audio:
        parser.error("Saved audio binding requires both --audio and --diarization.")
    if not args.speaker and (args.audio or args.diarization):
        parser.error("Full workflow obtains canonical audio automatically.")
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    try:
        values = read_environment(args.env_file)
        config = IntelligenceConfig.from_env(environ=values)
        if args.model:
            config = replace(config, model=args.model)
        if args.speaker:
            from meeting_assistant.diarization.serialization import (
                diarization_from_json,
                speaker_transcript_from_json,
            )
            from meeting_assistant.grounding.serialization import grounding_from_json

            refined = refined_from_json(read_json(args.input))
            speaker = speaker_transcript_from_json(read_json(args.speaker))
            grounding = grounding_from_json(read_json(args.grounding))
            canonical = args.audio
            diarization = (
                diarization_from_json(read_json(args.diarization)) if args.diarization else None
            )
            output = args.output_dir or Path("transcripts")
            backend = None  # Service validates all saved provenance before model construction/call.
        else:
            from .workflow import run_upstream

            backend = GroqMeetingIntelligenceBackend(config)
            refined, speaker, grounding, canonical, diarization, output = run_upstream(args, values)
        result = extract_meeting_record(
            refined,
            speaker,
            grounding,
            backend=backend,
            config=config,
            canonical_audio_path=canonical,
            diarization=diarization,
        )
        files = save_meeting_record(result, refined, output)
        from meeting_assistant.semantic_reasoning.service import run_optional

        run_optional(
            refined, speaker, result, output, environ=values, enabled=args.semantic_reasoning
        )
    except Exception as exc:
        from meeting_assistant.asr.exceptions import ASRError
        from meeting_assistant.audio.exceptions import AudioIngestionError
        from meeting_assistant.contextual_asr.exceptions import ContextualASRError
        from meeting_assistant.diarization.exceptions import DiarizationError
        from meeting_assistant.grounding.exceptions import GroundingError
        from meeting_assistant.refinement.exceptions import RefinementError

        if isinstance(
            exc,
            (
                IntelligenceError,
                ASRError,
                AudioIngestionError,
                DiarizationError,
                GroundingError,
                RefinementError,
                ContextualASRError,
            ),
        ):
            print(f"Meeting intelligence failed [{exc.code}]: {exc}", file=sys.stderr)
            return 1
        if isinstance(exc, OSError):
            print("Meeting intelligence failed: check file access/disk space.", file=sys.stderr)
            return 1
        raise
    info = result.processing_info
    print(
        f"Meeting intelligence complete: {result.model_info.provider} / {result.model_info.model}"
    )
    print(
        f"Summary: {len(result.summary)}; minutes: {len(result.minutes)}; decisions: {len(result.decisions)}; actions: {len(result.action_items)}"
    )
    print(
        f"Owners: {sum(a.owner is not None for a in result.action_items)}/{len(result.action_items)}; deadlines: {sum(a.deadline_text is not None for a in result.action_items)}/{len(result.action_items)}"
    )
    print(
        f"Evidence references: {sum(len(i.evidence_utterance_ids) for i in result.content.items)}; unresolved: 0"
    )
    print(
        f"API calls: {len(info.calls)}; chunks: {info.chunk_count}; consolidation calls: {sum(c.stage == 'consolidation' for c in info.calls)}; schema repairs: {sum(c.schema_repair and not c.transport_retry for c in info.calls)}; transport retries: {sum(c.transport_retry for c in info.calls)}"
    )
    print(
        f"Provider latency: {info.provider_latency_seconds:.3f} s; total: {info.total_seconds:.3f} s; input tokens: {info.usage('prompt_tokens')}; output tokens: {info.usage('completion_tokens')}; total tokens: {info.usage()}"
    )
    print(
        f"JSON: {files.json_path}\nMarkdown: {files.markdown_path}\nEvidence: {files.evidence_manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
