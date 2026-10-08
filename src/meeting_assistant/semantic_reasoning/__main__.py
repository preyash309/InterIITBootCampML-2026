"""Saved-evidence smoke test; no rerun of ASR, diarization, refinement or extraction."""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from meeting_assistant.asr.config import read_environment
from meeting_assistant.diarization.serialization import speaker_transcript_from_json
from meeting_assistant.intelligence.serialization import meeting_record_from_json
from meeting_assistant.refinement.__main__ import read_json
from meeting_assistant.refinement.serialization import refined_from_json

from .config import SemanticConfig
from .serialization import render_text, save_semantics
from .service import analyze_meeting_semantics


def main(argv=None):
    parser = argparse.ArgumentParser(description="Phase X shadow observations from saved evidence.")
    parser.add_argument("record", type=Path)
    parser.add_argument("--refined", type=Path, required=True)
    parser.add_argument("--speaker", type=Path, required=True)
    parser.add_argument("--speaker-reliability", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("transcripts"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--enable",
        action="store_true",
        help="Enable shadow judgments using the configured backend.",
    )
    args = parser.parse_args(argv)
    try:
        values = read_environment(args.env_file)
        config = SemanticConfig.from_env(environ=values)
        if args.enable:
            config = replace(config, enabled=True)
        reliability = None
        if args.speaker_reliability:
            from meeting_assistant.speaker_reliability.serialization import reliability_from_json

            reliability = reliability_from_json(read_json(args.speaker_reliability))
        result = analyze_meeting_semantics(
            refined_from_json(read_json(args.refined)),
            speaker_transcript_from_json(read_json(args.speaker)),
            meeting_record_from_json(read_json(args.record)),
            config=config,
            environ=values,
            speaker_reliability=reliability,
        )
        destination = save_semantics(result, args.output_dir)
        print(render_text(result))
        print(f"Bundle: {destination}")
        return 0 if result.availability in ("available", "partial", "disabled") else 1
    except Exception as exc:
        from .exceptions import SemanticError

        if isinstance(exc, SemanticError):
            print(f"Semantic reasoning failed [{exc.code}]: {exc}", file=sys.stderr)
        else:
            print(
                "Semantic reasoning failed: check saved evidence and file access.", file=sys.stderr
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
