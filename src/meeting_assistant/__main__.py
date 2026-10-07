"""Integrated developer workflow; later stages remain deliberately absent."""

import sys

from .asr.__main__ import main as transcribe_main


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if not arguments or arguments[0] in ("-h", "--help"):
        print("Usage: python -m meeting_assistant transcribe INPUT [ASR options]")
        print("Runs audio ingestion, then Whisper API transcription, and writes raw JSON/TXT.")
        print("Also: python -m meeting_assistant diarize INPUT [diarization options]")
        print("Also: python -m meeting_assistant ground INPUT [grounding options]")
        print("Also: python -m meeting_assistant refine INPUT [refinement options]")
        print("Also: python -m meeting_assistant intelligence INPUT [intelligence options]")
        print("Also: python -m meeting_assistant evidence RECORD ITEM_ID --refined JSON")
        return 0
    if arguments[0] == "intelligence":
        from .intelligence.__main__ import main as intelligence_main

        return intelligence_main(arguments[1:])
    if arguments[0] == "evidence":
        from .intelligence.__main__ import evidence_main

        return evidence_main(arguments[1:])
    if arguments[0] == "refine":
        from .refinement.__main__ import main as refine_main

        return refine_main(arguments[1:])
    if arguments[0] == "ground":
        from .grounding.__main__ import main as ground_main

        return ground_main(arguments[1:])
    if arguments[0] == "diarize":
        from .diarization.__main__ import main as diarize_main

        return diarize_main(arguments[1:])
    if arguments[0] != "transcribe":
        print("Unknown command. Use: transcribe INPUT", file=sys.stderr)
        return 2
    return transcribe_main([*arguments[1:], "--ingest"])


if __name__ == "__main__":
    raise SystemExit(main())
