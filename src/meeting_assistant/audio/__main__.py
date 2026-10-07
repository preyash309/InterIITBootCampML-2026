"""Minimal developer smoke test; no UI framework or CLI dependency."""

import argparse
import logging
import sys
from pathlib import Path

from .config import AudioIngestionConfig
from .exceptions import AudioIngestionError
from .ingestion import ingest_audio


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate meeting media and produce canonical audio."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--job-id", help="Reuse a UUID job with safe atomic replacement.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log internal diagnostics (may include filesystem paths).",
    )
    args = parser.parse_args(argv)
    if args.verbose:
        # Extra diagnostic fields remain available to application JSON handlers.
        class DiagnosticFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                message = super().format(record)
                stderr = getattr(record, "stderr", "")
                return f"{message}\n{stderr}" if stderr else message

        handler = logging.StreamHandler()
        handler.setFormatter(DiagnosticFormatter("%(levelname)s %(name)s %(message)s"))
        logging.basicConfig(level=logging.DEBUG, handlers=[handler])
    else:
        logging.getLogger("meeting_assistant.audio").addHandler(logging.NullHandler())
    try:
        config = AudioIngestionConfig.from_env()
        if args.work_dir is not None:
            from dataclasses import replace

            config = replace(config, work_dir=args.work_dir)
        result = ingest_audio(args.input, config=config, job_id=args.job_id)
    except AudioIngestionError as exc:
        print(f"Audio ingestion failed [{exc.code}]: {exc}", file=sys.stderr)
        return 1
    print("Audio ingestion succeeded")
    print(f"Job: {result.job_id}")
    for label, metadata in (
        ("Source", result.source_metadata),
        ("Canonical", result.canonical_metadata),
    ):
        print(f"\n{label}:")
        if label == "Canonical":
            print(f"  path: {result.canonical_audio_path}")
        print(f"  format: {metadata.container_format}")
        print(f"  codec: {metadata.codec}")
        duration = (
            f"{metadata.duration_seconds:.3f} s"
            if metadata.duration_seconds is not None
            else "unknown"
        )
        print(f"  duration: {duration}")
        print(
            f"  sample rate: {metadata.sample_rate if metadata.sample_rate is not None else 'unknown'} Hz"
        )
        print(f"  channels: {metadata.channels if metadata.channels is not None else 'unknown'}")
        print(f"  size: {metadata.file_size_bytes} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
