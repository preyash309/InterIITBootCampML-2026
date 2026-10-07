"""Developer CLI for canonical audio, or integrated ingestion with --ingest."""

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

from meeting_assistant.audio import AudioIngestionConfig, AudioIngestionError, ingest_audio

from .api_backend import WhisperAPIBackend
from .config import ASRConfig, TranscriptionOptions
from .exceptions import ASRError
from .serialization import save_transcript
from .service import transcribe_audio


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Produce raw timestamped Whisper JSON/TXT.")
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--ingest", action="store_true", help="Run Phase I on original audio/video."
    )
    parser.add_argument("--provider", choices=("groq", "openai"))
    parser.add_argument("--model", help="Provider-compatible Whisper model.")
    parser.add_argument("--language", help="ISO language code or auto (default: en).")
    parser.add_argument("--chunk-seconds", type=float)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--ffmpeg", help="FFmpeg executable override for integrated ingestion.")
    parser.add_argument("--ffprobe", help="ffprobe executable override for integrated ingestion.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    try:
        # Validate credentials/configuration before producing an ingestion job.
        from .config import read_environment

        values = read_environment(args.env_file)
        if args.provider is not None and args.provider != values.get("ASR_PROVIDER", "groq"):
            values["ASR_PROVIDER"] = args.provider
            values.pop("ASR_MODEL", None)
        if args.model is not None:
            values["ASR_MODEL"] = args.model
        config = ASRConfig.from_env(environ=values)
        if args.language is not None:
            config = replace(
                config,
                options=TranscriptionOptions(
                    language=None if args.language == "auto" else args.language,
                    temperature=config.options.temperature,
                ),
            )
        if args.chunk_seconds is not None:
            config = replace(config, chunk_duration_seconds=args.chunk_seconds)
        backend = WhisperAPIBackend(config)
        canonical = args.input
        output_dir = args.output_dir or Path(values.get("ASR_OUTPUT_DIR", "transcripts"))
        if args.ingest:
            audio_config = AudioIngestionConfig.from_env()
            overrides = {}
            for key, value in (
                ("work_dir", args.work_dir),
                ("ffmpeg_path", args.ffmpeg or values.get("FFMPEG_PATH")),
                ("ffprobe_path", args.ffprobe or values.get("FFPROBE_PATH")),
            ):
                if value is not None:
                    overrides[key] = value
            audio = ingest_audio(args.input, config=replace(audio_config, **overrides))
            canonical = audio.canonical_audio_path
            if args.output_dir is None and "ASR_OUTPUT_DIR" not in values:
                output_dir = canonical.parent.parent / "transcripts"
            print(f"Canonical audio: {canonical}")
        result = transcribe_audio(canonical, backend=backend)
        artifacts = save_transcript(result, output_dir)
    except (ASRError, AudioIngestionError) as exc:
        print(f"Transcription failed [{exc.code}]: {exc}", file=sys.stderr)
        return 1
    print("Transcription completed (uncorrected raw ASR)")
    print(f"Provider: {result.model_info.provider}")
    print(f"Model: {result.model_info.model}")
    print(f"Audio duration: {result.duration_seconds:.3f} s")
    print(f"API processing time: {result.processing_info.transcription_seconds:.3f} s")
    print(f"Total ASR time: {result.processing_info.total_seconds:.3f} s")
    print(f"RTF (includes network): {result.real_time_factor:.3f}")
    print(f"Language: {result.language or 'not reported consistently'}")
    print(f"Chunks: {result.processing_info.chunk_count}")
    print(f"Segments: {len(result.segments)}; timestamped words: {result.word_count}")
    print(f"JSON: {artifacts.json_path}")
    print(f"Text: {artifacts.text_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
