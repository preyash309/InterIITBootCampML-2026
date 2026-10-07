"""Explicit setup or developer workflow through Phases I–IV, with no text editing."""

import argparse
import json
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

from meeting_assistant.asr import ASRConfig, transcribe_audio
from meeting_assistant.asr.api_backend import WhisperAPIBackend
from meeting_assistant.asr.config import read_environment
from meeting_assistant.asr.exceptions import ASRError
from meeting_assistant.asr.serialization import save_transcript
from meeting_assistant.audio import AudioIngestionConfig, AudioIngestionError, ingest_audio
from meeting_assistant.diarization import DiarizationConfig, PyannoteBackend
from meeting_assistant.diarization.exceptions import DiarizationError
from meeting_assistant.diarization.reconciliation import reconcile_transcript
from meeting_assistant.diarization.serialization import (
    save_speaker_transcript,
    speaker_transcript_from_json,
)

from .config import GroundingConfig
from .embeddings import MiniLMBackend, prepare_model
from .exceptions import GroundingError
from .glossary import load_glossary, read_meeting_context
from .retrieval import GroundingRetriever
from .serialization import save_grounding
from .service import ground_transcript


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Retrieve terminology candidates without editing transcripts."
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        help="Meeting media, or saved speaker JSON with --speaker-transcript.",
    )
    parser.add_argument(
        "--speaker-transcript",
        action="store_true",
        help="Reuse Phase III JSON; no ASR, diarization or upload.",
    )
    parser.add_argument("--prepare-model", action="store_true")
    parser.add_argument("--verify-model", action="store_true")
    parser.add_argument("--glossary-stats", action="store_true")
    parser.add_argument("--project-glossary", type=Path)
    parser.add_argument("--meeting-context", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if sum((args.prepare_model, args.verify_model, args.glossary_stats)) > 1:
        parser.error("Choose one setup/inspection mode.")
    if not args.input and not (args.prepare_model or args.verify_model or args.glossary_stats):
        parser.error("Supply media or saved speaker JSON.")
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    try:
        values = read_environment(args.env_file)
        config = GroundingConfig.from_env(environ=values)
        if args.prepare_model:
            print(
                f"Pinned embedding cache prepared: {prepare_model(replace(config, offline_only=False))}"
            )
            return 0
        embeddings = MiniLMBackend(config)
        if args.verify_model:
            print(
                f"Offline CPU load: {embeddings.load_model():.3f} s; {config.model}@{config.revision}"
            )
            print(f"Embedding shape: {embeddings.encode(['Local terminology retrieval.']).shape}")
            return 0
        setup_started = time.perf_counter()
        meeting = read_meeting_context(args.meeting_context) if args.meeting_context else ()
        glossary = load_glossary(project_glossary=args.project_glossary, meeting_entries=meeting)
        glossary_seconds = time.perf_counter() - setup_started
        if args.glossary_stats:
            print(json.dumps(glossary.stats(), indent=2))
            return 0
        engine = GroundingRetriever(glossary, config, embeddings=embeddings)
        # Verify/cache local grounding before any upstream billed ASR request.
        engine.prepare()
        setup_seconds = time.perf_counter() - setup_started
        output = args.output_dir or Path("transcripts")
        if args.speaker_transcript:
            if args.input.stat().st_size > 128 * 1024**2:
                raise GroundingError("Speaker transcript JSON exceeds the 128 MiB read limit.")
            speaker = speaker_transcript_from_json(args.input.read_text(encoding="utf-8"))
        else:
            asr_backend = WhisperAPIBackend(ASRConfig.from_env(environ=values))
            diar_config = DiarizationConfig.from_env(environ=values)
            diar_backend = PyannoteBackend(diar_config)
            diar_backend.load_model()
            audio_config = AudioIngestionConfig.from_env()
            overrides = {
                name: values[key]
                for name, key in (
                    ("ffmpeg_path", "FFMPEG_PATH"),
                    ("ffprobe_path", "FFPROBE_PATH"),
                    ("work_dir", "AUDIO_WORK_DIR"),
                )
                if values.get(key)
            }
            audio = ingest_audio(args.input, config=replace(audio_config, **overrides))
            if args.output_dir is None:
                output = audio.canonical_audio_path.parent.parent / "transcripts"
            diarization = diar_backend.diarize(audio.canonical_audio_path)
            raw = transcribe_audio(audio.canonical_audio_path, backend=asr_backend)
            raw_files = save_transcript(raw, output)
            speaker = reconcile_transcript(raw, diarization, config=diar_config.reconciliation)
            speaker_files = save_speaker_transcript(speaker, diarization, output)
            print(f"Raw JSON: {raw_files.json_path}\nSpeaker JSON: {speaker_files.json_path}")
        result = ground_transcript(speaker, retriever=engine)
        result = replace(
            result,
            processing_info=replace(
                result.processing_info,
                embedding_model_load_seconds=engine.model_load_seconds,
                glossary_load_seconds=glossary_seconds,
                index_load_seconds=engine.index_load_seconds,
                total_seconds=result.processing_info.total_seconds + setup_seconds,
            ),
        )
        artifacts = save_grounding(result, output)
    except (GroundingError, ASRError, AudioIngestionError, DiarizationError) as exc:
        print(f"Grounding failed [{exc.code}]: {exc}", file=sys.stderr)
        return 1
    except OSError:
        print(
            "Grounding failed: check input/output access and available disk space.", file=sys.stderr
        )
        return 1
    info = result.processing_info
    print("Grounding completed — candidate evidence only; transcripts unchanged")
    print(f"Glossary: {len(glossary.entries)} entries; version: {glossary.version}")
    print(f"Embedding: {config.model}; CPU; offline local inference")
    print(
        f"Utterances: {info.utterance_count}; spans considered: {info.considered_spans}; records: {len(result.records)}"
    )
    print(
        f"Index setup: {engine.index_load_seconds:.3f} s; cache hit: {engine.cache_hit}; grounding: {info.grounding_seconds:.3f} s"
    )
    print(f"JSON: {artifacts.json_path}\nText: {artifacts.text_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
