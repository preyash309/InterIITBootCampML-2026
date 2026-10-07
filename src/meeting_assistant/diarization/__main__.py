"""Developer workflow: ingest, local diarization, raw ASR, separate speaker evidence."""

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

from meeting_assistant.asr import ASRConfig, transcribe_audio
from meeting_assistant.asr.api_backend import WhisperAPIBackend
from meeting_assistant.asr.config import read_environment
from meeting_assistant.asr.exceptions import ASRError
from meeting_assistant.asr.serialization import save_transcript, transcript_from_json
from meeting_assistant.audio import AudioIngestionConfig, AudioIngestionError, ingest_audio

from .cache import prepare_model
from .config import DiarizationConfig, DiarizationOptions
from .exceptions import DiarizationError
from .pyannote_backend import PyannoteBackend
from .reconciliation import reconcile_transcript
from .serialization import save_speaker_transcript


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create anonymous speaker evidence; raw ASR stays unchanged."
    )
    parser.add_argument("input", type=Path, nargs="?")
    parser.add_argument(
        "--canonical", action="store_true", help="Input is already Phase I canonical WAV."
    )
    parser.add_argument(
        "--raw-transcript", type=Path, help="Reuse saved Phase II JSON for this canonical input."
    )
    parser.add_argument(
        "--prepare-model",
        action="store_true",
        help="Explicit model download/verification; no audio inference.",
    )
    parser.add_argument(
        "--verify-model",
        action="store_true",
        help="Offline local loading only; no audio inference.",
    )
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--num-speakers", type=int)
    parser.add_argument("--min-speakers", type=int)
    parser.add_argument("--max-speakers", type=int)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.prepare_model and args.verify_model:
        parser.error("Choose --prepare-model or --verify-model.")
    if not args.input and not (args.prepare_model or args.verify_model):
        parser.error("An input recording is required.")
    if args.raw_transcript and not args.canonical:
        parser.error("--raw-transcript requires --canonical to identify the matching retained WAV.")
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    try:
        values = read_environment(args.env_file)
        config = DiarizationConfig.from_env(environ=values)
        if args.device:
            config = replace(config, device=args.device)
        if any(
            value is not None for value in (args.num_speakers, args.min_speakers, args.max_speakers)
        ):
            config = replace(
                config,
                options=DiarizationOptions(args.num_speakers, args.min_speakers, args.max_speakers),
            )
        if args.prepare_model:
            print(
                f"Verified Community-1 cache: {prepare_model(replace(config, offline_only=False))}"
            )
            return 0
        backend = PyannoteBackend(config)
        if args.verify_model:
            elapsed = backend.load_model()
            print(f"Local model loaded: {config.model}; device={config.device}; {elapsed:.3f} s")
            return 0
        raw = None
        asr_backend = None
        if args.raw_transcript:
            raw = transcript_from_json(args.raw_transcript.read_text(encoding="utf-8"))
        else:
            asr_backend = WhisperAPIBackend(ASRConfig.from_env(environ=values))
        canonical = args.input
        output_dir = args.output_dir or Path("transcripts")
        if not args.canonical:
            audio_config = AudioIngestionConfig.from_env()
            overrides = {
                name: value
                for name, value in (
                    ("work_dir", args.work_dir),
                    ("ffmpeg_path", values.get("FFMPEG_PATH")),
                    ("ffprobe_path", values.get("FFPROBE_PATH")),
                )
                if value is not None
            }
            audio = ingest_audio(args.input, config=replace(audio_config, **overrides))
            canonical = audio.canonical_audio_path
            if args.output_dir is None:
                output_dir = canonical.parent.parent / "transcripts"
        # Verify local inference before making any billed ASR request.
        diarization = backend.diarize(canonical)
        raw_artifacts = None
        if raw is None:
            raw = transcribe_audio(canonical, backend=asr_backend)
            raw_artifacts = save_transcript(raw, output_dir)
        result = reconcile_transcript(raw, diarization, config=config.reconciliation)
        artifacts = save_speaker_transcript(result, diarization, output_dir)
    except (DiarizationError, ASRError, AudioIngestionError) as exc:
        print(f"Diarization failed [{exc.code}]: {exc}", file=sys.stderr)
        return 1
    except OSError:
        print(
            "Diarization failed: check input/output access and available disk space.",
            file=sys.stderr,
        )
        return 1
    info = result.reconciliation_info
    print("Diarization completed (anonymous speaker evidence; uncorrected raw ASR)")
    print(f"Canonical audio: {canonical}")
    print(f"Model: {config.model}; device: {config.device}")
    print(
        f"Speakers: {result.detected_speaker_count}; regular/exclusive turns: "
        f"{len(diarization.regular_turns)}/{len(diarization.exclusive_turns)}"
    )
    print(
        f"Audio: {result.duration_seconds:.3f} s; model load: "
        f"{diarization.processing_info.model_load_seconds:.3f} s"
    )
    print(
        f"Diarization: {diarization.processing_info.diarization_seconds:.3f} s; RTF: "
        f"{diarization.real_time_factor:.3f}"
    )
    print(
        f"Words: {info.total_words}; direct: {info.direct_assignments}; "
        f"tolerance: {info.tolerance_assignments}; nearest: {info.nearest_assignments}; "
        f"unassigned: {info.unassigned_words}"
    )
    print(
        f"Assignment coverage: {info.assignment_coverage:.1%}"
        if info.assignment_coverage is not None
        else "Assignment coverage: unavailable (no word timestamps)"
    )
    if raw_artifacts:
        print(f"Raw JSON: {raw_artifacts.json_path}\nRaw text: {raw_artifacts.text_path}")
    else:
        print(f"Retained raw JSON: {args.raw_transcript}")
    print(f"Diarization JSON: {artifacts.diarization_path}")
    print(f"Speaker JSON: {artifacts.json_path}\nSpeaker text: {artifacts.text_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
