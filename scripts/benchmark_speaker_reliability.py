"""Offline sequential benchmark: existing 63.445-s fixture + six original SAPI fixtures.

No ASR/LLM/API calls. Controlled records contain no pretend ASR transcript: evaluation
uses authored SAPI word-event probes separately from the immutable transcript contract.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from meeting_assistant.asr.config import read_environment
from meeting_assistant.audio import AudioIngestionConfig, ingest_audio
from meeting_assistant.diarization import DiarizationConfig, PyannoteBackend, reconcile_transcript
from meeting_assistant.diarization.serialization import (
    diarization_from_json,
    diarization_to_json,
    speaker_transcript_from_json,
)
from meeting_assistant.speaker_reliability import (
    assess_speaker_reliability,
    save_speaker_reliability,
)
from meeting_assistant.speaker_reliability.evaluation import evaluate_reference
from meeting_assistant.speaker_reliability.serialization import to_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-root", type=Path, default=Path(".validation/phase9/fixtures"))
    parser.add_argument("--existing-audio", type=Path)
    parser.add_argument("--existing-primary", type=Path)
    parser.add_argument("--existing-speaker", type=Path)
    parser.add_argument("--existing-reference", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(".validation/phase9/benchmark"))
    args = parser.parse_args()
    values = read_environment()
    audio_config = replace(
        AudioIngestionConfig.from_env(),
        **{
            field: values[key]
            for field, key in (("ffmpeg_path", "FFMPEG_PATH"), ("ffprobe_path", "FFPROBE_PATH"))
            if values.get(key)
        },
    )
    reports = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.existing_audio:
        primary = diarization_from_json(args.existing_primary.read_text(encoding="utf-8"))
        speaker = speaker_transcript_from_json(args.existing_speaker.read_text(encoding="utf-8"))
        result = assess_speaker_reliability(args.existing_audio, primary, speaker)
        bundle = save_speaker_reliability(result, args.output_dir / "existing_63s")
        print("existing_63s", result.availability, bundle, flush=True)
        if args.existing_reference and result.availability == "available":
            reference = json.loads(args.existing_reference.read_text(encoding="utf-8-sig"))
            report = dict(
                fixture="existing_63s", **evaluate_reference(primary, speaker, result, reference)
            )
            reports.append(report)
            (args.output_dir / "existing_63s" / "evaluation.json").write_text(
                to_json(report), encoding="utf-8"
            )
    primary_backend = PyannoteBackend(DiarizationConfig.from_env())
    for directory in sorted(args.fixture_root.iterdir()):
        if not directory.is_dir():
            continue
        audio = ingest_audio(directory / "fixture.wav", config=audio_config)
        primary = primary_backend.diarize(audio.canonical_audio_path)
        # Empty actual-ASR record solely satisfies comparison input; no words fabricated.
        from uuid import uuid4

        from meeting_assistant.asr.models import ModelInfo, ProcessingInfo, TranscriptResult

        empty = TranscriptResult(
            str(uuid4()),
            "en",
            None,
            primary.duration_seconds,
            "",
            (),
            ModelInfo("fixture-metadata-only", "no-ASR-performed"),
            ProcessingInfo(0, 0, 0, 1),
        )
        speaker = reconcile_transcript(empty, primary)
        reference = json.loads((directory / "reference.json").read_text(encoding="utf-8"))
        reference.update(
            audio_sha256=primary.audio_sha256, duration_seconds=primary.duration_seconds
        )
        result = assess_speaker_reliability(audio.canonical_audio_path, primary, speaker)
        output = args.output_dir / directory.name
        output.mkdir(parents=True, exist_ok=True)
        bundle = save_speaker_reliability(result, output)
        (output / "primary.json").write_text(diarization_to_json(primary), encoding="utf-8")
        (output / "reference.json").write_text(to_json(reference), encoding="utf-8")
        if result.availability == "available":
            report = dict(
                fixture=directory.name, **evaluate_reference(primary, speaker, result, reference)
            )
            reports.append(report)
            (output / "evaluation.json").write_text(to_json(report), encoding="utf-8")
            print(
                directory.name,
                report["primary_count"],
                report["secondary_count"],
                report["agreement_fraction"],
                report["word_accuracy"],
                flush=True,
            )
        else:
            print(directory.name, "UNAVAILABLE", result.warnings, bundle, flush=True)
    (args.output_dir / "summary.json").write_text(to_json(reports), encoding="utf-8")


if __name__ == "__main__":
    main()
