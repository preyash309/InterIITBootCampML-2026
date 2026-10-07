"""Lightweight WER evaluation on developer-provided references. No dataset fabrication."""

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from .exceptions import ASRError
from .serialization import transcript_from_json
from .service import transcribe_audio


@dataclass(frozen=True)
class WERResult:
    wer: float | None
    word_errors: int
    reference_words: int
    hypothesis_words: int
    normalization: str = "casefold, remove punctuation, split whitespace; no number conversion"


def _tokens(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", "", text.casefold()).split()


def calculate_wer(reference: str, hypothesis: str) -> WERResult:
    """Levenshtein word distance, O(min(n,m)) memory. Original texts remain untouched."""
    ref, hyp = _tokens(reference), _tokens(hypothesis)
    longer, shorter = (ref, hyp) if len(ref) >= len(hyp) else (hyp, ref)
    previous = list(range(len(shorter) + 1))
    for row, word in enumerate(longer, 1):
        current = [row]
        for column, other in enumerate(shorter, 1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + (word != other),
                )
            )
        previous = current
    errors = previous[-1]
    # Undefined for empty reference and nonempty hypothesis; never divide by zero.
    rate = errors / len(ref) if ref else (0.0 if not hyp else None)
    return WERResult(rate, errors, len(ref), len(hyp))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate raw ASR against a provided reference.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--transcript", type=Path, help="Saved JSON; evaluation makes no API call.")
    source.add_argument("--audio", type=Path, help="Canonical WAV; makes a billed ASR API request.")
    parser.add_argument(
        "--reference", type=Path, help="UTF-8 reference, verified by the developer."
    )
    args = parser.parse_args(argv)
    try:
        result = (
            transcript_from_json(args.transcript.read_text(encoding="utf-8"))
            if args.transcript
            else transcribe_audio(args.audio)
        )
        metrics = {
            "audio_duration_seconds": result.duration_seconds,
            "transcription_seconds": result.processing_info.transcription_seconds,
            "real_time_factor": result.real_time_factor,
            "segments": len(result.segments),
            "timestamped_words": result.word_count,
        }
        if args.reference:
            reference = args.reference.read_text(encoding="utf-8-sig")
            metrics.update(asdict(calculate_wer(reference, result.text)))
        print(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False))
        if not args.reference:
            print("No reference supplied; reporting runtime metrics only.")
    except ASRError as exc:
        print(f"Evaluation failed [{exc.code}]: {exc}", file=sys.stderr)
        return 1
    except (OSError, UnicodeError):
        print("Evaluation failed: cannot read UTF-8 transcript/reference files.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
