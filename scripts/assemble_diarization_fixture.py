"""Assemble tiny SAPI fixtures, retaining exact synthetic scheduling provenance."""

import argparse
import json
import wave
from array import array
from pathlib import Path


def assemble(root: Path, *, overlap_seconds: float = 0.0) -> Path:
    manifest = json.loads((root / "provenance.json").read_text(encoding="utf-8-sig"))
    samples = array("h")
    schedule = []
    rate = None
    for index, turn in enumerate(manifest["turns"]):
        with wave.open(str(root / turn["filename"]), "rb") as audio:
            if audio.getnchannels() != 1 or audio.getsampwidth() != 2:
                raise ValueError("Synthetic fixtures require mono PCM 16-bit speech.")
            if rate is None:
                rate = audio.getframerate()
                samples.extend([0] * int(rate * 0.3))
            if audio.getframerate() != rate:
                raise ValueError("SAPI voices produced inconsistent sample rates.")
            spoken = array("h", audio.readframes(audio.getnframes()))
        # Controlled overlap only between the first two long utterances.
        overlap = int(rate * overlap_seconds) if index == 1 else 0
        if overlap:
            overlap = min(overlap, len(samples), len(spoken))
            start = len(samples) - overlap
            for position in range(overlap):
                samples[start + position] = max(
                    -32768, min(32767, samples[start + position] + spoken[position])
                )
            samples.extend(spoken[overlap:])
        else:
            start = len(samples)
            samples.extend(spoken)
        schedule.append({**turn, "start": start / rate, "end": (start + len(spoken)) / rate})
        samples.extend([0] * int(rate * (0.05 if index == 2 else 0.25)))
    name = "overlap" if overlap_seconds else "two_speakers"
    output = root / f"{name}.wav"
    with wave.open(str(output), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(samples.tobytes())
    (root / f"{name}-schedule.json").write_text(
        json.dumps(
            {
                "origin": manifest["origin"],
                "expected_speakers": 2,
                "turns": schedule,
                "overlap_seconds": overlap_seconds,
                "sample_rate": rate,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"{output}: {len(samples) / rate:.3f} s, two synthetic voices")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?", default=Path(".validation/phase3/fixture"))
    args = parser.parse_args()
    assemble(args.root)
    assemble(args.root, overlap_seconds=1.2)
