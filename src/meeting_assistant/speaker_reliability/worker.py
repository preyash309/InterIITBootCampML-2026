"""Standalone isolated NeMo worker. Exiting releases all secondary CUDA allocations."""

import argparse
import hashlib
import importlib.metadata
import json
import wave
from pathlib import Path
from time import perf_counter


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = perf_counter()
    model_sha = sha256(args.model)
    if model_sha != "bc74dfd8ca314240abcdc7e2949901eeaa72947a04ce1fab893e373d81f1e689":
        raise ValueError("Checkpoint does not match the pinned NVIDIA Sortformer release.")
    import torch
    from nemo.collections.asr.models import SortformerEncLabelModel

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Configured CUDA device is unavailable.")
    with wave.open(str(args.audio), "rb") as stream:
        if (
            stream.getnchannels(),
            stream.getsampwidth(),
            stream.getframerate(),
            stream.getcomptype(),
        ) != (1, 2, 16000, "NONE"):
            raise ValueError("Expected canonical PCM WAV.")
        duration = stream.getnframes() / 16000
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    load_started = perf_counter()
    model = SortformerEncLabelModel.restore_from(
        str(args.model), map_location=args.device, strict=True
    )
    model.eval()
    if args.device == "cuda":
        torch.cuda.synchronize()
    load_seconds = perf_counter() - load_started
    inference_started = perf_counter()
    with torch.inference_mode():
        output = model.diarize(audio=str(args.audio), batch_size=1, num_workers=0)
    if args.device == "cuda":
        torch.cuda.synchronize()
    inference_seconds = perf_counter() - inference_started
    segments = []
    for line in output[0]:
        begin, end, label = line.split()
        begin, end = max(0, float(begin)), min(duration, float(end))
        if end > begin:
            segments.append(dict(speaker_id=label, start=begin, end=end))
    segments.sort(key=lambda x: (x["start"], x["end"], x["speaker_id"]))
    result = dict(
        segments=segments,
        backend="nemo-sortformer",
        model="nvidia/diar_sortformer_4spk-v1",
        revision="2617bffbd820aa29d8f1fb6ab6f9ed7f0adbc996",
        package_version=importlib.metadata.version("nemo_toolkit"),
        device=args.device,
        audio_sha256=sha256(args.audio),
        duration_seconds=duration,
        model_sha256=model_sha,
        model_load_seconds=load_seconds,
        inference_seconds=inference_seconds,
        total_seconds=perf_counter() - started,
        peak_gpu_memory_bytes=torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
    )
    args.output.write_text(
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
