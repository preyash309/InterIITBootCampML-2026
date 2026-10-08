"""Atomic sidecar directory publication; canonical artifacts are never rewritten."""

import hashlib
import json
import os
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from uuid import uuid4

from .exceptions import InvalidReliability


def to_json(value):
    def encode(item):
        if isinstance(item, Path):
            return str(item)
        if is_dataclass(item):
            return asdict(item)
        raise TypeError("Unsupported sidecar value.")

    return (
        json.dumps(
            value, default=encode, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        + "\n"
    )


def fingerprint(value):
    return hashlib.sha256(to_json(value).encode("utf-8")).hexdigest()


def reliability_from_json(content):
    """Strict reconstruction of frozen version-1 sidecars, including nested tuples."""
    from types import UnionType
    from typing import Literal, Union, get_args, get_origin, get_type_hints

    from .models import SpeakerReliabilityResult

    def decode(value, kind):
        origin, args = get_origin(kind), get_args(kind)
        if origin in (UnionType, Union):
            if value is None and type(None) in args:
                return None
            return decode(value, next(x for x in args if x is not type(None)))
        if origin is tuple:
            if not isinstance(value, list):
                raise InvalidReliability("Serialized sequences must be arrays.")
            if len(args) == 2 and args[1] is Ellipsis:
                return tuple(decode(x, args[0]) for x in value)
            return tuple(decode(x, t) for x, t in zip(value, args, strict=True))
        if origin is Literal:
            if value not in args:
                raise InvalidReliability("Unknown serialized enum value.")
            return value
        if kind is Path:
            return Path(value)
        if is_dataclass(kind):
            hints = get_type_hints(kind)
            return kind(**{k: decode(v, hints[k]) for k, v in value.items()})
        if kind is float:
            if isinstance(value, bool) or not isinstance(value, (float, int)):
                raise InvalidReliability("Malformed serialized numeric value.")
            # Preserve JSON numeric spelling (0 vs 0.0) for fingerprint fidelity.
            return value
        if type(value) is not kind:
            raise InvalidReliability("Malformed serialized field type.")
        return value

    try:
        data = json.loads(content)
        if data["schema_version"] != "1.0":
            raise InvalidReliability("Unsupported speaker reliability schema version.")
        result = decode(data, SpeakerReliabilityResult)
        if (
            result.secondary is not None
            and result.provenance.secondary_result_sha256 != fingerprint(result.secondary)
        ):
            raise InvalidReliability("Secondary result fingerprint does not match its provenance.")
        return result
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise InvalidReliability("Malformed speaker reliability JSON.") from exc


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_text(result):
    lines = [
        "Speaker Reliability",
        "Temporal fractions are not probabilities of correctness.",
        f"Availability: {result.availability}",
        f"Primary: {result.provenance.primary_model}",
    ]
    if result.secondary:
        lines += [
            f"Secondary: {result.secondary.model}",
            f"Secondary runtime: {result.secondary.total_seconds:.3f} s",
        ]
    if result.comparison:
        c = result.comparison
        lines += [
            f"Speakers: primary {c.alignment.primary_count}, secondary {c.alignment.secondary_count}",
            f"Count mismatch: {c.alignment.count_mismatch}",
            f"Aligned agreement / speech union: {c.agreement_fraction:.3%}",
        ]
        lines += [
            f"Mapping: {x.secondary_id} -> {x.primary_id or 'UNMAPPED'} ({x.overlap_seconds:.3f} s)"
            for x in c.alignment.mappings
        ]
        lines += [
            f"[{x.start:.3f} --> {x.end:.3f}] {x.state} primary={x.primary_speakers} secondary={x.mapped_secondary_speakers}"
            for x in c.intervals
        ]
        lines += [
            f"Utterance {x.utterance_id}: {x.status}, agreement={x.agreement_fraction:.3%}, reasons={','.join(x.reasons)}"
            for x in result.utterances
        ]
    lines += [f"Warning: {x}" for x in result.warnings]
    return "\n".join(lines) + "\n"


def save_speaker_reliability(result, output_dir):
    """Same result/config repeats idempotently; conflicting reruns get separate bundles."""
    root = Path(output_dir) / "speaker_reliability"
    root.mkdir(parents=True, exist_ok=True)
    final = root / fingerprint(result)
    temporary = root / (".partial-" + str(uuid4()))
    payloads = {
        "secondary_diarization.json": to_json(
            result.secondary
            if result.secondary
            else {"availability": "unavailable", "warnings": result.warnings}
        ),
        "diarization_comparison.json": to_json(
            result.comparison
            if result.comparison
            else {"availability": "unavailable", "warnings": result.warnings}
        ),
        "speaker_reliability.json": to_json(result),
        "speaker_reliability.txt": render_text(result),
    }
    if final.exists():
        if all(
            (final / name).read_text(encoding="utf-8") == content
            for name, content in payloads.items()
        ):
            return final
        raise InvalidReliability("Existing reliability bundle is incomplete or changed.")
    try:
        temporary.mkdir()
        for name, content in payloads.items():
            with (temporary / name).open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        temporary.rename(final)
    except Exception:
        if temporary.is_dir():
            shutil.rmtree(temporary)
        raise
    return final
