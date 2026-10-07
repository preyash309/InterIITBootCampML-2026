"""Repeatable offline preservation/performance check on a supplied Phase III JSON."""

import argparse
import hashlib
import json
import socket
import time
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

from meeting_assistant.diarization.serialization import speaker_transcript_from_json
from meeting_assistant.grounding import (
    GroundingConfig,
    GroundingRetriever,
    ground_transcript,
    load_glossary,
    validate_grounding_source,
)
from meeting_assistant.grounding.embeddings import MiniLMBackend
from meeting_assistant.grounding.evaluate import evaluate_cases
from meeting_assistant.grounding.serialization import save_grounding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("speaker", type=Path)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--output", type=Path, default=Path(".validation/phase4/offline"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    originals = {path: path.read_bytes() for path in (args.speaker, args.raw) if path is not None}
    source = speaker_transcript_from_json(originals[args.speaker].decode("utf-8"))
    config = replace(GroundingConfig.from_env(), index_cache=args.output / "index")
    with patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden")):
        started = time.perf_counter()
        glossary = load_glossary()
        glossary_seconds = time.perf_counter() - started
        backend = MiniLMBackend(config)
        model_seconds = backend.load_model()
        engine = GroundingRetriever(glossary, config, embeddings=backend)
        engine.prepare()
        first_index_seconds = engine.index_load_seconds
        result = ground_transcript(source, retriever=engine)
        validate_grounding_source(result, source)
        files = save_grounding(result, args.output / "artifacts")
        again = GroundingRetriever(glossary, config, embeddings=backend)
        again.prepare()
        second = ground_transcript(source, retriever=again)
        validate_grounding_source(second, source)
        benchmark = evaluate_cases(
            json.loads(Path("data/grounding_benchmark.json").read_text(encoding="utf-8")), engine
        )
    if any(path.read_bytes() != before for path, before in originals.items()):
        raise AssertionError("An upstream artifact changed")
    report = {
        "network_blocked": True,
        "upstream_byte_preservation": True,
        "upstream_sha256": {
            str(path): hashlib.sha256(value).hexdigest() for path, value in originals.items()
        },
        "glossary": glossary.stats(),
        "glossary_load_seconds": glossary_seconds,
        "model_load_seconds": model_seconds,
        "first_index_seconds": first_index_seconds,
        "first_index_cache_hit": engine.cache_hit,
        "warm_index_seconds": again.index_load_seconds,
        "warm_index_cache_hit": again.cache_hit,
        "first_grounding": asdict(result.processing_info),
        "warm_grounding": asdict(second.processing_info),
        "records": len(result.records),
        "index_matrix_bytes": int(engine._matrix.nbytes),
        "artifacts": {name: str(value) for name, value in asdict(files).items()},
        "benchmark": benchmark,
    }
    try:
        import psutil

        report["observed_process_rss_bytes"] = psutil.Process().memory_info().rss
    except ImportError:
        pass
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("glossary", "benchmark")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
