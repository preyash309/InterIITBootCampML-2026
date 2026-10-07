"""Controlled retrieval metrics, not WER, correction accuracy or meeting quality."""

import argparse
import json
import time
from pathlib import Path

from .config import GroundingConfig
from .glossary import load_glossary
from .retrieval import GroundingRetriever


def evaluate_cases(data: dict, retriever: GroundingRetriever) -> dict:
    cases = data["positives"] + data["negatives"]
    started = time.perf_counter()
    scores = retriever.context_scores([c["context"] for c in cases])
    positives, negatives = [], []
    for index, case in enumerate(cases):
        candidates = retriever.candidates(
            case["span"], case["context"], semantic_scores=scores[index]
        )
        names = [c.canonical for c in candidates]
        if "expected" in case:
            rank = names.index(case["expected"]) + 1 if case["expected"] in names else None
            positives.append({**case, "rank": rank, "candidates": names})
        else:
            negatives.append(
                {
                    **case,
                    "candidates": names,
                    "forbidden_returned": sorted(set(names) & set(case["forbidden"])),
                }
            )
    count = len(positives)
    return {
        "metric": "controlled candidate retrieval, not transcript correction",
        "glossary_version": retriever.glossary.version,
        "embedding_model": retriever.embeddings.model,
        "embedding_revision": retriever.embeddings.revision,
        "top_k": retriever.config.top_k,
        "score_weights": {
            name: getattr(retriever.config, name)
            for name in ("lexical_weight", "phonetic_weight", "semantic_weight", "scope_weight")
        },
        "positive_cases": count,
        "negative_cases": len(negatives),
        **{
            f"recall_at_{k}": sum(bool(c["rank"] and c["rank"] <= k) for c in positives) / count
            if count
            else None
            for k in (1, 3, 5)
        },
        "mrr": sum(1 / c["rank"] if c["rank"] else 0 for c in positives) / count if count else None,
        "negative_forbidden_case_rate": sum(bool(c["forbidden_returned"]) for c in negatives)
        / len(negatives)
        if negatives
        else None,
        "negative_any_candidate_rate": sum(bool(c["candidates"]) for c in negatives)
        / len(negatives)
        if negatives
        else None,
        "seconds": time.perf_counter() - started,
        "positive_details": positives,
        "negative_details": negatives,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark controlled glossary retrieval cases.")
    parser.add_argument("--cases", type=Path, default=Path("data/grounding_benchmark.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    config = GroundingConfig.from_env()
    retriever = GroundingRetriever(load_glossary(), config)
    retriever.prepare()
    report = evaluate_cases(json.loads(args.cases.read_text(encoding="utf-8")), retriever)
    report["index_load_seconds"] = retriever.index_load_seconds
    report["cache_hit"] = retriever.cache_hit
    text = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in report.items() if not key.endswith("details")}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
