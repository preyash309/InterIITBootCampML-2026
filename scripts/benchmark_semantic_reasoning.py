"""Inventory by default; explicit live evaluation never substitutes authored answers."""

import argparse
import json
import os
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from meeting_assistant.asr.config import read_environment
from meeting_assistant.diarization.serialization import (
    speaker_transcript_from_json,
    speaker_transcript_to_json,
)
from meeting_assistant.intelligence import extract_meeting_record
from meeting_assistant.intelligence.evidence import resolve_meeting_record_evidence
from meeting_assistant.intelligence.fixtures import text_evidence
from meeting_assistant.intelligence.models import (
    ActionItem,
    ActionOwner,
    Decision,
    IntelligenceModelInfo,
    IntelligenceProcessingInfo,
    MeetingContent,
    MeetingRecord,
)
from meeting_assistant.intelligence.serialization import (
    meeting_record_from_json,
    meeting_record_to_json,
)
from meeting_assistant.intelligence.service import refined_digest
from meeting_assistant.refinement.serialization import refined_from_json, refined_to_json
from meeting_assistant.semantic_reasoning import (
    GLiNERBackend,
    JuliaBackend,
    SemanticConfig,
    analyze_meeting_semantics,
    save_semantics,
)
from meeting_assistant.semantic_reasoning.evaluation import evaluate_cases, evaluate_verification
from meeting_assistant.semantic_reasoning.provider import JevBackend
from meeting_assistant.semantic_reasoning.serialization import to_json
from meeting_assistant.semantic_reasoning.service import DecisionSession
from meeting_assistant.semantic_reasoning.verification import verify_items


def main(argv=None):
    with ExitStack() as resources:
        return run(argv, resources)


def run(argv, resources):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/semantic_benchmark.json"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("benchmarks/semantic/baseline"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/semantic/results"))
    parser.add_argument(
        "--create-baseline", action="store_true", help="Explicit billed Phase VI baseline calls."
    )
    parser.add_argument(
        "--live", action="store_true", help="Requires RUN_JEV_API_TESTS=1 and TypeSafe key."
    )
    parser.add_argument(
        "--local", action="store_true", help="Run cached Julia locally; no API key."
    )
    parser.add_argument("--local-provider", choices=("julia", "gliner"), default="julia")
    parser.add_argument(
        "--events-only",
        action="store_true",
        help="Use authored text and an explicitly empty record; no Phase VI calls or coverage quality claims.",
    )
    args = parser.parse_args(argv)
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    values = read_environment()
    inventory = {
        "cases": len(dataset["cases"]),
        "verification_challenges": len(dataset["verification_challenges"]),
        "reference": dataset["origin"],
        "live_access_configured": bool(values.get("TYPESAFE_API_KEY")),
        "live_validated": False,
        "results": evaluate_cases(dataset["cases"], {}),
    }
    if args.create_baseline:
        from meeting_assistant.intelligence.config import IntelligenceConfig
        from meeting_assistant.intelligence.groq_backend import GroqMeetingIntelligenceBackend

        backend = GroqMeetingIntelligenceBackend(IntelligenceConfig.from_env(environ=values))
        for case in dataset["cases"]:
            directory = args.baseline_dir / case["id"]
            directory.mkdir(parents=True, exist_ok=False)
            refined, speaker, grounding = text_evidence(case["turns"])
            record = extract_meeting_record(refined, speaker, grounding, backend=backend)
            for name, content in {
                "refined_transcript.json": refined_to_json(refined),
                "speaker_transcript.json": speaker_transcript_to_json(speaker),
                "meeting_record.json": meeting_record_to_json(record),
            }.items():
                (directory / name).write_text(content, encoding="utf-8")
    if args.live or args.local:
        if args.live and args.local:
            parser.error("Choose one semantic provider.")
        if args.live and (
            os.environ.get("RUN_JEV_API_TESTS") != "1" or not values.get("TYPESAFE_API_KEY")
        ):
            parser.error(
                "Live evaluation requires RUN_JEV_API_TESTS=1 and backend TYPESAFE_API_KEY."
            )
        provider = args.local_provider if args.local else "typesafe"
        config = replace(
            SemanticConfig.from_env(
                environ={
                    **values,
                    "SEMANTIC_PROVIDER": provider,
                    "SEMANTIC_MODEL": {
                        "julia": "SupersonicLabs/Julia-1",
                        "gliner": "fastino/GLiNER2.5-Decide",
                        "typesafe": "jev-1.13.0",
                    }[provider],
                    "SEMANTIC_REQUEST_TIMEOUT_SECONDS": "60",
                }
            ),
            enabled=True,
        )
        backend = (
            resources.enter_context(
                (JuliaBackend if provider == "julia" else GLiNERBackend)(config)
            )
            if args.local
            else JevBackend(values["TYPESAFE_API_KEY"], config)
        )
        results = {}
        for case in dataset["cases"]:
            directory = args.baseline_dir / case["id"]
            if args.events_only:
                refined, speaker, _ = text_evidence(case["turns"])
                record = authored_record(refined, MeetingContent(), "empty-evaluation-record")
            elif not (directory / "meeting_record.json").is_file():
                parser.error("Missing saved baseline; create/review Phase VI baseline first.")
            else:
                refined = refined_from_json(
                    (directory / "refined_transcript.json").read_text(encoding="utf-8")
                )
                speaker = speaker_transcript_from_json(
                    (directory / "speaker_transcript.json").read_text(encoding="utf-8")
                )
                record = meeting_record_from_json(
                    (directory / "meeting_record.json").read_text(encoding="utf-8")
                )
            if tuple(
                (u.utterance_id, u.speaker_id, u.refined_text) for u in refined.utterances
            ) != tuple((t["id"], t["speaker_id"], t["text"]) for t in case["turns"]):
                parser.error("Baseline text does not match the authored evaluation case.")
            result = analyze_meeting_semantics(
                refined, speaker, record, backend=backend, config=config, environ=values
            )
            save_semantics(result, args.output_dir / case["id"])
            results[case["id"]] = result
            print("Evaluated", case["id"], result.availability, flush=True)
        inventory["results"] = evaluate_cases(dataset["cases"], results)
        # Intentionally invalid claims remain isolated evaluation data, never a canonical artifact.
        observations = {}
        for challenge in dataset["verification_challenges"]:
            refined, speaker, _ = text_evidence([{"text": challenge["evidence"]}])
            if challenge["kind"] == "decision":
                content = MeetingContent(
                    decisions=(Decision("dec_0001", challenge["claim"], ("utt_000001",)),)
                )
            else:
                owner = challenge["owner"]
                owner = (
                    ActionOwner("speaker", speaker_id=owner)
                    if owner and owner.startswith("SPEAKER_")
                    else ActionOwner("named_entity", display_text=owner)
                    if owner
                    else None
                )
                content = MeetingContent(
                    action_items=(
                        ActionItem(
                            "act_0001",
                            challenge["claim"],
                            ("utt_000001",),
                            owner,
                            challenge["deadline"],
                        ),
                    )
                )
            record = authored_record(refined, content, "authored-verification-challenge")
            observation = verify_items(
                record,
                resolve_meeting_record_evidence(record, refined),
                DecisionSession(backend, config),
            )[0]
            observations[challenge["id"]] = observation
        inventory["verification"] = evaluate_verification(
            dataset["verification_challenges"], observations
        )
        inventory["runtime"] = getattr(backend, "runtime", {})
        inventory["provider"] = backend.provider
        inventory["model"] = backend.model
        inventory["provider_call_count"] = len(backend.calls)
        inventory["provider_latency_seconds"] = sum(c.latency_seconds for c in backend.calls)
        inventory["live_validated"] = any(r.processing.calls for r in results.values())
        inventory["comparisons"] = {
            "A": "Empty authored evaluation record; Phase VI quality not measured"
            if args.events_only
            else "Saved original Phase VI MeetingRecords remain unchanged",
            "B": "Event and relation metrics in results",
            "C": "Verification challenge metrics",
            "D": "Coverage requires manual links on actual baseline records; no fabricated coverage gold",
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "evaluation.json").write_text(to_json(inventory), encoding="utf-8")
    print(to_json(inventory))
    return 0


def authored_record(refined, content, model):
    return MeetingRecord(
        str(uuid4()),
        refined.id,
        refined_digest(refined),
        refined.source_speaker_transcript_id,
        refined.source_speaker_sha256,
        refined.grounding_result_id,
        refined.grounding_sha256,
        refined.source_raw_transcript_id,
        refined.source_raw_sha256,
        content,
        IntelligenceModelInfo("authored-evaluation", model),
        IntelligenceProcessingInfo(0, 0, ()),
    )


if __name__ == "__main__":
    raise SystemExit(main())
