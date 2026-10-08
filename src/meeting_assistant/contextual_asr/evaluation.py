"""Controlled A/B/C comparison on caller-supplied audio/reference; no fabricated scores."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from meeting_assistant.asr import ASRConfig, transcribe_audio
from meeting_assistant.asr.config import read_environment
from meeting_assistant.asr.evaluate import calculate_wer
from meeting_assistant.asr.serialization import (
    save_transcript,
    transcript_from_json,
    transcript_to_json,
)
from meeting_assistant.audio import AudioIngestionConfig, ingest_audio
from meeting_assistant.diarization import DiarizationConfig, PyannoteBackend, reconcile_transcript
from meeting_assistant.diarization.serialization import (
    save_speaker_transcript,
    speaker_transcript_from_json,
)
from meeting_assistant.grounding import (
    GroundingConfig,
    GroundingRetriever,
    ground_transcript,
    load_glossary,
)
from meeting_assistant.grounding.serialization import grounding_from_json, save_grounding
from meeting_assistant.refinement import (
    GroqRefinerBackend,
    RefinementConfig,
    refine_transcript,
    save_refined_transcript,
)
from meeting_assistant.refinement.exceptions import RefinementError, RefinementRateLimitError
from meeting_assistant.refinement.models import RefinementDecision, RefinementResponse
from meeting_assistant.refinement.prompt import build_messages
from meeting_assistant.refinement.serialization import refined_from_json
from meeting_assistant.refinement.service import build_requests, validate_refined_source
from meeting_assistant.refinement.validation import safety_violations

from .config import ContextualASRConfig
from .context import sha256
from .integration import ContextGroundingRetriever, context_retriever
from .serialization import contextual_asr_from_json, read_context, save_contextual_asr
from .service import contextual_retranscribe


class ComparisonBackend:
    """Reuse identical validated decision requests within one comparison, never across users."""

    def __init__(self, backend):
        self.backend, self.cache, self.cache_hits = backend, {}, 0
        self.config = backend.config
        self.model_info = backend.model_info

    def key(self, request):
        return sha256(json.dumps(build_messages(request, self.config), sort_keys=True))

    def seed(self, refined, source, grounding):
        validate_refined_source(refined, source, grounding)
        if refined.model_info != self.model_info:
            raise ValueError("Saved comparison uses a different refiner configuration.")
        for request in build_requests(source, grounding, self.config):
            decisions = tuple(
                RefinementDecision(
                    e.grounding_record_id, e.action, e.candidate_entry_id, e.model_reason_code
                )
                for e in refined.edit_log
                if e.utterance_id == request.target.utterance_id
            )
            self.cache[self.key(request)] = RefinementResponse(
                request.target.utterance_id, decisions, self.model_info
            )

    def refine(self, request):
        from dataclasses import replace

        key = self.key(request)
        if key in self.cache:
            self.cache_hits += 1
            return replace(self.cache[key], calls=())
        result = self.backend.refine(request)
        self.cache[key] = result
        return result


def recover_comparison(output, result):
    """Recover immutable stage bundles after an interrupted comparison, with exact provenance."""
    grounds = tuple(
        grounding_from_json(p.read_text(encoding="utf-8")) for p in output.glob("*/grounding.json")
    )
    contextual = next(g for g in grounds if g.grounding_id == result.source_grounding_id)
    others = tuple(g for g in grounds if g.grounding_id != result.source_grounding_id)
    if len(others) != 1:
        raise ValueError("Recovery requires exactly one baseline grounding bundle.")
    completed = {}
    for path in output.glob("*/refined_transcript.json"):
        refined = refined_from_json(path.read_text(encoding="utf-8"))
        name = (
            "context_only" if refined.grounding_result_id == contextual.grounding_id else "baseline"
        )
        if name in completed:
            raise ValueError("Recovery found ambiguous completed conditions.")
        completed[name] = (refined, path)
    return others[0], contextual, completed


def term_metrics(reference, hypothesis, terms):
    """Occurrence precision/recall against explicit intended vocabulary annotations."""
    import re

    def counts(text):
        return {
            term: len(re.findall(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text, re.I))
            for term in terms
        }

    expected, found = counts(reference), counts(hypothesis)
    true = sum(min(expected[t], found[t]) for t in terms)
    false = sum(max(0, found[t] - expected[t]) for t in terms)
    total, predicted = sum(expected.values()), sum(found.values())
    precision = true / predicted if predicted else None
    recall = true / total if total else None
    return {
        "true_terms": true,
        "false_terms": false,
        "reference_terms": total,
        "predicted_terms": predicted,
        "precision": precision,
        "recall": recall,
        "exact_accuracy": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None,
    }


def compare_saved_texts(raw_text, conditions, *, reference=None, terms=()):
    report = {}
    for name, text in conditions.items():
        report[name] = {"text": text, "changes_from_pass1": safety_violations(raw_text, text)}
        if reference is not None:
            report[name]["term_metrics"] = term_metrics(reference, text, terms)
            report[name]["wer"] = asdict(calculate_wer(reference, text))
    return report


def main(argv=None):
    from dataclasses import replace

    parser = argparse.ArgumentParser(
        description="Live A/B/C comparison; audio/text are uploaded and calls may be billed."
    )
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--canonical", action="store_true")
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument(
        "--reference", type=Path, help="Developer-verified reference text for WER/term scoring."
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        help="Authored intended-term/source-text JSON; not checked acoustic ground truth.",
    )
    parser.add_argument("--saved-raw", type=Path)
    parser.add_argument("--saved-speaker", type=Path)
    parser.add_argument("--scope", choices=("combined", "meeting", "global"), default="combined")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--recover",
        action="store_true",
        help="Reuse an interrupted run's immutable bundles; requires saved raw/speaker inputs.",
    )
    args = parser.parse_args(argv)
    if bool(args.saved_raw) != bool(args.saved_speaker):
        parser.error("Saved comparison requires both raw and speaker JSON.")
    if args.recover and not args.saved_raw:
        parser.error("Recovery requires saved raw and speaker artifacts.")
    values = read_environment()
    context = read_context(args.context)
    config = ASRConfig.from_env(environ=values)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "evaluation.json").exists():
        prior = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
        if not args.recover or prior.get("complete") is not False:
            parser.error("Completed evaluation output already exists; choose a new directory.")
    canonical = args.audio
    if not args.canonical:
        overrides = {
            name: values[key]
            for name, key in (("ffmpeg_path", "FFMPEG_PATH"), ("ffprobe_path", "FFPROBE_PATH"))
            if values.get(key)
        }
        canonical = ingest_audio(
            args.audio, config=replace(AudioIngestionConfig.from_env(), **overrides)
        ).canonical_audio_path
    if args.saved_raw:
        raw = transcript_from_json(args.saved_raw.read_text(encoding="utf-8"))
        speaker = speaker_transcript_from_json(args.saved_speaker.read_text(encoding="utf-8"))
        if speaker.source_raw_sha256 != sha256(transcript_to_json(raw)):
            parser.error("Saved raw/speaker evidence does not match.")
    else:
        diarizer = PyannoteBackend(DiarizationConfig.from_env(environ=values))
        diarizer.load_model()
        raw = transcribe_audio(canonical, config=config)
        save_transcript(raw, output)
        diarization = diarizer.diarize(canonical)
        speaker = reconcile_transcript(raw, diarization)
        save_speaker_transcript(speaker, diarization, output)
    recovered = {}
    if args.recover:
        bundles = tuple(output.glob("contextual_asr/*/contextual_asr.json"))
        if len(bundles) != 1:
            parser.error("Recovery requires exactly one contextual evidence bundle.")
        result = contextual_asr_from_json(bundles[0].read_text(encoding="utf-8"))
        from .serialization import context_to_json
        from .service import file_digest, validate_contextual_source

        if result.canonical_audio_sha256 != file_digest(canonical) or result.context_sha256 != (
            None if args.scope == "global" else sha256(context_to_json(context))
        ):
            parser.error("Recovery audio/context does not match the saved run.")
        baseline, contextual_grounding, recovered = recover_comparison(output, result)
        validate_contextual_source(result, speaker, contextual_grounding)
        bundle = bundles[0].parent
    else:
        baseline, contextual_grounding, result, bundle = prepare_comparison(
            canonical, speaker, context, args.scope, output, values, config
        )
    texts = {"pass1": raw.text}
    statistics = {}
    refiner_config = RefinementConfig.from_env(environ=values)
    backend = ComparisonBackend(GroqRefinerBackend(refiner_config))
    for name, (refined, _) in recovered.items():
        backend.seed(refined, speaker, contextual_grounding if name == "context_only" else baseline)
    for name, grounding, contextual in (
        ("baseline", baseline, None),
        ("context_only", contextual_grounding, None),
        ("closed_loop", contextual_grounding, result),
    ):
        started, cache_before = time.perf_counter(), backend.cache_hits
        kwargs = {"contextual_asr": contextual} if contextual else {}
        try:
            if name in recovered:
                refined, path = recovered[name]
            else:
                refined = refine_transcript(
                    speaker, grounding, backend=backend, config=refiner_config, **kwargs
                )
                path = save_refined_transcript(refined, output).json_path
        except RefinementError as exc:
            statistics[name] = {
                "status": "failed",
                "error_code": exc.code,
                "latency_seconds": time.perf_counter() - started,
            }
            # Keep completed conditions and contextual evidence. Do not retry the whole meeting.
            if isinstance(exc, RefinementRateLimitError):
                break
            continue
        texts[name] = "\n".join(u.refined_text for u in refined.utterances)
        statistics[name] = {
            "status": "completed",
            "recovered": name in recovered,
            "identical_requests_reused": backend.cache_hits - cache_before,
            "latency_seconds": refined.processing_info.total_seconds
            if name in recovered
            else time.perf_counter() - started,
            "recorded_provider_calls": len(refined.processing_info.calls),
            "refined_json": str(path.relative_to(output)),
            "applied_edits": sum(e.validation_status == "applied" for e in refined.edit_log),
            "rejected_edits": sum(e.validation_status == "rejected" for e in refined.edit_log),
        }
        if name == "closed_loop":
            # Original hypothesis bundle remains immutable; decision links are a new audit bundle.
            from dataclasses import replace
            from uuid import uuid4

            save_contextual_asr(
                replace(result, result_id=str(uuid4())),
                output,
                context=None if args.scope == "global" else context,
                refined=refined,
            )
    reference = args.reference.read_text(encoding="utf-8") if args.reference else None
    report = {
        "scope": args.scope,
        "complete": all(name in texts for name in ("baseline", "context_only", "closed_loop")),
        "conditions": compare_saved_texts(
            raw.text, texts, reference=reference, terms=tuple(t.canonical for t in context.terms)
        ),
        "processing": statistics,
        "contextual_calls": result.provider_calls,
        "contextual_audio_seconds": result.audio_seconds,
        "contextual_total_seconds": result.total_seconds,
        "contextual_failures": sum(h.status == "failed" for h in result.hypotheses),
        "contextual_bundle": str(bundle.relative_to(output)),
        "raw_sha256": sha256(transcript_to_json(raw)),
        "contextual_latency_seconds": sum(h.latency_seconds for h in result.hypotheses),
        "contextual_provider_retries": 0,
        "suspicious_spans": len(result.suspicious_spans),
        "grounding_records": len(contextual_grounding.records),
        "pass2_invocation_rate_per_grounding_record": result.provider_calls
        / len(contextual_grounding.records)
        if contextual_grounding.records
        else 0,
        "candidate_links": sum(len(h.candidate_links) for h in result.hypotheses),
        "skipped": [asdict(s) for s in result.skipped],
        "reference_status": "developer_supplied" if reference else "no_verified_reference",
    }
    if args.annotations:
        annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
        scripted = " ".join(c["reference_text"] for c in annotations["cases"])
        vocabulary = tuple(c["term"] for c in annotations["cases"] if c.get("term"))
        report["script_relative_term_metrics"] = {
            name: term_metrics(scripted, text, vocabulary) for name, text in texts.items()
        }
        report["annotation_status"] = (
            "Authored intended-term script; not manually verified acoustic ground truth. No WER scored from it."
        )
    temporary = output / ".evaluation.json.tmp"
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(output / "evaluation.json")
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in ("conditions", "processing")
            },
            indent=2,
        )
    )
    return 0 if report["complete"] else 2


def prepare_comparison(canonical, speaker, context, scope, output, values, config):
    """Prepare shared first-pass evidence once for all three controlled conditions."""
    from dataclasses import replace

    baseline_engine = GroundingRetriever(load_glossary(), GroundingConfig.from_env(environ=values))
    engine = context_retriever(context, baseline_engine)
    if scope == "meeting":
        from meeting_assistant.grounding.glossary import build_glossary

        from .context import context_entries

        engine = ContextGroundingRetriever(
            build_glossary(context_entries(context)),
            baseline_engine.config,
            embeddings=baseline_engine.embeddings,
            context_sources={t.id: t.source_ids for t in context.terms},
        )
    elif scope == "global":
        engine = baseline_engine
    baseline = ground_transcript(speaker, retriever=baseline_engine)
    contextual_grounding = ground_transcript(speaker, retriever=engine)
    save_grounding(baseline, output)
    save_grounding(contextual_grounding, output)
    casr_config = replace(
        ContextualASRConfig.from_env(environ=values),
        enabled=True,
        use_global_glossary=scope in ("combined", "global"),
    )
    result = contextual_retranscribe(
        canonical,
        speaker,
        contextual_grounding,
        context=None if scope == "global" else context,
        config=casr_config,
        asr_config=config,
    )
    bundle = save_contextual_asr(result, output, context=None if scope == "global" else context)
    return baseline, contextual_grounding, result, bundle


if __name__ == "__main__":
    raise SystemExit(main())
