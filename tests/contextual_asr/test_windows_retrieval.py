import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from meeting_assistant.contextual_asr import ContextualASRConfig, build_meeting_context
from meeting_assistant.contextual_asr.integration import context_retriever
from meeting_assistant.contextual_asr.models import AudioWindow, RetrievedTerm, SuspiciousSpan
from meeting_assistant.contextual_asr.retrieval import build_prompt, retrieve_terms
from meeting_assistant.contextual_asr.suspicion import detect_suspicious_spans
from meeting_assistant.contextual_asr.windows import select_windows
from meeting_assistant.grounding import GroundingConfig, GroundingRetriever
from meeting_assistant.grounding.glossary import build_glossary
from meeting_assistant.refinement.evaluate import controlled_evidence
from tests.grounding.helpers import StubEmbedding, entry

from .helpers import evidence


class SelectionTests(unittest.TestCase):
    def test_context_and_acoustic_signals_required(self):
        context, _, grounding = evidence()
        config = ContextualASRConfig(enabled=True)
        spans = detect_suspicious_spans(grounding, context, config)
        self.assertEqual(len(spans), 1)
        self.assertIn("supplied_context", spans[0].signals)
        self.assertFalse(detect_suspicious_spans(grounding, None, config))

    def test_weak_and_known_originals_do_not_trigger(self):
        for span, canonical, kwargs in (
            ("GPU", "GPU", {}),
            ("Drant", "Qdrant", {"phonetic_score": 0.2}),
            ("Drant", "Qdrant", {"lexical_score": 0.2}),
            ("Drant", "Qdrant", {"score": 0.1}),
        ):
            context, _, grounding = evidence(
                text="Use " + span, span=span, canonical=canonical, **kwargs
            )
            self.assertFalse(
                detect_suspicious_spans(grounding, context, ContextualASRConfig(enabled=True))
            )

    def test_protected_classes_not_retranscription_targets(self):
        for span in ("15%", "Friday", "not Monday", "might", "$500", "GPT-120B"):
            context, _, grounding = evidence(text="Use " + span, span=span)
            self.assertFalse(
                detect_suspicious_spans(grounding, context, ContextualASRConfig(enabled=True))
            )

    def test_ordinary_inflections_and_function_word_phrases_not_targets(self):
        for span, canonical in (
            ("quadrants", "Quadrant"),
            ("accelerates", "Accelerate"),
            ("discount at", "Discount rate"),
            ("GPU", "CPU"),
        ):
            context, _, grounding = evidence(text="Use " + span, span=span, canonical=canonical)
            self.assertFalse(
                detect_suspicious_spans(
                    grounding, context, ContextualASRConfig(enabled=True, use_global_glossary=True)
                )
            )

    def test_global_fallback_is_explicit(self):
        _, _, grounding = evidence()
        self.assertTrue(
            detect_suspicious_spans(
                grounding, None, ContextualASRConfig(enabled=True, use_global_glossary=True)
            )
        )

    def test_padding_clamps_and_aligns_frames(self):
        context, source, grounding = evidence()
        config = ContextualASRConfig(enabled=True)
        windows, _ = select_windows(
            detect_suspicious_spans(grounding, context, config),
            grounding,
            source.duration_seconds,
            config,
        )
        self.assertEqual(windows[0].start, 0)
        self.assertLessEqual(windows[0].end, source.duration_seconds)
        self.assertAlmostEqual(windows[0].end * 16000, round(windows[0].end * 16000))

    def test_nearby_regions_merge_one_call(self):
        source, grounding = controlled_evidence(
            "Use Drant and Pytorch",
            [
                {"span": "Drant", "candidates": [{"canonical": "Qdrant"}]},
                {"span": "Pytorch", "candidates": [{"canonical": "PyTorch"}]},
            ],
        )
        windows, _ = select_windows(
            tuple(SuspiciousSpan(r.id, 0.9, ()) for r in grounding.records),
            grounding,
            source.duration_seconds,
            ContextualASRConfig(),
        )
        self.assertEqual(len(windows), 1)
        self.assertEqual(len(windows[0].grounding_ids), 2)

    def test_window_duration_limit_records_skip(self):
        _, source, grounding = evidence()
        windows, skipped = select_windows(
            (SuspiciousSpan(grounding.records[0].id, 0.9, ()),),
            grounding,
            source.duration_seconds,
            ContextualASRConfig(max_window_seconds=1),
        )
        self.assertFalse(windows)
        self.assertEqual(skipped[0].reason, "window_duration_limit")

    def test_call_and_audio_budgets(self):
        _, source, grounding = evidence()
        records = tuple(
            replace(
                grounding.records[0],
                id=f"grnd_{i + 1:06d}",
                start=i * 20.0,
                end=i * 20.0 + 1.0,
                char_start=i * 20,
                char_end=i * 20 + 5,
            )
            for i in range(3)
        )
        grounding = replace(grounding, records=records)
        spans = tuple(SuspiciousSpan(r.id, 0.9 - i * 0.05, ()) for i, r in enumerate(records))
        for config, reason in (
            (ContextualASRConfig(max_windows=1), "call_budget"),
            (ContextualASRConfig(max_total_audio_seconds=5), "audio_budget"),
        ):
            windows, skipped = select_windows(spans, grounding, 60, config)
            self.assertEqual(len(windows), 1)
            self.assertTrue(any(s.reason == reason for s in skipped))

    def test_prompt_is_byte_bounded_deterministic_and_not_assertive(self):
        terms = tuple(
            RetrievedTerm(str(i), "模型" * 30 + str(i), "meeting", ("ctx_1",)) for i in range(8)
        )
        prompt, retained = build_prompt(terms)
        self.assertLessEqual(len(prompt.encode()), 223)
        self.assertEqual(build_prompt(terms), (prompt, retained))
        self.assertTrue(prompt.startswith("Possible vocabulary:"))

    def test_retrieval_small_and_traceable(self):
        context, source, grounding = evidence()
        terms = retrieve_terms(
            AudioWindow("win_000001", 0, source.duration_seconds, ("grnd_000001",)),
            grounding,
            context,
            ContextualASRConfig(top_k_terms=1),
        )
        self.assertEqual(terms[0].canonical, "Qdrant")
        self.assertEqual(terms[0].context_source_ids, context.terms[0].source_ids)

    def test_meeting_scope_outranks_equal_global_and_records_source(self):
        with tempfile.TemporaryDirectory() as directory:
            config = GroundingConfig(index_cache=Path(directory))
            baseline = GroundingRetriever(
                build_glossary([entry()]), config, embeddings=StubEmbedding()
            )
            context = build_meeting_context(terms=["Qdrant"], known_entries=())
            retriever = context_retriever(context, baseline)
            candidates = retriever.candidates("Qdrant", "vector database")
            self.assertEqual(candidates[0].scope, "meeting")
            self.assertIn("context_source:" + context.sources[0].id, candidates[0].reasons)
            self.assertEqual(len(baseline.glossary.entries), 1)
