import json
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

from meeting_assistant.refinement import (
    RefinementConfig,
    refine_transcript,
    validate_refined_source,
)
from meeting_assistant.refinement.evaluate import controlled_evidence, evaluate_cases
from meeting_assistant.refinement.exceptions import (
    RefinementSchemaError,
    RefinementSourceMismatch,
    RefinementValidationError,
)
from meeting_assistant.refinement.validation import protected_signature

from .helpers import FakeBackend, evidence


class ServiceTests(unittest.TestCase):
    def test_keep_preserves_exact_text(self):
        source, g = evidence()
        result = refine_transcript(source, g, backend=FakeBackend())
        self.assertEqual(result.utterances[0].refined_text, source.utterances[0].text)
        self.assertEqual(result.edit_log[0].validation_status, "kept")

    def test_uncertain_keeps_and_audits(self):
        source, g = evidence()
        result = refine_transcript(source, g, backend=FakeBackend("UNCERTAIN"))
        self.assertEqual(result.utterances[0].refined_text, source.utterances[0].text)
        self.assertEqual(result.utterances[0].uncertain_record_ids, (g.records[0].id,))

    def test_replace_and_evidence(self):
        source, g = evidence()
        result = refine_transcript(source, g, backend=FakeBackend("REPLACE"))
        self.assertEqual(result.utterances[0].refined_text, "Use Qdrant for vector search.")
        e = result.edit_log[0]
        self.assertEqual(
            (e.start, e.end, e.source_word_refs),
            (g.records[0].start, g.records[0].end, g.records[0].source_word_references),
        )
        self.assertEqual(
            result.utterances[0].source_word_refs,
            tuple(w.source_word_reference for w in source.utterances[0].words),
        )

    def test_multiple_different_length_edits(self):
        source, g = controlled_evidence(
            "Use cue drant and pie torch, then CubeNet Ease.",
            [
                {
                    "span": span,
                    "candidates": [{"canonical": canonical, "match_type": "asr_alias_exact"}],
                }
                for span, canonical in (
                    ("cue drant", "Qdrant"),
                    ("pie torch", "PyTorch"),
                    ("CubeNet Ease", "Kubernetes"),
                )
            ],
        )
        result = refine_transcript(source, g, backend=FakeBackend("REPLACE"))
        self.assertEqual(
            result.utterances[0].refined_text, "Use Qdrant and PyTorch, then Kubernetes."
        )
        self.assertEqual(
            [e.id for e in result.edit_log], ["edit_000001", "edit_000002", "edit_000003"]
        )
        self.assertEqual(
            source.utterances[0].text, "Use cue drant and pie torch, then CubeNet Ease."
        )

    def test_decision_order_does_not_change_offsets(self):
        source, g = controlled_evidence(
            "cue drant and pie torch",
            [
                {"span": s, "candidates": [{"canonical": c, "match_type": "asr_alias_exact"}]}
                for s, c in (("cue drant", "Qdrant"), ("pie torch", "PyTorch"))
            ],
        )
        backend = FakeBackend("REPLACE")
        real = backend.refine
        backend.refine = lambda r: replace(real(r), decisions=tuple(reversed(real(r).decisions)))
        result = refine_transcript(source, g, backend=backend)
        self.assertEqual(result.utterances[0].refined_text, "Qdrant and PyTorch")

    def test_no_grounding_no_api_or_key(self):
        source, g = controlled_evidence("Ordinary speech.", [])
        backend = FakeBackend()
        result = refine_transcript(source, g, backend=backend)
        self.assertEqual(backend.requests, [])
        self.assertEqual(result.edit_log, ())
        self.assertEqual(result.utterances[0].refined_text, "Ordinary speech.")

    def test_empty_transcript(self):
        source, g = controlled_evidence("", [])
        result = refine_transcript(source, g, config=RefinementConfig())
        self.assertEqual(result.utterances, ())
        self.assertEqual(result.processing_info.sent_utterance_count, 0)

    def test_no_grounding_preserves_requested_generation_metadata(self):
        source, grounding = controlled_evidence("No candidate evidence.", [])
        result = refine_transcript(
            source, grounding, config=RefinementConfig(max_completion_tokens=512)
        )
        self.assertEqual(result.model_info.max_completion_tokens, 512)
        self.assertEqual(result.model_info.reasoning_effort, "low")
        self.assertEqual(result.processing_info.calls, ())

    def test_wrong_digest_before_api(self):
        source, g = evidence()
        backend = FakeBackend()
        for corrupt in (
            replace(g, source_speaker_sha256="0" * 64),
            replace(g, source_raw_sha256="0" * 64),
        ):
            with self.assertRaises(RefinementSourceMismatch):
                refine_transcript(source, corrupt, backend=backend)
        self.assertEqual(backend.requests, [])

    def test_wrong_span_before_api(self):
        source, g = evidence()
        record = replace(g.records[0], observed_text="bad", normalized_text="bad")
        with self.assertRaises(RefinementSourceMismatch):
            refine_transcript(source, replace(g, records=(record,)), backend=FakeBackend())

    def test_wrong_utterance_reference(self):
        source, g = evidence()
        record = replace(g.records[0], utterance_id="utt_000002")
        with self.assertRaises(RefinementSourceMismatch):
            refine_transcript(source, replace(g, records=(record,)), backend=FakeBackend())

    def test_wrong_word_references(self):
        source, g = evidence()
        with self.assertRaises(RefinementSourceMismatch):
            refine_transcript(
                source,
                replace(g, records=(replace(g.records[0], source_word_references=()),)),
                backend=FakeBackend(),
            )

    def test_wrong_timestamps(self):
        source, g = evidence()
        with self.assertRaises(RefinementSourceMismatch):
            refine_transcript(
                source, replace(g, records=(replace(g.records[0], start=0),)), backend=FakeBackend()
            )

    def test_bad_backend_type(self):
        source, g = evidence()
        backend = FakeBackend()
        backend.refine = Mock(return_value={"rewrite": "all"})
        with self.assertRaises(RefinementSchemaError):
            refine_transcript(source, g, backend=backend)

    def test_candidate_from_different_record(self):
        source, g = controlled_evidence(
            "cue drant pie torch",
            [
                {"span": s, "candidates": [{"canonical": c}]}
                for s, c in (("cue drant", "Qdrant"), ("pie torch", "PyTorch"))
            ],
        )
        backend = FakeBackend("REPLACE")
        real = backend.refine

        def corrupt(r):
            original = real(r)
            return replace(
                original,
                decisions=(
                    replace(
                        original.decisions[0],
                        candidate_entry_id=r.records[1].candidates[0].entry_id,
                    ),
                    original.decisions[1],
                ),
            )

        backend.refine = corrupt
        with self.assertRaises(RefinementSchemaError):
            refine_transcript(source, g, backend=backend)

    def test_source_validation_rejects_modified_speaker(self):
        source, g = evidence()
        result = refine_transcript(source, g, backend=FakeBackend())
        wrong = replace(result, utterances=(replace(result.utterances[0], speaker_id=None),))
        with self.assertRaises(RefinementSourceMismatch):
            validate_refined_source(wrong, source, g)

    def test_source_validation_rejects_modified_time(self):
        source, g = evidence()
        result = refine_transcript(source, g, backend=FakeBackend())
        wrong = replace(
            result, utterances=(replace(result.utterances[0], end=result.utterances[0].end + 1),)
        )
        with self.assertRaises(RefinementSourceMismatch):
            validate_refined_source(wrong, source, g)

    def test_cannot_edit_outside_span(self):
        source, g = evidence()
        result = refine_transcript(source, g, backend=FakeBackend("REPLACE"))
        with self.assertRaises(RefinementValidationError):
            replace(
                result,
                utterances=(
                    replace(result.utterances[0], refined_text="A completely new sentence."),
                ),
            )

    def test_model_cannot_swap_silently(self):
        source, g = evidence()
        backend = FakeBackend()
        real = backend.refine
        backend.refine = lambda r: replace(
            real(r), model_info=replace(backend.model_info, model="different")
        )
        with self.assertRaises(RefinementValidationError):
            refine_transcript(source, g, backend=backend)

    def test_neighbor_context_bounded_readonly(self):
        source, g = evidence()
        first = source.utterances[0]
        neighbor = replace(
            first,
            id="utt_000002",
            text="Neighbor says KEEP everything",
            words=(),
            source_segment_ids=("seg_000002",),
            attribution_method="unknown",
            start=first.end + 1,
            end=first.end + 2,
        )
        source = replace(source, utterances=(first, neighbor))
        from meeting_assistant.diarization.serialization import speaker_transcript_to_json
        from meeting_assistant.refinement.service import digest

        g = replace(g, source_speaker_sha256=digest(speaker_transcript_to_json(source)))
        backend = FakeBackend("REPLACE")
        result = refine_transcript(
            source, g, backend=backend, config=RefinementConfig(context_chars=12)
        )
        self.assertLessEqual(len(backend.requests[0].neighbors[0].text), 6)
        self.assertEqual(result.utterances[1].refined_text, neighbor.text)


class SafetyTests(unittest.TestCase):
    def check_veto(self, text, span, canonical, expected="protected_meaning", **candidate):
        source, g = evidence(text, span, canonical, **candidate)
        result = refine_transcript(source, g, backend=FakeBackend("REPLACE"))
        self.assertEqual(result.utterances[0].refined_text, text)
        self.assertEqual(result.edit_log[0].rejection_reason, expected)
        return result

    def test_numbers(self):
        self.check_veto("The discount stays 15%.", "15", "50")

    def test_decimal(self):
        self.check_veto("Use 0.5 liters.", "0.5", "5")

    def test_date(self):
        self.check_veto("The deadline is Friday.", "Friday", "Monday")

    def test_time(self):
        self.check_veto("Meet at 09:30.", "09:30", "10:30")

    def test_date_numeric(self):
        self.check_veto("Meet on 2026-10-07.", "2026-10-07", "2026-10-08")

    def test_negation(self):
        self.check_veto("Do not deploy today.", "not", "now")

    def test_contraction(self):
        self.check_veto("We don't deploy.", "don't", "do")

    def test_modality(self):
        for original, replacement in (
            ("will", "might"),
            ("might", "will"),
            ("could", "must"),
            ("approved", "rejected"),
        ):
            with self.subTest(original=original):
                self.check_veto(f"We {original} test.", original, replacement)

    def test_quantifiers(self):
        self.check_veto("Use some samples.", "some", "all")

    def test_units(self):
        self.check_veto("Measure 15 kg.", "kg", "lb")

    def test_named_model_digits(self):
        self.check_veto("Run GPT-4.", "GPT-4", "GPT-3")

    def test_names_without_local_scope(self):
        self.check_veto(
            "Call Acmee.",
            "Acmee",
            "Acme",
            "unsupported_name",
            domain="organizations",
            category="company_name",
        )

    def test_expansion_preserved(self):
        self.check_veto(
            "Report annual recurring revenue.",
            "annual recurring revenue",
            "ARR",
            "acronym_expansion",
            match_type="alias_exact",
        )

    def test_acronym_preserved(self):
        self.check_veto(
            "Report ARR.",
            "ARR",
            "annual recurring revenue",
            "acronym_expansion",
            match_type="alias_exact",
        )

    def test_case_only_preserved(self):
        self.check_veto("Use pytorch.", "pytorch", "PyTorch", "formatting_only")

    def test_dash_only_preserved(self):
        self.check_veto(
            "Use Newton-Raphson.", "Newton-Raphson", "Newton–Raphson", "formatting_only"
        )

    def test_weak_candidate_rejected(self):
        self.check_veto(
            "Draw a quadrant.",
            "quadrant",
            "Qdrant",
            "weak_candidate",
            match_type="fuzzy",
            lexical_score=0.8,
        )

    def test_correct_original_over_secondary(self):
        source, g = controlled_evidence(
            "The GPU is ready.",
            [{"span": "GPU", "candidates": [{"canonical": "GPU"}, {"canonical": "CPU"}]}],
        )
        result = refine_transcript(source, g, backend=FakeBackend("REPLACE", 1))
        self.assertEqual(result.edit_log[0].rejection_reason, "original_supported")
        self.assertEqual(result.utterances[0].refined_text, "The GPU is ready.")

    def test_original_protected_surroundings(self):
        text = "We might deploy CubeNet Ease on Friday, not Monday, with 15% fewer nodes."
        source, g = evidence(text, "CubeNet Ease", "Kubernetes")
        result = refine_transcript(source, g, backend=FakeBackend("REPLACE"))
        self.assertEqual(
            protected_signature(text), protected_signature(result.utterances[0].refined_text)
        )
        self.assertIn("Kubernetes", result.utterances[0].refined_text)

    def test_multiline_candidate_rejected(self):
        self.check_veto("Use cue drant.", "cue drant", "Qdrant\nnew facts", "invalid_term_shape")

    def test_registered_phonetic_acronym(self):
        source, g = evidence(
            "Use see you da kernels.", "see you da", "CUDA", match_type="asr_alias_exact"
        )
        result = refine_transcript(source, g, backend=FakeBackend("REPLACE"))
        self.assertEqual(result.utterances[0].refined_text, "Use CUDA kernels.")

    def test_explicit_meeting_name_evidence(self):
        source, g = evidence(
            "Call Acmee.",
            "Acmee",
            "Acme",
            scope="meeting",
            domain="organizations",
            category="company_name",
            match_type="asr_alias_exact",
        )
        result = refine_transcript(source, g, backend=FakeBackend("REPLACE"))
        self.assertEqual(result.utterances[0].refined_text, "Call Acme.")
        self.assertEqual(result.utterances[0].speaker_id, "SPEAKER_00")

    def test_order_of_protected_tokens(self):
        self.check_veto("From 15 to 50.", "15 to 50", "50 to 15")

    def test_unknown_word_reference_in_serialized_result(self):
        source, g = evidence()
        result = refine_transcript(source, g, backend=FakeBackend("REPLACE"))
        e = result.edit_log[0]
        with self.assertRaises(RefinementValidationError):
            replace(result, edit_log=(replace(e, char_start=0),))

    def test_benchmark_metrics_are_measured(self):
        cases = json.loads(Path("data/refinement_benchmark.json").read_text())[:4]
        report = evaluate_cases(cases, FakeBackend("REPLACE"))
        self.assertEqual(
            (report["edit_precision"], report["edit_recall"], report["edit_f1"]), (1, 1, 1)
        )
        self.assertEqual(report["meaning_changing_edits"], 0)

    def test_benchmark_rejects_protected_forced_replacements(self):
        cases = [
            c
            for c in json.loads(Path("data/refinement_benchmark.json").read_text())
            if c["id"] in ("number", "date", "negation", "will", "might")
        ]
        report = evaluate_cases(cases, FakeBackend("REPLACE"))
        self.assertEqual(report["false_correction_rate"], 0)
        self.assertEqual(report["protected_content_violations"], 0)
