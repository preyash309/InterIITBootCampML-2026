import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from meeting_assistant.intelligence.models import (
    ActionItem,
    ActionOwner,
    Decision,
    MeetingContent,
    SummaryPoint,
)
from meeting_assistant.semantic_reasoning import (
    SemanticConfig,
    analyze_meeting_semantics,
    from_json,
    save_semantics,
    to_json,
    validate_semantic_source,
)
from meeting_assistant.semantic_reasoning.candidates import relation_candidates
from meeting_assistant.semantic_reasoning.exceptions import InvalidSemantics, SemanticSourceMismatch
from meeting_assistant.semantic_reasoning.graph import build_evolution, build_graph
from meeting_assistant.semantic_reasoning.ontology import EVENT_ONTOLOGY, RELATION_ONTOLOGY
from meeting_assistant.semantic_reasoning.service import DecisionSession, run_optional
from tests.semantic_reasoning.helpers import FakeBackend, evidence, scripted


class SemanticTests(unittest.TestCase):
    def run_case(
        self,
        texts,
        labels,
        content=None,
        relations=None,
        verification=None,
        coverage="YES",
        **config,
    ):
        sources = evidence(texts, content)
        backend = scripted(labels, relations, verification, coverage)
        result = analyze_meeting_semantics(
            *sources, backend=backend, config=SemanticConfig(enabled=True, **config), environ={}
        )
        return result, sources, backend

    def test_all_event_labels(self):
        for label, _ in EVENT_ONTOLOGY:
            with self.subTest(label=label):
                r, _, _ = self.run_case(["Evidence text."], [label])
                self.assertEqual(r.events[0].event_type, label)
                self.assertEqual(bool(r.event_graph.event_ids), label not in ("NONE", "AMBIGUOUS"))

    def test_proposal_unresolved_no_decision(self):
        r, _, _ = self.run_case(["We could use Redis."], ["PROPOSAL"])
        self.assertEqual(r.decision_evolution[0].current_decision_ids, ())
        self.assertEqual(r.decision_evolution[0].unresolved_proposal_ids, ("evt_000001",))

    def test_proposal_acceptance(self):
        a, b = "We could use Redis.", "Agreed, use Redis."
        r, _, _ = self.run_case([a, b], ["PROPOSAL", "DECISION"], relations={(a, b): "ACCEPTS"})
        self.assertEqual(r.decision_evolution[0].unresolved_proposal_ids, ())
        self.assertEqual(r.decision_evolution[0].current_decision_ids, ("evt_000002",))

    def test_proposal_rejection(self):
        a, b = "We could use Redis.", "No, reject Redis."
        r, _, _ = self.run_case([a, b], ["PROPOSAL", "REJECTION"], relations={(a, b): "REJECTS"})
        self.assertEqual(r.decision_evolution[0].unresolved_proposal_ids, ())
        self.assertEqual(r.decision_evolution[0].current_decision_ids, ())

    def test_decision_reversal(self):
        a, b = "We'll ship Friday.", "Actually ship Monday instead."
        r, _, _ = self.run_case([a, b], ["DECISION", "DECISION"], relations={(a, b): "SUPERSEDES"})
        chain = r.decision_evolution[0]
        self.assertEqual(chain.historical_decision_ids, ("evt_000001",))
        self.assertEqual(chain.current_decision_ids, ("evt_000002",))

    def test_question_answer(self):
        a, b = "What is the latency?", "The latency is 20 milliseconds."
        r, _, _ = self.run_case([a, b], ["QUESTION", "ANSWER"], relations={(a, b): "ANSWERS"})
        self.assertEqual(r.relations[0].relation_type, "ANSWERS")

    def test_separate_topics(self):
        r, _, _ = self.run_case(["Use Redis.", "Ship Monday."], ["DECISION", "DECISION"])
        self.assertEqual(len(r.event_graph.issue_threads), 2)

    def test_general_cycle_retained(self):
        a, b = "Use Redis.", "Ship Monday."
        r, _, _ = self.run_case([a, b], ["DECISION", "DECISION"], relations={(a, b): "SUPPORTS"})
        reverse = replace(
            r.relations[0],
            id="rel_000002",
            source_event_id="evt_000002",
            target_event_id="evt_000001",
        )
        graph = build_graph(r.events, r.relations + (reverse,))
        self.assertEqual(graph.cycles, (("evt_000001", "evt_000002"),))
        self.assertEqual(
            build_evolution(r.events, r.relations + (reverse,), graph)[0].ordered_event_ids,
            ("evt_000001", "evt_000002"),
        )

    def test_invalid_supersession_abstained(self):
        a, b = "Maybe Redis.", "Use Redis."
        r, _, _ = self.run_case([a, b], ["PROPOSAL", "DECISION"], relations={(a, b): "SUPERSEDES"})
        self.assertFalse(r.relations)
        self.assertIn("invalid_supersession_abstained", r.warnings)

    def test_supersession_backwards_rejected(self):
        a, b = "Ship Friday.", "Ship Monday."
        r, _, _ = self.run_case([a, b], ["DECISION", "DECISION"], relations={(a, b): "SUPERSEDES"})
        reverse = replace(
            r.relations[0], source_event_id="evt_000002", target_event_id="evt_000001"
        )
        with self.assertRaises(InvalidSemantics):
            build_graph(r.events, (reverse,))

    def test_unknown_event_rejected(self):
        r, _, _ = self.run_case(["Text."], ["INFORMATION"])
        with self.assertRaises(InvalidSemantics):
            replace(r.events[0], event_type="ALIEN")

    def test_unknown_relation_rejected(self):
        a, b = "Question?", "Answer."
        r, _, _ = self.run_case([a, b], ["QUESTION", "ANSWER"], relations={(a, b): "ANSWERS"})
        with self.assertRaises(InvalidSemantics):
            replace(r.relations[0], relation_type="ALIEN")

    def test_all_relations_closed(self):
        a, b = "Ship Friday.", "Actually ship Monday."
        for label, _ in RELATION_ONTOLOGY:
            r, _, _ = self.run_case([a, b], ["DECISION", "DECISION"], relations={(a, b): label})
            self.assertEqual(len(r.relations), int(label not in ("NONE", "AMBIGUOUS")))

    def test_immutable_round_trip_unicode(self):
        r, _, _ = self.run_case(["Café demo."], ["INFORMATION"])
        self.assertEqual(from_json(to_json(r)), r)
        self.assertIn("Café", to_json(r))
        with self.assertRaises(FrozenInstanceError):
            r.events[0].text = "changed"
        with self.assertRaises(InvalidSemantics):
            replace(r, events=list(r.events))

    def test_json_missing_extra_duplicate_fields(self):
        r, _, _ = self.run_case(["Text."], ["INFORMATION"])
        data = json.loads(to_json(r))
        for invalid in ({**data, "extra": 1}, {k: v for k, v in data.items() if k != "events"}):
            with self.assertRaises(InvalidSemantics):
                from_json(json.dumps(invalid))
        with self.assertRaises(InvalidSemantics):
            from_json('{"events": [], "events": []}')

    def test_atomic_idempotent_bundle(self):
        r, _, _ = self.run_case(["Use Redis."], ["DECISION"])
        with tempfile.TemporaryDirectory() as directory:
            destination = save_semantics(r, directory)
            self.assertEqual(destination, save_semantics(r, directory))
            self.assertEqual(len(list(destination.iterdir())), 9)
            self.assertFalse(list(destination.parent.glob(".partial-*")))
            (destination / "meeting_events.json").write_text("bad")
            with self.assertRaises(InvalidSemantics):
                save_semantics(r, directory)

    def test_atomic_write_failure_cleaned(self):
        r, _, _ = self.run_case(["Use Redis."], ["DECISION"])
        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "meeting_assistant.semantic_reasoning.serialization.os.fsync", side_effect=OSError
            ),
        ):
            with self.assertRaises(OSError):
                save_semantics(r, directory)
            self.assertFalse(list((Path(directory) / "semantic_reasoning").iterdir()))

    def test_source_wrong_hash_or_meeting(self):
        r, sources, _ = self.run_case(["Text."], ["INFORMATION"])
        for changed in (
            replace(r.provenance, refined_sha256="0" * 64),
            replace(r.provenance, meeting_record_id=str(uuid4())),
        ):
            with self.assertRaises(SemanticSourceMismatch):
                validate_semantic_source(replace(r, provenance=changed), *sources)

    def test_source_unknown_utterance_timestamp_text_speaker(self):
        r, sources, _ = self.run_case(["Text."], ["INFORMATION"])
        for changes in (
            {"evidence_utterance_ids": ("utt_999999",)},
            {"start": 0.01},
            {"text": "Invented text."},
            {"speaker_id": "SPEAKER_09"},
            {"context_utterance_ids": ("utt_999999",)},
        ):
            with self.assertRaises((SemanticSourceMismatch, InvalidSemantics)):
                validate_semantic_source(
                    replace(r, events=(replace(r.events[0], **changes),)), *sources
                )

    def test_cross_meeting_sources_rejected_before_calls(self):
        sources = evidence(["Text."])
        other = evidence(["Other."])
        backend = FakeBackend()
        with self.assertRaises(SemanticSourceMismatch):
            analyze_meeting_semantics(
                sources[0],
                other[1],
                sources[2],
                backend=backend,
                config=SemanticConfig(enabled=True),
                environ={},
            )
        self.assertFalse(backend.calls)

    def test_unknown_relation_evidence_rejected(self):
        a, b = "Question?", "Answer."
        r, sources, _ = self.run_case([a, b], ["QUESTION", "ANSWER"], relations={(a, b): "ANSWERS"})
        changed = replace(
            r.relations[0],
            evidence_utterance_ids=r.relations[0].evidence_utterance_ids + ("utt_999999",),
        )
        with self.assertRaises(SemanticSourceMismatch):
            validate_semantic_source(replace(r, relations=(changed,)), *sources)

    def test_disabled_default_no_provider(self):
        sources = evidence(["Text."])
        backend = FakeBackend()
        result = analyze_meeting_semantics(*sources, backend=backend, environ={})
        self.assertEqual(result.availability, "disabled")
        self.assertFalse(backend.calls)
        self.assertIsNone(run_optional(*sources, "ignored", environ={}))

    def test_missing_access_unavailable_without_record_mutation(self):
        d = Decision("dec_0001", "Use Redis.", ("utt_000001",))
        sources = evidence(["Use Redis."], MeetingContent(decisions=(d,)))
        before = to_json(sources)
        r = analyze_meeting_semantics(
            *sources,
            config=SemanticConfig(enabled=True, provider="typesafe", model="jev-1.13.0"),
            environ={},
        )
        self.assertEqual(r.availability, "unavailable")
        self.assertEqual(r.verification[0].status, "UNAVAILABLE")
        self.assertEqual(before, to_json(sources))
        self.assertFalse(r.processing.calls)

    def test_ambiguous_abstained_policy(self):
        for label, probability, outcome in (
            ("DECISION", 0.7, "ambiguous"),
            ("DECISION", 0.4, "abstained"),
        ):
            r, _, _ = self.run_case(["Maybe."], [(label, probability)])
            self.assertEqual(r.events[0].outcome, outcome)
            self.assertFalse(r.event_graph.event_ids)

    def test_stage_and_total_budgets(self):
        r, _, backend = self.run_case(
            ["One.", "Two.", "Three."], ["INFORMATION"] * 3, max_event_calls=2, max_total_calls=1
        )
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(r.availability, "partial")
        self.assertIn("event_candidates_truncated", r.warnings)

    def test_state_budget_no_silent_truncation(self):
        r, _, backend = self.run_case(["Evidence " * 100], ["INFORMATION"], max_state_chars=100)
        self.assertEqual(r.availability, "unavailable")
        self.assertFalse(backend.calls)

    def test_relation_window_bounded(self):
        r, _, _ = self.run_case(
            [f"Topic {n}." for n in range(40)], ["INFORMATION"] * 40, max_relation_calls=2
        )
        pairs, _ = relation_candidates(
            r.events, SemanticConfig(max_relation_calls=1000, relation_window=3)
        )
        self.assertLessEqual(len(pairs), 40 * 3)
        self.assertEqual(r.processing.evaluated_relation_pairs, 2)

    def test_dedup_per_meeting(self):
        from meeting_assistant.semantic_reasoning.models import Question
        from meeting_assistant.semantic_reasoning.verification import YES_NO

        backend = FakeBackend(lambda *_: "YES")
        session = DecisionSession(backend, SemanticConfig())
        q = (Question("q", "Question?", YES_NO),)
        first = session.ask({"text": "Evidence"}, q, "verification", "semantic_verification_v1")
        self.assertEqual(
            session.ask({"text": "Evidence"}, q, "verification", "semantic_verification_v1"), first
        )
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(session.cached, 1)

    def test_config_validation(self):
        for changes in (
            {"enabled": 1},
            {"acceptance_probability": float("nan")},
            {"max_event_calls": 0},
            {"relation_window": 100},
            {"model": "invented"},
        ):
            with self.assertRaises(InvalidSemantics):
                SemanticConfig(**changes)
        self.assertTrue(SemanticConfig.from_env(environ={"JEV_ENABLED": "true"}).enabled)
        with self.assertRaises(InvalidSemantics):
            SemanticConfig.from_env(environ={"JEV_ENABLED": "perhaps"})

    def test_valid_decision_verification(self):
        d = Decision("dec_0001", "Use Redis.", ("utt_000001",))
        r, _, _ = self.run_case(["Use Redis."], ["DECISION"], MeetingContent(decisions=(d,)))
        self.assertEqual(r.get_verification(d.id).status, "SUPPORTED")

    def test_invalid_decision_proposal_rejection_negation_number_scope(self):
        d = Decision("dec_0001", "Deploy 50 instances.", ("utt_000001",))
        for dimension in (
            "claim_supported",
            "item_type_supported",
            "negation_preserved",
            "numbers_preserved",
            "scope_preserved",
        ):
            r, _, _ = self.run_case(
                ["Do not deploy 15 instances."],
                ["DECISION"],
                MeetingContent(decisions=(d,)),
                verification={dimension: "NO"},
            )
            self.assertEqual(r.verification[0].status, "UNSUPPORTED")

    def test_action_existing_owner_deadline_only(self):
        action = ActionItem(
            "act_0001",
            "Benchmark models.",
            ("utt_000001",),
            ActionOwner("named_entity", display_text="Rahul"),
            "Friday",
        )
        r, _, backend = self.run_case(
            ["Rahul, benchmark models by Friday."],
            ["TASK_ASSIGNMENT"],
            MeetingContent(action_items=(action,)),
        )
        self.assertEqual(r.verification[0].status, "SUPPORTED")
        questions = [q.id for _, qs, p in backend.requests if p == "verification" for q in qs]
        self.assertIn("owner_explicit", questions)
        self.assertIn("deadline_explicit", questions)

    def test_missing_owner_deadline_not_invented(self):
        action = ActionItem("act_0001", "Benchmark models.", ("utt_000001",))
        r, _, backend = self.run_case(
            ["I will benchmark models."], ["COMMITMENT"], MeetingContent(action_items=(action,))
        )
        questions = [q.id for _, qs, p in backend.requests if p == "verification" for q in qs]
        self.assertNotIn("owner_explicit", questions)
        self.assertNotIn("deadline_explicit", questions)
        self.assertEqual(r.verification[0].status, "SUPPORTED")

    def test_invalid_owner_deadline_or_suggestion(self):
        action = ActionItem(
            "act_0001",
            "Benchmark.",
            ("utt_000001",),
            ActionOwner("named_entity", display_text="Rahul"),
            "Friday",
        )
        for dimension in ("owner_explicit", "deadline_explicit", "item_type_supported"):
            r, _, _ = self.run_case(
                ["Rahul mentioned Friday. Maybe someone should benchmark."],
                ["PROPOSAL"],
                MeetingContent(action_items=(action,)),
                verification={dimension: "NO"},
            )
            self.assertEqual(r.verification[0].status, "UNSUPPORTED")

    def test_verification_review(self):
        d = Decision("dec_0001", "Use Redis.", ("utt_000001",))
        r, _, _ = self.run_case(
            ["Use Redis."],
            ["DECISION"],
            MeetingContent(decisions=(d,)),
            verification={"claim_supported": "AMBIGUOUS"},
        )
        self.assertEqual(r.verification[0].status, "REVIEW")

    def test_coverage_exact_and_paraphrase(self):
        for claim in ("Use Redis.", "Select Redis for the cache."):
            d = Decision("dec_0001", claim, ("utt_000001",))
            r, _, _ = self.run_case(["Use Redis."], ["DECISION"], MeetingContent(decisions=(d,)))
            self.assertEqual(r.coverage[0].status, "covered")

    def test_missing_decision_action_commitment_blocker(self):
        for label in ("DECISION", "TASK_ASSIGNMENT", "COMMITMENT", "BLOCKER"):
            r, _, _ = self.run_case(["Explicit evidence."], [label])
            self.assertEqual(r.coverage[0].status, "missing")

    def test_coverage_duplicates(self):
        content = MeetingContent(
            decisions=(
                Decision("dec_0001", "Use Redis.", ("utt_000001",)),
                Decision("dec_0002", "Select Redis.", ("utt_000001",)),
            )
        )
        r, _, _ = self.run_case(["Use Redis."], ["DECISION"], content)
        self.assertEqual(len(r.coverage[0].matched_record_ids), 2)
        self.assertIn("multiple_record_matches_review_duplicates", r.coverage[0].warnings)

    def test_coverage_wrong_type(self):
        content = MeetingContent(summary=(SummaryPoint("sum_0001", "Use Redis.", ("utt_000001",)),))
        r, _, _ = self.run_case(["Use Redis."], ["DECISION"], content)
        self.assertEqual(r.coverage[0].status, "ambiguous")
        self.assertIn("wrong_record_type:sum_0001", r.coverage[0].warnings)

    def test_coverage_semantic_rejection(self):
        d = Decision("dec_0001", "Reject Redis.", ("utt_000001",))
        r, _, _ = self.run_case(
            ["Use Redis."], ["DECISION"], MeetingContent(decisions=(d,)), coverage="NO"
        )
        self.assertEqual(r.coverage[0].status, "missing")

    def test_optional_failure_does_not_propagate(self):
        with patch(
            "meeting_assistant.semantic_reasoning.service.analyze_meeting_semantics",
            side_effect=RuntimeError,
        ):
            self.assertIsNone(
                run_optional(None, None, None, "ignored", environ={"JEV_ENABLED": "true"})
            )

    def test_invalid_optional_config_no_provider(self):
        sources = evidence(["Text."])
        with tempfile.TemporaryDirectory() as directory:
            bundle = run_optional(
                *sources,
                directory,
                environ={
                    "JEV_ENABLED": "true",
                    "JEV_MODEL": "invalid",
                    "TYPESAFE_API_KEY": "never-send",
                },
            )
            r = from_json((bundle / "semantic_result.json").read_text())
            self.assertEqual(r.availability, "unavailable")
            self.assertFalse(r.processing.calls)
