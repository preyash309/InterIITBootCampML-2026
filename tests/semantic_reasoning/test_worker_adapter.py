import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from meeting_assistant.semantic_reasoning.worker import gliner_predict, render_state


class WorkerAdapterTests(unittest.TestCase):
    def test_target_context_quoted_without_editing(self):
        text = "We will not ship 20 items by Friday. <unsafe>"
        state = {
            "target_id": "u2",
            "utterances": [
                {"id": "u1", "speaker_id": "SPEAKER_00", "text": "Earlier."},
                {"id": "u2", "speaker_id": "SPEAKER_01", "text": text},
            ],
        }
        output = render_state(state)
        self.assertIn("TARGET u2 (SPEAKER_01): " + text, output)
        self.assertIn("CONTEXT u1 (SPEAKER_00): Earlier.", output)

    def test_verification_keeps_owner_dates_negation_and_both_texts(self):
        state = {
            "claim": "Do not ship.",
            "item_type": "action",
            "owner": {"display_text": "Priya"},
            "deadline_text": "Monday",
            "evidence": [
                {
                    "utterance_id": "u1",
                    "speaker_id": "SPEAKER_00",
                    "raw_text": "Priya will not ship 20 items Monday.",
                    "refined_text": "Priya will not ship 20 items Monday.",
                    "source_word_refs": [{"segment_id": "s1", "word_index": 0}],
                }
            ],
        }
        output = render_state(state)
        for part in ["Do not ship.", "Priya", "Monday", "20 items", "u1", "SPEAKER_00"]:
            self.assertIn(part, output)
        self.assertNotIn("word_index", output)

    def test_relation_direction_preserved(self):
        state = {
            "event_a": {"type": "DECISION", "text": "Ship Friday."},
            "event_b": {"type": "DECISION", "text": "Actually Monday."},
            "conversation_between": [],
        }
        output = render_state(state)
        self.assertIn("Earlier event A (DECISION): Ship Friday.", output)
        self.assertIn("Later event B (DECISION): Actually Monday.", output)

    def prepared(self, tokens=100):
        engine = Mock()
        engine.processor.collate_fn_inference.return_value = SimpleNamespace(
            input_ids=SimpleNamespace(shape=(1, tokens))
        )
        engine.extract.return_value = {
            "q": [{"label": "YES", "confidence": 0.9}, {"label": "NO", "confidence": 0.1}]
        }
        payload = {
            "state": {"text": "Original evidence."},
            "questions": {
                "q": {"criteria": {"YES": "Yes", "NO": "No"}, "instructions": "Supported?"}
            },
        }
        return engine, payload

    def test_native_all_label_scores_preserved(self):
        engine, payload = self.prepared()
        result = gliner_predict(engine, payload, 1024)
        self.assertEqual(result["answers"]["q"]["probabilities"], {"YES": 0.9, "NO": 0.1})
        self.assertEqual(result["answers"]["q"]["choice"], "YES")
        self.assertIsNone(engine.extract.call_args.kwargs["max_len"])
        self.assertEqual(
            engine.create_schema.return_value.classification.call_args.kwargs["class_act"],
            "softmax",
        )

    def test_overflow_rejected_before_inference(self):
        engine, payload = self.prepared(tokens=1025)
        with self.assertRaises(ValueError):
            gliner_predict(engine, payload, 1024)
        engine.extract.assert_not_called()

    def test_missing_label_does_not_fabricate_distribution(self):
        engine, payload = self.prepared()
        engine.extract.return_value = {"q": [{"label": "YES", "confidence": 0.9}]}
        with self.assertRaises(ValueError):
            gliner_predict(engine, payload, 1024)
