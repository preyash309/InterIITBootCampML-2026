import hashlib
import unittest
from dataclasses import FrozenInstanceError

from meeting_assistant.asr.models import TranscriptSegment
from meeting_assistant.asr.serialization import transcript_to_json
from meeting_assistant.diarization.config import ReconciliationConfig
from meeting_assistant.diarization.exceptions import ReconciliationError
from meeting_assistant.diarization.reconciliation import find_overlap_regions, reconcile_transcript

from .helpers import diar, raw, word


class ReconciliationTests(unittest.TestCase):
    def assigned(self, words, turns, **kwargs):
        return reconcile_transcript(raw(tuple(words)), diar(turns), **kwargs)

    def test_word_inside_turn(self):
        result = self.assigned([word()], (("a", 0, 10),))
        assigned = result.utterances[0].words[0]
        self.assertEqual(assigned.speaker_id, "SPEAKER_00")
        self.assertEqual(assigned.overlap_fraction, 1)
        self.assertEqual(assigned.asr_probability, 0.8)

    def test_maximum_overlap_at_boundary(self):
        result = self.assigned([word(start=1.9, end=2.5)], (("a", 0, 2), ("b", 2, 10)))
        self.assertEqual(result.utterances[0].speaker_id, "SPEAKER_01")
        self.assertAlmostEqual(result.utterances[0].words[0].overlap_fraction, 5 / 6)

    def test_exact_tie_uses_earliest_turn(self):
        result = self.assigned([word(start=1.5, end=2.5)], (("a", 0, 2), ("b", 2, 10)))
        self.assertEqual(result.utterances[0].speaker_id, "SPEAKER_00")

    def test_overlap_aggregates_same_speaker_intervals(self):
        result = self.assigned([word(start=0, end=3)], (("a", 0, 1), ("b", 1, 2), ("a", 2, 3)))
        assigned = result.utterances[0].words[0]
        self.assertEqual(assigned.speaker_id, "SPEAKER_00")
        self.assertAlmostEqual(assigned.overlap_fraction, 2 / 3)
        self.assertEqual(len(assigned.source_turn_ids), 2)

    def test_small_gap_tolerance(self):
        result = self.assigned([word(start=2.04, end=2.08)], (("a", 0, 2),))
        self.assertEqual(result.reconciliation_info.tolerance_assignments, 1)
        self.assertEqual(result.utterances[0].words[0].overlap_fraction, 0)

    def test_nearest_fallback(self):
        result = self.assigned([word(start=2.15, end=2.18)], (("a", 0, 2),))
        self.assertEqual(result.reconciliation_info.nearest_assignments, 1)

    def test_large_gap_remains_unknown(self):
        result = self.assigned([word(start=4, end=4.5)], (("a", 0, 2),))
        self.assertIsNone(result.utterances[0].speaker_id)
        self.assertEqual(result.reconciliation_info.unassigned_words, 1)
        self.assertEqual(result.reconciliation_info.assignment_coverage, 0)

    def test_disabled_fallbacks(self):
        result = self.assigned(
            [word(start=2.01, end=2.03)], (("a", 0, 2),), config=ReconciliationConfig(0, 0)
        )
        self.assertEqual(result.reconciliation_info.unassigned_words, 1)

    def test_segment_spanning_speakers_is_split(self):
        result = self.assigned(
            [word("We", 1, 1.4), word("agree", 2.5, 3)], (("a", 0, 2), ("b", 2, 10))
        )
        self.assertEqual([u.speaker_id for u in result.utterances], ["SPEAKER_00", "SPEAKER_01"])
        self.assertEqual([u.id for u in result.utterances], ["utt_000001", "utt_000002"])
        self.assertEqual(result.reconciliation_info.speaker_switches, 1)

    def test_short_backchannel_retained(self):
        result = self.assigned(
            [word("Deploy", 1, 1.4), word("yeah", 2.01, 2.1), word("tomorrow", 2.3, 2.7)],
            (("a", 0, 2), ("b", 2, 2.2), ("a", 2.2, 10)),
        )
        self.assertEqual(
            [u.speaker_id for u in result.utterances], ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]
        )
        self.assertEqual(result.utterances[1].text, "yeah")

    def test_missing_word_timestamps_falls_back_to_segment(self):
        segment = TranscriptSegment("seg_000001", 1, 3, " Original text. ")
        result = reconcile_transcript(raw(segments=(segment,)), diar())
        self.assertEqual(result.utterances[0].text, segment.text)
        self.assertEqual(result.utterances[0].attribution_method, "segment_overlap")
        self.assertEqual(result.reconciliation_info.segment_fallback_count, 1)
        self.assertIsNone(result.reconciliation_info.assignment_coverage)

    def test_regular_overlap_preserved_with_single_exclusive_assignment(self):
        result = reconcile_transcript(
            raw((word(start=2, end=2.5),)),
            diar((("a", 0, 3), ("b", 3, 10)), regular=(("a", 0, 4), ("b", 1, 10))),
        )
        self.assertTrue(result.utterances[0].overlap_present)
        self.assertEqual(result.utterances[0].speaker_id, "SPEAKER_00")
        self.assertEqual((result.overlap_regions[0].start, result.overlap_regions[0].end), (1, 4))

    def test_adjacent_speakers_are_not_overlap(self):
        self.assertEqual(find_overlap_regions(diar((("a", 0, 2), ("b", 2, 10))).regular_turns), ())

    def test_overlapping_turns_of_same_speaker_are_not_overlap(self):
        self.assertEqual(
            find_overlap_regions(diar(regular=(("a", 0, 5), ("a", 2, 10))).regular_turns), ()
        )

    def test_nested_regular_overlap_regions(self):
        regions = find_overlap_regions(
            diar((("a", 0, 10),), regular=(("a", 0, 10), ("b", 1, 8), ("c", 2, 4))).regular_turns
        )
        self.assertEqual(
            [(r.start, r.end, len(r.speakers)) for r in regions], [(1, 2, 2), (2, 4, 3), (4, 8, 2)]
        )

    def test_no_diarization_fails_cleanly(self):
        with self.assertRaises(ReconciliationError):
            reconcile_transcript(raw((word(),)), diar(()))

    def test_zero_words_empty_result(self):
        result = reconcile_transcript(raw(), diar(()))
        self.assertEqual(result.utterances, ())
        self.assertEqual(result.reconciliation_info.total_words, 0)

    def test_multiple_segments_preserve_references(self):
        segments = (
            TranscriptSegment("seg_000001", 1, 2, "Hello", (word("Hello", 1, 1.5),)),
            TranscriptSegment("seg_000002", 2, 3, " world!", (word(" world!", 2, 2.5),)),
        )
        source = raw(segments=segments)
        result = reconcile_transcript(source, diar())
        self.assertEqual(result.utterances[0].source_segment_ids, ("seg_000001", "seg_000002"))
        self.assertEqual(result.utterances[0].text, "Hello world!")
        self.assertIs(result.utterances[0].words[1].word, source.segments[1].words[0])

    def test_raw_record_unchanged_and_frozen(self):
        source = raw((word("DO NOT", 1, 1.5), word("modify.", 1.5, 2)))
        before = transcript_to_json(source)
        result = reconcile_transcript(source, diar())
        self.assertEqual(before, transcript_to_json(source))
        self.assertEqual(result.source_raw_sha256, hashlib.sha256(before.encode()).hexdigest())
        with self.assertRaises(FrozenInstanceError):
            result.utterances[0].words[0].word.text = "changed"

    def test_long_silence_splits_same_speaker(self):
        result = self.assigned([word(start=1, end=2), word(start=5, end=6)], (("a", 0, 10),))
        self.assertEqual(len(result.utterances), 2)
        self.assertEqual(result.reconciliation_info.speaker_switches, 0)

    def test_zero_duration_word_at_switch(self):
        result = self.assigned([word(start=2, end=2)], (("a", 0, 2), ("b", 2, 10)))
        self.assertEqual(result.utterances[0].speaker_id, "SPEAKER_01")
        self.assertIsNone(result.utterances[0].words[0].overlap_fraction)

    def test_duration_mismatch(self):
        with self.assertRaises(ReconciliationError):
            reconcile_transcript(raw((word(),)), diar(duration=12))

    def test_wrong_input_types(self):
        with self.assertRaises(ReconciliationError):
            reconcile_transcript({}, diar())

    def test_nearest_tie_uses_previous(self):
        result = self.assigned(
            [word(start=2.15, end=2.25)],
            (("a", 0, 2), ("b", 2.4, 10)),
            config=ReconciliationConfig(0, 0.2),
        )
        self.assertEqual(result.utterances[0].speaker_id, "SPEAKER_00")
