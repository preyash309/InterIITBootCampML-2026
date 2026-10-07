"""Regression for a real Groq boundary interval found during Phase III integration."""

import unittest

from meeting_assistant.asr.api_backend import parse_response


class SegmentMembershipTests(unittest.TestCase):
    def test_gap_word_overlaps_next_segment_preserving_native_timestamps(self):
        payload = {
            "text": "finish. I can review",
            "segments": [
                {"start": 19.199999, "end": 22.38, "text": " Let us check before we finish."},
                {"start": 23.16, "end": 26.42, "text": " I can review."},
            ],
            "words": [
                {"word": "finish.", "start": 21.88, "end": 22.3},
                {"word": "I", "start": 22.66, "end": 23.5},
                {"word": "can", "start": 23.5, "end": 23.7},
            ],
        }
        _, _, _, segments = parse_response(
            payload, offset=0, chunk_duration=27.687125, segment_offset=0
        )
        self.assertEqual([w.text for w in segments[1].words], ["I", "can"])
        self.assertEqual((segments[1].words[0].start, segments[1].words[0].end), (22.66, 23.5))
        self.assertEqual((segments[1].start, segments[1].end), (23.16, 26.42))
        self.assertEqual(segments[1].text, payload["segments"][1]["text"])
