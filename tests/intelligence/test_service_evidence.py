import hashlib
import json
import math
import tempfile
import unittest
import wave
from array import array
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

from meeting_assistant.intelligence.config import IntelligenceConfig
from meeting_assistant.intelligence.evidence import (
    bind_audio,
    evidence_manifest,
    extract_evidence_clip,
    get_item_evidence,
    resolve_meeting_record_evidence,
)
from meeting_assistant.intelligence.exceptions import (
    ConsolidationError,
    EvidenceAudioError,
    IntelligenceSourceMismatch,
    IntelligenceValidationError,
    IntelligenceWriteError,
    MeetingTooLargeError,
    UnknownEvidenceError,
)
from meeting_assistant.intelligence.fixtures import text_evidence
from meeting_assistant.intelligence.models import (
    EvidenceSpan,
    IntelligenceRequest,
    IntelligenceResponse,
    MeetingContent,
    SummaryPoint,
)
from meeting_assistant.intelligence.serialization import (
    meeting_record_from_json,
    meeting_record_to_json,
    render_meeting_markdown,
    save_meeting_record,
)
from meeting_assistant.intelligence.service import build_chunks, extract_meeting_record
from meeting_assistant.intelligence.validation import parse_consolidation

from .helpers import FakeBackend, evidence


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.refined, self.speaker, self.grounding = evidence()
        self.backend = FakeBackend()

    def run_stage(self, **options):
        return extract_meeting_record(
            self.refined, self.speaker, self.grounding, backend=self.backend, **options
        )

    def test_upstream_unchanged_and_ids(self):
        before = tuple(asdict(v) for v in (self.refined, self.speaker, self.grounding))
        result = self.run_stage()
        self.assertEqual(result.summary[0].id, "sum_0001")
        self.assertEqual(
            tuple(asdict(v) for v in (self.refined, self.speaker, self.grounding)), before
        )
        self.assertIsNone(result.audio)

    def test_source_mismatch_before_backend(self):
        self.grounding = replace(self.grounding, source_speaker_sha256="0" * 64)
        with self.assertRaises(IntelligenceSourceMismatch):
            self.run_stage()
        self.assertEqual(self.backend.requests, [])

    def test_wrong_refined_before_backend(self):
        self.refined = replace(self.refined, source_speaker_sha256="0" * 64)
        with self.assertRaises(IntelligenceSourceMismatch):
            self.run_stage()
        self.assertEqual(self.backend.requests, [])

    def test_empty_no_backend_calls(self):
        self.refined, self.speaker, self.grounding = text_evidence([])
        record = self.run_stage()
        self.assertEqual(record.content.items, ())
        self.assertEqual(self.backend.requests, [])
        self.assertEqual(record.processing_info.usage(), 0)

    def test_oversized_utterance_rejected_before_call(self):
        self.refined, self.speaker, self.grounding = evidence("speech " * 100)
        with self.assertRaises(MeetingTooLargeError):
            self.run_stage(config=IntelligenceConfig(max_chunk_chars=256))
        self.assertEqual(self.backend.requests, [])

    def test_chunk_budget_rejected_before_call(self):
        self.refined, self.speaker, self.grounding = text_evidence(
            [{"text": f"Point {n}: " + "speech " * 8} for n in range(6)]
        )
        with self.assertRaises(MeetingTooLargeError):
            self.run_stage(config=IntelligenceConfig(max_chunk_chars=350, max_chunks=1))
        self.assertEqual(self.backend.requests, [])

    def test_chunk_boundaries_overlap_dedup_consolidation(self):
        self.refined, self.speaker, self.grounding = text_evidence(
            [{"text": f"Point {n}: " + "speech " * 8} for n in range(6)]
        )
        config = IntelligenceConfig(max_chunk_chars=640, overlap_utterances=1)
        chunks = build_chunks(self.refined, config)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(
            {u.utterance_id for c in chunks for u in c.utterances},
            {u.utterance_id for u in self.refined.utterances},
        )
        self.assertTrue(all(u in self.refined.utterances for c in chunks for u in c.utterances))
        self.assertTrue(
            any(
                set(u.utterance_id for u in a.utterances)
                & set(u.utterance_id for u in b.utterances)
                for a, b in zip(chunks, chunks[1:])
            )
        )
        result = self.run_stage(config=config)
        self.assertEqual(len(result.summary), 6)
        self.assertEqual(self.backend.requests[-1].stage, "consolidation")
        self.assertEqual(
            tuple(s.evidence_utterance_ids[0] for s in result.summary),
            tuple(u.utterance_id for u in self.refined.utterances),
        )

    def test_consolidation_budget_failure_no_publication(self):
        self.refined, self.speaker, self.grounding = text_evidence(
            [{"text": f"Point {n}: " + "speech " * 8} for n in range(6)]
        )
        with self.assertRaises(MeetingTooLargeError):
            self.run_stage(
                config=IntelligenceConfig(max_chunk_chars=640, max_consolidation_chars=256)
            )
        self.assertFalse(any(r.stage == "consolidation" for r in self.backend.requests))

    def test_backend_identity_checked(self):
        request = IntelligenceRequest("wrong", self.refined.utterances)
        response = IntelligenceResponse("wrong", MeetingContent(), self.backend.model_info)
        with (
            patch.object(self.backend, "extract", return_value=response),
            self.assertRaises(IntelligenceValidationError),
        ):
            self.run_stage()
        self.assertEqual(request.request_id, "wrong")

    def test_backend_unknown_evidence_checked(self):
        self.backend = FakeBackend(
            lambda _: MeetingContent(
                summary=(SummaryPoint("sum_0001", "A claim", ("utt_999999",)),)
            )
        )
        with self.assertRaises(UnknownEvidenceError):
            self.run_stage()

    def test_resolver_exact_source_times_words_raw(self):
        result = self.run_stage()
        span = get_item_evidence(result, "sum_0001", self.refined)[0]
        source = self.refined.utterances[0]
        self.assertEqual(
            (
                span.start,
                span.end,
                span.speaker_id,
                span.raw_text,
                span.refined_text,
                span.source_word_refs,
            ),
            (
                source.start,
                source.end,
                source.speaker_id,
                source.raw_text,
                source.refined_text,
                source.source_word_refs,
            ),
        )
        self.assertEqual(resolve_meeting_record_evidence(result, self.refined)["sum_0001"], (span,))

    def test_wrong_refined_resolver_fails(self):
        result = self.run_stage()
        other, _, _ = evidence("Other speech.")
        with self.assertRaises(IntelligenceSourceMismatch):
            get_item_evidence(result, "sum_0001", other)

    def test_unknown_item(self):
        with self.assertRaises(UnknownEvidenceError):
            get_item_evidence(self.run_stage(), "../path", self.refined)

    def test_serialization_roundtrip(self):
        result = self.run_stage()
        self.assertEqual(meeting_record_from_json(meeting_record_to_json(result)), result)
        self.assertEqual(meeting_record_to_json(result), meeting_record_to_json(result))

    def test_saved_record_rejects_tampering(self):
        data = json.loads(meeting_record_to_json(self.run_stage()))
        for changes in ({"schema_version": "future"}, {"unknown": 1}, {"source_raw_sha256": "bad"}):
            with self.subTest(changes=changes), self.assertRaises(IntelligenceValidationError):
                meeting_record_from_json(json.dumps({**data, **changes}))

    def test_manifest_bound_and_derived(self):
        record = self.run_stage()
        manifest = evidence_manifest(record, self.refined)
        self.assertEqual(manifest["source_refined_sha256"], record.source_refined_sha256)
        self.assertEqual(
            manifest["items"]["sum_0001"][0]["refined_text"],
            self.refined.utterances[0].refined_text,
        )

    def test_markdown_untrusted_html_escaped(self):
        self.refined, self.speaker, self.grounding = evidence(
            "<script>alert('x')</script> [Click](bad)"
        )
        rendered = render_meeting_markdown(self.run_stage(), self.refined)
        self.assertNotIn("<script>", rendered)
        self.assertNotIn("[Click](bad)", rendered)

    def test_atomic_bundle_and_no_overwrite(self):
        record = self.run_stage()
        with tempfile.TemporaryDirectory() as root:
            files = save_meeting_record(record, self.refined, root)
            self.assertEqual(
                set(p.name for p in files.json_path.parent.iterdir()),
                {"meeting_record.json", "meeting_record.md", "evidence_manifest.json"},
            )
            with self.assertRaises(IntelligenceWriteError):
                save_meeting_record(record, self.refined, root)
            self.assertFalse(list(Path(root).glob(".pending-*")))

    def test_failed_publication_cleanup(self):
        with tempfile.TemporaryDirectory() as root:
            with (
                patch(
                    "meeting_assistant.intelligence.serialization.os.rename",
                    side_effect=OSError("denied"),
                ),
                self.assertRaises(IntelligenceWriteError),
            ):
                save_meeting_record(self.run_stage(), self.refined, root)
            self.assertEqual(list(Path(root).iterdir()), [])


class ConsolidationTests(unittest.TestCase):
    def setUp(self):
        self.items = (
            SummaryPoint("p0001_sum_0001", "Use PostgreSQL.", ("utt_000001",)),
            SummaryPoint("p0002_sum_0001", "Use PostgreSQL.", ("utt_000002",)),
        )
        self.request = IntelligenceRequest(
            "consolidation_0001", (), "consolidation", MeetingContent(summary=self.items)
        )

    def data(self, primary="p0001_sum_0001", merge=None, **extra):
        return json.dumps(
            {
                "summary": [
                    {
                        "primary_item_id": primary,
                        "merge_item_ids": merge if merge is not None else ["p0002_sum_0001"],
                        **extra,
                    }
                ],
                "minutes": [],
                "decisions": [],
                "action_items": [],
            }
        )

    def test_union_is_application_derived(self):
        item = parse_consolidation(self.data(), self.request).summary[0]
        self.assertEqual(item.text, self.items[0].text)
        self.assertEqual(item.evidence_utterance_ids, ("utt_000001", "utt_000002"))

    def test_new_evidence_claim_fields_rejected(self):
        for changes in ({"text": "New claim"}, {"evidence_utterance_ids": ["utt_999999"]}):
            with self.subTest(changes=changes), self.assertRaises(IntelligenceValidationError):
                parse_consolidation(self.data(**changes), self.request)

    def test_unknown_partial_id(self):
        with self.assertRaises(ConsolidationError):
            parse_consolidation(self.data(primary="new_id"), self.request)

    def test_duplicate_partial(self):
        with self.assertRaises(ConsolidationError):
            parse_consolidation(self.data(merge=["p0001_sum_0001"]), self.request)

    def test_superseded_selection(self):
        # Selection drops earlier candidate; it never synthesizes a new final claim.
        result = parse_consolidation(self.data(primary="p0002_sum_0001", merge=[]), self.request)
        self.assertEqual(result.summary, (self.items[1],))


class AudioEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "canonical.wav"
        self.samples = array("h", (n % 1000 for n in range(32000)))
        with wave.open(str(self.source), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            stream.writeframes(self.samples.tobytes())
        self.sha = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.span = EvidenceSpan(
            "utt_000001", "SPEAKER_00", 0.10001, 0.30001, "raw", "refined", (), (), ()
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_exact_frames_pcm_and_preservation(self):
        clip = extract_evidence_clip(self.source, self.span, self.root / "clip.wav")
        start = math.floor(0.10001 * 16000)
        end = math.ceil(0.30001 * 16000)
        self.assertEqual((clip.start_frame, clip.end_frame), (start, end))
        self.assertEqual((clip.evidence_start, clip.evidence_end), (0.10001, 0.30001))
        with wave.open(str(clip.path), "rb") as stream:
            self.assertEqual(
                (stream.getframerate(), stream.getnchannels(), stream.getsampwidth()), (16000, 1, 2)
            )
            self.assertEqual(stream.readframes(end - start), self.samples[start:end].tobytes())
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), self.sha)
        self.assertFalse(list(self.root.glob(".pending-*")))

    def test_padding_clamps(self):
        span = replace(self.span, start=0.1, end=2.1)
        clip = extract_evidence_clip(self.source, span, self.root / "clip.wav", padding_seconds=0.5)
        self.assertEqual((clip.start_frame, clip.end_frame, clip.duration_seconds), (0, 32000, 2))
        self.assertEqual(clip.evidence_end, 2.1)

    def test_zero_outside_and_invalid_ranges(self):
        for start, end in ((0.1, 0.1), (2, 3)):
            with self.subTest(start=start), self.assertRaises(EvidenceAudioError):
                extract_evidence_clip(
                    self.source, replace(self.span, start=start, end=end), self.root / "clip.wav"
                )
        for padding in (-1, float("nan"), True):
            with self.subTest(padding=padding), self.assertRaises(EvidenceAudioError):
                extract_evidence_clip(
                    self.source, self.span, self.root / "clip.wav", padding_seconds=padding
                )

    def test_no_overwrite_source_or_clip(self):
        with self.assertRaises(EvidenceAudioError):
            extract_evidence_clip(self.source, self.span, self.source)
        extract_evidence_clip(self.source, self.span, self.root / "clip.wav")
        with self.assertRaises(EvidenceAudioError):
            extract_evidence_clip(self.source, self.span, self.root / "clip.wav")

    def test_audio_binding_digest(self):
        from meeting_assistant.intelligence.models import AudioReference

        with self.assertRaises(EvidenceAudioError):
            extract_evidence_clip(
                self.source,
                self.span,
                self.root / "clip.wav",
                audio_reference=AudioReference("0" * 64, 2, 32000),
            )

    def test_noncanonical_missing_corrupt(self):
        bad = self.root / "bad.wav"
        bad.write_text("random")
        for path in (bad, self.root / "missing.wav", self.root):
            with self.subTest(path=path), self.assertRaises(EvidenceAudioError):
                extract_evidence_clip(path, self.span, self.root / "clip.wav")

    def test_failed_clip_cleanup(self):
        with patch(
            "meeting_assistant.intelligence.evidence.os.link", side_effect=OSError("denied")
        ):
            with self.assertRaises(EvidenceAudioError):
                extract_evidence_clip(self.source, self.span, self.root / "clip.wav")
        self.assertFalse(list(self.root.glob(".pending-*")))
        self.assertFalse((self.root / "clip.wav").exists())

    def test_bind_wrong_duration(self):
        _, speaker, _ = evidence()
        with self.assertRaises(EvidenceAudioError):
            bind_audio(self.source, speaker)
