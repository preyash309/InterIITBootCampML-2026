import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from meeting_assistant.asr.audio import inspect_audio
from meeting_assistant.asr.config import ASRConfig
from meeting_assistant.asr.exceptions import ASRRateLimitError
from meeting_assistant.contextual_asr import ContextualASRConfig, contextual_retranscribe
from meeting_assistant.contextual_asr.evaluation import compare_saved_texts, term_metrics
from meeting_assistant.contextual_asr.models import AudioWindow
from meeting_assistant.contextual_asr.service import crop_window
from meeting_assistant.contextual_asr.transcribe import GroqContextualASRBackend

from .helpers import FakeASR, evidence, wav


class EvaluationTests(unittest.TestCase):
    def test_evaluation_rate_limit_retains_partial_report_without_claiming_completion(self):
        import json

        from meeting_assistant.asr.models import (
            ModelInfo,
            ProcessingInfo,
            TranscriptResult,
            TranscriptSegment,
        )
        from meeting_assistant.contextual_asr.evaluation import main
        from meeting_assistant.contextual_asr.serialization import context_to_json
        from meeting_assistant.refinement.exceptions import RefinementRateLimitError
        from tests.refinement.helpers import FakeBackend

        context, source, grounding = evidence()
        words = tuple(w.word for w in source.utterances[0].words)
        raw = TranscriptResult(
            source.source_raw_transcript_id,
            "en",
            None,
            source.duration_seconds,
            source.utterances[0].text,
            (TranscriptSegment("seg_000001", 0, words[-1].end, source.utterances[0].text, words),),
            ModelInfo("synthetic_fixture", "original_text"),
            ProcessingInfo(0, 0, 0, 1),
        )
        backend = FakeBackend()
        backend.refine = Mock(side_effect=RefinementRateLimitError("limited", status_code=429))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "canonical.wav"
            wav(path, source.duration_seconds)
            cp = root / "context.json"
            cp.write_text(context_to_json(context))
            result = contextual_retranscribe(
                path,
                source,
                grounding,
                context=context,
                config=ContextualASRConfig(enabled=True),
                backend=FakeASR(),
            )
            with (
                patch(
                    "meeting_assistant.contextual_asr.evaluation.read_environment", return_value={}
                ),
                patch("meeting_assistant.contextual_asr.evaluation.PyannoteBackend"),
                patch(
                    "meeting_assistant.contextual_asr.evaluation.transcribe_audio", return_value=raw
                ),
                patch("meeting_assistant.contextual_asr.evaluation.save_speaker_transcript"),
                patch(
                    "meeting_assistant.contextual_asr.evaluation.reconcile_transcript",
                    return_value=source,
                ),
                patch(
                    "meeting_assistant.contextual_asr.evaluation.prepare_comparison",
                    return_value=(grounding, grounding, result, root / "bundle"),
                ),
                patch(
                    "meeting_assistant.contextual_asr.evaluation.GroqRefinerBackend",
                    return_value=backend,
                ),
            ):
                status = main(
                    [
                        "--audio",
                        str(path),
                        "--canonical",
                        "--context",
                        str(cp),
                        "--output-dir",
                        str(root),
                    ]
                )
            report = json.loads((root / "evaluation.json").read_text())
            self.assertEqual(status, 2)
            self.assertFalse(report["complete"])
            self.assertEqual(
                report["processing"]["baseline"]["error_code"], "refinement_rate_limit_error"
            )
            self.assertNotIn("closed_loop", report["conditions"])
            self.assertFalse(list(root.glob("*.tmp")))

    def test_comparison_reuses_only_identical_phase_v_requests(self):
        from meeting_assistant.contextual_asr.evaluation import ComparisonBackend
        from meeting_assistant.refinement import refine_transcript
        from tests.refinement.helpers import FakeBackend

        _, source, grounding = evidence()
        delegate = FakeBackend()
        backend = ComparisonBackend(delegate)
        first = refine_transcript(source, grounding, backend=backend)
        second = refine_transcript(source, grounding, backend=backend)
        self.assertEqual(first.utterances, second.utterances)
        self.assertEqual(len(delegate.requests), 1)
        self.assertEqual(backend.cache_hits, 1)
        seeded = ComparisonBackend(FakeBackend())
        seeded.seed(first, source, grounding)
        refine_transcript(source, grounding, backend=seeded)
        self.assertEqual(seeded.cache_hits, 1)
        self.assertEqual(seeded.backend.requests, [])

    def test_term_precision_recall_and_false_substitution(self):
        metrics = term_metrics(
            "Use Qdrant, not CUDA", "Use Qdrant and CUDA and CUDA", ("Qdrant", "CUDA")
        )
        self.assertEqual(metrics["recall"], 1)
        self.assertEqual(metrics["precision"], 2 / 3)
        self.assertEqual(metrics["false_terms"], 1)

    def test_term_matching_respects_boundaries_and_case(self):
        self.assertEqual(term_metrics("Qdrant", "qdrantish", ("Qdrant",))["true_terms"], 0)
        self.assertEqual(term_metrics("Qdrant", "qdrant", ("Qdrant",))["true_terms"], 1)

    def test_empty_reference_does_not_fabricate_metric(self):
        result = compare_saved_texts("Use GPU", {"baseline": "Use GPU"})
        self.assertNotIn("wer", result["baseline"])
        self.assertIsNone(term_metrics("ordinary speech", "ordinary speech", ("CUDA",))["recall"])

    def test_verified_reference_wer_and_meaning_checks(self):
        result = compare_saved_texts(
            "not Monday", {"closed_loop": "Monday"}, reference="not Monday"
        )
        self.assertEqual(result["closed_loop"]["wer"]["wer"], 0.5)
        self.assertEqual(result["closed_loop"]["changes_from_pass1"]["negation_changes"], 1)

    def test_native_timestamps_offset_without_raw_retiming(self):
        payload = {
            "text": "Qdrant",
            "duration": 2,
            "segments": [{"start": 0.2, "end": 1, "text": "Qdrant"}],
            "words": [{"start": 0.2, "end": 1, "word": "Qdrant"}],
        }
        backend = GroqContextualASRBackend(ASRConfig(api_key="fake-key"))
        with patch(
            "meeting_assistant.contextual_asr.transcribe.request_transcription",
            return_value=payload,
        ) as request:
            _, segments = backend.transcribe_window(
                Path("unused"),
                start=10,
                duration=2,
                prompt="Possible vocabulary: Qdrant.",
                timeout=5,
            )
        self.assertEqual(segments[0].words[0].start, 10.2)
        self.assertEqual(segments[0].words[0].end, 11)
        self.assertEqual(request.call_args.kwargs["prompt"], "Possible vocabulary: Qdrant.")

    def test_pcm_slice_exact_frames(self):
        import wave

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.wav"
            target = Path(directory) / "target.wav"
            wav(source, 3)
            audio = inspect_audio(source, ASRConfig())
            digest = crop_window(audio, AudioWindow("win_000001", 1, 2, ("grnd_000001",)), target)
            with wave.open(str(target)) as stream:
                self.assertEqual(stream.getnframes(), 16000)
                self.assertEqual(stream.readframes(16000), b"\x01\x00" * 16000)
            self.assertEqual(len(digest), 64)

    def test_rate_limit_stops_remaining_windows(self):
        context, source, grounding = evidence()
        # Select controlled windows while preserving real provenance checks on the source record.
        windows = (
            AudioWindow("win_000001", 0, 1, ("grnd_000001",)),
            AudioWindow("win_000002", 1, 2, ("grnd_000001",)),
        )
        backend = FakeASR(error=ASRRateLimitError("limited", status_code=429))
        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "meeting_assistant.contextual_asr.service.select_windows",
                return_value=(windows, ()),
            ),
        ):
            path = Path(directory) / "canonical.wav"
            wav(path, source.duration_seconds)
            result = contextual_retranscribe(
                path,
                source,
                grounding,
                context=context,
                backend=backend,
                config=ContextualASRConfig(enabled=True),
            )
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(result.skipped[0].reason, "provider_unavailable")

    def test_malformed_backend_becomes_enhancement_failure(self):
        context, source, grounding = evidence()
        backend = Mock(provider="fixture", model="fake")
        backend.transcribe_window.return_value = ("Qdrant", ())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "canonical.wav"
            wav(path, source.duration_seconds)
            result = contextual_retranscribe(
                path,
                source,
                grounding,
                context=context,
                backend=backend,
                config=ContextualASRConfig(enabled=True),
            )
        self.assertEqual(result.hypotheses[0].status, "failed")


@unittest.skipUnless(
    os.environ.get("RUN_CONTEXT_ASR_API_TESTS") == "1", "Opt-in billed contextual ASR test"
)
class LiveContextualTests(unittest.TestCase):
    def test_real_native_window_transcription(self):
        from meeting_assistant.contextual_asr.models import RetrievedTerm
        from meeting_assistant.contextual_asr.retrieval import build_prompt
        from meeting_assistant.contextual_asr.service import file_digest

        path = Path(os.environ["CONTEXT_ASR_TEST_AUDIO"])
        config = ASRConfig.from_env()
        audio = inspect_audio(path, config)
        backend = GroqContextualASRBackend(config)
        window = AudioWindow("win_000001", 0, min(12, audio.duration_seconds), ("grnd_000001",))
        terms = (RetrievedTerm("test.kubernetes", "Kubernetes", "meeting", ("ctx_test",)),)
        prompt, _ = build_prompt(terms)
        before = file_digest(path)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "window.wav"
            crop_window(audio, window, target)
            text, segments = backend.transcribe_window(
                target, start=0, duration=window.end, prompt=prompt, timeout=60
            )
        self.assertTrue(text.strip())
        self.assertTrue(any(segment.words for segment in segments))
        self.assertEqual(file_digest(path), before)
