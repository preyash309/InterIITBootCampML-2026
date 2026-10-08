import hashlib
import json
import os
import subprocess
import tempfile
import unittest
import wave
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from unittest.mock import Mock, patch

from meeting_assistant.diarization.reconciliation import reconcile_transcript
from meeting_assistant.diarization.serialization import (
    diarization_to_json,
    speaker_transcript_to_json,
)
from meeting_assistant.speaker_reliability import (
    SecondaryDiarizationResult,
    SecondarySpeakerSegment,
    SpeakerReliabilityConfig,
    assess_speaker_reliability,
    compare_diarization,
    save_speaker_reliability,
)
from meeting_assistant.speaker_reliability.alignment import optimal_assignment
from meeting_assistant.speaker_reliability.exceptions import (
    InvalidReliability,
    SecondaryTimeout,
    SecondaryUnavailable,
)
from meeting_assistant.speaker_reliability.secondary import SortformerBackend
from meeting_assistant.speaker_reliability.serialization import fingerprint, render_text, to_json
from meeting_assistant.speaker_reliability.service import run_optional
from tests.diarization.helpers import diar, raw, word


def secondary(turns, *, duration=10, sha="a" * 64):
    segments = tuple(
        sorted(
            (SecondarySpeakerSegment(*x) for x in turns),
            key=lambda x: (x.start, x.end, x.speaker_id),
        )
    )
    return SecondaryDiarizationResult(
        segments,
        "independent-fixture",
        "test-model",
        "revision",
        "1",
        "cpu",
        sha,
        duration,
        "b" * 64,
        0,
        0,
        0,
    )


def comparison(primary_turns, secondary_turns, *, words=None, regular=None):
    p = diar(primary_turns, regular=regular)
    speaker = reconcile_transcript(raw(tuple(words or [word()])), p)
    return compare_diarization(p, speaker, secondary(secondary_turns))


class AlignmentTests(unittest.TestCase):
    def test_dependency_free_assignment_matches_global_optimum(self):
        import builtins

        original = builtins.__import__

        def without_scipy(name, *args, **kwargs):
            if name == "scipy.optimize":
                raise ImportError("No optional SciPy installed")
            return original(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=without_scipy):
            self.assertEqual(optimal_assignment([[9, 8], [8, 0]]), (1, 0))
            self.assertEqual(optimal_assignment([[1, 1], [1, 1]]), (0, 1))
            self.assertEqual(optimal_assignment([[0, 0]]), (None,))

    def test_permutation(self):
        r = comparison((("a", 0, 5), ("b", 5, 10)), (("z", 0, 5), ("x", 5, 10)))
        self.assertEqual(
            [(x.secondary_id, x.primary_id) for x in r.comparison.alignment.mappings],
            [("x", "SPEAKER_01"), ("z", "SPEAKER_00")],
        )
        self.assertEqual(r.comparison.agreement_fraction, 1)

    def test_global_optimum_not_greedy(self):
        self.assertEqual(optimal_assignment([[9, 8], [8, 0]]), (1, 0))

    def test_tie_lexicographic(self):
        self.assertEqual(optimal_assignment([[1, 1], [1, 1]]), (0, 1))

    def test_zero_overlap_not_forced(self):
        self.assertEqual(optimal_assignment([[0, 0]]), (None,))

    def test_short_fragment_preserved(self):
        self.assertEqual(optimal_assignment([[0, 1e-8], [1e-8, 0]]), (1, 0))

    def test_more_secondary_speakers(self):
        r = comparison((("a", 0, 10),), (("x", 0, 4), ("y", 4, 10)))
        a = r.comparison.alignment
        self.assertTrue(a.count_mismatch)
        self.assertEqual(a.unmapped_secondary, ("x",))
        self.assertEqual(a.possible_splits, (("SPEAKER_00", ("x", "y")),))

    def test_fewer_secondary_merge(self):
        r = comparison((("a", 0, 5), ("b", 5, 10)), (("x", 0, 10),))
        self.assertEqual(r.comparison.alignment.unmapped_primary, ("SPEAKER_01",))
        self.assertTrue(r.comparison.alignment.possible_merges)

    def test_stability(self):
        p = diar((("a", 0, 5), ("b", 5, 10)))
        sp = reconcile_transcript(raw((word(),)), p)
        s = secondary((("y", 0, 5), ("x", 5, 10)))
        a, b = compare_diarization(p, sp, s), compare_diarization(p, sp, s)
        self.assertEqual(a.words, b.words)
        self.assertEqual(a.utterances, b.utterances)
        self.assertEqual(a.comparison.intervals, b.comparison.intervals)

    def test_empty_secondary(self):
        r = comparison((("a", 0, 10),), ())
        self.assertEqual(r.comparison.alignment.secondary_count, 0)
        self.assertEqual(r.comparison.intervals[0].state, "PRIMARY_ONLY")

    def test_duplicate_same_label_turns_not_double_counted(self):
        r = comparison((("a", 0, 10),), (("x", 0, 10), ("x", 1, 5)))
        self.assertEqual(r.comparison.alignment.mappings[0].overlap_seconds, 10)


class ObservationTests(unittest.TestCase):
    def test_fully_agreed_word_and_utterance(self):
        r = comparison((("a", 0, 10),), (("x", 0, 10),))
        self.assertEqual(r.words[0].agreement_fraction, 1)
        self.assertEqual(r.words[0].status, "RELIABLE")
        self.assertEqual(r.utterances[0].status, "RELIABLE")

    def test_partial_word_crossing_disagreement(self):
        r = comparison(
            (("a", 0, 5), ("b", 5, 10)),
            (("x", 0, 2), ("y", 2, 3), ("x", 3, 5), ("y", 5, 10)),
            words=[word(start=1.5, end=2.5)],
        )
        self.assertAlmostEqual(r.words[0].agreement_fraction, 0.5)
        self.assertAlmostEqual(r.words[0].disagreement_fraction, 0.5)
        self.assertEqual(r.utterances[0].status, "UNCERTAIN")

    def test_secondary_silence(self):
        r = comparison((("a", 0, 10),), (("x", 3, 10),))
        self.assertEqual(r.words[0].primary_only_fraction, 1)
        self.assertEqual(r.words[0].disagreement_fraction, 0)

    def test_secondary_only_and_shared_silence(self):
        r = comparison((("a", 1, 5),), (("x", 2, 6),))
        states = {x.state for x in r.comparison.intervals}
        self.assertEqual(states, {"SILENCE", "PRIMARY_ONLY", "AGREE", "SECONDARY_ONLY"})
        self.assertEqual(r.comparison.silence_seconds, 5)

    def test_overlap_agreement_keeps_exclusive(self):
        r = comparison(
            (("a", 0, 5), ("b", 5, 10)),
            (("x", 0, 6), ("y", 1, 10)),
            regular=(("a", 0, 6), ("b", 1, 10)),
            words=[word(start=2, end=3)],
        )
        self.assertIn("OVERLAP_AGREE", {x.state for x in r.comparison.intervals})
        self.assertEqual(r.words[0].primary_overlap_fraction, 1)
        self.assertEqual(r.words[0].secondary_overlap_fraction, 1)
        self.assertEqual(r.words[0].overlap_agreement_fraction, 1)
        self.assertEqual(r.words[0].status, "UNCERTAIN")

    def test_overlap_disagreement(self):
        r = comparison(
            (("a", 0, 5), ("b", 5, 10)),
            (("x", 0, 5), ("y", 5, 10)),
            regular=(("a", 0, 6), ("b", 1, 10)),
        )
        self.assertIn("OVERLAP_DISAGREE", {x.state for x in r.comparison.intervals})

    def test_unmapped_word(self):
        r = comparison((("a", 0, 10),), (("x", 0, 3), ("y", 3, 10)))
        self.assertEqual(r.words[0].unmapped_fraction, 1)
        self.assertIn("unmapped_secondary", r.words[0].reasons)

    def test_boundary_deltas_compare_turns_not_asr_utterance(self):
        r = comparison((("a", 0, 5), ("b", 5, 10)), (("x", 0.1, 5.2), ("y", 5.2, 10)))
        self.assertAlmostEqual(r.utterances[0].start_delta_seconds, 0.1)
        self.assertAlmostEqual(r.utterances[0].end_delta_seconds, 0.2)
        self.assertAlmostEqual(r.utterances[0].boundary_disagreement_seconds, 0.3)

    def test_zero_duration_word(self):
        r = comparison((("a", 0, 10),), (("x", 0, 10),), words=[word(start=1, end=1)])
        self.assertEqual(r.words[0].status, "UNCERTAIN")
        self.assertEqual(r.words[0].agreement_fraction, 0)

    def test_mixed_threshold(self):
        r = comparison(
            (("a", 0, 5), ("b", 5, 10)),
            (("x", 0, 1.7), ("y", 1.7, 2), ("x", 2, 5), ("y", 5, 10)),
            words=[word(start=1, end=2)],
        )
        self.assertEqual(r.words[0].status, "MIXED")

    def test_invariants_primary_unchanged(self):
        p = diar()
        sp = reconcile_transcript(raw((word(),)), p)
        before = (diarization_to_json(p), speaker_transcript_to_json(sp))
        r = compare_diarization(p, sp, secondary((("x", 0, 10),)))
        self.assertEqual(before, (diarization_to_json(p), speaker_transcript_to_json(sp)))
        self.assertEqual(
            r.words[0].source_word_reference, sp.utterances[0].words[0].source_word_reference
        )
        self.assertIsNotNone(r.get_speaker_reliability(sp.utterances[0].id))
        self.assertIsNone(r.get_speaker_reliability("missing"))

    def test_frozen(self):
        r = comparison((("a", 0, 10),), (("x", 0, 10),))
        with self.assertRaises(FrozenInstanceError):
            r.words[0].status = "UNCERTAIN"

    def test_nan_and_invalid_fraction(self):
        r = comparison((("a", 0, 10),), (("x", 0, 10),))
        for value in (float("nan"), -1, 2):
            with self.subTest(value=value), self.assertRaises(InvalidReliability):
                replace(r.words[0], agreement_fraction=value)

    def test_bad_secondary_segments(self):
        for start, end in ((-1, 2), (1, 1), (2, 1), (float("nan"), 2)):
            with self.subTest(start=start), self.assertRaises(InvalidReliability):
                SecondarySpeakerSegment("x", start, end)

    def test_source_mismatch(self):
        p = diar()
        sp = reconcile_transcript(raw((word(),)), p)
        with self.assertRaises(InvalidReliability):
            compare_diarization(p, sp, secondary((("x", 0, 10),), sha="c" * 64))

    def test_primary_link_mismatch(self):
        p = diar()
        sp = reconcile_transcript(raw((word(),)), p)
        with self.assertRaises(InvalidReliability):
            compare_diarization(diar(), sp, secondary((("x", 0, 10),)))


class ConfigAndOutputTests(unittest.TestCase):
    def test_default_disabled(self):
        self.assertFalse(SpeakerReliabilityConfig.from_env(environ={}).enabled)

    def test_env(self):
        c = SpeakerReliabilityConfig.from_env(
            environ={
                "SPEAKER_RELIABILITY_ENABLED": "true",
                "SECONDARY_DIARIZER_DEVICE": "cpu",
                "SPEAKER_RELIABILITY_TIMEOUT_SECONDS": "3",
            }
        )
        self.assertTrue(c.enabled)
        self.assertEqual(c.device, "cpu")
        self.assertEqual(c.timeout_seconds, 3)

    def test_invalid_env(self):
        for key, value in (
            ("SPEAKER_RELIABILITY_ENABLED", "yes"),
            ("SECONDARY_DIARIZER_DEVICE", "auto"),
            ("SPEAKER_RELIABILITY_TIMEOUT_SECONDS", "nan"),
            ("SPEAKER_RELIABILITY_MAX_SPEAKERS", "5"),
        ):
            with self.subTest(key=key), self.assertRaises(InvalidReliability):
                SpeakerReliabilityConfig.from_env(environ={key: value})

    def test_threshold_order(self):
        with self.assertRaises(InvalidReliability):
            SpeakerReliabilityConfig(reliable_threshold=0.5, mixed_threshold=0.8)

    def test_default_no_backend(self):
        with patch(
            "meeting_assistant.speaker_reliability.service.assess_speaker_reliability"
        ) as mock:
            self.assertIsNone(run_optional(None, None, None, None, environ={}))
            mock.assert_not_called()

    def test_serialization_atomic_idempotent_utf8(self):
        r = comparison((("a", 0, 10),), (("独立", 0, 10),))
        with tempfile.TemporaryDirectory() as root:
            a = save_speaker_reliability(r, root)
            self.assertEqual(a, save_speaker_reliability(r, root))
            self.assertEqual(len(list(a.iterdir())), 4)
            self.assertEqual(
                json.loads((a / "speaker_reliability.json").read_text(encoding="utf-8"))[
                    "availability"
                ],
                "available",
            )
            self.assertFalse(list(a.parent.glob(".partial-*")))
            self.assertIn("独立", (a / "speaker_reliability.txt").read_text(encoding="utf-8"))

    def test_failed_write_cleanup(self):
        r = comparison((("a", 0, 10),), (("x", 0, 10),))
        with (
            tempfile.TemporaryDirectory() as root,
            patch("pathlib.Path.rename", side_effect=OSError("disk")),
        ):
            with self.assertRaises(OSError):
                save_speaker_reliability(r, root)
            self.assertFalse(list((Path(root) / "speaker_reliability").iterdir()))

    def test_provenance_fingerprints(self):
        p = diar()
        sp = reconcile_transcript(raw((word(),)), p)
        s = secondary((("x", 0, 10),))
        r = compare_diarization(p, sp, s)
        self.assertEqual(r.provenance.secondary_result_sha256, fingerprint(s))
        self.assertEqual(
            r.provenance.primary_diarization_sha256,
            hashlib.sha256(diarization_to_json(p).encode()).hexdigest(),
        )
        self.assertEqual(to_json(r), to_json(r))
        self.assertIn("not probabilities", render_text(r))


class FailureTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "fixture.wav"
        with wave.open(str(self.path), "wb") as w:
            w.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            w.writeframes(b"\0\0" * 160000)
        self.sha = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.primary = replace(diar(), audio_sha256=self.sha)
        self.speaker = reconcile_transcript(raw((word(),)), self.primary)

    def assess(self, backend):
        return assess_speaker_reliability(self.path, self.primary, self.speaker, backend=backend)

    def test_success(self):
        r = self.assess(Mock(diarize=Mock(return_value=secondary((("x", 0, 10),), sha=self.sha))))
        self.assertEqual(r.availability, "available")

    def test_optional_failures_sanitized(self):
        for error in (
            SecondaryUnavailable("C:/private/path"),
            RuntimeError("load"),
            SecondaryTimeout("timeout"),
            ValueError("inference"),
        ):
            with self.subTest(error=error):
                r = self.assess(Mock(diarize=Mock(side_effect=error)))
                self.assertEqual(r.availability, "unavailable")
                self.assertNotIn("private", to_json(r))
                self.assertEqual(self.speaker.utterances[0].speaker_id, "SPEAKER_00")

    def test_malformed_backend(self):
        self.assertEqual(
            self.assess(Mock(diarize=Mock(return_value={}))).availability, "unavailable"
        )

    def test_missing_input(self):
        self.path.unlink()
        self.assertEqual(self.assess(Mock()).availability, "unavailable")

    def test_noncanonical(self):
        with wave.open(str(self.path), "wb") as w:
            w.setparams((2, 2, 8000, 0, "NONE", "not compressed"))
            w.writeframes(b"\0" * 32000)
        self.assertEqual(self.assess(Mock()).availability, "unavailable")

    def test_missing_secondary_environment(self):
        backend = SortformerBackend(SpeakerReliabilityConfig(python_path=Path("missing-python")))
        with self.assertRaises(SecondaryUnavailable):
            backend.diarize(self.path)

    def test_missing_model(self):
        backend = SortformerBackend(
            SpeakerReliabilityConfig(python_path=self.path, model_path=Path("missing-checkpoint"))
        )
        with self.assertRaises(SecondaryUnavailable):
            backend.diarize(self.path)

    def test_subprocess_timeout(self):
        config = SpeakerReliabilityConfig(python_path=self.path, model_path=self.path)
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("worker", 0.01)):
            with self.assertRaises(SecondaryTimeout):
                SortformerBackend(config).diarize(self.path)

    def test_subprocess_arguments_not_shell_and_no_credentials(self):
        config = SpeakerReliabilityConfig(python_path=self.path, model_path=self.path)
        with (
            patch.dict(os.environ, {"GROQ_API_KEY": "test-secret"}),
            patch("subprocess.run") as call,
        ):
            call.return_value = subprocess.CompletedProcess([], 1, "", "failed")
            with self.assertRaises(SecondaryUnavailable):
                SortformerBackend(config).diarize(self.path)
            kwargs = call.call_args.kwargs
            self.assertFalse(kwargs["shell"])
            self.assertEqual(kwargs["env"]["HF_HUB_OFFLINE"], "1")
            self.assertNotIn("GROQ_API_KEY", kwargs["env"])
            command = call.call_args.args[0]
            owned_output = Path(command[command.index("--output") + 1])
            # Other real jobs may be running; assert cleanup only for this invocation.
            self.assertFalse(owned_output.parent.exists())


@unittest.skipUnless(
    os.environ.get("RUN_SECONDARY_DIARIZER_TESTS") == "1", "Opt-in local Sortformer inference"
)
class RealSecondaryTests(unittest.TestCase):
    def test_offline_model_and_source_unchanged(self):
        path = Path(os.environ["SECONDARY_TEST_AUDIO"])
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        result = SortformerBackend(SpeakerReliabilityConfig.from_env()).diarize(path)
        self.assertGreater(len(result.segments), 0)
        self.assertEqual(result.audio_sha256, before)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)


if __name__ == "__main__":
    unittest.main()
