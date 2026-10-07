import contextlib
import hashlib
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from meeting_assistant.diarization import (
    DiarizationConfig,
    PyannoteBackend,
    diarize_audio,
    diarize_transcript,
)
from meeting_assistant.diarization.audio import ensure_unchanged, inspect_audio
from meeting_assistant.diarization.cache import REQUIRED_FILES, prepare_model, validate_model_cache
from meeting_assistant.diarization.exceptions import (
    DiarizationAccessError,
    DiarizationDeviceError,
    DiarizationInferenceError,
    DiarizationModelLoadError,
    DiarizationModelUnavailable,
    DiarizationOutOfMemory,
    InvalidDiarizationAudio,
    InvalidDiarizationResult,
    ReconciliationError,
)
from meeting_assistant.diarization.pyannote_backend import convert_turns
from meeting_assistant.diarization.service import _default_backend
from tests.audio.helpers import write_wav

from .helpers import diar, raw, word


def annotation(turns):
    return SimpleNamespace(
        itertracks=lambda **kwargs: iter(
            (SimpleNamespace(start=start, end=end), index, label)
            for index, (start, end, label) in enumerate(turns)
        )
    )


def output(regular=((0, 0.5, "arbitrary-label"),), exclusive=None):
    return SimpleNamespace(
        speaker_diarization=annotation(regular),
        exclusive_speaker_diarization=annotation(regular if exclusive is None else exclusive),
    )


class AudioBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_canonical_unicode_and_silence(self):
        path = write_wav(self.root / "会议 test.wav", silent=True)
        result = inspect_audio(path, 10)
        self.assertEqual(result.duration_seconds, 0.5)
        self.assertEqual(result.sha256, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_invalid_paths(self):
        for path in (self.root, self.root / "missing.wav"):
            with self.subTest(path=path), self.assertRaises(InvalidDiarizationAudio):
                inspect_audio(path, 10)

    def test_empty_and_corrupt_input(self):
        path = self.root / "invalid.wav"
        for content in (b"", b"not wave"):
            path.write_bytes(content)
            with self.assertRaises(InvalidDiarizationAudio):
                inspect_audio(path, 10)

    def test_noncanonical_and_zero_frames(self):
        for params in ({"rate": 44100}, {"channels": 2}, {"duration": 0}):
            with self.subTest(params=params), self.assertRaises(InvalidDiarizationAudio):
                inspect_audio(write_wav(self.root / "audio.wav", **params), 10)

    def test_truncated_wav(self):
        path = write_wav(self.root / "audio.wav")
        path.write_bytes(path.read_bytes()[:-20])
        with self.assertRaises(InvalidDiarizationAudio):
            inspect_audio(path, 10)

    def test_duration_limit(self):
        with self.assertRaises(InvalidDiarizationAudio):
            inspect_audio(write_wav(self.root / "audio.wav"), 0.1)

    def test_changed_audio(self):
        path = write_wav(self.root / "audio.wav")
        info = inspect_audio(path, 10)
        path.write_bytes(b"changed")
        with self.assertRaises(InvalidDiarizationAudio):
            ensure_unchanged(info)


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = write_wav(Path(self.temporary.name) / "audio.wav")
        self.pipeline = MagicMock(return_value=output())
        self.pipeline_class = MagicMock()
        self.pipeline_class.from_pretrained.return_value = self.pipeline
        self.torch = MagicMock()
        self.torch.cuda.is_available.return_value = True
        self.torch.cuda.max_memory_allocated.return_value = 1234
        self.torch.inference_mode.side_effect = contextlib.nullcontext
        self.runtime = patch(
            "meeting_assistant.diarization.pyannote_backend._load_runtime",
            return_value=(self.torch, self.pipeline_class, "4.0.7"),
        )
        self.runtime.start()
        self.addCleanup(self.runtime.stop)
        self.cache = patch(
            "meeting_assistant.diarization.pyannote_backend.validate_model_cache",
            return_value=Path("local-cache"),
        )
        self.cache.start()
        self.addCleanup(self.cache.stop)
        self.waveform = patch(
            "meeting_assistant.diarization.pyannote_backend.load_waveform", return_value="waveform"
        )
        self.waveform.start()
        self.addCleanup(self.waveform.stop)
        self.backend = PyannoteBackend(DiarizationConfig())

    def test_lazy_load_and_reuse(self):
        self.assertIsNone(self.backend.model_info)
        first = self.backend.diarize(self.path)
        second = self.backend.diarize(self.path)
        self.pipeline_class.from_pretrained.assert_called_once_with(
            Path("local-cache"), token=False
        )
        self.assertEqual(second.processing_info.model_load_seconds, 0)
        self.assertEqual(first.speakers, ("SPEAKER_00",))
        self.assertEqual(self.backend.model_info.device, "cuda")

    def test_explicit_cpu_no_gpu_calls(self):
        backend = PyannoteBackend(DiarizationConfig(device="cpu"))
        result = backend.diarize(self.path)
        self.assertIsNone(result.processing_info.peak_gpu_memory_bytes)
        self.torch.cuda.is_available.assert_not_called()
        self.torch.cuda.synchronize.assert_not_called()

    def test_cuda_unavailable_no_silent_fallback(self):
        self.torch.cuda.is_available.return_value = False
        with self.assertRaises(DiarizationDeviceError):
            self.backend.load_model()
        self.pipeline_class.from_pretrained.assert_not_called()

    def test_model_load_error_cause_and_sanitization(self):
        error = RuntimeError("private-token and private filesystem")
        self.pipeline_class.from_pretrained.side_effect = error
        with self.assertRaises(DiarizationModelLoadError) as caught:
            self.backend.load_model()
        self.assertIs(caught.exception.__cause__, error)
        self.assertNotIn("private-token", str(caught.exception))
        self.assertIsNone(self.backend.model_info)

    def test_failed_load_can_retry(self):
        self.pipeline_class.from_pretrained.side_effect = [RuntimeError("failed"), self.pipeline]
        with self.assertRaises(DiarizationModelLoadError):
            self.backend.load_model()
        self.backend.load_model()
        self.assertIsNotNone(self.backend.model_info)

    def test_conversion_output_malformed(self):
        self.pipeline.return_value = {"turns": []}
        with self.assertRaises(InvalidDiarizationResult):
            self.backend.diarize(self.path)

    def test_exclusive_overlaps_rejected(self):
        self.pipeline.return_value = output(((0, 0.4, "a"), (0.2, 0.5, "b")))
        with self.assertRaises(InvalidDiarizationResult):
            self.backend.diarize(self.path)

    def test_inference_error_translation(self):
        for error, kind in (
            (RuntimeError("CUDA out of memory"), DiarizationOutOfMemory),
            (MemoryError(), DiarizationOutOfMemory),
            (RuntimeError("CUDA driver failed"), DiarizationDeviceError),
            (RuntimeError("generic failure"), DiarizationInferenceError),
        ):
            self.pipeline.side_effect = error
            with self.subTest(error=error), self.assertRaises(kind):
                self.backend.diarize(self.path)

    def test_invalid_audio_does_not_load_model(self):
        with self.assertRaises(InvalidDiarizationAudio):
            self.backend.diarize(self.path.parent / "missing")
        self.pipeline_class.from_pretrained.assert_not_called()

    def test_concurrent_calls_share_one_model(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(self.backend.diarize, [self.path] * 8))
        self.assertEqual(len(results), 8)
        self.pipeline_class.from_pretrained.assert_called_once()
        self.assertEqual(sum(r.processing_info.model_load_seconds == 0 for r in results), 7)

    def test_stable_labels_first_observation(self):
        speakers, regular, exclusive = convert_turns(output(((0, 1, "Z"), (1, 2, "A"))))
        self.assertEqual(speakers, ("SPEAKER_00", "SPEAKER_01"))
        self.assertEqual(regular[0].speaker_id, "SPEAKER_00")
        self.assertEqual(exclusive[1].speaker_id, "SPEAKER_01")

    def test_empty_output_typed(self):
        self.pipeline.return_value = output(())
        self.assertEqual(self.backend.diarize(self.path).speakers, ())


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = DiarizationConfig(model_cache=self.root)
        self.hashes = {}
        for name in REQUIRED_FILES:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())
            self.hashes[name] = hashlib.sha256(name.encode()).hexdigest()
        (self.root / "source-manifest.json").write_text(
            json.dumps(
                {
                    "model": self.config.model,
                    "revision": self.config.revision,
                    "files": list(REQUIRED_FILES),
                }
            )
        )
        self.hash_patch = patch("meeting_assistant.diarization.cache.PINNED_HASHES", self.hashes)
        self.hash_patch.start()
        self.addCleanup(self.hash_patch.stop)

    def test_valid_offline_cache(self):
        self.assertEqual(validate_model_cache(self.config), self.root.resolve())
        self.assertEqual(prepare_model(self.config), self.root.resolve())

    def test_missing_offline_cache_no_network(self):
        (self.root / "segmentation/pytorch_model.bin").unlink()
        with self.assertRaises(DiarizationModelUnavailable):
            prepare_model(self.config)

    def test_corrupt_weights(self):
        (self.root / "embedding/pytorch_model.bin").write_bytes(b"corrupt")
        with self.assertRaises(DiarizationModelUnavailable):
            validate_model_cache(self.config)

    def test_wrong_revision(self):
        with self.assertRaises(DiarizationModelUnavailable):
            validate_model_cache(replace(self.config, revision="a" * 40))

    def test_corrupt_manifest(self):
        for content in ("nonsense", "null", "[]"):
            (self.root / "source-manifest.json").write_text(content)
            with self.assertRaises(DiarizationModelUnavailable):
                validate_model_cache(self.config)

    def test_missing_token_setup(self):
        (self.root / "source-manifest.json").unlink()
        with self.assertRaises(DiarizationAccessError):
            prepare_model(replace(self.config, offline_only=False))

    def test_setup_failure_cleans_lock_and_no_ready_manifest(self):
        (self.root / "source-manifest.json").unlink()
        fake_hub = SimpleNamespace(
            snapshot_download=MagicMock(side_effect=OSError("network failed")),
            constants=SimpleNamespace(HF_HUB_OFFLINE=False),
        )
        with patch.dict("sys.modules", {"huggingface_hub": fake_hub}):
            with self.assertRaises(DiarizationModelUnavailable):
                prepare_model(replace(self.config, offline_only=False, hf_token="secret"))
        self.assertFalse((self.root / ".acquisition.lock").exists())
        self.assertFalse((self.root / "source-manifest.json").exists())
        self.assertFalse(list(self.root.glob(".manifest-*")))


class ServiceTests(unittest.TestCase):
    def test_invalid_path_domain_error(self):
        with self.assertRaises(InvalidDiarizationAudio):
            diarize_audio(None, backend=MagicMock())

    def test_backend_injection(self):
        backend = SimpleNamespace(diarize=lambda path, options: diar())
        result = diarize_transcript("audio.wav", raw((word(),)), backend=backend)
        self.assertEqual(result.detected_speaker_count, 1)

    def test_bad_backend_result(self):
        with self.assertRaises(InvalidDiarizationResult):
            diarize_audio("audio.wav", backend=SimpleNamespace(diarize=lambda *args: {}))

    def test_mutually_exclusive_backend_config(self):
        with self.assertRaises(ReconciliationError):
            diarize_audio("audio.wav", backend=object(), config=DiarizationConfig())

    def test_invalid_transcript_before_inference(self):
        backend = MagicMock()
        with self.assertRaises(ReconciliationError):
            diarize_transcript("audio.wav", {}, backend=backend)
        backend.diarize.assert_not_called()

    def test_default_backend_reuse(self):
        _default_backend.cache_clear()
        with patch("meeting_assistant.diarization.service.PyannoteBackend") as factory:
            factory.return_value.diarize.return_value = diar()
            config = DiarizationConfig()
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(lambda _: diarize_audio("audio.wav", config=config), range(8)))
            factory.assert_called_once()
        _default_backend.cache_clear()
