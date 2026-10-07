import io
import stat
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from meeting_assistant.audio import AudioIngestionConfig, AudioIngestionResult, ingest_audio
from meeting_assistant.audio.__main__ import main
from meeting_assistant.audio.exceptions import (
    AudioConversionFailed,
    CorruptMedia,
    EmptyInputFile,
    ExternalCommandFailed,
    ExternalCommandTimeout,
    FileAccessError,
    InputFileNotFound,
    InputTooLarge,
    InvalidCanonicalAudio,
    InvalidInputFile,
    InvalidJobId,
    JobBusy,
    MetadataExtractionFailed,
    NoAudioStream,
    UnsupportedMedia,
)
from meeting_assistant.audio.ingestion import _is_link
from meeting_assistant.audio.metadata import extract_metadata
from meeting_assistant.audio.validation import validate_canonical
from tests.audio.helpers import probe_data, write_wav


class IngestionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = write_wav(self.root / "source.wav")
        self.config = AudioIngestionConfig(work_dir=self.root / "jobs")
        self.patcher = patch("meeting_assistant.audio.ingestion.FFmpegTools")
        self.tools = self.patcher.start().return_value
        self.addCleanup(self.patcher.stop)
        self.tools.probe.side_effect = lambda path: probe_data()
        self.tools.convert.side_effect = lambda source, dest, index: write_wav(dest)

    def test_success_typed_path_contract_and_events(self):
        with self.assertLogs("meeting_assistant.audio.ingestion", level="INFO") as logs:
            result = ingest_audio(self.source, config=self.config)
        self.assertIsInstance(result, AudioIngestionResult)
        self.assertIsInstance(result.canonical_audio_path, Path)
        self.assertEqual(result.status, "success")
        self.assertEqual(
            result.canonical_audio_path,
            self.config.work_dir / result.job_id / "audio" / "canonical.wav",
        )
        self.assertTrue(self.source.exists())
        self.assertEqual(
            list(result.canonical_audio_path.parent.iterdir()), [result.canonical_audio_path]
        )
        self.assertFalse((result.canonical_audio_path.parent.parent / ".ingestion.lock").exists())
        for event in (
            "ingestion_started",
            "source_validated",
            "metadata_extracted",
            "conversion_started",
            "conversion_completed",
            "canonical_validation_passed",
        ):
            self.assertIn(event, [record.event for record in logs.records])

    def test_file_level_failures_before_tools(self):
        zero = self.root / "zero.wav"
        zero.touch()
        cases = (
            (self.root / "missing.wav", InputFileNotFound),
            (zero, EmptyInputFile),
            (self.root, InvalidInputFile),
        )
        for path, expected in cases:
            with self.subTest(path=path), self.assertRaises(expected):
                ingest_audio(path, config=self.config)
        self.tools.probe.assert_not_called()
        self.assertFalse(self.config.work_dir.exists())

    def test_input_size_limit(self):
        with self.assertRaises(InputTooLarge):
            ingest_audio(self.source, config=replace(self.config, max_input_size_bytes=10))

    def test_permission_error_sanitized(self):
        with (
            patch(
                "meeting_assistant.audio.ingestion.validate_input",
                side_effect=PermissionError("private/path"),
            ),
            self.assertRaises(FileAccessError) as caught,
        ):
            ingest_audio(self.source, config=self.config)
        self.assertNotIn("private/path", str(caught.exception))
        self.assertIsInstance(caught.exception.__cause__, PermissionError)

    def test_invalid_probe_and_unknown_media(self):
        self.tools.probe.side_effect = ExternalCommandFailed(
            "failed", stderr="details", returncode=1
        )
        with self.assertRaises(CorruptMedia):
            ingest_audio(self.source, config=self.config)
        unknown = self.root / "random.bin"
        unknown.write_text("not media")
        with self.assertRaises(UnsupportedMedia):
            ingest_audio(unknown, config=self.config)

    def test_no_audio_and_malformed_metadata(self):
        self.tools.probe.side_effect = None
        self.tools.probe.return_value = {
            "streams": [{"codec_type": "video"}],
            "format": {"format_name": "mov"},
        }
        with self.assertRaises(NoAudioStream):
            ingest_audio(self.source, config=self.config)
        self.tools.probe.return_value = {}
        with self.assertRaises(MetadataExtractionFailed):
            ingest_audio(self.source, config=self.config)

    def test_safe_job_id(self):
        for job in ("../../escape", "C:/escape", "not-a-uuid"):
            with self.subTest(job=job), self.assertRaises(InvalidJobId):
                ingest_audio(self.source, config=self.config, job_id=job)
        chosen = uuid4()
        result = ingest_audio(self.source, config=self.config, job_id=chosen.hex)
        self.assertEqual(result.job_id, str(chosen))

    def test_job_busy_preserves_lock(self):
        chosen = str(uuid4())
        job = self.config.work_dir / chosen
        job.mkdir(parents=True)
        lock = job / ".ingestion.lock"
        lock.touch()
        with self.assertRaises(JobBusy):
            ingest_audio(self.source, config=self.config, job_id=chosen)
        self.assertTrue(lock.exists())

    def test_repeat_and_failed_retry_preserves_valid_artifact(self):
        result = ingest_audio(self.source, config=self.config)
        before = result.canonical_audio_path.read_bytes()
        repeated = ingest_audio(self.source, config=self.config, job_id=result.job_id)
        self.assertEqual(repeated.canonical_audio_path, result.canonical_audio_path)
        self.assertEqual(repeated.canonical_audio_path.read_bytes(), before)

        def fail(source, destination, index):
            destination.write_bytes(b"partial output")
            raise AudioConversionFailed("conversion failed")

        self.tools.convert.side_effect = fail
        with self.assertRaises(AudioConversionFailed):
            ingest_audio(self.source, config=self.config, job_id=result.job_id)
        self.assertEqual(result.canonical_audio_path.read_bytes(), before)
        self.assertEqual(
            list(result.canonical_audio_path.parent.iterdir()), [result.canonical_audio_path]
        )
        self.assertFalse((result.canonical_audio_path.parent.parent / ".ingestion.lock").exists())

    def test_missing_and_zero_output_cleanup(self):
        for delete in (False, True):

            def no_output(source, destination, index):
                if delete:
                    destination.unlink()

            self.tools.convert.side_effect = no_output
            chosen = str(uuid4())
            with self.subTest(delete=delete), self.assertRaises(InvalidCanonicalAudio):
                ingest_audio(self.source, config=self.config, job_id=chosen)
            self.assertEqual(list((self.config.work_dir / chosen / "audio").iterdir()), [])

    def test_invalid_output_and_probe_failure_cleanup(self):
        self.tools.convert.side_effect = lambda source, dest, index: dest.write_bytes(b"bad WAV")
        chosen = str(uuid4())
        with self.assertRaises(InvalidCanonicalAudio):
            ingest_audio(self.source, config=self.config, job_id=chosen)
        self.assertEqual(list((self.config.work_dir / chosen / "audio").iterdir()), [])
        self.tools.convert.side_effect = lambda source, dest, index: write_wav(dest)
        self.tools.probe.side_effect = [probe_data(), ExternalCommandFailed("bad output")]
        with self.assertRaises(InvalidCanonicalAudio):
            ingest_audio(self.source, config=self.config, job_id=chosen)
        self.assertEqual(list((self.config.work_dir / chosen / "audio").iterdir()), [])

    def test_post_conversion_timeout_remains_distinguishable(self):
        chosen = str(uuid4())
        self.tools.probe.side_effect = [probe_data(), ExternalCommandTimeout("ffprobe timed out")]
        with self.assertRaises(ExternalCommandTimeout):
            ingest_audio(self.source, config=self.config, job_id=chosen)
        self.assertEqual(list((self.config.work_dir / chosen / "audio").iterdir()), [])
        self.assertFalse((self.config.work_dir / chosen / ".ingestion.lock").exists())

    def test_source_cannot_be_destination(self):
        result = ingest_audio(self.source, config=self.config)
        before = result.canonical_audio_path.read_bytes()
        with self.assertRaises(InvalidInputFile):
            ingest_audio(result.canonical_audio_path, config=self.config, job_id=result.job_id)
        self.assertEqual(result.canonical_audio_path.read_bytes(), before)

    def test_symlink_job_is_rejected(self):
        chosen = str(uuid4())
        self.config.work_dir.mkdir()
        link = self.config.work_dir / chosen
        outside = self.root / "outside"
        outside.mkdir()
        try:
            link.symlink_to(outside, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"Symlink creation unavailable: {type(exc).__name__}")
        self.addCleanup(link.unlink)
        with self.assertRaises(FileAccessError):
            ingest_audio(self.source, config=self.config, job_id=chosen)
        self.assertEqual(list(outside.iterdir()), [])

    def test_atomic_publish_failure_cleans_temp(self):
        chosen = str(uuid4())
        with (
            patch("meeting_assistant.audio.ingestion.os.replace", side_effect=PermissionError),
            self.assertRaises(FileAccessError),
        ):
            ingest_audio(self.source, config=self.config, job_id=chosen)
        self.assertEqual(list((self.config.work_dir / chosen / "audio").iterdir()), [])

    def test_links_in_audio_directory_or_destination_are_rejected(self):
        chosen = str(uuid4())
        job = self.config.work_dir / chosen
        audio = job / "audio"
        canonical = audio / "canonical.wav"
        for unsafe in (audio, canonical):
            with (
                self.subTest(unsafe=unsafe),
                patch(
                    "meeting_assistant.audio.ingestion._is_link",
                    side_effect=lambda path: path == unsafe,
                ),
                self.assertRaises(FileAccessError),
            ):
                ingest_audio(self.source, config=self.config, job_id=chosen)
            self.assertFalse((job / ".ingestion.lock").exists())
            self.tools.convert.assert_not_called()

    def test_hardlinked_source_cannot_be_overwritten(self):
        result = ingest_audio(self.source, config=self.config)
        linked = self.root / "linked-source.wav"
        try:
            linked.hardlink_to(result.canonical_audio_path)
        except OSError as exc:
            self.skipTest(f"Hard links unavailable: {type(exc).__name__}")
        before = linked.read_bytes()
        with self.assertRaises(InvalidInputFile):
            ingest_audio(linked, config=self.config, job_id=result.job_id)
        self.assertEqual(linked.read_bytes(), before)

    def test_windows_reparse_point_detection(self):
        attributes = SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
        with (
            patch.object(Path, "lstat", return_value=attributes),
            patch(
                "meeting_assistant.audio.ingestion.stat.FILE_ATTRIBUTE_REPARSE_POINT",
                0x400,
                create=True,
            ),
        ):
            self.assertTrue(_is_link(self.root))

    def test_lock_cleanup_failure_is_reported(self):
        original_unlink = Path.unlink

        def deny_lock_removal(path, *args, **kwargs):
            if path.name == ".ingestion.lock":
                raise PermissionError("cannot remove lock")
            return original_unlink(path, *args, **kwargs)

        with (
            patch.object(Path, "unlink", deny_lock_removal),
            self.assertRaises(FileAccessError) as caught,
        ):
            ingest_audio(self.source, config=self.config)
        self.assertIsInstance(caught.exception.__cause__, PermissionError)

    def test_cli_success_and_failure(self):
        result = ingest_audio(self.source, config=self.config)
        output = io.StringIO()
        with (
            patch("meeting_assistant.audio.__main__.ingest_audio", return_value=result),
            redirect_stdout(output),
        ):
            self.assertEqual(main([str(self.source)]), 0)
        self.assertIn("Audio ingestion succeeded", output.getvalue())
        self.assertIn("pcm_s16le", output.getvalue())
        error_output = io.StringIO()
        with (
            patch(
                "meeting_assistant.audio.__main__.ingest_audio",
                side_effect=CorruptMedia("Damaged recording."),
            ),
            redirect_stderr(error_output),
        ):
            self.assertEqual(main([str(self.source)]), 1)
        self.assertIn("corrupt_media", error_output.getvalue())
        self.assertNotIn("Traceback", error_output.getvalue())


class CanonicalValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = write_wav(Path(self.temp.name) / "canonical.wav")
        self.metadata = extract_metadata(probe_data(), self.path.stat().st_size)
        self.config = AudioIngestionConfig()

    def test_wrong_canonical_properties(self):
        for fields in (
            {"codec": "aac"},
            {"sample_rate": 8000},
            {"channels": 2},
            {"bits_per_sample": 8},
            {"container_format": "mp3"},
            {"duration_seconds": None},
            {"duration_seconds": 0.01},
        ):
            with self.subTest(fields=fields), self.assertRaises(InvalidCanonicalAudio):
                validate_canonical(
                    self.path, replace(self.metadata, **fields), self.metadata, self.config
                )

    def test_header_truncation(self):
        data = self.path.read_bytes()
        self.path.write_bytes(data[:-100])
        with self.assertRaises(InvalidCanonicalAudio):
            validate_canonical(self.path, self.metadata, self.metadata, self.config)

    def test_digital_silence(self):
        write_wav(self.path, silent=True)
        with self.assertRaises(InvalidCanonicalAudio):
            validate_canonical(self.path, self.metadata, self.metadata, self.config)

    def test_duration_consistency_and_tolerance(self):
        validate_canonical(
            self.path, self.metadata, replace(self.metadata, duration_seconds=0.6), self.config
        )
        with self.assertRaises(InvalidCanonicalAudio):
            validate_canonical(
                self.path, self.metadata, replace(self.metadata, duration_seconds=30), self.config
            )

    def test_video_container_duration_is_not_audio_duration(self):
        source = replace(
            self.metadata,
            container_format="mov,mp4,m4a,3gp,3g2,mj2",
            duration_seconds=30,
            duration_source="container",
        )
        validate_canonical(self.path, self.metadata, source, self.config)

    def test_probe_header_duration_disagreement(self):
        with self.assertRaises(InvalidCanonicalAudio):
            validate_canonical(
                self.path, replace(self.metadata, duration_seconds=0.6), self.metadata, self.config
            )
