"""Real codecs/executables; skipped only when FFmpeg/ffprobe are unavailable."""

import json
import os
import subprocess
import tempfile
import unittest
import wave
from dataclasses import replace
from pathlib import Path

from meeting_assistant.audio import AudioIngestionConfig, ingest_audio
from meeting_assistant.audio.exceptions import (
    AudioIngestionError,
    CorruptMedia,
    FFmpegUnavailable,
    FFprobeUnavailable,
    InvalidCanonicalAudio,
    NoAudioStream,
    UnsupportedMedia,
)
from meeting_assistant.audio.ffmpeg import FFmpegTools
from tests.audio.helpers import write_wav


class FFmpegIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = AudioIngestionConfig.from_env()
        try:
            cls.tools = FFmpegTools(cls.settings)
        except (FFmpegUnavailable, FFprobeUnavailable) as exc:
            if os.environ.get("REQUIRE_FFMPEG_TESTS") == "1":
                raise
            raise unittest.SkipTest(str(exc)) from exc

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = replace(self.settings, work_dir=self.root / "jobs")

    def encode(self, *args):
        completed = subprocess.run(
            [
                self.tools.ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                *map(str, args),
            ],
            capture_output=True,
            text=True,
            timeout=20,
            shell=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def assert_canonical(self, result, duration=0.5):
        path = result.canonical_audio_path
        self.assertGreater(path.stat().st_size, 44)
        with wave.open(str(path), "rb") as wav:
            self.assertEqual(
                (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getcomptype()),
                (16000, 1, 2, "NONE"),
            )
            self.assertAlmostEqual(wav.getnframes() / wav.getframerate(), duration, delta=0.15)
        # Independent command, rather than the production probe/parser.
        completed = subprocess.run(
            [
                self.tools.ffprobe,
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
            shell=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        metadata = json.loads(completed.stdout)
        stream = metadata["streams"][0]
        self.assertEqual(metadata["format"]["format_name"], "wav")
        self.assertEqual(
            (
                stream["codec_name"],
                stream["sample_rate"],
                stream["channels"],
                stream["bits_per_sample"],
            ),
            ("pcm_s16le", "16000", 1, 16),
        )
        self.assertEqual(list(path.parent.iterdir()), [path])

    def test_mono_wav(self):
        self.assert_canonical(ingest_audio(write_wav(self.root / "mono.wav"), config=self.config))

    def test_stereo_and_non_16khz_wav(self):
        for rate, channels in ((16000, 2), (44100, 1), (48000, 2)):
            source = write_wav(self.root / f"{rate}-{channels}.wav", rate=rate, channels=channels)
            with self.subTest(rate=rate, channels=channels):
                result = ingest_audio(source, config=self.config)
                self.assertEqual(result.source_metadata.sample_rate, rate)
                self.assertEqual(result.source_metadata.channels, channels)
                self.assert_canonical(result)

    def test_spaces_unicode_and_hostile_filename(self):
        for name in (
            "meeting with spaces.wav",
            "बैठक 音声.wav",
            "$(echo injected) & meeting.wav",
            "-leading-option.wav",
        ):
            with self.subTest(name=name):
                self.assert_canonical(ingest_audio(write_wav(self.root / name), config=self.config))

    def test_valid_media_with_misleading_extension(self):
        self.assert_canonical(
            ingest_audio(write_wav(self.root / "recording.txt"), config=self.config)
        )

    def test_mp3(self):
        source = write_wav(self.root / "original.wav", rate=44100, channels=2)
        mp3 = self.root / "meeting.mp3"
        self.encode("-i", source, "-c:a", "libmp3lame", mp3)
        result = ingest_audio(mp3, config=self.config)
        self.assertEqual(result.source_metadata.codec, "mp3")
        self.assert_canonical(result)

    def test_additional_audio_containers(self):
        source = write_wav(self.root / "original.wav", rate=48000)
        for extension, codec in (
            ("m4a", "aac"),
            ("aac", "aac"),
            ("flac", "flac"),
            ("ogg", "libopus"),
            ("webm", "libopus"),
        ):
            destination = self.root / f"meeting.{extension}"
            with self.subTest(extension=extension):
                self.encode("-i", source, "-c:a", codec, destination)
                self.assert_canonical(ingest_audio(destination, config=self.config))

    def test_video_with_audio(self):
        source = write_wav(self.root / "original.wav", rate=44100)
        video = self.root / "meeting.mp4"
        self.encode(
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=32x32:r=10:d=0.5",
            "-i",
            source,
            "-c:v",
            "mpeg4",
            "-c:a",
            "aac",
            "-shortest",
            video,
        )
        self.assert_canonical(ingest_audio(video, config=self.config))

    def test_video_without_audio(self):
        video = self.root / "silent-video.mp4"
        self.encode("-f", "lavfi", "-i", "color=c=black:s=32x32:r=10:d=0.5", "-c:v", "mpeg4", video)
        with self.assertRaises(NoAudioStream):
            ingest_audio(video, config=self.config)
        self.assertFalse(self.config.work_dir.exists())

    def test_corrupt_and_unsupported_media(self):
        for name, expected in (("fake.wav", CorruptMedia), ("unknown.bin", UnsupportedMedia)):
            path = self.root / name
            path.write_text("random text pretending to be media")
            with self.subTest(name=name), self.assertRaises(expected):
                ingest_audio(path, config=self.config)
        self.assertFalse(self.config.work_dir.exists())

    def test_truncated_wav_rejected_and_cleaned(self):
        path = write_wav(self.root / "truncated.wav")
        path.write_bytes(path.read_bytes()[:1000])
        with self.assertRaises(AudioIngestionError):
            ingest_audio(path, config=self.config)
        self.assertEqual(list(self.root.rglob(".canonical-*")), [])
        self.assertEqual(list(self.root.rglob("canonical.wav")), [])
        self.assertEqual(list(self.root.rglob(".ingestion.lock")), [])
        self.assertTrue(path.exists())

    def test_silence_and_too_short(self):
        for name, options in (("silent.wav", {"silent": True}), ("short.wav", {"duration": 0.01})):
            path = write_wav(self.root / name, **options)
            with self.subTest(name=name), self.assertRaises(InvalidCanonicalAudio):
                ingest_audio(path, config=self.config)
        self.assertEqual(list(self.root.rglob(".canonical-*")), [])
        self.assertEqual(list(self.root.rglob("canonical.wav")), [])

    def test_playlist_rejected(self):
        source = write_wav(self.root / "private.wav")
        playlist = self.root / "meeting.ffconcat"
        playlist.write_text(f"ffconcat version 1.0\nfile '{source.as_posix()}'\n")
        with self.assertRaises(UnsupportedMedia):
            ingest_audio(playlist, config=self.config)

    def test_idempotent_real_conversion(self):
        source = write_wav(self.root / "original.wav")
        first = ingest_audio(source, config=self.config)
        before = first.canonical_audio_path.read_bytes()
        second = ingest_audio(source, config=self.config, job_id=first.job_id)
        self.assertEqual(first.canonical_audio_path, second.canonical_audio_path)
        self.assertEqual(before, second.canonical_audio_path.read_bytes())
