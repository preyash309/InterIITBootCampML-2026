import os
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch

from meeting_assistant.audio import AudioIngestionConfig
from meeting_assistant.audio.exceptions import (
    ConfigurationError,
    MetadataExtractionFailed,
    NoAudioStream,
    UnsupportedMedia,
)
from meeting_assistant.audio.metadata import extract_metadata
from tests.audio.helpers import probe_data


class ConfigTests(unittest.TestCase):
    def test_defaults_and_immutability(self):
        config = AudioIngestionConfig()
        self.assertEqual(config.max_input_size_bytes, 4 * 1024**3)
        with self.assertRaises(FrozenInstanceError):
            config.work_dir = Path("changed")

    def test_environment(self):
        with patch.dict(
            os.environ,
            {
                "AUDIO_WORK_DIR": "test-jobs",
                "FFMPEG_PATH": "chosen-ffmpeg",
                "FFPROBE_PATH": "chosen-ffprobe",
                "AUDIO_MAX_INPUT_SIZE": "100",
                "AUDIO_FFMPEG_TIMEOUT": "12",
                "AUDIO_FFPROBE_TIMEOUT": "5",
                "AUDIO_MIN_DURATION": "0.2",
                "AUDIO_DURATION_TOLERANCE_SECONDS": "2",
                "AUDIO_DURATION_TOLERANCE_RATIO": "0.04",
            },
            clear=True,
        ):
            config = AudioIngestionConfig.from_env()
        self.assertEqual(config.work_dir, Path("test-jobs"))
        self.assertEqual(config.ffmpeg_path, "chosen-ffmpeg")
        self.assertEqual(config.ffprobe_path, "chosen-ffprobe")
        self.assertEqual(config.max_input_size_bytes, 100)
        self.assertEqual(config.ffmpeg_timeout_seconds, 12)
        self.assertEqual(config.ffprobe_timeout_seconds, 5)
        self.assertEqual(config.min_audio_duration_seconds, 0.2)
        self.assertEqual(config.duration_tolerance_seconds, 2)
        self.assertEqual(config.duration_tolerance_ratio, 0.04)

    def test_invalid_environment(self):
        for value in ("invalid", "nan", "inf", "0", "-1"):
            with (
                self.subTest(value=value),
                patch.dict(os.environ, {"AUDIO_FFMPEG_TIMEOUT": value}, clear=True),
            ):
                with self.assertRaises(ConfigurationError):
                    AudioIngestionConfig.from_env()

    def test_invalid_limits(self):
        for values in (
            {"max_input_size_bytes": 0},
            {"duration_tolerance_ratio": -1},
            {"min_audio_duration_seconds": float("nan")},
            {"ffmpeg_timeout_seconds": "invalid"},
            {"ffprobe_timeout_seconds": True},
            {"work_dir": None},
        ):
            with self.subTest(values=values), self.assertRaises(ConfigurationError):
                AudioIngestionConfig(**values)


class MetadataTests(unittest.TestCase):
    def test_complete_metadata(self):
        metadata = extract_metadata(probe_data(), 16044)
        self.assertEqual(metadata.codec, "pcm_s16le")
        self.assertEqual(metadata.sample_rate, 16000)
        self.assertEqual(metadata.channels, 1)
        self.assertEqual(metadata.duration_seconds, 0.5)
        self.assertEqual(metadata.duration_source, "stream")
        self.assertEqual(metadata.file_size_bytes, 16044)

    def test_optional_fields_absent(self):
        data = probe_data()
        for key in ("duration", "sample_rate", "channels", "bits_per_sample"):
            del data["streams"][0][key]
        del data["format"]["duration"]
        metadata = extract_metadata(data, 1)
        self.assertIsNone(metadata.duration_seconds)
        self.assertIsNone(metadata.sample_rate)
        self.assertIsNone(metadata.channels)
        self.assertEqual(metadata.duration_source, "unknown")

    def test_container_duration_fallback(self):
        data = probe_data()
        del data["streams"][0]["duration"]
        data["format"]["duration"] = "N/A"
        self.assertIsNone(extract_metadata(data, 1).duration_seconds)
        data["format"]["duration"] = "1.25"
        self.assertEqual(extract_metadata(data, 1).duration_source, "container")

    def test_time_base_duration(self):
        data = probe_data()
        stream = data["streams"][0]
        del stream["duration"]
        stream.update(duration_ts=8000, time_base="1/16000")
        self.assertEqual(extract_metadata(data, 1).duration_seconds, 0.5)
        for time_base in ("0/1", "1/0", "garbage"):
            stream["time_base"] = time_base
            with self.subTest(time_base=time_base), self.assertRaises(MetadataExtractionFailed):
                extract_metadata(data, 1)

    def test_default_track_selection(self):
        data = probe_data()
        second = dict(data["streams"][0], index=2, disposition={"default": 1})
        data["streams"].append(second)
        self.assertEqual(extract_metadata(data, 1).audio_stream_index, 2)

    def test_no_audio(self):
        data = probe_data()
        data["streams"] = [{"codec_type": "video"}]
        with self.assertRaises(NoAudioStream):
            extract_metadata(data, 1)

    def test_malformed_structure(self):
        for data in (
            {},
            {"streams": {}, "format": {}},
            {"streams": [None], "format": {}},
            {"streams": [], "format": {"format_name": 4}},
        ):
            with self.subTest(data=data), self.assertRaises(MetadataExtractionFailed):
                extract_metadata(data, 1)

    def test_invalid_numeric_fields(self):
        for key, value in (
            ("sample_rate", "nan"),
            ("channels", 0),
            ("index", None),
            ("channels", True),
            ("index", "1.5"),
            ("duration", -1),
            ("duration", {}),
        ):
            data = probe_data()
            data["streams"][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(MetadataExtractionFailed):
                extract_metadata(data, 1)

    def test_unidentified_codec(self):
        data = probe_data(codec="unknown")
        with self.assertRaises(UnsupportedMedia):
            extract_metadata(data, 1)
