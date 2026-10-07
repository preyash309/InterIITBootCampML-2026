import json
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from meeting_assistant.audio.config import AudioIngestionConfig
from meeting_assistant.audio.exceptions import (
    AudioConversionFailed,
    ExternalCommandFailed,
    ExternalCommandTimeout,
    FFmpegUnavailable,
    FFprobeUnavailable,
    FileAccessError,
    MetadataExtractionFailed,
)
from meeting_assistant.audio.ffmpeg import FFmpegTools, resolve_executable, run_command
from tests.audio.helpers import probe_data


class CommandTests(unittest.TestCase):
    def test_resolve_path_or_explicit_override(self):
        with patch(
            "meeting_assistant.audio.ffmpeg.shutil.which",
            return_value=str(Path("ffmpeg.exe").absolute()),
        ) as which:
            self.assertTrue(Path(resolve_executable("ffmpeg", "/explicit/ffmpeg")).is_absolute())
            which.assert_called_once_with("/explicit/ffmpeg")

    def test_missing_binaries(self):
        with patch("meeting_assistant.audio.ffmpeg.shutil.which", return_value=None):
            with self.assertRaises(FFmpegUnavailable):
                resolve_executable("ffmpeg", None)
            with self.assertRaises(FFprobeUnavailable):
                resolve_executable("ffprobe", None)
            with self.assertRaises(FFmpegUnavailable):
                resolve_executable("ffmpeg", "invalid-explicit-override")

    def test_shell_free_list_and_timeout(self):
        command = ["ffmpeg", "-i", "spaces & $(not a command) 音.wav"]
        with patch(
            "meeting_assistant.audio.ffmpeg.subprocess.run",
            return_value=subprocess.CompletedProcess(command, 0, "ok", "diagnostic"),
        ) as run:
            run_command(command, timeout=10, tool="ffmpeg")
        self.assertEqual(run.call_args.args[0], command)
        self.assertFalse(run.call_args.kwargs["shell"])
        self.assertEqual(run.call_args.kwargs["timeout"], 10)
        self.assertEqual(run.call_args.kwargs["stdin"], subprocess.DEVNULL)

    def test_failed_command_retains_diagnostics(self):
        with patch(
            "meeting_assistant.audio.ffmpeg.subprocess.run",
            return_value=subprocess.CompletedProcess([], 2, "", "sensitive/full/path: invalid"),
        ):
            with self.assertRaises(ExternalCommandFailed) as caught:
                run_command(["ffprobe"], timeout=1, tool="ffprobe")
        self.assertEqual(caught.exception.returncode, 2)
        self.assertIn("sensitive/full/path", caught.exception.stderr)
        self.assertNotIn("sensitive/full/path", str(caught.exception))

    def test_timeout(self):
        with patch(
            "meeting_assistant.audio.ffmpeg.subprocess.run",
            side_effect=subprocess.TimeoutExpired(["ffmpeg"], 1, stderr=b"partial diagnostics"),
        ):
            with self.assertRaises(ExternalCommandTimeout) as caught:
                run_command(["ffmpeg"], timeout=1, tool="ffmpeg")
        self.assertEqual(caught.exception.stderr, "partial diagnostics")
        self.assertIsInstance(caught.exception.__cause__, subprocess.TimeoutExpired)

    def test_start_failures(self):
        for tool, expected in (("ffmpeg", FFmpegUnavailable), ("ffprobe", FFprobeUnavailable)):
            with (
                patch(
                    "meeting_assistant.audio.ffmpeg.subprocess.run", side_effect=FileNotFoundError
                ),
                self.assertRaises(expected),
            ):
                run_command([tool], timeout=1, tool=tool)
        with (
            patch("meeting_assistant.audio.ffmpeg.subprocess.run", side_effect=PermissionError),
            self.assertRaises(FileAccessError),
        ):
            run_command(["ffmpeg"], timeout=1, tool="ffmpeg")


class ToolTests(unittest.TestCase):
    def setUp(self):
        with patch(
            "meeting_assistant.audio.ffmpeg.resolve_executable",
            side_effect=lambda name, override: name,
        ):
            self.tools = FFmpegTools(AudioIngestionConfig())

    def test_probe_json(self):
        with patch(
            "meeting_assistant.audio.ffmpeg.run_command",
            return_value=subprocess.CompletedProcess([], 0, json.dumps(probe_data()), ""),
        ) as run:
            data = self.tools.probe(Path("recording.wav"))
        self.assertEqual(data["streams"][0]["codec_name"], "pcm_s16le")
        command = run.call_args.args[0]
        self.assertIn("json", command)
        self.assertIn("-format_whitelist", command)

    def test_malformed_probe_response(self):
        for stdout in ("not json", "[]", "null"):
            with (
                self.subTest(stdout=stdout),
                patch(
                    "meeting_assistant.audio.ffmpeg.run_command",
                    return_value=subprocess.CompletedProcess([], 0, stdout, ""),
                ),
                self.assertRaises(MetadataExtractionFailed),
            ):
                self.tools.probe(Path("test.wav"))

    def test_canonical_command(self):
        with patch("meeting_assistant.audio.ffmpeg.run_command") as run:
            self.tools.convert(Path("input.wav"), Path("temp.wav"), 2)
        command = run.call_args.args[0]
        for flag, value in (
            ("-ac", "1"),
            ("-ar", "16000"),
            ("-c:a", "pcm_s16le"),
            ("-f", "wav"),
            ("-map", "0:2"),
        ):
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertNotIn("-af", command)
        self.assertIn("-xerror", command)

    def test_conversion_failure(self):
        error = ExternalCommandFailed("failed", stderr="full diagnostics", returncode=1)
        with (
            patch("meeting_assistant.audio.ffmpeg.run_command", side_effect=error),
            self.assertRaises(AudioConversionFailed) as caught,
        ):
            self.tools.convert(Path("input.wav"), Path("temp.wav"), 0)
        self.assertEqual(caught.exception.stderr, "full diagnostics")
        self.assertIs(caught.exception.__cause__, error)
