import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

from meeting_assistant.asr.serialization import transcript_to_json
from meeting_assistant.diarization.__main__ import main
from meeting_assistant.diarization.exceptions import DiarizationModelUnavailable

from .helpers import diar, raw, word


class CLITests(unittest.TestCase):
    def test_saved_raw_workflow_no_asr_call(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "raw.json"
            source = transcript_to_json(raw((word(),)))
            path.write_text(source, encoding="utf-8")
            backend = MagicMock()
            backend.diarize.return_value = diar()
            with (
                patch(
                    "meeting_assistant.diarization.__main__.PyannoteBackend", return_value=backend
                ),
                patch("meeting_assistant.diarization.__main__.WhisperAPIBackend") as asr,
                redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(
                    main(
                        [
                            "canonical.wav",
                            "--canonical",
                            "--raw-transcript",
                            str(path),
                            "--env-file",
                            str(Path(folder) / "no-env"),
                            "--output-dir",
                            str(Path(folder) / "outputs"),
                        ]
                    ),
                    0,
                )
            asr.assert_not_called()
            self.assertEqual(path.read_text(encoding="utf-8"), source)

    def test_missing_model_clean_failure_no_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "raw.json"
            path.write_text(transcript_to_json(raw((word(),))))
            backend = MagicMock()
            backend.diarize.side_effect = DiarizationModelUnavailable("Prepare the local model.")
            output = Path(folder) / "outputs"
            with (
                patch(
                    "meeting_assistant.diarization.__main__.PyannoteBackend", return_value=backend
                ),
                redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(
                    main(
                        [
                            "canonical.wav",
                            "--canonical",
                            "--raw-transcript",
                            str(path),
                            "--output-dir",
                            str(output),
                        ]
                    ),
                    1,
                )
            self.assertFalse(output.exists())

    def test_download_only(self):
        with (
            patch(
                "meeting_assistant.diarization.__main__.prepare_model", return_value=Path("cache")
            ) as prepare,
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(main(["--prepare-model"]), 0)
        self.assertFalse(prepare.call_args.args[0].offline_only)

    def test_root_dispatch(self):
        from meeting_assistant.__main__ import main as root_main

        with patch("meeting_assistant.diarization.__main__.main", return_value=0) as implementation:
            self.assertEqual(root_main(["diarize", "test.wav"]), 0)
        implementation.assert_called_once_with(["test.wav"])
