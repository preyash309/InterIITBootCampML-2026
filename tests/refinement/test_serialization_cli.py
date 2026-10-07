import contextlib
import io
import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from meeting_assistant.__main__ import main as root_main
from meeting_assistant.diarization.serialization import speaker_transcript_to_json
from meeting_assistant.grounding.serialization import grounding_to_json
from meeting_assistant.refinement import (
    refine_transcript,
    refined_from_json,
    refined_to_json,
    render_refined_text,
    save_refined_transcript,
    validate_refined_source,
)
from meeting_assistant.refinement.__main__ import main
from meeting_assistant.refinement.exceptions import RefinementValidationError, RefinementWriteError
from meeting_assistant.refinement.models import ProviderCall

from .helpers import FakeBackend, evidence


class SerializationTests(unittest.TestCase):
    def setUp(self):
        self.source, self.g = evidence()
        self.result = refine_transcript(self.source, self.g, backend=FakeBackend("REPLACE"))

    def test_roundtrip(self):
        text = refined_to_json(self.result)
        self.assertEqual(refined_to_json(refined_from_json(text)), text)
        validate_refined_source(refined_from_json(text), self.source, self.g)

    def test_unicode_offsets_and_output_directory(self):
        source, grounding = evidence("Meet in café; use cue drant.")
        result = refine_transcript(source, grounding, backend=FakeBackend("REPLACE"))
        self.assertEqual(result.utterances[0].refined_text, "Meet in café; use Qdrant.")
        with TemporaryDirectory() as tmp:
            files = save_refined_transcript(result, Path(tmp) / "meeting notes 演示")
            payload = files.json_path.read_text(encoding="utf-8")
            self.assertIn("café", payload)
            self.assertEqual(refined_from_json(payload), result)

    def test_malformed_schema(self):
        for text in ("no", "[]", "{}", '{"schema_version":"9"}'):
            with self.subTest(text=text), self.assertRaises(RefinementValidationError):
                refined_from_json(text)

    def test_text_raw_and_refined_separate(self):
        self.assertEqual(self.result.utterances[0].raw_text, self.source.utterances[0].text)
        text = render_refined_text(self.result)
        self.assertIn("SPEAKER_00: Use Qdrant", text)
        self.assertIn("00:00:00.000 -->", text)
        self.assertNotIn("cue drant", text)

    def test_atomic_bundle(self):
        with TemporaryDirectory() as tmp:
            files = save_refined_transcript(self.result, tmp)
            self.assertEqual(
                {p.name for p in files.json_path.parent.iterdir()},
                {"refined_transcript.json", "refined_transcript.txt", "edit_log.json"},
            )
            self.assertEqual(refined_from_json(files.json_path.read_text()), self.result)
            self.assertEqual(
                json.loads(files.edit_log_path.read_text())["edits"][0]["candidate_entry_id"],
                self.g.records[0].candidates[0].entry_id,
            )
            self.assertFalse(any(p.name.startswith(".pending") for p in Path(tmp).iterdir()))

    def test_no_overwrite(self):
        with TemporaryDirectory() as tmp:
            files = save_refined_transcript(self.result, tmp)
            original = files.json_path.read_bytes()
            with self.assertRaises(RefinementWriteError):
                save_refined_transcript(self.result, tmp)
            self.assertEqual(files.json_path.read_bytes(), original)

    def test_failed_publish_cleanup(self):
        with (
            TemporaryDirectory() as tmp,
            patch(
                "meeting_assistant.refinement.serialization.os.rename",
                side_effect=OSError("disk problem"),
            ),
        ):
            with self.assertRaises(RefinementWriteError):
                save_refined_transcript(self.result, tmp)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_failed_write_cleanup(self):
        with (
            TemporaryDirectory() as tmp,
            patch(
                "meeting_assistant.refinement.serialization.os.fsync",
                side_effect=OSError("disk problem"),
            ),
        ):
            with self.assertRaises(RefinementWriteError):
                save_refined_transcript(self.result, tmp)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_upstream_bytes_unchanged(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            speaker = root / "speaker_transcript.json"
            ground = root / "grounding.json"
            speaker.write_text(speaker_transcript_to_json(self.source), encoding="utf-8")
            ground.write_text(grounding_to_json(self.g), encoding="utf-8")
            before = (speaker.read_bytes(), ground.read_bytes())
            save_refined_transcript(self.result, root)
            self.assertEqual((speaker.read_bytes(), ground.read_bytes()), before)

    def test_usage_nullable_not_fabricated(self):
        info = replace(
            self.result.processing_info,
            calls=(
                ProviderCall("utt_000001", 1, 1, 1, 200, total_tokens=10),
                ProviderCall("utt_000001", 1, 1, 1, 503),
            ),
        )
        self.assertIsNone(info.usage("total_tokens"))
        with self.assertRaises(ValueError):
            info.usage("cost")

    def test_usage_exact(self):
        info = replace(
            self.result.processing_info,
            calls=(ProviderCall("utt_000001", 1, 1, 1, 200, total_tokens=10),),
        )
        self.assertEqual(info.usage("total_tokens"), 10)


class CLITests(unittest.TestCase):
    def test_saved_evidence_workflow(self):
        source, g = evidence()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            speaker = root / "speaker.json"
            ground = root / "grounding.json"
            speaker.write_text(speaker_transcript_to_json(source), encoding="utf-8")
            ground.write_text(grounding_to_json(g), encoding="utf-8")
            before = (speaker.read_bytes(), ground.read_bytes())
            with (
                patch(
                    "meeting_assistant.refinement.groq_backend.GroqRefinerBackend",
                    return_value=FakeBackend("REPLACE"),
                ),
                contextlib.redirect_stdout(io.StringIO()) as output,
            ):
                code = main(
                    [
                        str(speaker),
                        "--grounding",
                        str(ground),
                        "--output-dir",
                        str(root / "outputs"),
                    ]
                )
            self.assertEqual(code, 0)
            self.assertIn("Applied: 1", output.getvalue())
            self.assertEqual((speaker.read_bytes(), ground.read_bytes()), before)

    def test_invalid_path_no_output(self):
        with TemporaryDirectory() as tmp, contextlib.redirect_stderr(io.StringIO()):
            code = main(
                [
                    str(Path(tmp) / "missing.json"),
                    "--grounding",
                    str(Path(tmp) / "missing2.json"),
                    "--output-dir",
                    str(Path(tmp) / "out"),
                ]
            )
            self.assertEqual(code, 1)
            self.assertFalse((Path(tmp) / "out").exists())

    def test_root_dispatch(self):
        with patch("meeting_assistant.refinement.__main__.main", return_value=0) as target:
            self.assertEqual(root_main(["refine", "meeting.mp4"]), 0)
        target.assert_called_once_with(["meeting.mp4"])

    def test_help(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(root_main(["--help"]), 0)
        self.assertIn("refine", output.getvalue())
